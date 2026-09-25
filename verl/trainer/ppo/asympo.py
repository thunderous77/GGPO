# Copyright 2026 OpenAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core ASymPO Eq. (19) loss."""

from __future__ import annotations

import math
from typing import Any

import torch


def compute_asympo_loss(
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    global_batch_size: int | None = None,
    dp_size: int = 1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute response-equal ASymPO using FP32 response NLL scales.

    ``advantages`` must contain one raw-reward, mean-centered response
    advantage repeated over that response's valid tokens. The denominator is
    the exact detached response NLL: no clipping or numerical floor is used.
    """
    if log_prob.ndim != 2:
        raise ValueError("ASymPO log_prob must have shape [batch, response_length].")
    if advantages.shape != log_prob.shape or response_mask.shape != log_prob.shape:
        raise ValueError("ASymPO log_prob, advantages, and response_mask shapes must be identical.")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("ASymPO response_mask must be binary.")

    local_batch_size = log_prob.shape[0]
    if global_batch_size is None:
        global_batch_size = local_batch_size
    if isinstance(global_batch_size, bool) or not isinstance(global_batch_size, int):
        raise ValueError("ASymPO global_batch_size must be an integer.")
    if isinstance(dp_size, bool) or not isinstance(dp_size, int):
        raise ValueError("ASymPO dp_size must be an integer.")
    if global_batch_size <= 0 or dp_size <= 0 or local_batch_size > global_batch_size:
        raise ValueError("ASymPO requires positive, consistent global_batch_size and dp_size values.")
    if not math.isfinite(float(global_batch_size)) or not math.isfinite(float(dp_size)):
        raise ValueError("ASymPO global batch geometry must be finite.")

    mask = response_mask.detach().bool()
    lengths = mask.sum(dim=-1)
    if not torch.all(lengths > 0):
        raise ValueError("Every ASymPO response must contain at least one valid token.")
    if not torch.isfinite(log_prob).all():
        raise FloatingPointError("ASymPO current-policy log probabilities are not finite.")
    if not torch.isfinite(advantages).all():
        raise FloatingPointError("ASymPO advantages are not finite.")

    log_prob_fp32 = log_prob.float()
    advantages_fp32 = advantages.detach().float()
    lengths_fp32 = lengths.float()
    response_advantages = (advantages_fp32 * mask).sum(dim=-1) / lengths_fp32
    expanded_advantages = response_advantages.unsqueeze(-1).expand_as(advantages_fp32)
    if not torch.equal(advantages_fp32[mask], expanded_advantages[mask]):
        raise ValueError("ASymPO requires one response-level advantage repeated across all valid tokens.")

    response_nll = -(log_prob_fp32 * mask).sum(dim=-1) / lengths_fp32
    if not torch.isfinite(response_nll).all():
        raise FloatingPointError("ASymPO response NLL is not finite.")
    if not torch.all(response_nll > 0):
        raise ValueError("Every ASymPO response NLL scale must be strictly positive.")

    inverse_scale = response_nll.detach().reciprocal()
    contributions = response_advantages * response_nll / response_nll.detach()
    # The engine sums micro-batch losses, then averages them over DP ranks.
    # Scale each local response contribution so that the accumulated gradient
    # is exactly the response-equal global mean from Eq. (19).
    loss = contributions.sum() * dp_size / global_batch_size
    if not torch.isfinite(loss):
        raise FloatingPointError("ASymPO loss is not finite.")

    positive = response_advantages > 0
    negative = response_advantages < 0
    detached_contributions = contributions.detach()
    metrics = {
        # Scientific metrics are combined separately with response-count
        # weights, so expose the local response mean rather than the
        # engine-scaled backward scalar.
        "asympo_loss": detached_contributions.mean(),
        "asympo_response_nll_mean": response_nll.detach().mean(),
        "asympo_response_nll_min": response_nll.detach().min(),
        "asympo_response_nll_max": response_nll.detach().max(),
        "asympo_inverse_scale_mean": inverse_scale.mean(),
        "asympo_inverse_scale_max": inverse_scale.max(),
        # Normalize both signed masses by every local response so response-count
        # weighted metric aggregation remains exact across arbitrary microbatch
        # and DP splits. The two metrics therefore sum to ``asympo_loss``.
        "asympo_positive_contribution": (detached_contributions * positive).sum() / local_batch_size,
        "asympo_negative_contribution": (detached_contributions * negative).sum() / local_batch_size,
        "asympo_gradient_coefficient_abs_mean": (response_advantages * inverse_scale).abs().mean(),
    }
    return loss, metrics
