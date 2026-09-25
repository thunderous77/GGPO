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

"""Off-policy AGRO Eq. (4) loss."""

from __future__ import annotations

import math
from typing import Any

import torch


def _validate_agro_inputs(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    beta: float,
    group_size: int,
    global_batch_size: int,
    dp_size: int,
) -> None:
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError(f"AGRO beta must be positive and finite, got {beta}.")
    if group_size <= 1:
        raise ValueError(f"AGRO group_size must be greater than 1, got {group_size}.")
    if dp_size <= 0:
        raise ValueError(f"AGRO dp_size must be positive, got {dp_size}.")
    if global_batch_size <= 0 or global_batch_size % group_size:
        raise ValueError("AGRO global_batch_size must be positive and divisible by group_size.")
    if log_prob.ndim != 2:
        raise ValueError("AGRO log_prob must have shape [batch, response_length].")
    if ref_log_prob.shape != log_prob.shape or response_mask.shape != log_prob.shape:
        raise ValueError("AGRO log_prob, ref_log_prob, and response_mask shapes must be identical.")
    batch_size = log_prob.shape[0]
    if rewards.shape != (batch_size,) or group_ids.shape != (batch_size,):
        raise ValueError("AGRO rewards and group_ids must contain one value per response.")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("AGRO response_mask must be binary.")
    if not response_mask.any(dim=-1).all():
        raise ValueError("Every AGRO response must contain at least one valid token.")
    if not torch.isfinite(log_prob).all():
        raise FloatingPointError("AGRO current-policy log probabilities are not finite.")
    if not torch.isfinite(ref_log_prob).all():
        raise FloatingPointError("AGRO reference-policy log probabilities are not finite.")
    if not torch.isfinite(rewards).all():
        raise FloatingPointError("AGRO rewards are not finite.")
    if group_ids.is_floating_point() and (
        not torch.isfinite(group_ids).all() or not torch.equal(group_ids, group_ids.trunc())
    ):
        raise ValueError("AGRO group_ids must contain finite integers.")
    _, counts = torch.unique(group_ids.long(), return_counts=True)
    if counts.numel() == 0 or not torch.all(counts == group_size):
        raise ValueError(f"Every AGRO group must contain exactly group_size={group_size} responses.")


def compute_agro_loss(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    beta: float,
    group_size: int,
    global_batch_size: int,
    batch_num_tokens: int,
    dp_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the off-policy AGRO Eq. (4) surrogate divided by beta.

    Response log probabilities are always summed. The leave-one-out delta is
    detached, so the only differentiated factor is the current-policy
    sequence log probability. No PPO ratio, clipping, or importance sampling
    appears in this objective. The outer gradient beta is removed; beta still
    controls the reference penalty in the regularized rewards. The final
    gradient is averaged over the global valid-token count, as in GRPO.
    """
    response_mask = response_mask.detach()
    rewards = rewards.detach().float()
    ref_log_prob = ref_log_prob.detach().float()
    group_ids = group_ids.detach()
    _validate_agro_inputs(
        log_prob,
        ref_log_prob,
        rewards,
        response_mask,
        group_ids,
        beta=beta,
        group_size=group_size,
        global_batch_size=global_batch_size,
        dp_size=dp_size,
    )

    mask = response_mask.bool()
    current_sequence_log_prob = (log_prob.float() * mask).sum(dim=-1)
    reference_sequence_log_prob = (ref_log_prob * mask).sum(dim=-1)
    regularized_rewards = rewards - beta * (current_sequence_log_prob - reference_sequence_log_prob)

    order = torch.argsort(group_ids.long(), stable=True)
    grouped_rewards = regularized_rewards[order].reshape(-1, group_size)
    grouped_current_log_prob = current_sequence_log_prob[order].reshape(-1, group_size)
    group_mean = grouped_rewards.mean(dim=-1, keepdim=True)
    loo_mean = (grouped_rewards.sum(dim=-1, keepdim=True) - grouped_rewards) / (group_size - 1)
    loo_delta = grouped_rewards - loo_mean

    per_group_surrogate = -(loo_delta.detach() * grouped_current_log_prob).mean(dim=-1)
    per_group_surrogate = per_group_surrogate * (global_batch_size / batch_num_tokens)
    global_group_count = global_batch_size // group_size
    loss = per_group_surrogate.sum() * dp_size / global_group_count
    if not torch.isfinite(loss):
        raise FloatingPointError("AGRO loss is not finite.")

    centered = grouped_rewards - group_mean
    sample_variance_objective = centered.square().sum(dim=-1) / (2 * (group_size - 1))
    detached_rewards = grouped_rewards.detach().float()
    detached_delta = loo_delta.detach().float()
    log_ratio = (current_sequence_log_prob.detach() - reference_sequence_log_prob).float()
    metrics = {
        "agro_surrogate_loss": per_group_surrogate.detach().mean(),
        "agro_sample_variance": sample_variance_objective.detach().mean(),
        "agro_regularized_reward_mean": detached_rewards.mean(),
        "agro_regularized_reward_min": detached_rewards.min(),
        "agro_regularized_reward_max": detached_rewards.max(),
        "agro_loo_delta_mean": detached_delta.mean(),
        "agro_loo_delta_abs_mean": detached_delta.abs().mean(),
        "agro_loo_delta_abs_max": detached_delta.abs().max(),
        "agro_sequence_log_ratio_mean": log_ratio.mean(),
        "agro_sequence_log_ratio_abs_max": log_ratio.abs().max(),
        "agro_reference_sequence_log_prob_mean": reference_sequence_log_prob.detach().float().mean(),
    }
    return loss, metrics
