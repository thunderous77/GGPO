# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Pure group-level GGPO objective."""

from __future__ import annotations

import math
import operator
from typing import Any, NamedTuple

import torch


# Private batch keys used by the fully-async, memory-safe GGPO actor update.
# The current-policy tensor exists only between the diagnostic pass and the
# actor update. It is consumed on the driver and never sent to actor workers.
GGPO_SURROGATE_CURRENT_LOG_PROBS = "_ggpo_memory_safe_current_log_probs"
GGPO_SURROGATE_WEIGHTS = "_ggpo_memory_safe_surrogate_weights"
GGPO_SURROGATE_METRICS_META = "_ggpo_memory_safe_surrogate_metrics"
GGPO_RAW_CORRECTNESS = "_ggpo_raw_correctness"
POLICY_GROUP_IDS = "policy_group_ids"


# One authoritative schema for the direct loss path and loss-masked padding
# records. The tied-max family is intentionally optional because shaped
# rewards do not support its binary-reward KL tie-break interpretation.
GGPO_BASE_METRIC_NAMES = (
    "ggpo_loss",
    "ggpo_raw_loss",
    "ggpo_normalized_loss_to_raw_loss_ratio",
    "ggpo_global_valid_token_count",
    "ggpo_batch_mean_response_length",
    "ggpo_adjusted_reward_mean",
    "ggpo_adjusted_reward_mean_microbatch_std",
    "ggpo_adjusted_reward_min",
    "ggpo_adjusted_reward_mean_microbatch_p05",
    "ggpo_adjusted_reward_mean_microbatch_p50",
    "ggpo_adjusted_reward_mean_microbatch_p95",
    "ggpo_adjusted_reward_max",
    "ggpo_all_zero_group_frac",
    "ggpo_all_one_group_frac",
    "ggpo_informative_group_frac",
    "ggpo_sequence_log_ratio_mean",
    "ggpo_sequence_log_ratio_mean_microbatch_std",
    "ggpo_sequence_log_ratio_abs_max",
    "ggpo_raw_sequence_log_ratio_mean",
    "ggpo_raw_sequence_log_ratio_mean_microbatch_std",
    "ggpo_raw_sequence_log_ratio_abs_max",
    "ggpo_const_group_coefficient_rms",
    "ggpo_informative_group_coefficient_rms",
    "ggpo_const_group_squared_mass_share",
    "ggpo_const_group_kl_alignment",
    "ggpo_q_entropy",
    "ggpo_q_max",
    "ggpo_q_ess",
    "ggpo_q_normalized_ess",
    "ggpo_normalized_coefficient_rms",
    "ggpo_q_loss_normalization_max_abs_diff",
    "ggpo_q_unchanged_after_loss_normalization",
)
GGPO_TIED_MAX_REWARD_SUBSET_METRIC_NAMES = (
    "ggpo_mean_tied_max_reward_k_in_informative_groups",
    "ggpo_informative_group_frac_tied_max_reward_k_ge_2",
    "ggpo_tied_max_reward_subset_delta_spread",
    "ggpo_tied_max_reward_subset_ess_ratio",
    "ggpo_tied_max_reward_subset_delta_weighted_mean",
    "ggpo_tied_max_reward_subset_delta_spread_over_all_groups",
    "ggpo_tied_max_reward_subset_ess_ratio_over_all_groups",
    "ggpo_tied_max_reward_subset_delta_weighted_mean_over_all_groups",
    "ggpo_tied_max_reward_subset_group_count_mean",
    "ggpo_groups_per_microbatch",
    "ggpo_tied_max_reward_subset_group_frac",
)
GGPO_METRIC_NAMES = GGPO_BASE_METRIC_NAMES + GGPO_TIED_MAX_REWARD_SUBSET_METRIC_NAMES

# Conditional means use the selected group count, not the enclosing
# micro-batch group count, when worker metrics are combined.
GGPO_CONDITIONAL_METRIC_FRACTIONS = {
    "ggpo_tied_max_reward_subset_delta_spread": "ggpo_tied_max_reward_subset_group_frac",
    "ggpo_tied_max_reward_subset_ess_ratio": "ggpo_tied_max_reward_subset_group_frac",
    "ggpo_tied_max_reward_subset_delta_weighted_mean": "ggpo_tied_max_reward_subset_group_frac",
    "ggpo_mean_tied_max_reward_k_in_informative_groups": "ggpo_informative_group_frac",
    "ggpo_informative_group_frac_tied_max_reward_k_ge_2": "ggpo_informative_group_frac",
    "ggpo_const_group_coefficient_rms": ("ggpo_all_zero_group_frac", "ggpo_all_one_group_frac"),
    "ggpo_informative_group_coefficient_rms": "ggpo_informative_group_frac",
    "ggpo_const_group_kl_alignment": ("ggpo_all_zero_group_frac", "ggpo_all_one_group_frac"),
}


def ggpo_conditional_metric_fraction(name: str, metrics: dict[str, Any]) -> float | None:
    """Return the selected-population fraction for a conditional metric."""
    fraction_names = GGPO_CONDITIONAL_METRIC_FRACTIONS.get(name)
    if fraction_names is None:
        return None
    if isinstance(fraction_names, str):
        fraction_names = (fraction_names,)
    return sum(float(metrics[fraction_name]) for fraction_name in fraction_names)


def build_policy_group_ids(uids, group_size: int) -> torch.Tensor:
    """Build contiguous group IDs from repeated prompt UIDs.

    GGPO expects rollout responses for each prompt to stay adjacent, with one
    complete group occupying each consecutive ``group_size`` block.
    """
    try:
        group_size = operator.index(group_size)
    except TypeError as exc:
        raise ValueError(f"Policy-loss group_size must be an integer greater than 1, got {group_size!r}") from exc
    if group_size <= 1:
        raise ValueError(f"Policy-loss group_size must be greater than 1, got {group_size}")

    uid_values = list(uids)

    response_count = len(uid_values)
    if response_count % group_size != 0:
        raise ValueError(
            f"Policy-loss response count ({response_count}) must be divisible by group_size ({group_size})"
        )

    group_ids: list[int] = []
    seen_uids: list[Any] = []
    for block_start in range(0, response_count, group_size):
        block = uid_values[block_start : block_start + group_size]
        first_uid = block[0]
        if any(not _ggpo_uid_equal(uid, first_uid) for uid in block):
            raise ValueError(
                "Policy-loss uid values must form a contiguous block of "
                f"group_size={group_size} identical values; block starting at index {block_start} is interleaved."
            )
        if any(_ggpo_uid_equal(uid, first_uid) for uid in seen_uids):
            raise ValueError(
                f"Policy-loss uid {first_uid!r} appears in a non-contiguous block; each uid must form exactly one group."
            )

        seen_uids.append(first_uid)
        group_ids.extend([len(seen_uids) - 1] * group_size)

    return torch.tensor(group_ids, dtype=torch.long)


def _ggpo_uid_equal(left: Any, right: Any) -> bool:
    try:
        equal = left == right
    except Exception as exc:
        raise ValueError("GGPO uid values must be scalar-comparable; found an ambiguous uid value.") from exc

    shape = getattr(equal, "shape", None)
    if shape is not None and tuple(shape) != ():
        raise ValueError("GGPO uid values must be scalar-comparable; found an array-valued uid value.")

    try:
        return bool(equal)
    except Exception as exc:
        raise ValueError("GGPO uid values must be scalar-comparable; found an ambiguous uid value.") from exc


def _validate_inputs(
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
    sequence_log_prob_reduction: str,
) -> None:
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError(f"GGPO beta must be positive and finite, got {beta}")
    if sequence_log_prob_reduction not in {"sum", "mean"}:
        raise ValueError(
            f"GGPO sequence_log_prob_reduction must be one of ['mean', 'sum'], got {sequence_log_prob_reduction!r}"
        )
    if group_size <= 1:
        raise ValueError(f"GGPO group_size must be greater than 1, got {group_size}")
    if dp_size <= 0:
        raise ValueError(f"GGPO dp_size must be positive, got {dp_size}")
    if global_batch_size <= 0:
        raise ValueError(f"GGPO global_batch_size must be positive, got {global_batch_size}")
    if global_batch_size % group_size != 0:
        raise ValueError("GGPO global_batch_size must be divisible by group_size")
    if global_batch_size // group_size <= 0:
        raise ValueError("GGPO global group count must be positive")
    try:
        batch_num_tokens = operator.index(batch_num_tokens)
    except TypeError as exc:
        raise ValueError(
            f"GGPO batch_num_tokens must be a positive integer, got {batch_num_tokens!r}"
        ) from exc
    if batch_num_tokens <= 0:
        raise ValueError(f"GGPO batch_num_tokens must be positive, got {batch_num_tokens}")
    if batch_num_tokens < global_batch_size:
        raise ValueError(
            "GGPO batch_num_tokens cannot be smaller than global_batch_size because every response "
            "must contain at least one valid token"
        )

    if log_prob.ndim != 2:
        raise ValueError("GGPO tensor shapes require log_prob to have shape [batch, response_length]")
    if ref_log_prob.shape != log_prob.shape or response_mask.shape != log_prob.shape:
        raise ValueError("log_prob, ref_log_prob, and response_mask shapes must be identical")
    batch_size = log_prob.shape[0]
    if rewards.shape != (batch_size,) or group_ids.shape != (batch_size,):
        raise ValueError("rewards and group_ids shapes must contain one value per response")
    if group_ids.is_floating_point() and (
        not torch.isfinite(group_ids).all() or not torch.all(group_ids == group_ids.trunc())
    ):
        raise ValueError("GGPO group_ids must contain finite integral group_ids")
    if not torch.isfinite(rewards).all():
        raise ValueError("GGPO rewards must be finite")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("GGPO response_mask must be binary values in {0, 1}")
    if not response_mask.any(dim=-1).all():
        raise ValueError("every GGPO response must contain at least one valid token")

    _, counts = torch.unique(group_ids, return_counts=True)
    if counts.numel() == 0 or not torch.all(counts == group_size):
        raise ValueError(f"every GGPO group must contain exactly group_size={group_size} responses")


def _group_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of per-group ``values`` over ``mask``, zero when nothing is selected.

    The worker aggregation assigns zero scientific weight to the empty case,
    so this finite placeholder can never affect a defined conditional mean.
    """
    selected = mask.float()
    count = selected.sum()
    total = (values.float() * selected).sum()
    return total / count.clamp_min(1.0)


class _GGPOObjective(NamedTuple):
    sequence_log_prob: torch.Tensor
    reference_sequence_log_prob: torch.Tensor
    grouped_adjusted_rewards: torch.Tensor
    beta_sqrt_normalized_per_group: torch.Tensor
    normalized_per_group: torch.Tensor
    normalized_to_raw_ratio: float
    q: torch.Tensor
    normalized_coefficients: torch.Tensor
    derivative_by_group: torch.Tensor


def _group_objective_terms(
    sequence_log_prob: torch.Tensor,
    reference_sequence_log_prob: torch.Tensor,
    rewards: torch.Tensor,
    order: torch.Tensor,
    *,
    beta: float,
    group_size: int,
    reward_kl: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    adjusted_rewards = rewards
    if reward_kl:
        adjusted_rewards = rewards - beta * (sequence_log_prob - reference_sequence_log_prob)
    grouped = adjusted_rewards[order].reshape(-1, group_size)
    grouped_centered = grouped - grouped.mean(dim=-1, keepdim=True)
    beta_sqrt_normalized_per_group = (
        torch.logsumexp(grouped_centered / beta, dim=-1) - math.log(group_size)
    ) / math.sqrt(group_size - 1)
    q = torch.softmax(grouped.detach().float() / beta, dim=-1)
    normalized_coefficients = (group_size * q - 1.0) / math.sqrt(group_size - 1)
    if not reward_kl:
        # Reward-only q still drives the same normalized policy-gradient update.
        grouped_log_prob = sequence_log_prob[order].reshape(-1, group_size)
        beta_sqrt_normalized_per_group = -(normalized_coefficients * grouped_log_prob).mean(dim=-1)
    return grouped, beta_sqrt_normalized_per_group, q, normalized_coefficients


def _seqsum_objective(
    raw_sequence_log_prob: torch.Tensor,
    raw_reference_sequence_log_prob: torch.Tensor,
    rewards: torch.Tensor,
    order: torch.Tensor,
    *,
    beta: float,
    group_size: int,
    reward_kl: bool,
    global_batch_size: int,
    batch_num_tokens: int,
    dp_size: int,
) -> _GGPOObjective:
    """Define SeqSum with a mean over the global valid-token count."""
    sequence_log_prob = raw_sequence_log_prob
    reference_sequence_log_prob = raw_reference_sequence_log_prob
    grouped, beta_sqrt_normalized_per_group, q, normalized_coefficients = _group_objective_terms(
        sequence_log_prob,
        reference_sequence_log_prob,
        rewards,
        order,
        beta=beta,
        group_size=group_size,
        reward_kl=reward_kl,
    )

    global_token_mean = global_batch_size / batch_num_tokens
    normalized_per_group = beta_sqrt_normalized_per_group * global_token_mean
    normalized_to_raw_ratio = global_token_mean / (beta * math.sqrt(group_size - 1))
    derivative_by_group = -normalized_coefficients / group_size
    derivative_by_group *= dp_size / (global_batch_size // group_size)
    derivative_by_group *= global_token_mean
    return _GGPOObjective(
        sequence_log_prob,
        reference_sequence_log_prob,
        grouped,
        beta_sqrt_normalized_per_group,
        normalized_per_group,
        normalized_to_raw_ratio,
        q,
        normalized_coefficients,
        derivative_by_group,
    )


def _seqmean_objective(
    raw_sequence_log_prob: torch.Tensor,
    raw_reference_sequence_log_prob: torch.Tensor,
    response_lengths: torch.Tensor,
    rewards: torch.Tensor,
    order: torch.Tensor,
    *,
    beta: float,
    group_size: int,
    reward_kl: bool,
    global_batch_size: int,
    dp_size: int,
) -> _GGPOObjective:
    """Define SeqMean with a mean over the global response count."""
    sequence_log_prob = raw_sequence_log_prob / response_lengths
    reference_sequence_log_prob = raw_reference_sequence_log_prob / response_lengths
    grouped, beta_sqrt_normalized_per_group, q, normalized_coefficients = _group_objective_terms(
        sequence_log_prob,
        reference_sequence_log_prob,
        rewards,
        order,
        beta=beta,
        group_size=group_size,
        reward_kl=reward_kl,
    )

    normalized_per_group = beta_sqrt_normalized_per_group
    normalized_to_raw_ratio = 1.0 / (beta * math.sqrt(group_size - 1))
    derivative_by_group = -normalized_coefficients / group_size
    derivative_by_group *= dp_size / (global_batch_size // group_size)
    return _GGPOObjective(
        sequence_log_prob,
        reference_sequence_log_prob,
        grouped,
        beta_sqrt_normalized_per_group,
        normalized_per_group,
        normalized_to_raw_ratio,
        q,
        normalized_coefficients,
        derivative_by_group,
    )


def _resolve_objective(
    sequence_log_prob_reduction: str,
    raw_sequence_log_prob: torch.Tensor,
    raw_reference_sequence_log_prob: torch.Tensor,
    response_lengths: torch.Tensor,
    rewards: torch.Tensor,
    order: torch.Tensor,
    *,
    beta: float,
    group_size: int,
    reward_kl: bool,
    global_batch_size: int,
    batch_num_tokens: int,
    dp_size: int,
) -> _GGPOObjective:
    if sequence_log_prob_reduction == "sum":
        return _seqsum_objective(
            raw_sequence_log_prob,
            raw_reference_sequence_log_prob,
            rewards,
            order,
            beta=beta,
            group_size=group_size,
            reward_kl=reward_kl,
            global_batch_size=global_batch_size,
            batch_num_tokens=batch_num_tokens,
            dp_size=dp_size,
        )
    return _seqmean_objective(
        raw_sequence_log_prob,
        raw_reference_sequence_log_prob,
        response_lengths,
        rewards,
        order,
        beta=beta,
        group_size=group_size,
        reward_kl=reward_kl,
        global_batch_size=global_batch_size,
        dp_size=dp_size,
    )


def compute_ggpo_loss(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    raw_correctness: torch.Tensor,
    beta: float,
    group_size: int,
    global_batch_size: int,
    batch_num_tokens: int,
    dp_size: int,
    sequence_log_prob_reduction: str = "sum",
    reward_kl: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the exact, directly differentiated group-level GGPO loss.

    ``sum`` selects SeqSum, whose sequence-sum gradient is averaged over the
    global valid-token count. ``mean`` selects SeqMean, whose length-normalized
    sequence gradient is averaged over the global response count. Verifier
    token scores have already been reduced to one reward per response.
    ``raw_correctness`` is the unshaped, strictly binary verifier result used
    only for group classification.
    With ``reward_kl=False``, q is softmax(reward / beta) and the returned loss
    is a policy-gradient surrogate with the same coefficient normalization.
    """
    rewards = rewards.detach().float()
    raw_correctness = raw_correctness.detach().float()
    ref_log_prob = ref_log_prob.detach().float()
    response_mask = response_mask.detach()
    group_ids = group_ids.detach()

    _validate_inputs(
        log_prob,
        ref_log_prob,
        rewards,
        response_mask,
        group_ids,
        beta=beta,
        group_size=group_size,
        global_batch_size=global_batch_size,
        batch_num_tokens=batch_num_tokens,
        dp_size=dp_size,
        sequence_log_prob_reduction=sequence_log_prob_reduction,
    )
    if raw_correctness.shape != rewards.shape:
        raise ValueError("GGPO raw_correctness must contain one value per response")
    if not torch.isfinite(raw_correctness).all() or not torch.logical_or(
        raw_correctness == 0, raw_correctness == 1
    ).all():
        raise ValueError("GGPO raw_correctness must be finite and exactly binary (0 or 1)")
    response_mask = response_mask.bool()
    group_ids = group_ids.long()

    response_lengths = response_mask.sum(dim=-1).float()
    raw_sequence_log_prob = (log_prob.float() * response_mask).sum(dim=-1)
    raw_reference_sequence_log_prob = (ref_log_prob * response_mask).sum(dim=-1)
    order = torch.argsort(group_ids, stable=True)
    objective = _resolve_objective(
        sequence_log_prob_reduction,
        raw_sequence_log_prob,
        raw_reference_sequence_log_prob,
        response_lengths,
        rewards,
        order,
        beta=beta,
        group_size=group_size,
        reward_kl=reward_kl,
        global_batch_size=global_batch_size,
        batch_num_tokens=batch_num_tokens,
        dp_size=dp_size,
    )
    sequence_log_prob = objective.sequence_log_prob
    reference_sequence_log_prob = objective.reference_sequence_log_prob
    grouped = objective.grouped_adjusted_rewards
    beta_sqrt_normalized_per_group = objective.beta_sqrt_normalized_per_group
    normalized_per_group = objective.normalized_per_group
    grouped_rewards = rewards[order].reshape(-1, group_size).detach().float()
    grouped_raw_correctness = raw_correctness[order].reshape(-1, group_size)
    constant_negative_mask = (grouped_raw_correctness == 0).all(dim=-1)
    constant_positive_mask = (grouped_raw_correctness == 1).all(dim=-1)
    informative_group_mask = ~(constant_negative_mask | constant_positive_mask)

    batch_mean_response_length = batch_num_tokens / global_batch_size
    global_group_count = global_batch_size // group_size
    loss = normalized_per_group.sum() * dp_size / global_group_count
    if not torch.isfinite(loss):
        raise FloatingPointError("GGPO loss is not finite")

    grouped_detached = grouped.detach().float()
    raw_sequence_log_ratio = raw_sequence_log_prob.detach() - raw_reference_sequence_log_prob
    sequence_log_ratio = sequence_log_prob.detach() - reference_sequence_log_prob
    raw_per_group = beta_sqrt_normalized_per_group.detach() * beta * math.sqrt(group_size - 1)
    metric_device = grouped_detached.device
    metrics = {
        "ggpo_loss": normalized_per_group.detach().mean(),
        "ggpo_raw_loss": raw_per_group.mean(),
        "ggpo_normalized_loss_to_raw_loss_ratio": torch.tensor(
            objective.normalized_to_raw_ratio, dtype=torch.float32, device=metric_device
        ),
        "ggpo_global_valid_token_count": torch.tensor(
            batch_num_tokens, dtype=torch.float32, device=metric_device
        ),
        "ggpo_batch_mean_response_length": torch.tensor(
            batch_mean_response_length, dtype=torch.float32, device=metric_device
        ),
        "ggpo_adjusted_reward_mean": grouped_detached.mean(),
        "ggpo_adjusted_reward_mean_microbatch_std": grouped_detached.std(unbiased=False),
        "ggpo_adjusted_reward_min": grouped_detached.min(),
        "ggpo_adjusted_reward_mean_microbatch_p05": torch.quantile(grouped_detached, 0.05),
        "ggpo_adjusted_reward_mean_microbatch_p50": torch.quantile(grouped_detached, 0.50),
        "ggpo_adjusted_reward_mean_microbatch_p95": torch.quantile(grouped_detached, 0.95),
        "ggpo_adjusted_reward_max": grouped_detached.max(),
        "ggpo_all_zero_group_frac": constant_negative_mask.float().mean(),
        "ggpo_all_one_group_frac": constant_positive_mask.float().mean(),
        "ggpo_informative_group_frac": informative_group_mask.float().mean(),
        "ggpo_sequence_log_ratio_mean": sequence_log_ratio.mean(),
        "ggpo_sequence_log_ratio_mean_microbatch_std": sequence_log_ratio.float().std(unbiased=False),
        "ggpo_sequence_log_ratio_abs_max": sequence_log_ratio.float().abs().max(),
        "ggpo_raw_sequence_log_ratio_mean": raw_sequence_log_ratio.mean(),
        "ggpo_raw_sequence_log_ratio_mean_microbatch_std": raw_sequence_log_ratio.float().std(unbiased=False),
        "ggpo_raw_sequence_log_ratio_abs_max": raw_sequence_log_ratio.float().abs().max(),
    }
    q = objective.q
    q_ess = q.square().sum(dim=-1).reciprocal()
    normalized_coefficients = objective.normalized_coefficients

    # Diagnostics only. Everything below is detached and never enters the loss.
    grouped_delta = sequence_log_ratio[order].reshape(-1, group_size).float()
    group_max_reward = grouped_rewards.max(dim=-1, keepdim=True).values
    tied_max_reward_mask = (grouped_rewards == group_max_reward).float()
    tied_max_reward_count = tied_max_reward_mask.sum(dim=-1)
    # Inside the max-reward subset of an informative group the reward cancels
    # exactly, so q there is driven by exp(-Delta_i) with KL, or is uniform without KL.
    tied_max_reward_subset_mask = informative_group_mask & (tied_max_reward_count >= 2)
    # q renormalized on the subset, evaluated in log space. Dividing the
    # already-exponentiated q by its subset mass underflows whenever the
    # subset sits far below the group's best exponent, which is reachable at
    # beta=1e-3 once within-group Delta spread approaches r/beta.
    tied_max_reward_subset_logits = (grouped_detached / beta).masked_fill(
        tied_max_reward_mask == 0, float("-inf")
    )
    q_tilde = torch.softmax(tied_max_reward_subset_logits, dim=-1)
    subset_delta_mean = (
        (grouped_delta * tied_max_reward_mask).sum(dim=-1, keepdim=True)
        / tied_max_reward_count.clamp_min(1.0).unsqueeze(-1)
    )
    subset_delta_centered = (grouped_delta - subset_delta_mean) * tied_max_reward_mask
    subset_delta_spread = (
        subset_delta_centered.square().sum(dim=-1) / tied_max_reward_count.clamp_min(1.0)
    ).sqrt()
    subset_ess_ratio = q_tilde.square().sum(dim=-1).reciprocal() / tied_max_reward_count.clamp_min(1.0)
    subset_delta_weighted_mean = (q_tilde * subset_delta_centered).sum(dim=-1)

    constant_group_mask = ~informative_group_mask
    group_coefficient_mean_square = normalized_coefficients.square().mean(dim=-1)
    constant_squared_mass = (group_coefficient_mean_square * constant_group_mask.float()).sum()
    total_squared_mass = group_coefficient_mean_square.sum()
    group_delta_centered = grouped_delta - grouped_delta.mean(dim=-1, keepdim=True)
    kl_alignment_terms = normalized_coefficients * group_delta_centered
    constant_rows = constant_group_mask.unsqueeze(-1).float()
    kl_alignment_numerator = -(kl_alignment_terms * constant_rows).sum()
    kl_alignment_denominator = (kl_alignment_terms.abs() * constant_rows).sum() + 1e-12
    group_count_scalar = torch.tensor(float(q.shape[0]), dtype=torch.float32, device=metric_device)
    metrics.update(
        {
            # Constant-reward-group diagnostics. Empty whenever dynamic
            # filtering already removed every constant group.
            "ggpo_const_group_coefficient_rms": _group_masked_mean(
                group_coefficient_mean_square, constant_group_mask
            ).sqrt(),
            "ggpo_informative_group_coefficient_rms": _group_masked_mean(
                group_coefficient_mean_square, informative_group_mask
            ).sqrt(),
            "ggpo_const_group_squared_mass_share": torch.where(
                total_squared_mass > 0,
                constant_squared_mass / total_squared_mass.clamp_min(1e-30),
                torch.zeros_like(total_squared_mass),
            ),
            "ggpo_const_group_kl_alignment": torch.where(
                constant_group_mask.any(),
                kl_alignment_numerator / kl_alignment_denominator,
                torch.zeros_like(kl_alignment_numerator),
            ),
        }
    )
    # Emitted unconditionally. Inside the tied-max subset the reward is constant
    # whatever its value: subset q is proportional to exp(-Delta_i) with KL,
    # or uniform without KL. The reward cancellation does not depend on
    # the reward being binary; only the old "correct subset" name did.
    metrics.update(
        {
            "ggpo_mean_tied_max_reward_k_in_informative_groups": _group_masked_mean(
                tied_max_reward_count, informative_group_mask
            ),
            "ggpo_informative_group_frac_tied_max_reward_k_ge_2": _group_masked_mean(
                (tied_max_reward_count >= 2).float(), informative_group_mask
            ),
            "ggpo_tied_max_reward_subset_delta_spread": _group_masked_mean(
                subset_delta_spread, tied_max_reward_subset_mask
            ),
            "ggpo_tied_max_reward_subset_ess_ratio": _group_masked_mean(
                subset_ess_ratio, tied_max_reward_subset_mask
            ),
            "ggpo_tied_max_reward_subset_delta_weighted_mean": _group_masked_mean(
                subset_delta_weighted_mean, tied_max_reward_subset_mask
            ),
            # These companions remain group-weighted so their ratio with
            # group_frac reconstructs the conditional mean after reduction.
            "ggpo_tied_max_reward_subset_delta_spread_over_all_groups": (
                subset_delta_spread * tied_max_reward_subset_mask.float()
            ).mean(),
            "ggpo_tied_max_reward_subset_ess_ratio_over_all_groups": (
                subset_ess_ratio * tied_max_reward_subset_mask.float()
            ).mean(),
            "ggpo_tied_max_reward_subset_delta_weighted_mean_over_all_groups": (
                subset_delta_weighted_mean * tied_max_reward_subset_mask.float()
            ).mean(),
            "ggpo_tied_max_reward_subset_group_count_mean": tied_max_reward_subset_mask.float().sum(),
            "ggpo_groups_per_microbatch": group_count_scalar,
            "ggpo_tied_max_reward_subset_group_frac": (
                tied_max_reward_subset_mask.float().sum() / group_count_scalar
            ),
        }
    )
    metrics.update(
        {
            "ggpo_q_entropy": (-(q * q.clamp_min(1e-12).log()).sum(dim=-1)).mean(),
            "ggpo_q_max": q.max(dim=-1).values.mean(),
            "ggpo_q_ess": q_ess.mean(),
            "ggpo_q_normalized_ess": (q_ess / q.shape[-1]).mean(),
            "ggpo_normalized_coefficient_rms": normalized_coefficients.square().mean().sqrt(),
            # The loss scale is deliberately external to q. This receipt is
            # structurally exact rather than a numerical tolerance check.
            "ggpo_q_loss_normalization_max_abs_diff": torch.zeros((), device=metric_device),
            "ggpo_q_unchanged_after_loss_normalization": torch.ones((), device=metric_device),
        }
    )
    if metrics.keys() != set(GGPO_METRIC_NAMES):
        raise AssertionError("GGPO metric schema and live values differ")
    return loss, metrics


def compute_ggpo_surrogate_weights(
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
    sequence_log_prob_reduction: str = "sum",
    reward_kl: bool = True,
) -> torch.Tensor:
    """Return detached per-response weights for an exact GGPO gradient surrogate.

    GGPO's value needs a complete rollout group, but its derivative with
    respect to each response's sequence log-probability is a scalar evaluated
    at the pre-update policy. Once those scalars are calculated, the actor can
    safely split a group across dynamic token-packed micro-batches and use
    ``sum_i weight_i * sequence_log_prob_i``. Its parameter gradient is
    exactly the direct GGPO gradient at the pre-update policy, while avoiding
    an unsafe requirement that all ``group_size`` long responses coexist in
    one FSDP micro-batch.
    """
    log_prob = log_prob.detach()
    _validate_inputs(
        log_prob,
        ref_log_prob,
        rewards,
        response_mask,
        group_ids,
        beta=beta,
        group_size=group_size,
        global_batch_size=global_batch_size,
        batch_num_tokens=batch_num_tokens,
        dp_size=dp_size,
        sequence_log_prob_reduction=sequence_log_prob_reduction,
    )

    response_mask = response_mask.detach().bool()
    ref_log_prob = ref_log_prob.detach().float()
    rewards = rewards.detach().float()
    group_ids = group_ids.detach().long()
    response_lengths = response_mask.sum(dim=-1).float()
    raw_sequence_log_prob = (log_prob.float() * response_mask).sum(dim=-1)
    raw_reference_sequence_log_prob = (ref_log_prob * response_mask).sum(dim=-1)
    order = torch.argsort(group_ids, stable=True)
    objective = _resolve_objective(
        sequence_log_prob_reduction,
        raw_sequence_log_prob,
        raw_reference_sequence_log_prob,
        response_lengths,
        rewards,
        order,
        beta=beta,
        group_size=group_size,
        reward_kl=reward_kl,
        global_batch_size=global_batch_size,
        batch_num_tokens=batch_num_tokens,
        dp_size=dp_size,
    )
    weights = torch.empty_like(objective.sequence_log_prob)
    weights[order] = objective.derivative_by_group.reshape(-1)
    if not torch.isfinite(weights).all():
        raise FloatingPointError("GGPO surrogate weights are not finite")
    return weights.detach()
