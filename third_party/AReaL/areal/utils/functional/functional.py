# SPDX-License-Identifier: Apache-2.0

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from areal.api.cli_args import RejectionSamplingConfig
from areal.utils.data import KLEstimator


@torch.no_grad()
def masked_normalization(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    dim=None,
    unbiased=False,
    eps=1e-5,
    high_precision=True,
    all_reduce=True,
    reduce_group=None,
):
    dtype = torch.float64 if high_precision else torch.float32
    x = x.to(dtype)
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        factor = torch.tensor(
            np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
        )
    else:
        mask = mask.to(dtype)
        x = x * mask
        factor = mask.sum(dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    if dist.is_initialized() and all_reduce:
        dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(
            x_sum_sq,
            op=dist.ReduceOp.SUM,
            group=reduce_group,
        )
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


def _compute_sequence_level_ratio_and_advantages(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sequence-level geometric mean ratios and average advantages per sequence (GSPO).

    Args:
        log_ratio: Log of probability ratios (logprobs - proximal_logprobs)
        advantages: Per-token advantages
        loss_mask: Boolean mask indicating valid tokens
        cu_seqlens: Cumulative sequence lengths. Required for 1D tensors (packed format).
            Shape: [batch_size + 1], where cu_seqlens[i] marks the start of sequence i.
            For a single sequence, use cu_seqlens=torch.tensor([0, seq_len]).

    Returns:
        ratio: Sequence-level importance sampling ratios (broadcast to all tokens)
        advantages: Sequence-averaged advantages (broadcast to all tokens)
            Note: We use mean instead of sum to keep gradient magnitude independent
            of sequence length. When multiplied by ratio and summed over tokens,
            this gives the correct total gradient contribution per sequence.
    """
    # Handle both 1D (packed) and 2D (padded) tensor shapes
    if log_ratio.ndim == 1:
        # For 1D tensors (packed format), cu_seqlens is required
        if cu_seqlens is None:
            raise ValueError(
                "cu_seqlens is required for 1D tensors (packed format). "
                "In AReaL, 1D tensors are produced by pack_tensor_dict() and always have cu_seqlens. "
                "For a single sequence, use cu_seqlens=torch.tensor([0, seq_len], dtype=torch.int32)."
            )

        # Packed sequences: use cu_seqlens boundaries
        batch_size = cu_seqlens.shape[0] - 1
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        # Create sequence index for each token: [0,0,0,1,1,2,2,2,2,...]
        sequence_idx = torch.arange(
            batch_size, device=log_ratio.device
        ).repeat_interleave(seq_lengths)

        # Use scatter_add for vectorized summation per sequence (faster than Python loop)
        masked_log_ratio = torch.where(loss_mask, log_ratio, 0.0)
        log_ratio_sum_per_seq = torch.zeros(
            batch_size, device=log_ratio.device, dtype=log_ratio.dtype
        ).scatter_add_(0, sequence_idx, masked_log_ratio)

        masked_advantages = torch.where(loss_mask, advantages, 0.0)
        advantages_sum_per_seq = torch.zeros(
            batch_size, device=advantages.device, dtype=advantages.dtype
        ).scatter_add_(0, sequence_idx, masked_advantages)

        valid_count_per_seq = (
            torch.zeros(batch_size, device=loss_mask.device, dtype=torch.int32)
            .scatter_add_(0, sequence_idx, loss_mask.int())
            .clamp(min=1)
        )

        # Compute sequence-level means
        log_ratio_mean_per_seq = log_ratio_sum_per_seq / valid_count_per_seq.to(
            log_ratio.dtype
        )
        adv_mean_per_seq = advantages_sum_per_seq / valid_count_per_seq.to(
            advantages.dtype
        )

        # Broadcast sequence-level values back to token-level
        ratio = torch.exp(log_ratio_mean_per_seq)[sequence_idx]
        ratio = torch.where(loss_mask, ratio, 0.0)

        advantages = adv_mean_per_seq[sequence_idx]
        advantages = torch.where(loss_mask, advantages, 0.0)
    else:
        # For 2D tensors (padded sequences)
        # Input shape: [batch_size, seq_len]
        # Compute mean log ratio over sequence length for each sample
        seq_log_ratio_mean = torch.where(loss_mask, log_ratio, 0.0).sum(dim=1) / (
            loss_mask.sum(dim=1).clamp(min=1)
        )
        # Broadcast back to original shape: each sequence gets its own geometric mean ratio
        ratio = torch.exp(seq_log_ratio_mean.unsqueeze(1).expand_as(log_ratio))
        # Apply mask
        ratio = torch.where(loss_mask, ratio, 0.0)

        # Average token advantages per sequence
        # This ensures gradient magnitude is independent of sequence length
        seq_lengths = loss_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        masked_advantages = torch.where(loss_mask, advantages, 0.0)
        advantages = (
            masked_advantages.sum(dim=-1, keepdim=True) / seq_lengths
        ).expand_as(log_ratio)
        advantages = torch.where(loss_mask, advantages, 0.0)

    return ratio, advantages


@dataclass
class RejectionSamplingResult:
    """Result of rejection sampling, used by ppo_actor_loss_fn.

    Attributes:
        loss_mask: Updated loss mask (mask mode) or original loss mask (clamp mode).
        behave_imp_weight: Importance weight (clamped in clamp mode, raw in mask mode).
        filtered_fraction: Fraction of valid tokens that were filtered/clamped (for logging).
    """

    loss_mask: torch.Tensor
    behave_imp_weight: torch.Tensor
    filtered_fraction: float


def _check_bounds(
    metric: torch.Tensor, config: RejectionSamplingConfig
) -> torch.Tensor:
    """Check if metric values are within configured bounds.

    Args:
        metric: Per-token or per-sequence metric values.
        config: Rejection sampling configuration with upper and optional lower bounds.

    Returns:
        Boolean tensor, True where metric is within bounds.
    """
    if config.lower is not None:
        return (metric >= config.lower) & (metric <= config.upper)
    else:
        return metric <= config.upper


def apply_rejection_sampling(
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    config: RejectionSamplingConfig,
) -> RejectionSamplingResult:
    """Apply rejection sampling based on divergence between proximal and behavior policy.

    Supports two action modes:
    - 'mask': zero out loss_mask for tokens/sequences exceeding threshold (rejection)
    - 'clamp': clamp importance weight to bounds for tokens/sequences exceeding
      threshold (truncation, tokens still participate in gradient)

    Args:
        proximal_logprobs: Proximal policy log-probabilities,
            shape [batch, seq_len] (2D padded) or [total_tokens] (1D packed).
        old_logprobs: Behavior policy log-probabilities from inference engine,
            same shape as proximal_logprobs.
        loss_mask: Original loss mask (1 for valid tokens), same shape as proximal_logprobs.
        cu_seqlens: Cumulative sequence lengths for 1D packed format. Shape: [batch_size + 1].
            Required when inputs are 1D. None for 2D padded inputs.
        config: Configuration for rejection sampling.

    Returns:
        RejectionSamplingResult with updated loss_mask, behave_imp_weight, and filtered_fraction.
    """
    # Step 0: Validate input shapes.
    if proximal_logprobs.shape != old_logprobs.shape:
        raise ValueError(
            f"proximal_logprobs shape {proximal_logprobs.shape} != "
            f"old_logprobs shape {old_logprobs.shape}"
        )
    if proximal_logprobs.shape != loss_mask.shape:
        raise ValueError(
            f"proximal_logprobs shape {proximal_logprobs.shape} != "
            f"loss_mask shape {loss_mask.shape}"
        )
    if proximal_logprobs.ndim not in (1, 2):
        raise ValueError(
            f"Expected 1D (packed) or 2D (padded) tensors, "
            f"got ndim={proximal_logprobs.ndim}"
        )

    # Step 1: Compute log ratio = log(π_proximal / π_behave)
    # Upcast operands to fp32 before subtraction to avoid precision loss in bf16/fp16.
    log_ratio = proximal_logprobs.detach().float() - old_logprobs.detach().float()
    # Sanitize non-finite values (e.g. -inf - (-inf) = NaN) to prevent NaN propagation.
    log_ratio = torch.where(torch.isfinite(log_ratio), log_ratio, 0.0)

    # Step 2: Compute metric value (reuse existing KLEstimator sign conventions)
    if config.metric == "ratio":
        # Direct ratio π_proximal / π_behave
        metric = torch.exp(log_ratio)
    elif config.metric in ("kl_k1", "kl_k2", "kl_k3"):
        # Use existing KLEstimator (note: _compute_approx_kl takes log_probs, log_probs_base)
        estimator_name = config.metric.replace("kl_", "")  # "k1", "k2", "k3"
        metric = KLEstimator._compute_approx_kl(
            log_probs=proximal_logprobs.detach(),
            log_probs_base=old_logprobs.detach(),
            kl_estimator=estimator_name,
            apply_clamp=False,  # Don't clamp; threshold check handles bounds
        )
    else:
        raise ValueError(f"Unknown metric: {config.metric}")

    # Step 3: Compute behave_imp_weight (needed for both modes)
    behave_imp_weight = torch.exp(log_ratio)
    # Save original weight before any clamping, to compute clamped fraction later.
    original_weight = behave_imp_weight

    # Step 4: Aggregate and filter
    #
    # For ratio metric, aggregate in log space (geometric mean) to match GSPO
    # semantics and avoid the "length trap" where arithmetic mean inflates
    # sequence-level ratios. For KL metrics, aggregate in metric space
    # (arithmetic) since KL divergence is additive.
    _use_log_agg = config.metric == "ratio"

    if config.level == "sequence":
        # Pre-compute sequence indexing (shared by filtering and weight broadcast).
        if loss_mask.ndim == 1:
            # 1D packed format: use cu_seqlens
            if cu_seqlens is None:
                raise ValueError(
                    "cu_seqlens is required for 1D packed tensors "
                    "in sequence-level filtering."
                )
            batch_size = cu_seqlens.shape[0] - 1
            seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            sequence_idx = torch.arange(
                batch_size, device=metric.device
            ).repeat_interleave(seq_lengths)

            # For ratio metric: aggregate log_ratio (geometric); else: aggregate metric (arithmetic).
            agg_values = log_ratio if _use_log_agg else metric
            masked_agg = torch.where(loss_mask.bool(), agg_values, 0.0)
            valid_count_per_seq = (
                torch.zeros(batch_size, device=loss_mask.device, dtype=torch.int32)
                .scatter_add_(0, sequence_idx, loss_mask.int())
                .clamp(min=1)
            )

            # Ratio metric + sequence level: use geometric mean as uniform weight
            # for all tokens (matches old sequence_mask/sequence_truncate semantics).
            if _use_log_agg:
                # masked_agg is already log_ratio masked by loss_mask (computed above).
                seq_log_sum = torch.zeros(
                    batch_size, device=log_ratio.device, dtype=log_ratio.dtype
                ).scatter_add_(0, sequence_idx, masked_agg)
                seq_log_mean = seq_log_sum / valid_count_per_seq.to(log_ratio.dtype)
                behave_imp_weight = torch.exp(seq_log_mean)[sequence_idx]
                original_weight = behave_imp_weight

            if config.agg == "sum":
                seq_agg = torch.zeros(
                    batch_size, device=metric.device, dtype=agg_values.dtype
                ).scatter_add_(0, sequence_idx, masked_agg)
            elif config.agg == "mean":
                seq_agg_sum = torch.zeros(
                    batch_size, device=metric.device, dtype=agg_values.dtype
                ).scatter_add_(0, sequence_idx, masked_agg)
                seq_agg = seq_agg_sum / valid_count_per_seq.to(agg_values.dtype)
            elif config.agg == "max":
                agg_for_max = agg_values.masked_fill(~loss_mask.bool(), float("-inf"))
                seq_agg = torch.full(
                    (batch_size,),
                    float("-inf"),
                    device=metric.device,
                    dtype=agg_values.dtype,
                ).scatter_reduce_(0, sequence_idx, agg_for_max, reduce="amax")
                # All-masked sequences stay -inf; treat them as in-bounds (no valid
                # tokens to filter, and their loss_mask is already all-zero).
                # Recompute from raw counts to detect true zero.
                raw_valid = torch.zeros(
                    batch_size, device=loss_mask.device, dtype=torch.int32
                ).scatter_add_(0, sequence_idx, loss_mask.int())
                all_masked = raw_valid == 0
                seq_agg = torch.where(all_masked, torch.zeros_like(seq_agg), seq_agg)
            else:
                raise ValueError(f"Unknown agg method: {config.agg}")

            # Convert back to metric space for threshold comparison.
            seq_metric = torch.exp(seq_agg) if _use_log_agg else seq_agg

            # Check each sequence against bounds
            in_bounds_per_seq = _check_bounds(seq_metric, config)

            if config.action == "mask":
                # Broadcast back to token level, filter entire sequence
                in_bounds = in_bounds_per_seq[sequence_idx]
            else:
                # clamp mode: clamp tokens in out-of-bounds sequences
                out_of_bounds = (~in_bounds_per_seq)[sequence_idx]
                behave_imp_weight = torch.where(
                    out_of_bounds,
                    behave_imp_weight.clamp(
                        min=config.lower if config.lower is not None else 0.0,
                        max=config.upper,
                    ),
                    behave_imp_weight,
                )
        else:
            # 2D padded format
            agg_values = log_ratio if _use_log_agg else metric
            masked_agg = torch.where(loss_mask.bool(), agg_values, 0.0)
            valid_count = loss_mask.sum(dim=-1, keepdim=True).clamp(min=1)

            # Ratio metric + sequence level: geometric mean as uniform weight.
            if _use_log_agg:
                seq_log_mean = masked_agg.sum(dim=-1, keepdim=True) / valid_count
                behave_imp_weight = torch.exp(seq_log_mean).expand_as(log_ratio)
                original_weight = behave_imp_weight

            if config.agg == "sum":
                seq_agg = masked_agg.sum(dim=-1, keepdim=True)
            elif config.agg == "mean":
                seq_agg = masked_agg.sum(dim=-1, keepdim=True) / valid_count
            elif config.agg == "max":
                agg_for_max = agg_values.masked_fill(~loss_mask.bool(), float("-inf"))
                seq_agg = agg_for_max.max(dim=-1, keepdim=True)[0]
                # All-masked sequences stay -inf; treat them as in-bounds (no valid
                # tokens to filter, and their loss_mask is already all-zero).
                all_masked = loss_mask.sum(dim=-1, keepdim=True) == 0
                seq_agg = torch.where(all_masked, torch.zeros_like(seq_agg), seq_agg)
            else:
                raise ValueError(f"Unknown agg method: {config.agg}")

            # Convert back to metric space for threshold comparison.
            seq_metric = torch.exp(seq_agg) if _use_log_agg else seq_agg

            if config.action == "mask":
                in_bounds = _check_bounds(seq_metric, config).expand_as(loss_mask)
            else:
                # clamp mode: clamp tokens in out-of-bounds sequences
                out_of_bounds = (~_check_bounds(seq_metric, config)).expand_as(
                    loss_mask
                )
                behave_imp_weight = torch.where(
                    out_of_bounds,
                    behave_imp_weight.clamp(
                        min=config.lower if config.lower is not None else 0.0,
                        max=config.upper,
                    ),
                    behave_imp_weight,
                )
    else:
        # Token level
        if config.action == "mask":
            in_bounds = _check_bounds(metric, config)
        else:
            # clamp mode: directly clamp importance weight
            behave_imp_weight = behave_imp_weight.clamp(
                min=config.lower if config.lower is not None else 0.0,
                max=config.upper,
            )

    # Step 5: Update loss_mask or keep it based on action mode
    if config.action == "mask":
        candidates = loss_mask.bool()
        updated_mask = (candidates & in_bounds).to(loss_mask.dtype)
        filtered_count = (candidates & ~in_bounds).sum().item()
        total_count = candidates.sum().item()
        filtered_fraction = filtered_count / max(total_count, 1)
    else:
        # clamp mode: loss_mask unchanged
        updated_mask = loss_mask
        # Report fraction of clamped tokens (for logging)
        clamped_count = (
            (loss_mask.bool() & (original_weight != behave_imp_weight)).sum().item()
        )
        total_count = loss_mask.bool().sum().item()
        filtered_fraction = clamped_count / max(total_count, 1)

    # Apply loss_mask to behave_imp_weight
    behave_imp_weight = torch.where(updated_mask.bool(), behave_imp_weight, 0.0)

    return RejectionSamplingResult(
        loss_mask=updated_mask,
        behave_imp_weight=behave_imp_weight,
        filtered_fraction=filtered_fraction,
    )


def _frpo_reverse_cumsum_per_seq(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Per-sequence tail-sum: out[t] = sum_{t' >= t in same seq} x[t'].

    Handles 2D [bs, seq] padded tensors (no cu_seqlens needed) and 1D packed
    tensors (require cu_seqlens of shape [n_seq + 1]).
    """
    if x.dim() == 2:
        return torch.flip(torch.cumsum(torch.flip(x, dims=[1]), dim=1), dims=[1])
    n = x.shape[0]
    if cu_seqlens is None:
        return torch.flip(torch.cumsum(torch.flip(x, dims=[0]), dim=0), dims=[0])
    # Global reverse cumsum, padded with 0 at the end as the "after-end" reference.
    global_rcs = torch.flip(torch.cumsum(torch.flip(x, dims=[0]), dim=0), dims=[0])
    global_rcs = torch.cat(
        [global_rcs, torch.zeros(1, device=x.device, dtype=x.dtype)]
    )
    # For each token t, the index of "one past the end of its own sequence" is
    # cu_seqlens[seg_id + 1] where seg_id is the segment containing t. The
    # per-token tail-sum within its segment is then
    #   global_rcs[t] - global_rcs[seg_end[t]].
    seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    seg_end = torch.repeat_interleave(cu_seqlens[1:], seg_lens)
    return global_rcs[:n] - global_rcs[seg_end]


def _fipo_decayed_reverse_cumsum_per_seq(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    """Per-sequence decayed tail-sum: out[t] = sum_{k>=t in same seq} gamma**(k-t) * x[k].

    FIPO §4.2.1 "soft decay window" with horizon tau = -1 / log2(gamma).
    For gamma=1.0 this reduces to _frpo_reverse_cumsum_per_seq.

    Vectorised via the scale-then-cumsum identity: let r_k = k - seg_start(k),
    g_k = gamma**r_k. Then with y_k = g_k * x_k and Y[t] = sum_{k>=t in same seg} y_k
    (an ordinary reverse cumsum), out[t] = Y[t] / g_t. No Python loops.

    Numerical caveat: for gamma=0.99 and T=4096, gamma**T ~ 1e-18 which still
    fits float32. For very small gamma + long sequences, scale_min may underflow;
    we clamp to 1e-30 to keep the division finite.
    """
    if gamma >= 1.0 - 1e-9:
        return _frpo_reverse_cumsum_per_seq(x, cu_seqlens)

    if x.dim() == 2:
        bs, T = x.shape
        r = torch.arange(T, device=x.device, dtype=x.dtype)
        g = torch.pow(x.new_tensor(gamma), r)  # [T]
        y = x * g.unsqueeze(0)  # [bs, T]
        Y = torch.flip(torch.cumsum(torch.flip(y, dims=[1]), dim=1), dims=[1])
        return Y / g.unsqueeze(0).clamp(min=1e-30)

    n = x.shape[0]
    if cu_seqlens is None:
        r = torch.arange(n, device=x.device, dtype=x.dtype)
        g = torch.pow(x.new_tensor(gamma), r)
        y = x * g
        Y = torch.flip(torch.cumsum(torch.flip(y, dims=[0]), dim=0), dims=[0])
        return Y / g.clamp(min=1e-30)

    # Packed: within-segment position r_k = k - seg_start(k).
    seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    seg_id = torch.repeat_interleave(
        torch.arange(seg_lens.numel(), device=x.device), seg_lens
    )
    seg_start_per_token = cu_seqlens[:-1][seg_id]
    abs_idx = torch.arange(n, device=x.device)
    r = (abs_idx - seg_start_per_token).to(x.dtype)
    g = torch.pow(x.new_tensor(gamma), r)
    y = x * g
    # Use the existing per-segment reverse cumsum on y, then divide by g.
    Y = _frpo_reverse_cumsum_per_seq(y, cu_seqlens)
    return Y / g.clamp(min=1e-30)


def ppo_actor_loss_fn(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    eps_clip_higher: float | None = None,
    c_clip: float | None = None,
    rejection_sampling: RejectionSamplingConfig | None = None,
    importance_sampling_level: str = "token",
    cu_seqlens: torch.Tensor | None = None,
    use_frpo_loss_term: bool = False,
    frpo_clip_eps: float = 0.2,
    frpo_beta: float = 0.0,
    frpo_normalize_tail: bool = False,
    # FRPO anchor — "prox" uses proximal_logprobs (recomputed every PPO inner step,
    # AReaL default but drifts as theta moves) or "ref" uses ref_logprobs (frozen
    # SFT/base, matches FRPO paper Eq. (5.1) where delta = log pi_theta / pi_ref).
    frpo_anchor: str = "prox",
    ref_logprobs: torch.Tensor | None = None,
    # FIPO (Future-KL Induced Policy Optimization, arXiv:2603.19835): a contemporaneous
    # framework that uses the SAME per-token primitive Δlog p as FRPO but couples it
    # MULTIPLICATIVELY to the advantage and applies a soft-decay window + dual-clip
    # mask. The two are mutually exclusive: setting use_fipo=True disables the FRPO
    # subtractive coupling above. Defaults below are the paper's near-defaults adjusted
    # for 1.7B + 8K-tok max_new (smaller scale than the paper's 32B + 8K).
    use_fipo: bool = False,
    fipo_beta: float = 1.0,
    fipo_gamma: float = 0.99,         # soft-decay; effective horizon = -1/log2(0.99) ~ 69 tokens
    fipo_dual_clip_c: float = 10.0,   # zero out summand whose ratio > c (FIPO Eq. 5)
    fipo_eps_low: float = 0.2,        # f_t = clip(exp(...), 1-eps_low, 1+eps_high)
    fipo_eps_high: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """PPO actor loss function with optional rejection sampling.

    The ``rejection_sampling`` parameter replaces the removed
    ``behave_imp_weight_cap`` / ``behave_imp_weight_mode``.

    - ``action='mask'``: modifies loss_mask before loss computation (rejection)
    - ``action='clamp'``: clamps importance weight to bounds (truncation)

    When decoupled loss is disabled:
    1. if recompute logp, both old_logprobs and proximal_logprobs are recomputed logp;
    2. if no recomputation, both old_logp and proximal_logprobs are produced by the inference backend.

    When decoupled loss is enabled, proximal_logprobs is the recomputed logp,
    old_logprobs is produced by the inference engine.

    Note: ``importance_sampling_level`` controls PPO ratio (π_θ/π_proximal)
    aggregation (GSPO), which is orthogonal to ``rejection_sampling.level``
    that controls staleness filtering (π_proximal/π_behave) granularity.

    Args:
        logprobs: Current policy log-probabilities (π_θ).
        proximal_logprobs: Proximal policy log-probabilities (π_proximal).
        old_logprobs: Behavior policy log-probabilities from inference (π_behave).
        advantages: Per-token advantage estimates.
        eps_clip: PPO clipping factor for policy ratio.
        loss_mask: Mask for valid tokens (1 = valid).
        eps_clip_higher: Upper clipping factor (decoupled clipping). None = use eps_clip.
        c_clip: Dual clipping factor, must be > 1.0. None disables dual clipping.
        rejection_sampling: Rejection sampling configuration. None disables filtering.
        importance_sampling_level: Level at which to compute importance sampling ratios.
            - 'token': Per-token ratios (standard PPO)
            - 'sequence': Sequence-level geometric mean of per-token ratios (GSPO)
        cu_seqlens: Cumulative sequence lengths for packed sequences (1D tensors).
            Required when inputs are 1D and importance_sampling_level='sequence'.
            Shape: [batch_size + 1], where cu_seqlens[i] marks the start of sequence i.
            Not needed for 2D padded inputs (sequences identified by batch dimension).
    """
    # Save original count BEFORE rejection sampling may modify loss_mask.
    # This keeps the denominator consistent with loss_weight_fn in actor.py,
    # which always uses the original loss_mask from input_data. Without this,
    # mask mode would inflate per-token gradients by N_original / N_kept.
    loss_mask_count = loss_mask.count_nonzero() or 1

    # === Apply rejection sampling (replaces old compute_behave_imp_weight) ===
    if rejection_sampling is not None:
        rs_result = apply_rejection_sampling(
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            loss_mask=loss_mask,
            cu_seqlens=cu_seqlens,
            config=rejection_sampling,
        )
        # mask mode updates loss_mask; clamp mode keeps it unchanged
        loss_mask = rs_result.loss_mask
        behave_imp_weight = rs_result.behave_imp_weight
        filtered_fraction = rs_result.filtered_fraction
    else:
        filtered_fraction = 0.0

    # FRPO advantage modification (loss-side, per PPO inner step):
    #   A^FRPO_t = A_t - frpo_beta * sum_{t' >= t in same seq}
    #                clip(log(pi_theta.detach() / pi_prox), -frpo_clip_eps, +frpo_clip_eps) * loss_mask_{t'}
    # When frpo_normalize_tail=True the tail-sum is additionally divided by the FULL
    # valid-token count of the token's trajectory (T_i), not by the remaining count
    # (T_i - t). Trajectory-level normalisation removes systematic length-dependence
    # between trajectories while preserving the per-position structure of the
    # tail-sum within a trajectory.
    # Detached log-ratio so the FRPO term contributes only as a scalar shift on
    # the advantage; gradient still flows through the ratio in the pg loss.
    if use_frpo_loss_term and frpo_beta != 0 and frpo_clip_eps > 0:
        with torch.no_grad():
            # Pick FRPO anchor: pi_prox (AReaL default, recomputed per outer step) or
            # pi_ref (frozen, paper-faithful). If "ref" is requested but ref_logprobs
            # wasn't supplied, fall back to prox with a no-op so training doesn't
            # silently train against a zero tensor.
            if frpo_anchor == "ref":
                if ref_logprobs is None:
                    raise ValueError(
                        "frpo_anchor='ref' requires ref_logprobs to be passed in. "
                        "Make sure the yaml has a `ref:` engine configured so the "
                        "trainer computes ref_logp each rollout."
                    )
                anchor_logp = ref_logprobs.detach().float()
            else:
                anchor_logp = proximal_logprobs.detach().float()
            clipped_logratio = (
                logprobs.detach().float() - anchor_logp
            ).clamp(-frpo_clip_eps, frpo_clip_eps) * loss_mask.float()
            tail = _frpo_reverse_cumsum_per_seq(clipped_logratio, cu_seqlens)
            if frpo_normalize_tail:
                # Per-token denominator = total valid tokens in THIS token's trajectory
                # (constant across tokens within a segment). Broadcast a per-segment
                # mask-sum back to each token.
                if cu_seqlens is None:
                    traj_total = loss_mask.float().sum().clamp(min=1.0)
                    tail = tail / traj_total
                else:
                    seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
                    n_segs = seg_lens.numel()
                    seg_id = torch.repeat_interleave(
                        torch.arange(n_segs, device=loss_mask.device), seg_lens
                    )
                    seg_totals = torch.zeros(
                        n_segs, device=loss_mask.device, dtype=torch.float32
                    )
                    seg_totals.scatter_add_(0, seg_id, loss_mask.float())
                    traj_total_per_token = seg_totals[seg_id].clamp(min=1.0)
                    tail = tail / traj_total_per_token
            frpo_correction = (frpo_beta * tail).to(advantages.dtype)
            # Local-token correction term β·f'(ρ_i)·ρ_i from the full w_i derivation
            # (blog Eq. "w_i = (r - v) - β·Σ_{j≥i} f(ρ_j) + β·f'(ρ_i)·ρ_i").
            # For f = -ln, f'(ρ)·ρ ≡ -1, so β·f'(ρ_i)·ρ_i = -β — a per-token constant.
            # We were previously dropping it. With the FRPO setting f = -ln, the full
            # advantage modification is:
            #   A_t  ← A_t  -  β·tail_sum_t  -  β            (this latter -β term is new)
            # The constant survives our adv_norm (which is batch-z-score) only if
            # adv_norm is applied BEFORE this block; if it's applied here-and-now to
            # a zero-mean advantage we shift the mean by -β. Either way, mathematical
            # parity with the derivation requires its presence.
            frpo_local_correction = (
                frpo_beta * loss_mask.to(advantages.dtype)
            )
            advantages = advantages - frpo_correction - frpo_local_correction

    # FIPO advantage modification (mutually exclusive with FRPO above):
    #   Delta_logp_t = log pi_theta - log pi_old  (using pi_behave as the
    #     rollout-time anchor, NOT the per-inner-step pi_prox; FIPO's anchor
    #     is fixed for the whole outer step which is critical to its design.)
    #   FutureKL_t   = sum_{k>=t in same seg} M_k * gamma^(k-t) * Delta_logp_k
    #     where M_k = indicator(pi_theta(o_k)/pi_old(o_k) <= dual_clip_c)
    #     zeros out outlier tokens (FIPO Eq. 5).
    #   f_t          = clip(exp(beta * FutureKL_t), 1-eps_low, 1+eps_high)
    #     multiplied INTO the advantage (FIPO Eq. 7).
    if use_fipo and fipo_beta != 0:
        with torch.no_grad():
            # Use the rollout-time policy (pi_behave / old_logprobs) as the anchor.
            # AReaL's pi_prox is recomputed every PPO inner step which would make
            # the anchor track theta — exactly the failure mode our FRPO experiments
            # diagnosed. old_logprobs is frozen for the whole outer step.
            delta_logp = (logprobs.detach().float() - old_logprobs.detach().float())
            # Dual-clip mask: importance ratio > c => set summand to 0.
            ratio_for_mask = torch.exp(delta_logp)
            outlier_mask = (ratio_for_mask <= fipo_dual_clip_c).float()
            masked = delta_logp * outlier_mask * loss_mask.float()
            fipo_tail = _fipo_decayed_reverse_cumsum_per_seq(
                masked, cu_seqlens, fipo_gamma
            )
            f_t_raw = torch.exp(fipo_beta * fipo_tail)
            f_t = f_t_raw.clamp(1.0 - fipo_eps_low, 1.0 + fipo_eps_high)
            f_t = f_t.to(advantages.dtype)
            advantages = advantages * f_t

    if importance_sampling_level == "sequence":
        # GSPO: Compute sequence-level geometric mean of probability ratios
        log_ratio = logprobs - proximal_logprobs
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(
            log_ratio, advantages, loss_mask, cu_seqlens
        )
    elif importance_sampling_level == "token":
        # Standard PPO: per-token ratio
        ratio = torch.where(loss_mask, torch.exp(logprobs - proximal_logprobs), 0)
    else:
        raise ValueError(
            f"Invalid importance_sampling_level: {importance_sampling_level}. "
            "Must be 'token' or 'sequence'."
        )

    clipped_ratio = torch.clamp(
        ratio,
        1.0 - eps_clip,
        1.0 + (eps_clip if eps_clip_higher is None else eps_clip_higher),
    )

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    clip_mask = pg_loss1.detach() < pg_loss2.detach()
    pg_loss = torch.max(pg_loss1, pg_loss2)
    if c_clip is not None:
        assert c_clip > 1.0, c_clip
        pg_loss3 = torch.sign(advantages) * c_clip * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)

    # Apply behavioural importance weight from rejection sampling
    if rejection_sampling is not None:
        behave_approx_kl = proximal_logprobs.detach() - old_logprobs.detach()
        behave_mask = (behave_imp_weight > 0).logical_and(loss_mask.bool())
        behave_approx_kl = torch.where(behave_mask, behave_approx_kl, 0.0)
        pg_loss = pg_loss * behave_imp_weight

    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count
    clip_mask.logical_and_(loss_mask)
    dual_clip_mask.logical_and_(loss_mask)
    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=(logprobs - proximal_logprobs).detach(),
        clip_mask=clip_mask,
        dual_clip_mask=dual_clip_mask,
    )
    if use_frpo_loss_term and frpo_beta != 0 and frpo_clip_eps > 0:
        stat.update(
            frpo_clipped_logratio=clipped_logratio.detach(),
            frpo_tail=tail.detach(),
            frpo_correction=frpo_correction.detach(),
            frpo_local_correction=frpo_local_correction.detach(),
        )
    if use_fipo and fipo_beta != 0:
        stat.update(
            fipo_delta_logp=delta_logp.detach(),
            fipo_outlier_mask=outlier_mask.detach(),
            fipo_tail=fipo_tail.detach(),
            fipo_f_t=f_t.detach(),
        )
    if rejection_sampling is not None:
        stat.update(
            behave_approx_kl=behave_approx_kl.detach(),
            behave_imp_weight=behave_imp_weight.detach(),
            behave_mask=behave_mask,
            filtered_fraction=filtered_fraction,
        )
    return pg_loss, stat


def sapo_loss_fn(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    tau_pos: float,
    tau_neg: float,
    loss_mask: torch.Tensor,
    importance_sampling_level: str = "token",
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """SAPO (Soft Adaptive Policy Optimization) loss with asymmetric sigmoid gates.

    SAPO replaces PPO clipping with soft sigmoid gates, providing smooth gradients.
    Note: SAPO requires use_decoupled_loss=False.

    Args:
        logprobs: Current policy log probabilities
        old_logprobs: Old policy log probabilities
        advantages: Advantage values
        tau_pos: Temperature for positive advantages (higher = sharper gate)
        tau_neg: Temperature for negative advantages (higher = sharper gate)
        loss_mask: Mask for valid tokens
        importance_sampling_level: "token" or "sequence" level importance sampling
        cu_seqlens: Cumulative sequence lengths for sequence-level IS

    Returns:
        Tuple of (loss, statistics dict compatible with PPO)
    """
    if tau_pos <= 0 or tau_neg <= 0:
        raise ValueError("SAPO temperatures (tau_pos, tau_neg) must be positive.")
    loss_mask_count = loss_mask.count_nonzero() or 1
    advantages = advantages.detach()
    log_ratio = logprobs - old_logprobs

    if importance_sampling_level == "sequence":
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(
            log_ratio, advantages, loss_mask, cu_seqlens
        )
    elif importance_sampling_level == "token":
        ratio = torch.exp(log_ratio)
    else:
        raise ValueError(
            f"Invalid importance_sampling_level: {importance_sampling_level}. "
            "Must be 'token' or 'sequence'."
        )

    # SAPO: Asymmetric sigmoid gates with 4/τ gradient normalization
    gate_pos = torch.sigmoid(tau_pos * (ratio - 1.0))
    gate_neg = torch.sigmoid(tau_neg * (ratio - 1.0))
    scale_pos = 4.0 / tau_pos
    scale_neg = 4.0 / tau_neg
    scaled_gate_pos = gate_pos * scale_pos
    scaled_gate_neg = gate_neg * scale_neg

    # Select gate based on advantage sign
    is_positive = advantages > 0
    soft_gate = torch.where(is_positive, scaled_gate_pos, scaled_gate_neg)

    # Compute loss
    pg_loss = -soft_gate * advantages
    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count

    # Return stat dict compatible with PPO (fake clip_mask for logging compatibility)
    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=log_ratio.detach(),
        clip_mask=torch.zeros_like(loss_mask, dtype=torch.bool),  # SAPO doesn't clip
        dual_clip_mask=torch.zeros_like(loss_mask, dtype=torch.bool),
        # SAPO-specific stats (scaled gates for consistency)
        sapo_soft_gate=soft_gate.detach(),
        sapo_scaled_gate_pos=scaled_gate_pos.detach(),
        sapo_scaled_gate_neg=scaled_gate_neg.detach(),
    )

    return pg_loss, stat


def dpo_pair_logratios(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    cu_seqlens: torch.Tensor,
    loss_mask: torch.Tensor,
    valid_pairs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate per-sequence masked logprobs over a packed batch and pair them up.

    Sequences are packed and interleaved as ``[chosen_0, rejected_0, chosen_1,
    rejected_1, ...]``. ``loss_mask`` is shifted by one to align with
    next-token logprobs, and each sequence's final position is zeroed (no next
    token exists for it). Aggregation uses ``fp64`` scatter-add to avoid
    precision loss on long (~2k+ tok) pairs where ``fp32`` accumulation can
    flip the log-ratio sign.

    Returns:
        ``(policy_logps, ref_logps, completion_lens)`` each of shape ``(K, 2)``
        where ``K = valid_pairs.sum()``, column 0 is chosen, column 1 is
        rejected. ``completion_lens`` counts the shifted-mask response tokens
        per sequence (needed by IPO for per-token normalization).
    """
    device = logprobs.device

    shifted_mask = torch.zeros_like(loss_mask)
    if loss_mask.shape[0] > 1:
        shifted_mask[:-1] = loss_mask[1:]
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    shifted_mask.index_fill_(0, (cu_seqlens[1:] - 1)[seqlens > 0], False)

    n_seqs = seqlens.shape[0]
    seq_ids = torch.repeat_interleave(torch.arange(n_seqs, device=device), seqlens)
    masked = shifted_mask.to(dtype=logprobs.dtype)
    policy_logps = torch.zeros(n_seqs, dtype=torch.float64, device=device)
    ref_logps = torch.zeros(n_seqs, dtype=torch.float64, device=device)
    policy_logps.index_add_(0, seq_ids, (logprobs * masked).double())
    ref_logps.index_add_(0, seq_ids, (ref_logprobs * masked).double())

    completion_lens = torch.zeros(n_seqs, dtype=torch.float64, device=device)
    completion_lens.index_add_(0, seq_ids, shifted_mask.double())

    return (
        policy_logps.view(-1, 2)[valid_pairs],
        ref_logps.view(-1, 2)[valid_pairs],
        completion_lens.view(-1, 2)[valid_pairs],
    )


def dpo_preference_loss(
    logits: torch.Tensor, *, beta: float, loss_type: str = "sigmoid"
) -> torch.Tensor:
    """Per-pair preference loss from DPO log-ratio logits.

    For ``"sigmoid"``, ``logits`` is the un-normalized pair delta.
    For ``"ipo"``, ``logits`` must be **per-token averaged** (length-normalized)
    before being passed here, matching trl's confirmed-with-authors convention.
    """
    if loss_type == "sigmoid":
        return -torch.nn.functional.logsigmoid(beta * logits.float())
    if loss_type == "ipo":
        return (logits.float() - 1.0 / (2.0 * beta)) ** 2
    raise ValueError(f"Unsupported DPO loss_type: {loss_type!r}")


def _huber_loss(x: torch.Tensor, y: torch.Tensor, delta: float):
    diff = torch.abs(x - y)
    return torch.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))


def _mse_loss(x: torch.Tensor, y: torch.Tensor):
    return 0.5 * (x - y) ** 2


def ppo_critic_loss_fn(
    value: torch.FloatTensor,
    old_value: torch.FloatTensor,
    target_value: torch.FloatTensor,
    value_eps_clip: float,
    loss_mask: torch.Tensor | None = None,
    loss_fn_type: str = "mse",
) -> tuple[torch.Tensor, dict]:
    """Compute PPO critic loss function given padded batch inputs.

    There is no shape requirements for the inputs, but they must have the same shape.
    Either [bs, max_seqlen] for batch padded inputs or [tot_seqlen] for padded inputs.

    Args:
        value (torch.FloatTensor): Values. The position of the final token is not included.
            (The whole generated sequence is not a state.)
        old_value (torch.FloatTensor): Old values.
        target_value (torch.FloatTensor): Returns computed by GAE.
        value_eps_clip (float): Clip ratio.
        loss_mask (Optional[torch.Tensor], optional): Mask for loss computation.
            1 if valid else 0. Defaults to None.
        loss_fn_type (str, optional): Type of loss function. Defaults to 'mse'.

    Returns:
        Tuple[torch.Tensor, Dict]: Scalar loss and statistics.
    """
    assert value.dtype == torch.float32
    assert old_value.dtype == torch.float32
    assert target_value.dtype == torch.float32

    if loss_fn_type == "huber":
        loss_fn = functools.partial(_huber_loss, delta=10.0)
    elif loss_fn_type == "mse":
        loss_fn = _mse_loss
    else:
        raise NotImplementedError(f"Unknown loss fn type: {loss_fn_type}")

    if target_value.is_inference():
        target_value = target_value.clone()  # clone a inference tensor

    value_loss_original = loss_fn(value, target_value)

    value_clipped = old_value + (value - old_value).clamp(
        -value_eps_clip, value_eps_clip
    )

    value_loss_clipped = loss_fn(value_clipped, target_value)

    value_loss = torch.max(value_loss_original, value_loss_clipped)

    with torch.no_grad():
        clip_mask = value_loss_clipped.detach() > value_loss_original.detach()
        if loss_mask is not None:
            clip_mask.logical_and_(loss_mask)

        stat = dict(clip_mask=clip_mask, loss=value_loss.detach())

    if loss_mask is not None:
        value_loss = (
            torch.where(loss_mask, value_loss, 0).sum() / loss_mask.count_nonzero()
        )
    else:
        value_loss = value_loss.mean()

    return value_loss, stat


# code modified from VERL: https://github.com/volcengine/verl/blob/main/verl/workers/reward_manager/dapo.py
def reward_overlong_penalty(
    data: dict[str, Any],
    overlong_tokens: int,
    overlong_penalty_factor: float,
    max_response_length: int,
) -> dict[str, Any]:
    reward_score = data["rewards"]
    input_ids = data["input_ids"]
    response_lengths = (data["loss_mask"].sum(dim=-1)).long()
    batch_size = input_ids.shape[0]
    for sample_idx in range(batch_size):
        reward_score_cur = reward_score[sample_idx]
        response_length_cur = response_lengths[sample_idx]
        expected_len = max_response_length - overlong_tokens
        exceed_len = response_length_cur - expected_len
        overlong_reward = min(
            -exceed_len / overlong_tokens * overlong_penalty_factor, 0
        )
        reward_score_cur += overlong_reward
        reward_score[sample_idx] = reward_score_cur

    data["rewards"] = reward_score
    return data
