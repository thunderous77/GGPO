# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Exact whole-training-batch rollout/current-policy mismatch statistics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


INTERNAL_MISMATCH_STATS_KEY = "__rollout_training_mismatch_stats__"
MISMATCH_AGGREGATION_META = "_rollout_training_mismatch_aggregation"
MISMATCH_PREFIX = "rollout_training_mismatch/"
RAW_IS_PREFIX = f"{MISMATCH_PREFIX}raw_is/"


@dataclass(frozen=True)
class MetricAggregation:
    """Explicit instructions for merging one scalar across learner steps.

    ``statistic`` is required only for records backed by raw samples.  Keeping
    the sufficient statistics beside the emitted scalar prevents a later
    window aggregator from guessing semantics from the metric name.
    """

    kind: str
    weight: int | float = 1
    total: float | None = None
    sum_of_squares: float | None = None
    samples: tuple[float, ...] = ()
    statistic: str | None = None


class MetricsWithAggregation(dict[str, float | int | None]):
    """Scalar metrics plus their explicit cross-step aggregation metadata."""

    def __init__(self, values: dict[str, float | int | None], aggregation: dict[str, MetricAggregation]):
        super().__init__(values)
        if values.keys() != aggregation.keys():
            raise ValueError("every emitted metric must carry aggregation metadata")
        self.aggregation = aggregation


ALP_MATH_VALIDATION_KEYS = {
    "val-core/alp_math500/acc/mean@32",
    "val-core/alp_math500/acc/best@32/mean",
    "val-core/alp_math500/acc/best@32/std",
}


def prune_sync_alp_math_validation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep the deliberately small, reference-compatible ALP Math schema."""
    if not any("/alp_math500/" in key for key in metrics):
        return metrics
    return {key: value for key, value in metrics.items() if key in ALP_MATH_VALIDATION_KEYS}


def prune_sync_alp_math_training_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """ALP Math is single-turn and has no tool-call health dimension."""
    omitted_dimensions = {"num_turns", "tool_calls", "tool_call_counts"}
    return {
        key: value
        for key, value in metrics.items()
        if omitted_dimensions.isdisjoint(key.split("/"))
    }


def compute_mismatch_stats(
    current_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, Any]:
    """Build mergeable statistics from the log-probs used by one loss forward."""
    if current_log_probs.shape != rollout_log_probs.shape or current_log_probs.shape != response_mask.shape:
        raise ValueError(
            "current log-probs, rollout log-probs, and the loss response mask must have identical shapes: "
            f"{tuple(current_log_probs.shape)}, {tuple(rollout_log_probs.shape)}, {tuple(response_mask.shape)}"
        )
    mask = response_mask.to(bool)
    if not mask.any():
        raise ValueError("The loss response mask contains no tokens for rollout/training mismatch measurement.")

    current = current_log_probs.detach().float()
    rollout = rollout_log_probs.detach().float()
    log_ratio_matrix = current - rollout
    log_ratio = log_ratio_matrix[mask].double()
    probs_diff = (current.exp() - rollout.exp()).abs()[mask].double()
    k3 = torch.exp(log_ratio) - log_ratio - 1.0
    active_responses = mask.any(dim=-1)
    sequence_log_ratio = (log_ratio_matrix.double() * mask).sum(dim=-1)[active_responses]

    if not (
        torch.isfinite(log_ratio).all()
        and torch.isfinite(probs_diff).all()
        and torch.isfinite(k3).all()
        and torch.isfinite(sequence_log_ratio).all()
    ):
        raise FloatingPointError("Non-finite rollout/training mismatch statistic.")

    return {
        "token_count": int(log_ratio.numel()),
        "probs_diff_sum": probs_diff.sum().item(),
        "probs_diff_sq_sum": probs_diff.square().sum().item(),
        "log_ratio_sum": log_ratio.sum().item(),
        "log_ratio_sq_sum": log_ratio.square().sum().item(),
        "log_ratio_abs_sum": log_ratio.abs().sum().item(),
        "k3_sum": k3.sum().item(),
        "sequence_log_ratios": sequence_log_ratio.cpu().tolist(),
    }


def empty_mismatch_stats() -> dict[str, Any]:
    """Return the merge identity for a padding-only actor microbatch."""
    return {
        "token_count": 0,
        "probs_diff_sum": 0.0,
        "probs_diff_sq_sum": 0.0,
        "log_ratio_sum": 0.0,
        "log_ratio_sq_sum": 0.0,
        "log_ratio_abs_sum": 0.0,
        "k3_sum": 0.0,
        "sequence_log_ratios": [],
    }


def _iter_stats(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_stats(item)
    else:
        raise TypeError(f"Unexpected mismatch accumulator value: {type(value)!r}")


def finalize_mismatch_metrics(value: Any) -> MetricsWithAggregation:
    """Merge all microbatch/minibatch/DP sufficient statistics exactly once."""
    total = {
        "token_count": 0,
        "probs_diff_sum": 0.0,
        "probs_diff_sq_sum": 0.0,
        "log_ratio_sum": 0.0,
        "log_ratio_sq_sum": 0.0,
        "log_ratio_abs_sum": 0.0,
        "k3_sum": 0.0,
    }
    sequence_values: list[float] = []
    for stats in _iter_stats(value):
        for key in total:
            total[key] += stats[key]
        sequence_values.extend(stats["sequence_log_ratios"])

    count = int(total["token_count"])
    if count <= 0 or not sequence_values:
        raise ValueError("Cannot finalize an empty rollout/training mismatch accumulator.")

    probs_mean = total["probs_diff_sum"] / count
    log_ratio_mean = total["log_ratio_sum"] / count
    probs_var = max(0.0, total["probs_diff_sq_sum"] / count - probs_mean * probs_mean)
    log_ratio_var = max(0.0, total["log_ratio_sq_sum"] / count - log_ratio_mean * log_ratio_mean)
    sequences = np.asarray(sequence_values, dtype=np.float64)
    metrics: dict[str, float] = {
        f"{MISMATCH_PREFIX}probs_diff_mean": probs_mean,
        f"{MISMATCH_PREFIX}probs_diff_std": float(np.sqrt(probs_var)),
        f"{MISMATCH_PREFIX}log_ratio_mean": log_ratio_mean,
        f"{MISMATCH_PREFIX}log_ratio_std": float(np.sqrt(log_ratio_var)),
        f"{MISMATCH_PREFIX}log_ratio_abs_mean": total["log_ratio_abs_sum"] / count,
        f"{MISMATCH_PREFIX}k3_kl": total["k3_sum"] / count,
        f"{MISMATCH_PREFIX}sequence_log_ratio_p95": float(np.quantile(sequences, 0.95)),
        f"{MISMATCH_PREFIX}sequence_log_ratio_max": float(sequences.max()),
    }

    # Passive sequence-level IS diagnostics are measurement only.  No weight
    # is dispatched back to the policy loss.  Stable log-space arithmetic is
    # essential here because the large-mismatch phase is expected to produce
    # extremely skewed raw weights.
    sequence_count = int(sequences.size)
    log_weight_sum = float(np.logaddexp.reduce(sequences))
    log_weight_square_sum = float(np.logaddexp.reduce(2.0 * sequences))
    log_count = float(np.log(sequence_count))
    normalized_weights = np.exp(sequences - log_weight_sum)
    descending_weights = np.sort(normalized_weights)[::-1]
    cumulative_weight = np.cumsum(descending_weights)

    def coverage_fraction(target: float) -> float:
        return float((np.searchsorted(cumulative_weight, target, side="left") + 1) / sequence_count)

    def top_weight_mass(fraction: float) -> float:
        selected = max(1, int(np.ceil(sequence_count * fraction)))
        return float(descending_weights[:selected].sum())

    raw_is_metrics = {
        f"{RAW_IS_PREFIX}sequence_count": float(sequence_count),
        f"{RAW_IS_PREFIX}sequence_log_ratio_mean": float(sequences.mean()),
        f"{RAW_IS_PREFIX}sequence_log_ratio_std": float(sequences.std()),
        f"{RAW_IS_PREFIX}sequence_log_ratio_p95": float(np.quantile(sequences, 0.95)),
        f"{RAW_IS_PREFIX}sequence_log_ratio_p99": float(np.quantile(sequences, 0.99)),
        f"{RAW_IS_PREFIX}sequence_log_ratio_p999": float(np.quantile(sequences, 0.999)),
        f"{RAW_IS_PREFIX}sequence_log_ratio_max": float(sequences.max()),
        f"{RAW_IS_PREFIX}positive_log_ratio_fraction": float(np.mean(sequences > 0.0)),
        f"{RAW_IS_PREFIX}log_mean_importance_weight": log_weight_sum - log_count,
        f"{RAW_IS_PREFIX}d2_log_second_moment_proxy": log_weight_square_sum - log_count,
        f"{RAW_IS_PREFIX}ess_normalized": float(
            np.exp(min(0.0, 2.0 * log_weight_sum - log_weight_square_sum - log_count))
        ),
        f"{RAW_IS_PREFIX}max_normalized_weight": float(descending_weights[0]),
        f"{RAW_IS_PREFIX}top_1pct_weight_mass": top_weight_mass(0.01),
        f"{RAW_IS_PREFIX}top_5pct_weight_mass": top_weight_mass(0.05),
        f"{RAW_IS_PREFIX}responses_for_50pct_weight_mass_fraction": coverage_fraction(0.50),
        f"{RAW_IS_PREFIX}responses_for_90pct_weight_mass_fraction": coverage_fraction(0.90),
    }
    metrics.update(raw_is_metrics)

    if not all(np.isfinite(metric) for metric in metrics.values()):
        raise FloatingPointError("The finalized rollout/training mismatch record is incomplete or non-finite.")

    samples = tuple(float(item) for item in sequences)
    aggregation = {
        f"{MISMATCH_PREFIX}probs_diff_mean": MetricAggregation(
            kind="mean", weight=count, total=float(total["probs_diff_sum"])
        ),
        f"{MISMATCH_PREFIX}probs_diff_std": MetricAggregation(
            kind="std",
            weight=count,
            total=float(total["probs_diff_sum"]),
            sum_of_squares=float(total["probs_diff_sq_sum"]),
        ),
        f"{MISMATCH_PREFIX}log_ratio_mean": MetricAggregation(
            kind="mean", weight=count, total=float(total["log_ratio_sum"])
        ),
        f"{MISMATCH_PREFIX}log_ratio_std": MetricAggregation(
            kind="std",
            weight=count,
            total=float(total["log_ratio_sum"]),
            sum_of_squares=float(total["log_ratio_sq_sum"]),
        ),
        f"{MISMATCH_PREFIX}log_ratio_abs_mean": MetricAggregation(
            kind="mean", weight=count, total=float(total["log_ratio_abs_sum"])
        ),
        f"{MISMATCH_PREFIX}k3_kl": MetricAggregation(
            kind="mean", weight=count, total=float(total["k3_sum"])
        ),
        f"{MISMATCH_PREFIX}sequence_log_ratio_p95": MetricAggregation(
            kind="raw_samples", samples=samples, statistic="p95"
        ),
        f"{MISMATCH_PREFIX}sequence_log_ratio_max": MetricAggregation(kind="max"),
    }
    raw_is_statistics = {
        "sequence_count": "count",
        "sequence_log_ratio_mean": "mean",
        "sequence_log_ratio_std": "std",
        "sequence_log_ratio_p95": "p95",
        "sequence_log_ratio_p99": "p99",
        "sequence_log_ratio_p999": "p999",
        "sequence_log_ratio_max": "max",
        "positive_log_ratio_fraction": "positive_fraction",
        "log_mean_importance_weight": "log_mean_weight",
        "d2_log_second_moment_proxy": "d2_log_second_moment",
        "ess_normalized": "ess_normalized",
        "max_normalized_weight": "max_normalized_weight",
        "top_1pct_weight_mass": "top_1pct_weight_mass",
        "top_5pct_weight_mass": "top_5pct_weight_mass",
        "responses_for_50pct_weight_mass_fraction": "coverage_50",
        "responses_for_90pct_weight_mass_fraction": "coverage_90",
    }
    aggregation.update(
        {
            f"{RAW_IS_PREFIX}{name}": MetricAggregation(
                kind="raw_samples",
                samples=samples,
                statistic=statistic,
            )
            for name, statistic in raw_is_statistics.items()
        }
    )
    return MetricsWithAggregation(metrics, aggregation)
