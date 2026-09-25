# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.trainer.ppo.rollout_training_mismatch import MetricAggregation, MetricsWithAggregation


@dataclass
class RolloutSample:
    """Enhanced rollout sample containing both original batch info and AgentLoopOutput"""

    # Original batch information
    full_batch: Any

    # Metadata
    sample_id: str
    epoch: int

    # Processing metadata
    rollout_status: dict[str, Any]


@dataclass(frozen=True)
class RolloutGroupAdmission:
    """Result of validating one prompt group's rollout payload."""

    accepted: bool
    reason: str
    policy_version: int | None = None


def _parse_integral_policy_versions(values: np.ndarray) -> np.ndarray | None:
    """Return exact int64 policy versions without lossy coercion.

    NumPy's ``astype(int64)`` accepts values such as ``3.5``, ``"3"`` and
    booleans.  Version metadata is a safety boundary for asynchronous GGPO, so
    accept only genuine Python/NumPy integers and fail closed otherwise.
    """
    parsed: list[int] = []
    int64 = np.iinfo(np.int64)
    for value in values:
        if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
            return None
        value = int(value)
        if value < int64.min or value > int64.max:
            return None
        parsed.append(value)
    return np.asarray(parsed, dtype=np.int64)


def validate_rollout_group_admission(
    rollout_sample: RolloutSample,
    *,
    expected_group_size: int,
    learner_version: int,
    max_policy_version_lag: int | None,
    require_single_policy_version: bool,
    seen_group_ids: set[Any] | None = None,
) -> RolloutGroupAdmission:
    """Validate group atomicity and behavior-policy version before dequeue.

    A message-queue item represents one prompt group.  Version ``-1`` is
    reserved for prompt tokens by AReaL and is therefore ignored here; the
    rollouter stores response-token extrema in ``min_global_steps`` and
    ``max_global_steps``.
    """
    if expected_group_size < 1:
        raise ValueError(f"expected_group_size must be positive, got {expected_group_size}")
    if max_policy_version_lag is not None and max_policy_version_lag < 0:
        raise ValueError(f"max_policy_version_lag must be non-negative or null, got {max_policy_version_lag}")

    batch = rollout_sample.full_batch
    if len(batch) != expected_group_size:
        return RolloutGroupAdmission(False, "incomplete_group")

    uids = np.asarray(batch.non_tensor_batch.get("uid", []), dtype=object)
    # A nested UID payload such as ``[["g"], ["g"]]`` must not be flattened
    # into apparently valid scalar IDs.  Queue group identity is a safety
    # boundary, so require the canonical one-dimensional representation.
    if uids.ndim != 1:
        return RolloutGroupAdmission(False, "invalid_group_uid")
    if uids.size != expected_group_size:
        return RolloutGroupAdmission(False, "mixed_group_uid")
    uid_values = uids.tolist()
    try:
        for uid in uid_values:
            hash(uid)
    except TypeError:
        return RolloutGroupAdmission(False, "invalid_group_uid")
    if len(set(uid_values)) != 1:
        return RolloutGroupAdmission(False, "mixed_group_uid")
    group_id = uid_values[0]
    if seen_group_ids is not None and group_id in seen_group_ids:
        return RolloutGroupAdmission(False, "duplicate_group")

    if "rollout_policy_versions" not in batch.batch or "response_mask" not in batch.batch:
        return RolloutGroupAdmission(False, "missing_policy_version")
    versions = batch.batch["rollout_policy_versions"].detach()
    response_mask = batch.batch["response_mask"].detach().bool()
    if versions.shape != response_mask.shape or versions.ndim != 2 or versions.shape[0] != expected_group_size:
        return RolloutGroupAdmission(False, "invalid_policy_version_shape")
    if versions.is_floating_point() and (
        not torch.isfinite(versions).all() or not torch.equal(versions, versions.trunc())
    ):
        return RolloutGroupAdmission(False, "invalid_policy_version")
    versions = versions.long()
    if torch.any(versions[response_mask] < 0) or torch.any(versions[~response_mask] != -1):
        return RolloutGroupAdmission(False, "invalid_policy_version")
    valid_versions = versions[response_mask]
    if valid_versions.numel() == 0:
        return RolloutGroupAdmission(False, "missing_policy_version")
    if require_single_policy_version and torch.unique(valid_versions).numel() != 1:
        return RolloutGroupAdmission(False, "mixed_policy_version")
    behavior_version = int(valid_versions.max().item())
    if behavior_version > learner_version:
        return RolloutGroupAdmission(False, "future_policy_version", behavior_version)
    if max_policy_version_lag is not None and learner_version - int(valid_versions.min().item()) > max_policy_version_lag:
        return RolloutGroupAdmission(False, "stale_policy_version", behavior_version)

    if seen_group_ids is not None:
        seen_group_ids.add(group_id)
    return RolloutGroupAdmission(True, "accepted", behavior_version)


def compute_policy_version_metrics(
    learner_version: int,
    rollout_versions: Any,
    reference_delta: int = 8,
    response_mask: Any | None = None,
) -> MetricsWithAggregation:
    """Summarize the signed learner-minus-rollout policy-version delta.

    ``staleness_threshold`` in fully-async VERL is a sample-budget multiplier,
    not a policy-version cap.  Keeping this calculation separate makes the
    actual off-policy distance observable without changing official sampling
    or dropping behavior.
    """
    versions_array = np.asarray(rollout_versions)
    if response_mask is not None:
        mask = np.asarray(response_mask, dtype=bool)
        if mask.shape != versions_array.shape:
            raise ValueError("rollout_versions and response_mask must have identical shapes")
        if np.any(versions_array[~mask] != -1):
            raise ValueError("policy-version padding must be -1")
        selected_versions = versions_array[mask]
    else:
        selected_versions = versions_array.reshape(-1)
    versions = _parse_integral_policy_versions(selected_versions)
    if versions is None:
        raise ValueError("rollout policy versions must be exact int64 integers")
    if versions.size == 0:
        return MetricsWithAggregation({}, {})

    deltas = int(learner_version) - versions
    metrics: dict[str, int | float] = {
        "fully_async/version/learner_version": int(learner_version),
        "fully_async/version/rollout_version_min": int(versions.min()),
        "fully_async/version/rollout_version_max": int(versions.max()),
        "fully_async/version/rollout_version_mean": float(versions.mean()),
        "fully_async/version/rollout_version_diversity": int(np.unique(versions).size),
        "fully_async/version/delta_min": int(deltas.min()),
        "fully_async/version/delta_max": int(deltas.max()),
        "fully_async/version/delta_mean": float(deltas.mean()),
        "fully_async/version/delta_p50": float(np.percentile(deltas, 50)),
        "fully_async/version/delta_p95": float(np.percentile(deltas, 95)),
        "fully_async/version/future_rollout_count": int(np.count_nonzero(deltas < 0)),
        "fully_async/version/over_age_count": int(np.count_nonzero(deltas > reference_delta)),
        "fully_async/version/reference_delta": int(reference_delta),
    }
    for delta in range(reference_delta + 1):
        metrics[f"fully_async/version/delta_{delta}_count"] = int(np.count_nonzero(deltas == delta))
    version_samples = tuple(float(value) for value in versions)
    delta_samples = tuple(float(value) for value in deltas)
    token_count = int(versions.size)
    aggregation = {
        "fully_async/version/learner_version": MetricAggregation(kind="last"),
        "fully_async/version/rollout_version_min": MetricAggregation(kind="min"),
        "fully_async/version/rollout_version_max": MetricAggregation(kind="max"),
        "fully_async/version/rollout_version_mean": MetricAggregation(
            kind="mean", weight=token_count, total=float(versions.sum())
        ),
        "fully_async/version/rollout_version_diversity": MetricAggregation(
            kind="raw_samples", samples=version_samples, statistic="unique_count"
        ),
        "fully_async/version/delta_min": MetricAggregation(kind="min"),
        "fully_async/version/delta_max": MetricAggregation(kind="max"),
        "fully_async/version/delta_mean": MetricAggregation(
            kind="mean", weight=token_count, total=float(deltas.sum())
        ),
        "fully_async/version/delta_p50": MetricAggregation(
            kind="raw_samples", samples=delta_samples, statistic="p50"
        ),
        "fully_async/version/delta_p95": MetricAggregation(
            kind="raw_samples", samples=delta_samples, statistic="p95"
        ),
        "fully_async/version/future_rollout_count": MetricAggregation(kind="sum"),
        "fully_async/version/over_age_count": MetricAggregation(kind="sum"),
        "fully_async/version/reference_delta": MetricAggregation(kind="last"),
        **{
            f"fully_async/version/delta_{delta}_count": MetricAggregation(kind="sum")
            for delta in range(reference_delta + 1)
        },
    }
    if response_mask is not None:
        spans = []
        for row_versions, row_mask in zip(versions_array, np.asarray(response_mask, dtype=bool), strict=True):
            valid = row_versions[row_mask]
            spans.append(int(valid.max() - valid.min()))
        metrics["fully_async/version/partial_response_ratio"] = float(np.mean(np.asarray(spans) > 0))
        metrics["fully_async/version/response_version_span_mean"] = float(np.mean(spans))
        metrics["fully_async/version/response_version_span_max"] = int(np.max(spans))
        response_count = len(spans)
        aggregation.update(
            {
                "fully_async/version/partial_response_ratio": MetricAggregation(
                    kind="mean",
                    weight=response_count,
                    total=float(np.count_nonzero(np.asarray(spans) > 0)),
                ),
                "fully_async/version/response_version_span_mean": MetricAggregation(
                    kind="mean", weight=response_count, total=float(np.sum(spans))
                ),
                "fully_async/version/response_version_span_max": MetricAggregation(kind="max"),
            }
        )
    return MetricsWithAggregation(metrics, aggregation)


def compute_lagged_log_prob_metrics(
    *,
    learner_version: int,
    rollout_versions: Any,
    behavior_log_probs: torch.Tensor,
    current_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    max_lag: int = 4,
) -> MetricsWithAggregation:
    """Compute detached behavior/current diagnostics separately for each lag.

    Empty lag buckets are explicit ``count=0``/``None`` records.  The helper
    never returns a graph-connected tensor and therefore cannot alter the
    policy objective or gradients.  Response-level mismatch uses the signed
    raw sum over valid response tokens; it is never length-normalized.
    """
    versions = torch.as_tensor(rollout_versions, device=response_mask.device)
    behavior = behavior_log_probs.detach()
    current = current_log_probs.detach()
    mask = response_mask.detach().bool()
    if behavior.shape != current.shape or behavior.shape != mask.shape:
        raise ValueError("behavior/current log probabilities and response_mask must have identical shapes")
    if versions.shape != mask.shape:
        raise ValueError("rollout_versions must contain one policy version per response token")
    if torch.any(versions[~mask] != -1) or torch.any(versions[mask] < 0):
        raise ValueError("valid token versions must be non-negative and padding must be -1")
    versions = versions.long()

    metrics: dict[str, int | float | None] = {}
    aggregation: dict[str, MetricAggregation] = {}
    token_delta = current - behavior
    response_log_ratio_sum = torch.where(
        mask,
        token_delta.double(),
        torch.zeros_like(token_delta, dtype=torch.float64),
    ).sum(dim=-1)
    response_stat_names = ("mean", "std", "p50", "p95", "p99", "max")

    def add_response_stats(prefix: str, values: torch.Tensor) -> None:
        count = int(values.numel())
        count_name = f"{prefix}/response_log_ratio_sum_count"
        metrics[count_name] = count
        aggregation[count_name] = MetricAggregation(kind="sum")
        if count == 0:
            for name in response_stat_names:
                metric_name = f"{prefix}/response_log_ratio_sum_{name}"
                metrics[metric_name] = None
                aggregation[metric_name] = MetricAggregation(kind="single_step_only")
            return

        values = values.double()
        total = float(values.sum().item())
        sum_of_squares = float(values.square().sum().item())
        samples = tuple(float(item) for item in values.cpu().tolist())
        quantiles = torch.quantile(
            values,
            torch.tensor((0.50, 0.95, 0.99), dtype=values.dtype, device=values.device),
        )
        mean_name = f"{prefix}/response_log_ratio_sum_mean"
        std_name = f"{prefix}/response_log_ratio_sum_std"
        metrics[mean_name] = float(values.mean().item())
        aggregation[mean_name] = MetricAggregation(kind="mean", weight=count, total=total)
        metrics[std_name] = float(values.std(unbiased=False).item())
        aggregation[std_name] = MetricAggregation(
            kind="std", weight=count, total=total, sum_of_squares=sum_of_squares
        )
        for index, quantile_name in enumerate(("p50", "p95", "p99")):
            metric_name = f"{prefix}/response_log_ratio_sum_{quantile_name}"
            metrics[metric_name] = float(quantiles[index].item())
            aggregation[metric_name] = MetricAggregation(
                kind="raw_samples", samples=samples, statistic=quantile_name
            )
        max_name = f"{prefix}/response_log_ratio_sum_max"
        metrics[max_name] = float(values.max().item())
        aggregation[max_name] = MetricAggregation(kind="max")

    # Retain an all-version response-level view alongside the version buckets.
    # The mandatory aggregate GGPO Set-A record is computed independently; this
    # diagnostic exists to make its async behavior-version composition visible.
    add_response_stats("fully_async/mismatch/all", response_log_ratio_sum)
    for lag in range(max_lag + 1):
        token_mask = mask & versions.eq(int(learner_version) - lag)
        row_mask = token_mask.any(dim=-1)
        count = int(row_mask.sum().item())
        token_count = int(token_mask.sum().item())
        prefix = f"fully_async/mismatch/lag_{lag}"
        trajectory_count_name = f"{prefix}/trajectory_count"
        token_count_name = f"{prefix}/token_count"
        metrics[trajectory_count_name] = count
        metrics[token_count_name] = token_count
        aggregation[trajectory_count_name] = MetricAggregation(kind="sum")
        aggregation[token_count_name] = MetricAggregation(kind="sum")
        stratum_response_sum = torch.where(token_mask, token_delta.double(), 0.0).sum(dim=-1)
        add_response_stats(prefix, stratum_response_sum[row_mask])
        if token_count == 0:
            for suffix in ("abs_log_ratio_mean", "log_ratio_mean", "chi2", "k3_kl"):
                metric_name = f"{prefix}/{suffix}"
                metrics[metric_name] = None
                aggregation[metric_name] = MetricAggregation(kind="single_step_only")
            continue

        delta = token_delta[token_mask].double()
        ratio = torch.exp(delta)
        token_values = {
            "abs_log_ratio_mean": delta.abs(),
            "log_ratio_mean": delta,
            "chi2": torch.square(ratio - 1.0),
            "k3_kl": ratio - delta - 1.0,
        }
        for suffix, values in token_values.items():
            metric_name = f"{prefix}/{suffix}"
            metrics[metric_name] = float(values.mean().item())
            aggregation[metric_name] = MetricAggregation(
                kind="mean",
                weight=token_count,
                total=float(values.double().sum().item()),
            )
    return MetricsWithAggregation(metrics, aggregation)


def prepare_single_generation_data(batch_dict, config) -> DataProto:
    """
    Similar to the logic of ray_trainer._prepare_generate_batch, but for a single sample.
    Separate the data used for generation from the original data.

    Returns:
        tuple: (original_batch_dict, gen_data_for_single_sample)
    """

    full_batch = DataProto.from_single_dict(batch_dict)

    # Setting selected agent, that supports partial
    if not config.actor_rollout_ref.rollout.multi_turn.enable:
        full_batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(full_batch), dtype=object)

    # Add global step count to generated data
    full_batch = full_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
    return full_batch


def assemble_batch_from_rollout_samples(rollout_samples: list[RolloutSample], balance_batch=None) -> DataProto:
    """
    Assemble gen_batch_output from RolloutSample objects
    Assembles batches from RolloutSample objects, similar to the _post_generate_batch logic in ray_trainer.

    Args:
        rollout_samples: List of RolloutSample objects
        balance_batch: Whether to balance the batch (simplified version)

    Returns:
        DataProto: Assembled gen_batch_output

    Raises:
        ValueError: If rollout_samples is empty
    """
    start_time = time.time()

    if not rollout_samples:
        raise ValueError("Empty rollout_samples provided for batch assembly")

    print(f"[BatchUtils] Assembling batch from {len(rollout_samples)} RolloutSample objects")

    rollout_samples_batch = []
    # Rollout status is cumulative. Queue order is assembly order, so the last
    # accepted sample carries the newest snapshot for this assembled batch.
    rollout_status = rollout_samples[-1].rollout_status
    # Add a prefix to all rollout_status keys
    rollout_status = {f"fully_async/{key}": value for key, value in rollout_status.items()}

    for rs in rollout_samples:
        batch = rs.full_batch
        metrics = batch.meta_info.pop("metrics")  # List[Dict[str, str]]
        batch.non_tensor_batch["processing_times"] = [item["generate_sequences"] for item in metrics]
        batch.non_tensor_batch["tool_calls_times"] = [item["tool_calls"] for item in metrics]
        rollout_samples_batch.append(batch)
    final_batch = DataProto.concat(rollout_samples_batch)

    # Calculate response_mask (if not present)
    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    if balance_batch:
        balance_batch(final_batch, metrics={})

    # Calculate the global valid token number
    if "attention_mask" in final_batch.batch:
        final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

    processing_times = final_batch.non_tensor_batch["processing_times"]
    tool_calls = final_batch.non_tensor_batch["tool_calls_times"]
    # Collect statistics
    processing_time_stats = {
        "processing_time/avg": np.mean(processing_times),
        "processing_time/max": np.max(processing_times),
        "processing_time/min": np.min(processing_times),
        "processing_time/tp50": np.percentile(processing_times, 50),
        "processing_time/tp99": np.percentile(processing_times, 99),
        "processing_time/tp95": np.percentile(processing_times, 95),
    }
    tool_calls_stats = {}
    if len(tool_calls) > 0:
        tool_calls_stats = {
            "timing_s/agent_loop/tool_calls/max": np.max(tool_calls),
            "timing_s/agent_loop/tool_calls/min": np.min(tool_calls),
            "timing_s/agent_loop/tool_calls/mean": np.mean(tool_calls),
        }
    processing_time_stats = {f"fully_async/{key}": value for key, value in processing_time_stats.items()}

    param_version_start = final_batch.non_tensor_batch["min_global_steps"]
    param_version_end = final_batch.non_tensor_batch["max_global_steps"]
    param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
    num_diff0 = param_version_diff.count(0)
    partial_stats = {
        "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
        "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff),
        "fully_async/partial/max_partial_span": max(param_version_diff),
    }
    # add meta_info
    trajectory_param_versions = final_batch.non_tensor_batch["max_global_steps"]

    final_batch.meta_info.update(
        {
            "param_version_diversity": len(set(trajectory_param_versions)),
            "trajectory_param_versions": trajectory_param_versions,
            **processing_time_stats,
            **rollout_status,
            **partial_stats,
            **tool_calls_stats,
        }
    )

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return final_batch


_FULLY_ASYNC_LAST_METRICS = frozenset(
    {
        "fully_async/count/total_generated_samples",
        "fully_async/count/stale_trajectory_processed",
        "fully_async/count/current_param_version",
        "fully_async/count/completed_optimizer_steps",
        "fully_async/count/dropped_stale_samples",
        "fully_async/monitor/queue/mq_total_produced",
        "fully_async/monitor/queue/mq_total_consumed",
        "fully_async/monitor/queue/mq_dropped_samples",
        "training/global_step",
        "training/epoch",
        "actor/lr",
    }
)
_FULLY_ASYNC_PROMPT_WEIGHTED_MEANS = frozenset(
    {
        "critic/score/mean",
        "actor/pg_loss",
        "actor/entropy",
        "actor/grad_norm",
        "actor/ggpo_loss",
        "actor/ggpo_raw_loss",
        "timing_s/agent_loop/tool_calls/mean",
    }
)
_FULLY_ASYNC_SUM_METRICS = frozenset(
    {
        "perf/time_per_step",
        "perf/total_num_tokens",
        "fully_async/actor_update_padding/logical_responses",
        "fully_async/actor_update_padding/physical_responses",
        "fully_async/actor_update_padding/padding_responses",
    }
)
_FULLY_ASYNC_MIN_METRICS = frozenset(
    {
        "fully_async/health/all_update_gates_passed",
        "actor/optimizer_applied",
        "timing_s/agent_loop/tool_calls/min",
    }
)
_FULLY_ASYNC_MAX_METRICS = frozenset(
    {
        "timing_s/agent_loop/tool_calls/max",
    }
)


def declare_fully_async_step_metrics(
    metrics: dict[str, Any],
    *,
    prompt_count: int,
    declared: dict[str, MetricAggregation] | None = None,
) -> dict[str, MetricAggregation]:
    """Attach an explicit window rule to every learner-step metric.

    Producers of distribution metrics pass mergeable statistics in
    ``declared``.  The small stable set below has intentionally explicit
    semantics.  Any other scalar is safe for a one-step publication only and
    is omitted when a policy-version window contains multiple learner steps.
    """
    if prompt_count <= 0:
        raise ValueError("fully-async prompt_count must be positive")
    aggregation = dict(declared or {})
    unknown_declared = aggregation.keys() - metrics.keys()
    if unknown_declared:
        raise ValueError(f"aggregation metadata has no emitted metric: {sorted(unknown_declared)}")
    for name in metrics:
        if name in aggregation:
            continue
        if name in _FULLY_ASYNC_LAST_METRICS or name.startswith("fully_async/admission/cumulative_"):
            aggregation[name] = MetricAggregation(kind="last")
        elif name in _FULLY_ASYNC_PROMPT_WEIGHTED_MEANS:
            aggregation[name] = MetricAggregation(kind="mean", weight=prompt_count)
        elif name in _FULLY_ASYNC_MIN_METRICS:
            aggregation[name] = MetricAggregation(kind="min")
        elif name in _FULLY_ASYNC_MAX_METRICS:
            aggregation[name] = MetricAggregation(kind="max")
        elif name in _FULLY_ASYNC_SUM_METRICS or name.startswith("timing_s/"):
            aggregation[name] = MetricAggregation(kind="sum")
        else:
            aggregation[name] = MetricAggregation(kind="single_step_only")
    return aggregation


class MetricsAggregator:
    """Merge learner-step metrics only from producer-declared statistics."""

    def __init__(self, total_gpus: int):
        self.metric_values: dict[str, list[tuple[float, MetricAggregation]]] = defaultdict(list)
        self.step_count = 0
        self.total_gpus = total_gpus

    def add_step_metrics(
        self,
        metrics: dict[str, Any],
        aggregation: dict[str, MetricAggregation],
    ) -> None:
        """Add one step after checking that every scalar declares its merge rule."""
        if metrics.keys() != aggregation.keys():
            missing = metrics.keys() - aggregation.keys()
            extra = aggregation.keys() - metrics.keys()
            raise ValueError(
                "fully-async metrics and aggregation metadata differ: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        self.step_count += 1
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(f"fully-async metric {key} must be scalar")
                value = value.detach().item()
            if not isinstance(value, int | float | np.number):
                raise TypeError(f"fully-async metric {key} has unsupported value {type(value)!r}")
            numeric = float(value)
            if not np.isfinite(numeric):
                raise FloatingPointError(f"fully-async metric {key} is non-finite")
            self.metric_values[key].append((numeric, aggregation[key]))

    @staticmethod
    def _raw_sample_statistic(samples: np.ndarray, statistic: str) -> float:
        if samples.size == 0:
            raise ValueError("raw-sample metric aggregation requires at least one sample")
        if statistic == "count":
            return float(samples.size)
        if statistic == "mean":
            return float(samples.mean())
        if statistic == "std":
            return float(samples.std())
        if statistic == "max":
            return float(samples.max())
        if statistic in {"p50", "p95", "p99", "p999"}:
            return float(np.quantile(samples, float(statistic[1:]) / (1000 if statistic == "p999" else 100)))
        if statistic == "positive_fraction":
            return float(np.mean(samples > 0.0))
        if statistic == "unique_count":
            return float(np.unique(samples).size)

        log_count = float(np.log(samples.size))
        log_weight_sum = float(np.logaddexp.reduce(samples))
        log_weight_square_sum = float(np.logaddexp.reduce(2.0 * samples))
        normalized_weights = np.exp(samples - log_weight_sum)
        descending = np.sort(normalized_weights)[::-1]
        if statistic == "log_mean_weight":
            return log_weight_sum - log_count
        if statistic == "d2_log_second_moment":
            return log_weight_square_sum - log_count
        if statistic == "ess_normalized":
            return float(np.exp(min(0.0, 2.0 * log_weight_sum - log_weight_square_sum - log_count)))
        if statistic == "max_normalized_weight":
            return float(descending[0])
        if statistic in {"top_1pct_weight_mass", "top_5pct_weight_mass"}:
            fraction = 0.01 if statistic.startswith("top_1") else 0.05
            count = max(1, int(np.ceil(samples.size * fraction)))
            return float(descending[:count].sum())
        if statistic in {"coverage_50", "coverage_90"}:
            target = 0.5 if statistic == "coverage_50" else 0.9
            count = int(np.searchsorted(np.cumsum(descending), target, side="left")) + 1
            return float(count / samples.size)
        raise ValueError(f"unsupported raw-sample statistic {statistic!r}")

    @classmethod
    def _aggregate_single_metric(cls, metric_name: str, records: list[tuple[float, MetricAggregation]]) -> float | None:
        if not records:
            return None
        kinds = {metadata.kind for _, metadata in records}
        statistics = {metadata.statistic for _, metadata in records}
        if len(kinds) != 1 or len(statistics) != 1:
            raise ValueError(f"fully-async metric {metric_name} changed aggregation metadata within one window")
        kind = next(iter(kinds))
        values = [value for value, _ in records]
        metadata = [item for _, item in records]
        if kind == "single_step_only":
            return values[0] if len(values) == 1 else None
        if kind == "last":
            return values[-1]
        if kind == "sum":
            return float(sum(values))
        if kind == "max":
            return max(values)
        if kind == "min":
            return min(values)
        if kind == "mean":
            weight = float(sum(item.weight for item in metadata))
            if weight <= 0:
                return None
            total = sum(
                item.total if item.total is not None else value * item.weight
                for value, item in records
            )
            return float(total / weight)
        if kind == "std":
            count = float(sum(item.weight for item in metadata))
            if count <= 0:
                return None
            if any(item.total is None or item.sum_of_squares is None for item in metadata):
                raise ValueError(f"fully-async std metric {metric_name} lacks sufficient statistics")
            total = float(sum(item.total for item in metadata if item.total is not None))
            sum_of_squares = float(
                sum(item.sum_of_squares for item in metadata if item.sum_of_squares is not None)
            )
            variance = max(0.0, sum_of_squares / count - (total / count) ** 2)
            return float(np.sqrt(variance))
        if kind == "raw_samples":
            samples = np.asarray(
                [sample for item in metadata for sample in item.samples],
                dtype=np.float64,
            )
            statistic = next(iter(statistics))
            if statistic is None:
                raise ValueError(f"fully-async raw metric {metric_name} lacks a statistic")
            return cls._raw_sample_statistic(samples, statistic)
        raise ValueError(f"fully-async metric {metric_name} has unsupported aggregation kind {kind!r}")

    def get_aggregated_metrics(self) -> dict[str, Any]:
        """aggregated metrics"""
        t = time.time()
        if self.step_count == 0:
            return {}

        aggregated = {}

        # Aggregate all metrics
        for metric_name, values in self.metric_values.items():
            if self.step_count != 1 and values and values[0][1].kind == "single_step_only":
                continue
            value = self._aggregate_single_metric(metric_name, values)
            if value is not None:
                aggregated[metric_name] = value

        # Aggregate special metrics
        aggregated = self._special_metrics_aggergate(aggregated)

        print(f"aggregated metrics done. cost {time.time() - t:.4f} seconds.")

        return aggregated

    def _special_metrics_aggergate(self, aggregated: dict[str, Any]) -> dict[str, Any]:
        """calculate special metrics"""

        # global_seqlen/minmax_diff
        if "global_seqlen/minmax_diff" in aggregated.keys():
            aggregated["global_seqlen/minmax_diff"] = aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]

        # perf/throughput
        REQUIRED_PERF_KEYS = {"perf/total_num_tokens", "perf/time_per_step"}
        if REQUIRED_PERF_KEYS.issubset(aggregated):
            aggregated["perf/throughput"] = aggregated["perf/total_num_tokens"] / (
                aggregated["perf/time_per_step"] * self.total_gpus
            )

        # trainer/idle_ratio
        if "timing_s/gen" in aggregated.keys() and "timing_s/step" in aggregated.keys():
            aggregated["fully_async/trainer/idle_ratio"] = aggregated["timing_s/gen"] / aggregated["timing_s/step"]

        return aggregated

    def reset(self):
        """Reset Aggregator"""
        self.metric_values.clear()
        self.step_count = 0


def task_exception_handler(task: asyncio.Task):
    """Handle task exceptions and log them"""
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task was cancelled, this is expected
    except Exception as e:
        print(f"Task {task.get_name()} failed with exception: {e}")
        raise e


def safe_create_task(coro, name: str, task_set: set = None):
    """Safely create a task with exception handling

    Args:
        coro: The coroutine to run
        name: Name for the task
        task_set: Optional set to add the task to

    Returns:
        The created asyncio.Task
    """
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(task_exception_handler)
    if task_set is not None:
        task_set.add(task)
    return task
