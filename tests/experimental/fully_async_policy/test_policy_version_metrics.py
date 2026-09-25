# Copyright 2026 OpenAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import pytest

from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    compute_policy_version_metrics,
)


def test_compute_policy_version_metrics_reports_signed_distribution():
    metrics = compute_policy_version_metrics(
        learner_version=7,
        rollout_versions=[7, 6, 3, 2, 8],
        reference_delta=4,
    )

    assert metrics["fully_async/version/learner_version"] == 7
    assert metrics["fully_async/version/rollout_version_min"] == 2
    assert metrics["fully_async/version/rollout_version_max"] == 8
    assert metrics["fully_async/version/rollout_version_diversity"] == 5
    assert metrics["fully_async/version/delta_min"] == -1
    assert metrics["fully_async/version/delta_max"] == 5
    assert metrics["fully_async/version/delta_mean"] == 1.8
    assert metrics["fully_async/version/delta_p50"] == 1.0
    assert metrics["fully_async/version/delta_p95"] == 4.8
    assert metrics["fully_async/version/future_rollout_count"] == 1
    assert metrics["fully_async/version/delta_0_count"] == 1
    assert metrics["fully_async/version/delta_1_count"] == 1
    assert metrics["fully_async/version/delta_4_count"] == 1
    assert metrics["fully_async/version/over_age_count"] == 1


def test_compute_policy_version_metrics_rejects_fractional_versions():
    with pytest.raises(ValueError, match="exact int64 integers"):
        compute_policy_version_metrics(learner_version=7, rollout_versions=[7, 6.5])


def test_compute_policy_version_metrics_accepts_empty_versions():
    assert compute_policy_version_metrics(learner_version=0, rollout_versions=[]) == {}


def test_policy_version_window_pools_token_age_distribution():
    first = compute_policy_version_metrics(learner_version=7, rollout_versions=[7])
    second = compute_policy_version_metrics(learner_version=7, rollout_versions=[6, 5, 4])
    aggregator = MetricsAggregator(total_gpus=1)
    aggregator.add_step_metrics(first, aggregation=first.aggregation)
    aggregator.add_step_metrics(second, aggregation=second.aggregation)

    merged = aggregator.get_aggregated_metrics()
    pooled_deltas = np.asarray([0, 1, 2, 3])
    assert merged["fully_async/version/delta_mean"] == pytest.approx(pooled_deltas.mean())
    assert merged["fully_async/version/delta_p95"] == pytest.approx(np.quantile(pooled_deltas, 0.95))
    assert merged["fully_async/version/rollout_version_diversity"] == 4
