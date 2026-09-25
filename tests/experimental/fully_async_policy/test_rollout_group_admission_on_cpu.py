import numpy as np
import pytest
import torch

from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    RolloutSample,
    assemble_batch_from_rollout_samples,
    compute_lagged_log_prob_metrics,
    declare_fully_async_step_metrics,
    validate_rollout_group_admission,
)


class _Batch:
    def __init__(self, *, uids, min_versions, max_versions):
        self.non_tensor_batch = {"uid": np.asarray(uids, dtype=object)}
        self.batch = {}
        if min_versions is not None and max_versions is not None:
            self.batch["rollout_policy_versions"] = torch.tensor(
                [[lo, hi] for lo, hi in zip(min_versions, max_versions)]
            )
            self.batch["response_mask"] = torch.ones((len(uids), 2), dtype=torch.bool)

    def __len__(self):
        return len(self.non_tensor_batch["uid"])


def _sample(*, uids=("g", "g"), min_versions=(3, 3), max_versions=(3, 3)):
    return RolloutSample(
        full_batch=_Batch(uids=uids, min_versions=min_versions, max_versions=max_versions),
        sample_id="g",
        epoch=0,
        rollout_status={},
    )


def _admit(sample, *, learner_version=3, max_lag=0, single_version=True):
    return validate_rollout_group_admission(
        sample,
        expected_group_size=2,
        learner_version=learner_version,
        max_policy_version_lag=max_lag,
        require_single_policy_version=single_version,
    )


def test_accepts_exact_same_uid_same_version_group():
    result = _admit(_sample())
    assert result.accepted is True
    assert result.reason == "accepted"
    assert result.policy_version == 3


@pytest.mark.parametrize(
    ("sample", "reason"),
    [
        (_sample(uids=("g",)), "incomplete_group"),
        (_sample(uids=("g", "other")), "mixed_group_uid"),
        (_sample(min_versions=None), "missing_policy_version"),
        (_sample(min_versions=(-1, 3)), "invalid_policy_version"),
        (_sample(min_versions=(3.5, 3)), "invalid_policy_version"),
        (_sample(uids=(["g"], ["g"])), "invalid_group_uid"),
    ],
)
def test_rejects_malformed_groups_atomically(sample, reason):
    result = _admit(sample)
    assert result.accepted is False
    assert result.reason == reason


def test_ggpo_accepts_a_group_generated_across_policy_versions_when_single_version_is_not_required():
    result = _admit(
        _sample(min_versions=(2, 3), max_versions=(2, 3)),
        max_lag=1,
        single_version=False,
    )
    assert result.accepted is True


def test_agro_rejects_a_prompt_group_with_multiple_behavior_policy_versions():
    result = _admit(
        _sample(min_versions=(2, 3), max_versions=(2, 3)),
        max_lag=1,
        single_version=True,
    )
    assert (result.accepted, result.reason) == (False, "mixed_policy_version")


def test_rejects_a_duplicate_group_without_poisoning_the_seen_set():
    seen = set()
    first = validate_rollout_group_admission(
        _sample(),
        expected_group_size=2,
        learner_version=3,
        max_policy_version_lag=0,
        require_single_policy_version=True,
        seen_group_ids=seen,
    )
    duplicate = validate_rollout_group_admission(
        _sample(),
        expected_group_size=2,
        learner_version=3,
        max_policy_version_lag=0,
        require_single_policy_version=True,
        seen_group_ids=seen,
    )
    assert first.accepted is True
    assert (duplicate.accepted, duplicate.reason) == (False, "duplicate_group")
    assert seen == {"g"}


def test_rejects_future_and_stale_groups_against_actual_learner_version():
    future = _admit(_sample(min_versions=(4, 4), max_versions=(4, 4)), learner_version=3, max_lag=1)
    stale = _admit(_sample(min_versions=(1, 1), max_versions=(1, 1)), learner_version=3, max_lag=1)
    assert (future.accepted, future.reason) == (False, "future_policy_version")
    assert (stale.accepted, stale.reason) == (False, "stale_policy_version")


def test_invalid_lag_configuration_fails_instead_of_silently_disabling_the_gate():
    with pytest.raises(ValueError, match="non-negative"):
        _admit(_sample(), max_lag=-1)


def test_cumulative_admission_metrics_keep_the_latest_counter_value():
    aggregator = MetricsAggregator(total_gpus=1)
    key = "fully_async/admission/cumulative_accepted_groups"
    for value, count in ((2, 2), (7, 5)):
        metrics = {key: value}
        aggregator.add_step_metrics(
            metrics,
            aggregation=declare_fully_async_step_metrics(metrics, prompt_count=count),
        )

    assert aggregator.get_aggregated_metrics()[key] == 7


def test_batch_assembly_attaches_the_newest_cumulative_rollout_status():
    def sample(sample_id, version, status):
        batch = DataProto.from_single_dict(
            {
                "responses": torch.tensor([[1]]),
                "response_mask": torch.ones(1, 1, dtype=torch.bool),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
                "min_global_steps": np.asarray([version]),
                "max_global_steps": np.asarray([version]),
            }
        )
        batch.meta_info["metrics"] = [{"generate_sequences": 1.0, "tool_calls": 0.0}]
        return RolloutSample(
            full_batch=batch,
            sample_id=sample_id,
            epoch=0,
            rollout_status=status,
        )

    assembled = assemble_batch_from_rollout_samples(
        [
            sample("old", 1, {"count/total_generated_samples": 10}),
            sample("new", 2, {"count/total_generated_samples": 12}),
        ]
    )

    assert assembled.meta_info["fully_async/count/total_generated_samples"] == 12


def test_lagged_mismatch_metrics_are_detached_and_empty_buckets_are_null():
    behavior = torch.tensor([[0.0, -1.0], [-0.5, -0.5]], requires_grad=True)
    current = torch.tensor([[0.1, -1.2], [-0.5, -0.25]], requires_grad=True)
    metrics = compute_lagged_log_prob_metrics(
        learner_version=4,
        # DataProto concatenation produces a one-dimensional object array in
        # the real fully-async trainer.
        rollout_versions=torch.tensor([[4, 4], [3, 3]]),
        behavior_log_probs=behavior,
        current_log_probs=current,
        response_mask=torch.ones_like(behavior),
        max_lag=4,
    )
    assert metrics["fully_async/mismatch/lag_0/trajectory_count"] == 1
    assert metrics["fully_async/mismatch/lag_1/trajectory_count"] == 1
    assert metrics["fully_async/mismatch/lag_2/trajectory_count"] == 0
    assert metrics["fully_async/mismatch/lag_2/abs_log_ratio_mean"] is None
    expected_delta = current.detach() - behavior.detach()
    expected_k3 = torch.exp(expected_delta[0]) - expected_delta[0] - 1.0
    assert metrics["fully_async/mismatch/lag_0/k3_kl"] == pytest.approx(expected_k3.mean().item())
    assert all("kl_behavior_current" not in key for key in metrics)
    assert all("perplexity_ratio" not in key for key in metrics)
    assert all(not isinstance(value, torch.Tensor) for value in metrics.values())


def test_lag_window_uses_bucket_specific_counts_and_pooled_raw_samples():
    first = compute_lagged_log_prob_metrics(
        learner_version=4,
        rollout_versions=torch.tensor([[4, 4]]),
        behavior_log_probs=torch.zeros((1, 2)),
        current_log_probs=torch.tensor([[1.0, 1.0]]),
        response_mask=torch.ones((1, 2), dtype=torch.bool),
        max_lag=0,
    )
    second = compute_lagged_log_prob_metrics(
        learner_version=4,
        rollout_versions=torch.tensor([[4, -1], [4, -1]]),
        behavior_log_probs=torch.zeros((2, 2)),
        current_log_probs=torch.tensor([[3.0, 0.0], [5.0, 0.0]]),
        response_mask=torch.tensor([[1, 0], [1, 0]], dtype=torch.bool),
        max_lag=0,
    )
    aggregator = MetricsAggregator(total_gpus=1)
    aggregator.add_step_metrics(first, aggregation=first.aggregation)
    aggregator.add_step_metrics(second, aggregation=second.aggregation)

    merged = aggregator.get_aggregated_metrics()
    response_values = np.asarray([2.0, 3.0, 5.0])
    token_values = np.asarray([1.0, 1.0, 3.0, 5.0])
    assert merged["fully_async/mismatch/lag_0/response_log_ratio_sum_mean"] == pytest.approx(
        response_values.mean()
    )
    assert merged["fully_async/mismatch/lag_0/response_log_ratio_sum_std"] == pytest.approx(
        response_values.std()
    )
    assert merged["fully_async/mismatch/lag_0/response_log_ratio_sum_p95"] == pytest.approx(
        np.quantile(response_values, 0.95)
    )
    assert merged["fully_async/mismatch/lag_0/log_ratio_mean"] == pytest.approx(token_values.mean())


def test_lagged_mismatch_metrics_reject_invalid_padding_policy_versions():
    values = torch.zeros((2, 1))
    with pytest.raises(ValueError, match="padding"):
        compute_lagged_log_prob_metrics(
            learner_version=4,
            rollout_versions=torch.tensor([[4], [3]]),
            behavior_log_probs=values,
            current_log_probs=values,
            response_mask=torch.tensor([[1], [0]], dtype=torch.bool),
            max_lag=4,
        )


def test_token_age_boundary_and_padding_are_checked_per_token():
    accepted = _admit(
        _sample(min_versions=(2, 2), max_versions=(10, 10)),
        learner_version=10,
        max_lag=8,
        single_version=False,
    )
    stale = _admit(
        _sample(min_versions=(1, 2), max_versions=(10, 10)),
        learner_version=10,
        max_lag=8,
        single_version=False,
    )
    future = _admit(
        _sample(min_versions=(2, 2), max_versions=(11, 10)),
        learner_version=10,
        max_lag=8,
        single_version=False,
    )
    assert accepted.accepted
    assert stale.reason == "stale_policy_version"
    assert future.reason == "future_policy_version"
