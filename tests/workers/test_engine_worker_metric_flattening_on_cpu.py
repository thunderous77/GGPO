import pytest

from verl.trainer.ppo.rollout_training_mismatch import INTERNAL_MISMATCH_STATS_KEY
from verl.utils.metric import AggregationType, Metric
from verl.workers.engine_workers import _normalize_gathered_metric_value


def test_dp_gathered_scalar_metrics_remain_a_flat_rank_list():
    values = [1.0, 2.0, 3.0, 4.0]
    assert _normalize_gathered_metric_value("loss", values) == values


def test_dp_gathered_legacy_nested_lists_are_flattened_once():
    values = [[1.0, 2.0], [3.0], [4.0]]
    assert _normalize_gathered_metric_value("loss", values) == [1.0, 2.0, 3.0, 4.0]


def test_dp_gathered_weighted_metrics_use_metric_aggregation():
    values = [
        Metric(AggregationType.MEAN, value=1.0, weight=1),
        Metric(AggregationType.MEAN, value=5.0, weight=3),
    ]
    assert _normalize_gathered_metric_value("agro_sample_variance", values) == pytest.approx(4.0)


def test_mixed_nested_and_scalar_rank_values_fail_closed():
    with pytest.raises(TypeError, match="mixes nested and scalar"):
        _normalize_gathered_metric_value("loss", [[1.0], 2.0])


def test_internal_mismatch_payload_is_not_flattened():
    values = [[{"count": 1}], [{"count": 2}]]
    assert _normalize_gathered_metric_value(INTERNAL_MISMATCH_STATS_KEY, values) is values
