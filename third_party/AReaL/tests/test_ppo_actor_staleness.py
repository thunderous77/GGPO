import torch

from areal.trainer.ppo.actor import _log_version_staleness_stats
from areal.utils import stats_tracker


def test_theta_staleness_histogram_covers_delta_four_boundary(monkeypatch):
    """The per-token histogram must expose exact and out-of-bound deltas."""
    tracker = stats_tracker.DistributedStatsTracker()
    monkeypatch.setattr(stats_tracker, "scope", tracker.scope)
    monkeypatch.setattr(stats_tracker, "denominator", tracker.denominator)
    monkeypatch.setattr(stats_tracker, "stat", tracker.stat)
    monkeypatch.setattr(stats_tracker, "scalar", tracker.scalar)
    versions = torch.tensor([[-1, 0, 1, 2, 3, 4, 5, 9]], dtype=torch.int32)
    mask = torch.ones_like(versions, dtype=torch.bool)

    _log_version_staleness_stats(
        versions=versions,
        current_version=6,
        version_metrics_mask=mask,
    )
    metrics = tracker.export()

    prefix = "version_stats/sample_staleness_theta_delta_"
    assert metrics[f"{prefix}lt_0_count"] == 1.0
    assert metrics[f"{prefix}0_count"] == 0.0
    for delta in range(1, 5):
        assert metrics[f"{prefix}{delta}_count"] == 1.0
    assert metrics[f"{prefix}gt_4_count"] == 2.0
    assert metrics["version_stats/sample_staleness_theta_max"] == 6.0
    assert metrics["version_stats/sample_staleness_theta_min"] == -3.0
