# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import inspect

import numpy as np
import pytest
import torch

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.rollout_training_mismatch import (
    ALP_MATH_VALIDATION_KEYS,
    compute_mismatch_stats,
    empty_mismatch_stats,
    finalize_mismatch_metrics,
    prune_sync_alp_math_training_metrics,
    prune_sync_alp_math_validation_metrics,
)


def test_mismatch_uses_exact_loss_mask_and_merges_nested_accumulators():
    current = torch.log(torch.tensor([[0.5, 0.4, 0.3], [0.2, 0.1, 0.8]]))
    rollout = torch.log(torch.tensor([[0.4, 0.2, 0.3], [0.1, 0.5, 0.4]]))
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool)

    left = compute_mismatch_stats(current[:1], rollout[:1], mask[:1])
    right = compute_mismatch_stats(current[1:], rollout[1:], mask[1:])
    actual = finalize_mismatch_metrics([[[left]], [right]])

    log_ratio_matrix = current - rollout
    log_ratio = log_ratio_matrix[mask].double()
    probs_diff = (current.exp() - rollout.exp()).abs()[mask].double()
    sequence_log_ratio = (log_ratio_matrix.double() * mask).sum(dim=-1).numpy()
    expected = {
        "rollout_training_mismatch/probs_diff_mean": probs_diff.mean().item(),
        "rollout_training_mismatch/probs_diff_std": probs_diff.std(correction=0).item(),
        "rollout_training_mismatch/log_ratio_mean": log_ratio.mean().item(),
        "rollout_training_mismatch/log_ratio_std": log_ratio.std(correction=0).item(),
        "rollout_training_mismatch/log_ratio_abs_mean": log_ratio.abs().mean().item(),
        "rollout_training_mismatch/k3_kl": (log_ratio.exp() - log_ratio - 1).mean().item(),
        "rollout_training_mismatch/sequence_log_ratio_p95": np.quantile(sequence_log_ratio, 0.95),
        "rollout_training_mismatch/sequence_log_ratio_max": sequence_log_ratio.max(),
    }
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value)
    weights = np.exp(sequence_log_ratio - np.logaddexp.reduce(sequence_log_ratio))
    expected_ess = weights.sum() ** 2 / (len(weights) * np.square(weights).sum())
    assert actual["rollout_training_mismatch/raw_is/sequence_count"] == 2
    assert actual["rollout_training_mismatch/raw_is/ess_normalized"] == pytest.approx(expected_ess)
    assert actual["rollout_training_mismatch/raw_is/sequence_log_ratio_p99"] == pytest.approx(
        np.quantile(sequence_log_ratio, 0.99)
    )
    assert 0 < actual["rollout_training_mismatch/raw_is/responses_for_90pct_weight_mass_fraction"] <= 1
    assert actual.keys() == actual.aggregation.keys()


def test_mismatch_rejects_empty_loss_mask():
    values = torch.zeros((1, 2))
    with pytest.raises(ValueError, match="contains no tokens"):
        compute_mismatch_stats(values, values, torch.zeros_like(values, dtype=torch.bool))


def test_mismatch_sequence_distribution_excludes_padding_only_responses():
    current = torch.tensor([[1.0, 2.0], [100.0, 100.0]])
    rollout = torch.zeros_like(current)
    mask = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)

    stats = compute_mismatch_stats(current, rollout, mask)
    metrics = finalize_mismatch_metrics(stats)

    assert stats["sequence_log_ratios"] == [3.0]
    assert metrics["rollout_training_mismatch/raw_is/sequence_count"] == 1.0


def test_mismatch_merge_ignores_padding_only_microbatch_identity():
    values = torch.tensor([[0.5]])
    real = compute_mismatch_stats(values, torch.zeros_like(values), torch.ones_like(values, dtype=torch.bool))

    expected = finalize_mismatch_metrics(real)
    actual = finalize_mismatch_metrics([[empty_mismatch_stats()], [real]])

    assert actual == expected


def test_sync_trainer_uses_one_actor_update_and_no_diagnostic_probe_forward():
    source = inspect.getsource(RayPPOTrainer._update_actor)
    assert source.count("actor_rollout_wg.update_actor(") == 1
    assert "_compute_old_log_prob" not in source
    assert '"collect_ggpo_theory_current_log_probs": collect_ggpo_theory_log_probs' in source
    assert "self._record_ggpo_theory_metrics(batch, actor_output)" in source


def test_sync_alp_math_metric_schema_is_deliberately_small():
    validation = {key: index for index, key in enumerate(sorted(ALP_MATH_VALIDATION_KEYS))}
    validation.update(
        {
            "val-aux/alp_math500/acc/mean@32": 1,
            "val-aux/alp_math500/reward/mean@32": 2,
            "val-aux/alp_math500/valid_response_length/average@32": 3,
            "val-aux/alp_math500/valid_response_length/best@32/mean": 4,
            "val-aux/num_turns/mean": 5,
        }
    )
    assert prune_sync_alp_math_validation_metrics(validation) == {
        key: validation[key] for key in ALP_MATH_VALIDATION_KEYS
    }

    training = {
        "training/reward/mean": 1,
        "num_turns/mean": 2,
        "tool_calls/mean": 3,
        "tool_call_counts/max": 4,
        "timing_s/agent_loop/tool_calls/mean": 5,
        "timing_s/agent_loop/slowest/tool_calls": 6,
    }
    assert prune_sync_alp_math_training_metrics(training) == {"training/reward/mean": 1}
