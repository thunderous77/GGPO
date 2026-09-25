# Copyright 2026 OpenAI

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.offline_rl.contract import validate_checkpoint_dataset_identity
from verl.trainer.offline_rl.dataset import seeded_pool_round_schedule
from verl.trainer.offline_rl.evaluate import OfflineMathEvaluator
from verl.trainer.offline_rl.trainer import (
    OfflineCursorLoader,
    OfflineRLTrainer,
    assert_finite_metrics,
    prepare_offline_ggpo_batch,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.engine_workers import classify_actor_rollout_ref_role


def test_one_pass_prefix_schedule_consumes_shards_zero_through_ninety_nine_once():
    schedule = [step - 1 for step in range(1, 101)]
    assert schedule == list(range(100))
    cursor = OfflineCursorLoader(100)
    cursor.set_global_step(100)
    assert cursor.state_dict() == {
        "schedule": "one_pass_prefix",
        "completed_optimizer_steps": 100,
        "next_batch_index": 100,
        "batch_count": 100,
    }
    restored = OfflineCursorLoader(100)
    restored.load_state_dict(cursor.state_dict())
    assert restored.global_step == 100


def test_cursor_rejects_duplicate_or_skipped_next_batch():
    cursor = OfflineCursorLoader(100)
    with pytest.raises(ValueError, match="inconsistent next batch"):
        cursor.load_state_dict(
            {
                "schedule": "one_pass_prefix",
                "completed_optimizer_steps": 20,
                "next_batch_index": 21,
                "batch_count": 100,
            }
        )


def test_seeded_pool_cycle_has_exact_multi_pass_schedule_and_resume_cursor():
    schedule = seeded_pool_round_schedule(
        pool_round_count=3,
        total_round_count=8,
        seed=7,
    )
    assert schedule == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (1, 0),
        (2, 2),
        (2, 0),
    ]
    assert [pass_index for pass_index, _ in schedule] == [0, 0, 0, 1, 1, 1, 2, 2]
    assert [pool_index for _, pool_index in schedule] == [0, 1, 2, 1, 2, 0, 2, 0]

    interrupted = OfflineCursorLoader(8, schedule="seeded_pool_cycle")
    interrupted.set_global_step(5)
    resumed = OfflineCursorLoader(8, schedule="seeded_pool_cycle")
    resumed.load_state_dict(interrupted.state_dict())
    assert resumed.global_step == 5
    assert schedule[resumed.global_step] == (1, 0)


def test_checkpoint_dataset_identity_is_an_optional_label(monkeypatch):
    save_source = inspect.getsource(OfflineRLTrainer._save_checkpoint)
    load_source = inspect.getsource(OfflineRLTrainer._load_checkpoint)
    evaluation_source = inspect.getsource(OfflineMathEvaluator.fit)

    assert '"dataset_id": str(self.config.offline_data.dataset_id)' in save_source
    assert "validate_checkpoint_dataset_identity" in load_source
    assert "validate_checkpoint_dataset_identity" in evaluation_source

    monkeypatch.delenv("GGPO_DATASET_ID", raising=False)
    validate_checkpoint_dataset_identity(
        {"dataset_id": "synthetic-checkpoint-dataset"},
        dataset_id="synthetic-configured-dataset",
    )

    monkeypatch.setenv("GGPO_DATASET_ID", "synthetic-optional-label")
    validate_checkpoint_dataset_identity(
        {"dataset_id": "synthetic-matching-dataset"},
        dataset_id="synthetic-matching-dataset",
    )
    with pytest.raises(ValueError, match="dataset ID"):
        validate_checkpoint_dataset_identity(
            {"dataset_id": "synthetic-checkpoint-dataset"},
            dataset_id="synthetic-configured-dataset",
        )


def test_offline_metric_gate_rejects_every_supported_nonfinite_scalar_kind():
    assert_finite_metrics({"float": 1.0, "numpy": np.array([2.0]), "torch": torch.tensor(3.0)})
    for value in (float("nan"), np.array([float("inf")]), torch.tensor(float("-inf"))):
        with pytest.raises(FloatingPointError, match="bad"):
            assert_finite_metrics({"bad": value})


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("actor", (True, False, False)),
        ("rollout", (False, True, False)),
        ("ref", (False, False, True)),
        ("actor_ref", (True, False, True)),
        ("actor_rollout", (True, True, False)),
        ("actor_rollout_ref", (True, True, True)),
    ],
)
def test_actor_rollout_ref_role_membership(role, expected):
    assert classify_actor_rollout_ref_role(role) == expected


def test_offline_fit_has_no_generation_or_rollout_weight_update():
    source = inspect.getsource(OfflineRLTrainer.fit)
    assert "generate_sequences" not in source
    assert "checkpoint_manager.update_weights" not in source
    assert "batch_index = step - 1" in source
    assert "_update_actor(batch)" in source
    assert "_compute_current_log_prob" not in source
    assert "_compute_old_log_prob" not in source


def test_offline_fit_logs_the_deterministic_pool_schedule():
    source = inspect.getsource(OfflineRLTrainer.fit)
    assert "self.train_dataset.schedule_entry(batch_index)" in source
    assert '"offline/dataset_pass": dataset_pass' in source
    assert '"offline/pool_round_index": pool_round_index' in source


def test_offline_entropy_diagnostic_uses_the_actor_config_switch():
    offline_source = inspect.getsource(OfflineRLTrainer.fit)
    actor_source = inspect.getsource(RayPPOTrainer._update_actor)
    assert '"calculate_entropy": False' not in offline_source
    assert "self.config.actor_rollout_ref.actor.calculate_entropy" in actor_source


def test_offline_reward_audit_does_not_run_a_detached_policy_forward():
    source = inspect.getsource(OfflineRLTrainer._offline_reward_metrics)
    assert "compute_log_prob" not in source
    assert 'batch.batch["token_level_scores"]' in source


def test_offline_ggpo_exposes_binary_correctness_and_requires_prefiltered_mixed_groups():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"policy_loss": {"loss_mode": "ggpo"}, "ppo_mini_batch_size": 1},
            },
            "algorithm": {"filter_groups": {"enable": True, "metric": "raw_correctness"}},
            "trainer": {"nnodes": 1, "n_gpus_per_node": 1},
        }
    )
    batch = DataProto.from_single_dict(
        {
            "token_level_scores": torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]]),
            "old_log_probs": torch.arange(6, dtype=torch.float32).unsqueeze(-1),
            "sample_id": np.array(["a0", "a1", "b0", "b1", "c0", "c1"], dtype=object),
        }
    )
    batch.meta_info["responses_per_group"] = 2

    metrics = prepare_offline_ggpo_batch(batch, config)

    assert metrics == {
        "offline/filter_kept_groups": 1,
        "offline/filter_dropped_groups": 2,
        "offline/filter_kept_fraction": 1 / 3,
        "offline/filter_truncated_groups": 0,
        "offline/filter_final_responses": 2,
    }
    assert len(batch) == 2
    torch.testing.assert_close(batch.batch["token_level_scores"], torch.tensor([[0.0], [1.0]]))
    torch.testing.assert_close(batch.batch["old_log_probs"], torch.tensor([[2.0], [3.0]]))
    np.testing.assert_array_equal(batch.non_tensor_batch["acc"], np.array([0.0, 1.0]))
    np.testing.assert_array_equal(batch.non_tensor_batch["sample_id"], np.array(["b0", "b1"], dtype=object))

    constant = DataProto.from_single_dict(
        {"token_level_scores": torch.tensor([[0.0], [0.0], [1.0], [1.0]])}
    )
    constant.meta_info["responses_per_group"] = 2
    with pytest.raises(RuntimeError, match="zero mixed raw-correctness groups"):
        prepare_offline_ggpo_batch(constant, config)


def test_offline_ggpo_truncates_trailing_groups_to_actor_dispatch_unit():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"policy_loss": {"loss_mode": "ggpo"}, "ppo_mini_batch_size": 32},
            },
            "algorithm": {"filter_groups": {"enable": True, "metric": "raw_correctness"}},
            "trainer": {"nnodes": 1, "n_gpus_per_node": 4},
        }
    )
    # 35 mixed groups of 8 = 280 responses; the actor dispatch unit for
    # mini=32 prompts x 8 responses on dp=4 is one whole global mini-batch
    # (256 responses), so the trailing 3 groups must be truncated.
    scores = torch.tensor([[value] for _ in range(35) for value in (0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)])
    batch = DataProto.from_single_dict(
        {
            "token_level_scores": scores,
            "old_log_probs": torch.arange(280, dtype=torch.float32).unsqueeze(-1),
            "sample_id": np.array([f"sample-{index}" for index in range(280)], dtype=object),
        }
    )
    batch.meta_info["responses_per_group"] = 8

    metrics = prepare_offline_ggpo_batch(batch, config)

    assert metrics == {
        "offline/filter_kept_groups": 35,
        "offline/filter_dropped_groups": 0,
        "offline/filter_kept_fraction": 1.0,
        "offline/filter_truncated_groups": 3,
        "offline/filter_final_responses": 256,
    }
    assert len(batch) == 256
    torch.testing.assert_close(batch.batch["token_level_scores"], scores[:256])
    torch.testing.assert_close(batch.batch["old_log_probs"], torch.arange(256, dtype=torch.float32).unsqueeze(-1))
    np.testing.assert_array_equal(batch.non_tensor_batch["acc"], np.tile([0.0, 1.0], 128))
    np.testing.assert_array_equal(
        batch.non_tensor_batch["sample_id"],
        np.array([f"sample-{index}" for index in range(256)], dtype=object),
    )


def test_offline_ggpo_fails_when_dispatch_truncation_would_empty_batch():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"policy_loss": {"loss_mode": "ggpo"}, "ppo_mini_batch_size": 4},
            },
            "algorithm": {"filter_groups": {"enable": True, "metric": "raw_correctness"}},
            "trainer": {"nnodes": 1, "n_gpus_per_node": 2},
        }
    )
    batch = DataProto.from_single_dict(
        {"token_level_scores": torch.tensor([[0.0], [1.0], [0.0], [0.0]])}
    )
    batch.meta_info["responses_per_group"] = 2

    with pytest.raises(RuntimeError, match="zero responses after truncating"):
        prepare_offline_ggpo_batch(batch, config)


@pytest.mark.parametrize("loss_mode", ["ggpo", "agro", "asympo"])
def test_mixed_filtered_strict_guard_accepts_full_mixed_round_for_every_loss(loss_mode):
    config = OmegaConf.create(
        {
            "offline_data": {"contract": "sealed_mixedfilt"},
            "actor_rollout_ref": {
                "actor": {"policy_loss": {"loss_mode": loss_mode}, "ppo_mini_batch_size": 1},
            },
            "algorithm": {"filter_groups": {"enable": True, "metric": "raw_correctness"}},
            "trainer": {"nnodes": 1, "n_gpus_per_node": 1},
        }
    )
    scores = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    batch = DataProto.from_single_dict(
        {
            "token_level_scores": scores.clone(),
            "old_log_probs": torch.arange(4, dtype=torch.float32).unsqueeze(-1),
        }
    )
    batch.meta_info["responses_per_group"] = 2

    metrics = prepare_offline_ggpo_batch(batch, config)

    assert metrics["offline/filter_kept_fraction"] == 1.0
    assert metrics["offline/filter_truncated_groups"] == 0
    assert len(batch) == 4
    torch.testing.assert_close(batch.batch["token_level_scores"], scores)


def test_mixed_filtered_strict_guard_rejects_degenerate_groups_and_truncation():
    config = OmegaConf.create(
        {
            "offline_data": {"contract": "sealed_mixedfilt"},
            "actor_rollout_ref": {
                "actor": {"policy_loss": {"loss_mode": "agro"}, "ppo_mini_batch_size": 1},
            },
            "algorithm": {},
            "trainer": {"nnodes": 1, "n_gpus_per_node": 1},
        }
    )
    degenerate = DataProto.from_single_dict(
        {"token_level_scores": torch.tensor([[0.0], [0.0], [1.0], [0.0]])}
    )
    degenerate.meta_info["responses_per_group"] = 2
    with pytest.raises(RuntimeError, match="strict guard found a non-mixed"):
        prepare_offline_ggpo_batch(degenerate, config)

    config.actor_rollout_ref.actor.ppo_mini_batch_size = 2
    truncating = DataProto.from_single_dict(
        {"token_level_scores": torch.tensor([[0.0], [1.0], [1.0], [0.0], [0.0], [1.0]])}
    )
    truncating.meta_info["responses_per_group"] = 2
    with pytest.raises(RuntimeError, match="forbids actor-dispatch truncation"):
        prepare_offline_ggpo_batch(truncating, config)


def test_offline_ggpo_theory_identity_keeps_the_fixed_behavior_at_step_zero():
    trainer = OfflineRLTrainer.__new__(OfflineRLTrainer)
    trainer.global_steps = 7
    batch = DataProto.from_single_dict({"responses": torch.zeros(4, 1, dtype=torch.long)})

    identity = trainer._ggpo_theory_policy_identity(batch)

    assert identity["actor_update_step"] == 7
    assert identity["training_policy_version"] == 6
    torch.testing.assert_close(identity["behavior_policy_versions"], torch.zeros(4, dtype=torch.long))
    assert identity["behavior_version_stratification"] is True


def test_offline_ggpo_checkpoint_refreshes_and_restores_the_moving_reference(tmp_path, monkeypatch):
    checkpoint_root = tmp_path / "checkpoints"
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {"loss_mode": "ggpo"},
                    "checkpoint": {"async_save": False},
                }
            },
            "offline_data": {
                "dataset_id": "dataset",
                "implementation_commit": "implementation",
                "wandb_step_offset": 0,
                "segment_optimizer_steps": 20,
            },
            "trainer": {
                "default_local_dir": str(checkpoint_root),
                "default_hdfs_dir": None,
                "experiment_name": "offline-ggpo",
                "max_actor_ckpt_to_keep": None,
                "ggpo_moving_reference": {"enabled": True, "update_freq": 20},
                "resume_mode": "disable",
                "resume_from_path": None,
                "del_local_ckpt_after_load": False,
            },
        }
    )
    monkeypatch.setenv("WANDB_RUN_ID", "offline-ggpo-run")
    monkeypatch.delenv("GGPO_DATASET_ID", raising=False)

    trainer = OfflineRLTrainer.__new__(OfflineRLTrainer)
    trainer.config = config
    trainer.global_steps = 5
    trainer._ggpo_moving_reference_enabled = True
    trainer._ggpo_reference_version = 0
    trainer.train_dataloader = OfflineCursorLoader(100)

    def save_actor(path, **_kwargs):
        Path(path).mkdir(parents=True)

    trainer.actor_rollout_wg = SimpleNamespace(save_checkpoint=MagicMock(side_effect=save_actor))
    trainer.ref_policy_wg = SimpleNamespace(load_reference_checkpoint=MagicMock())
    trainer._save_checkpoint()

    checkpoint_5 = checkpoint_root / "global_step_5"
    state_5 = json.loads((checkpoint_5 / "offline_state.json").read_text())
    assert state_5["ggpo_reference_version_after_step"] == 0
    trainer.ref_policy_wg.load_reference_checkpoint.assert_not_called()
    assert trainer._ggpo_reference_version == 0

    trainer.global_steps = 20
    trainer._save_checkpoint()

    checkpoint = checkpoint_root / "global_step_20"
    state = json.loads((checkpoint / "offline_state.json").read_text())
    assert state["ggpo_reference_version_after_step"] == 20
    trainer.ref_policy_wg.load_reference_checkpoint.assert_called_once_with(str(checkpoint / "actor"))
    assert trainer._ggpo_reference_version == 20

    resumed = OfflineRLTrainer.__new__(OfflineRLTrainer)
    resumed.config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    resumed.config.offline_data.wandb_step_offset = 1
    resumed.config.trainer.resume_mode = "resume_path"
    resumed.config.trainer.resume_from_path = str(checkpoint)
    resumed._ggpo_moving_reference_enabled = True
    resumed._ggpo_reference_version = 0
    resumed.train_dataloader = OfflineCursorLoader(100)
    resumed.actor_rollout_wg = SimpleNamespace(load_checkpoint=MagicMock())
    resumed.ref_policy_wg = SimpleNamespace(load_reference_checkpoint=MagicMock())
    resumed._load_checkpoint()

    resumed.ref_policy_wg.load_reference_checkpoint.assert_called_once_with(str(checkpoint / "actor"))
    assert resumed._ggpo_reference_version == 20


def test_smoke_path_contains_no_validation_or_checkpoint_save():
    source = inspect.getsource(OfflineRLTrainer.fit)
    assert "_validate()" not in source
    assert "if bool(self.config.offline_data.smoke)" in source
    assert source.index("if bool(self.config.offline_data.smoke)") < source.index("self._save_checkpoint()")


def test_offline_fit_checkpoints_at_save_cadence_and_at_a_final_partial_round():
    source = inspect.getsource(OfflineRLTrainer.fit)
    assert "step % save_freq == 0" in source
    assert "step == self.total_training_steps and save_final" in source


def test_evaluator_has_no_actor_update():
    source = inspect.getsource(OfflineMathEvaluator.fit)
    assert "_update_actor" not in source
    assert "_validate()" in source


def test_identity_match_ignores_legacy_source_sha256():
    import json as _json

    from verl.trainer.offline_rl.dataset import _identity_for_config_match

    expected = {
        "implementation_commit": "78172e1a",
        "behavior_model": {"id": "Qwen/Qwen3-1.7B-Base", "revision": "ea980cb0"},
    }
    legacy = {
        "implementation_commit": "78172e1a",
        "behavior_model": {"id": "Qwen/Qwen3-1.7B-Base", "revision": "ea980cb0"},
        "source": {"path": "/old/host/train.parquet", "sha256": "deadbeef" * 8},
    }
    current = _json.loads(_json.dumps(legacy))
    del current["source"]["sha256"]
    current["source"]["path"] = "/new/host/train.parquet"

    assert _identity_for_config_match(legacy, expected) == _identity_for_config_match(current, expected)
