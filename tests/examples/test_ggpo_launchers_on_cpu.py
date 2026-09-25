from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo import ray_trainer as module
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def make_trainer():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "consume_collection_minibatches": True,
                "total_epochs": 100,
                "project_name": "test",
                "experiment_name": "test",
                "logger": [],
                "val_before_train": False,
                "balance_batch": False,
                "critic_warmup": 0,
                "save_freq": -1,
                "test_freq": 3,
                "esi_redundant_time": 0,
            },
            "data": {"train_batch_size": 4},
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": 2, "policy_loss": {"loss_mode": "vanilla"}},
                "rollout": {"n": 2},
            },
            "algorithm": {
                "filter_groups": {"enable": True},
                "adv_estimator": "grpo",
                "gamma": 1,
                "lam": 1,
                "rollout_correction": {
                    "bypass_mode": False,
                    "rollout_is": "token",
                    "rollout_is_threshold": 2.0,
                    "rollout_is_batch_normalize": False,
                    "rollout_rs": None,
                },
            },
            "global_profiler": {"steps": None, "profile_continuous_steps": False},
        }
    )
    trainer._collection_remainder = None
    trainer._current_dynamic_filter_epoch = 0
    trainer._fixed_policy_lag = None
    trainer._ggpo_moving_reference_enabled = False
    return trainer


def collection(index):
    return DataProto.from_single_dict(
        {
            "rollout_log_probs": torch.zeros(8, 2),
            "response_mask": torch.ones(8, 2),
            "attention_mask": torch.ones(8, 2),
            "token_level_scores": torch.zeros(8, 2),
            "uid": np.array([f"{index}:{i // 2}" for i in range(8)], dtype=object),
            "multi_modal_inputs": np.array([{} for _ in range(8)], dtype=object),
        }
    )


def test_fit_freezes_collection_tis_across_updates_and_resume(monkeypatch, tmp_path):
    trainer = make_trainer()
    trainer.total_training_steps = 5
    trainer._dump_executor = SimpleNamespace(_shutdown=False)
    trainer._load_checkpoint = lambda: None
    trainer._shutdown_dump_executor = lambda: None
    trainer.checkpoint_manager = MagicMock()
    awake = False

    def sync(*args):
        nonlocal awake
        awake = True

    def sleep():
        nonlocal awake
        assert awake, "rollout must not be put to sleep twice"
        awake = False

    trainer.checkpoint_manager.update_weights.side_effect = sync
    trainer.checkpoint_manager.sleep_replicas.side_effect = sleep
    trainer.actor_rollout_wg = SimpleNamespace()
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)
    trainer.train_dataloader = [{}, {}, {}]
    trainer.train_dataset = MagicMock()
    trainer.use_reference_policy = trainer.use_critic = False
    trainer._start_profiling = trainer._stop_profiling = lambda *args: None
    trainer._validate = MagicMock(return_value={"validation": 1.0})
    trainer._save_checkpoint = MagicMock()
    generated, consumed, weights, old_probs, scored = [], [], [], [], []

    def produce(batch_dict, iterator, epoch, *args):
        sleep()  # Full collection generation leaves replicas asleep.
        generated.append(len(generated))
        return collection(generated[-1]), iterator, epoch

    trainer._produce_training_batch = produce

    def score(batch):
        scored.append((len(consumed), len(batch)))
        return DataProto.from_single_dict(
            {"old_log_probs": torch.full_like(batch.batch["rollout_log_probs"], np.log(1 + len(consumed)))}
        )

    trainer._compute_old_log_prob = score

    def update(batch):
        consumed.append(list(batch.non_tensor_batch["uid"]))
        weights.append(batch.batch["rollout_is_weights"].clone())
        old_probs.append(batch.batch["old_log_probs"].clone())
        assert torch.equal(batch.batch["rollout_log_probs"], torch.zeros(4, 2))
        return DataProto.from_single_dict({}, meta_info={"metrics": {"actor/grad_norm": 1.0}})

    trainer._update_actor = update
    monkeypatch.setattr(module, "compute_advantage", lambda batch, **kwargs: batch)
    monkeypatch.setattr(module, "maybe_apply_group_reference_advantage_bypass", lambda *args: False)
    monkeypatch.setattr(module, "should_save_ckpt_esi", lambda **kwargs: False)
    for name in (
        "compute_data_metrics",
        "compute_timing_metrics",
        "compute_throughout_metrics",
        "compute_variance_proxy_metrics",
        "compute_spec_decode_metrics",
    ):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "SkipManager", MagicMock())
    monkeypatch.setattr("verl.utils.tracking.Tracking", MagicMock())
    trainer.fit()

    assert len(generated) == 3  # Last collection is full, but only half is consumed.
    assert consumed == [
        [f"{c}:{g}" for g in pair for _ in range(2)]
        for c, pair in [(0, (0, 1)), (0, (2, 3)), (1, (0, 1)), (1, (2, 3)), (2, (0, 1))]
    ]
    assert scored == [(0, 8), (2, 8), (4, 8)]
    for index, old in enumerate(old_probs):
        assert torch.allclose(old, torch.full((4, 2), np.log(1 + 2 * (index // 2))))
    assert all(torch.equal(w, torch.ones(4, 2)) for w in weights[:2])
    assert all(torch.equal(w, torch.full((4, 2), 2.0)) for w in weights[2:])
    assert len(trainer._collection_remainder) == 4
    assert "global_token_num" not in trainer._collection_remainder.meta_info
    assert [call.args[0] for call in trainer.checkpoint_manager.update_weights.call_args_list] == [0, 2, 3, 4, 5]
    assert trainer._validate.call_count == 2  # update 3 and final update 5
    assert trainer.train_dataset.on_batch_end.call_count == 2

    trainer._save_collection_state(str(tmp_path))
    trainer._collection_remainder = None

    def resume():
        trainer._restore_collection_state(str(tmp_path))
        trainer.global_steps = 5
        return str(tmp_path)

    trainer._load_checkpoint = resume
    trainer.total_training_steps = 6
    trainer.fit()
    assert scored == [(0, 8), (2, 8), (4, 8)]  # No rescore after restoring the last minibatch.
    assert torch.equal(old_probs[5], old_probs[4])
    assert torch.equal(weights[5], weights[4])
    assert trainer._collection_remainder is None


def test_checkpoint_resumes_unconsumed_groups_and_epoch(tmp_path):
    trainer = make_trainer()
    trainer._take_collection_minibatch(collection(0))
    trainer._current_dynamic_filter_epoch = 7
    trainer._save_collection_state(str(tmp_path))
    restored = make_trainer()
    restored._restore_collection_state(str(tmp_path))
    batch = restored._take_collection_minibatch(restored._collection_remainder)
    assert list(batch.non_tensor_batch["uid"]) == ["0:2", "0:2", "0:3", "0:3"]
    assert restored._restored_dynamic_filter_epoch == 7
    assert restored._collection_remainder is None


def launcher_config(**env):
    import os
    import subprocess
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(root / "examples/ggpo/run_math.sh")],
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "DATA_ROOT": "/tmp/ggpo-data", "WANDB_MODE": "disabled", "DRY_RUN": "1", **env},
        check=True,
    )
    overrides = [line.removeprefix("OVERRIDE ") for line in result.stdout.splitlines()]
    with initialize_config_dir(config_dir=str(root / "verl/trainer/config"), version_base=None):
        return compose(config_name="ppo_trainer", overrides=overrides)


def test_paper_launchers_compose_and_pass_runtime_validation():
    from verl.utils.config import validate_config, validate_policy_loss_trainer_compatibility

    for method in ("ggpo", "agro", "grpo", "grpo-tis"):
        for mode in ("online", "offline", "collect"):
            config = launcher_config(ALGORITHM=method, MODE=mode, POOL_DIR="/tmp/ggpo-pool", SEED="1234")
            validate_policy_loss_trainer_compatibility(config)
            validate_config(config, use_reference_policy=True, use_critic=False)
            assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 32
            assert config.actor_rollout_ref.ref.log_prob_use_dynamic_bsz
            assert config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu == 16384
            assert config.data.seed == config.actor_rollout_ref.rollout.seed == 1234
            assert (
                config.actor_rollout_ref.actor.data_loader_seed
                == config.actor_rollout_ref.actor.fsdp_config.seed
                == 1234
            )
            assert config.trainer.total_training_steps == (8000 if mode == "online" else 400)
            assert config.trainer.test_freq == (160 if mode == "online" else 80)
            assert config.trainer.save_freq == 160
            assert config.data.train_batch_size == (32 if mode == "collect" else 256)
            if mode == "offline" and method == "grpo-tis":
                assert config.trainer.trainer_class == "verl.trainer.ppo.paper_offline.OfflineTrainer"
                assert config.trainer.consume_collection_minibatches
                assert config.data.train_batch_size == 8 * config.actor_rollout_ref.actor.ppo_mini_batch_size
                assert not config.algorithm.rollout_correction.bypass_mode
                assert config.algorithm.rollout_correction.rollout_is == "token"
                assert config.algorithm.rollout_correction.rollout_is_threshold == 2
                assert not config.algorithm.rollout_correction.rollout_is_batch_normalize
            if mode != "collect" and method in ("ggpo", "agro"):
                assert config.actor_rollout_ref.actor.policy_loss.beta == 0.01
                assert config.actor_rollout_ref.actor.policy_loss.sequence_log_prob_reduction == "sum"
                assert config.trainer.ggpo_moving_reference.refresh_every_updates == 160
            else:
                assert config.actor_rollout_ref.actor.kl_loss_coef == 0.01
                assert (
                    config.actor_rollout_ref.actor.clip_ratio_low
                    == config.actor_rollout_ref.actor.clip_ratio_high
                    == 0.2
                )
                assert config.actor_rollout_ref.actor.clip_ratio_c == 10
                assert not config.trainer.ggpo_moving_reference.enabled
    config = launcher_config(ALGORITHM="grpo-tis", N="128")
    assert config.data.train_batch_size == 4096
    assert not config.algorithm.rollout_correction.bypass_mode
    assert config.algorithm.rollout_correction.rollout_is_threshold == 2
    assert not config.algorithm.rollout_correction.rollout_is_batch_normalize


def test_offline_pool_keeps_behavior_and_order_without_generating(tmp_path):
    from verl.trainer.ppo.paper_offline import CollectRollouts, OfflineTrainer

    pool = tmp_path / "pool"
    collector = CollectRollouts.__new__(CollectRollouts)
    collector.config = OmegaConf.create({"trainer": {"offline_pool_dir": str(pool)}})
    collector.train_dataloader = range(9)
    collector.total_training_steps = 9
    collector.checkpoint_manager = MagicMock()
    collector._shutdown_dump_executor = lambda: None
    collector._next_training_candidate = lambda iterator, epoch: (next(iterator), iterator, epoch)
    collector._produce_training_batch = lambda candidate, iterator, epoch, *args: (
        collection(candidate)[:4], iterator, epoch,
    )
    collector._update_actor = MagicMock(side_effect=AssertionError("collection must not update the actor"))
    collector.fit()
    assert [call.args for call in collector.checkpoint_manager.update_weights.call_args_list] == [(0,)] * 9
    collector._update_actor.assert_not_called()
    trainer = OfflineTrainer.__new__(OfflineTrainer)
    trainer.__dict__.update(make_trainer().__dict__)
    trainer.config.trainer.offline_pool_dir = str(pool)
    trainer.config.data.train_batch_size = 16  # Eight two-prompt minibatches.
    trainer.total_training_steps = 9
    trainer.checkpoint_manager = MagicMock()
    trainer.async_rollout_manager = MagicMock()
    scored = []

    def score(batch):
        scored.append((trainer.global_steps, len(batch)))
        return DataProto.from_single_dict({
            "old_log_probs": torch.full_like(batch.batch["rollout_log_probs"], np.log(trainer.global_steps)),
        })

    trainer._compute_old_log_prob = score
    for index in range(9):
        trainer.global_steps = index + 1
        block = trainer._collection_remainder
        if block is None:
            block, _, _ = trainer._produce_training_batch(None, None, 0, True, {}, {}, False)
            block, _ = trainer._prepare_behavior_log_probs(block, {})
        batch = trainer._take_collection_minibatch(block)
        expected = collection(index)[:4]
        assert batch.non_tensor_batch["uid"].tolist() == expected.non_tensor_batch["uid"].tolist()
        assert torch.equal(batch.batch["rollout_log_probs"], expected.batch["rollout_log_probs"])
        assert torch.allclose(batch.batch["old_log_probs"], torch.full((4, 2), np.log(1 if index < 8 else 9)))
        assert torch.equal(batch.batch["rollout_is_weights"], torch.full((4, 2), 1.0 if index < 8 else 2.0))
        if index == 3:  # Resume halfway through the first block without rescoring it.
            trainer._save_collection_state(str(tmp_path))
            trainer._collection_remainder = None
            trainer._restore_collection_state(str(tmp_path))
    assert scored == [(1, 32), (9, 4)]
    assert trainer._collection_remainder is None
    trainer.async_rollout_manager.generate_sequences.assert_not_called()


def test_offline_diagnostics_pool_eight_updates_across_resume(monkeypatch, tmp_path):
    from verl.trainer.ppo.paper_offline import OfflineTrainer
    from verl.trainer.ppo.rollout_training_mismatch import (
        MISMATCH_AGGREGATION_META,
        compute_mismatch_stats,
        finalize_mismatch_metrics,
    )

    trainer = OfflineTrainer.__new__(OfflineTrainer)
    trainer._collection_remainder = None
    trainer._current_dynamic_filter_epoch = 0
    stats = []

    def update(self, batch, **kwargs):
        current = torch.linspace(-2, -1, 256).unsqueeze(1).repeat(1, 2) + self.global_steps / 10
        mask = torch.ones_like(current)
        mask[::2, 1] = self.global_steps % 2
        stats.append(compute_mismatch_stats(current, torch.full_like(current, -2), mask))
        metrics = finalize_mismatch_metrics(stats[-1])
        return DataProto.from_single_dict({}, meta_info={
            "metrics": dict(metrics), MISMATCH_AGGREGATION_META: metrics.aggregation,
        })

    monkeypatch.setattr(RayPPOTrainer, "_update_actor", update)
    for step in range(1, 9):
        trainer.global_steps = step
        metrics = trainer._update_actor(None).meta_info["metrics"]
        if step < 8:
            assert not any(key.startswith("offline_block/") for key in metrics)
        if step == 3:
            trainer._save_collection_state(str(tmp_path))
            trainer._collection_metrics = None
            trainer._restore_collection_state(str(tmp_path))
    expected = finalize_mismatch_metrics(stats)
    for key, value in expected.items():
        assert np.isclose(metrics[f"offline_block/{key}"], value)
    assert metrics["offline_block/rollout_training_mismatch/raw_is/sequence_count"] == 2048
    assert trainer._collection_metrics.step_count == 0


def test_evaluation_schema_and_problem_weighted_pass_at_k(tmp_path):
    import json

    from examples.data_preprocess.alp_math import convert_eval_row
    from examples.ggpo.summarize_eval import summarize

    assert (
        convert_eval_row({"problem": "Q", "solution": r"\boxed{2}"}, 0, "aime24")["reward_model"]["ground_truth"] == "2"
    )
    assert (
        convert_eval_row({"question": "Q", "final_answer": [r"\boxed{\frac{1}{2}}"]}, 0, "olympiadbench")[
            "reward_model"
        ]["ground_truth"]
        == r"\frac{1}{2}"
    )
    path = tmp_path / "0.jsonl"
    rows = [
        {"data_source": source, "uid": str(index), "score": score}
        for source, index, score in [("math500", 0, 1), ("math500", 1, 1), ("aime24", 0, 0)]
        for _ in range(32)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    result = summarize(path)
    assert result["weighted_average"]["problems"] == 3
    assert result["weighted_average"]["pass@1"] == 200 / 3
    assert result["weighted_average"]["pass@32"] == 200 / 3
