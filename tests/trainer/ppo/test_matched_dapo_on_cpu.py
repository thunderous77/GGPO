import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.dapo import soft_overlong_penalty
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ggpo import compute_ggpo_loss
from verl.trainer.ppo.interactive_segment import resolve_segment_end
from verl.trainer.ppo.matched_contract import (
    expected_scientific_config,
    run_display_name,
    validate_matched_config,
)
from verl.trainer.ppo.mixed_reward_filter import (
    raw_correctness_tensor,
    require_complete_mixed_batch,
    select_mixed_reward_group_indices,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.skip.skip_manager import SkipManager

MODEL_REVISION = "a" * 40


@pytest.fixture(autouse=True)
def _synthetic_model_revision(monkeypatch):
    monkeypatch.setenv("GGPO_MODEL_REVISION", MODEL_REVISION)


def _pinned_model_stub(tmp_path: Path) -> Path:
    snapshot = tmp_path / "cache" / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True)
    return snapshot


def _matched_data_stub(tmp_path: Path) -> Path:
    root = tmp_path / "alp-math"
    root.mkdir(exist_ok=True)
    (root / "train.parquet").write_bytes(b"matched-train")
    (root / "math500.parquet").write_bytes(b"matched-validation")
    return root


def test_mixed_filter_uses_raw_correctness_not_shaped_reward():
    raw = raw_correctness_tensor(
        {"acc": [0, 1, 0, 0, 1, 1, 1, 1]},
        expected_responses=8,
    )
    shaped = torch.tensor([-1.0, 0.2, 0.0, -0.5, 0.3, 0.6, 0.9, 1.0])

    indices, stats = select_mixed_reward_group_indices(raw, group_size=4, max_groups=2)

    assert indices.tolist() == [0, 1, 2, 3]
    assert shaped[4:].unique().numel() > 1  # shaped variation must not admit the raw all-1 group
    assert stats == {
        "generated_groups": 2,
        "mixed_groups": 1,
        "all_zero_groups": 0,
        "all_one_groups": 1,
        "selected_groups": 1,
        "discarded_mixed_groups": 0,
    }


def _refill_gate_trainer(loss_mode, max_num_gen_batches):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "filter_groups": {
                    "enable": True,
                    "metric": "raw_correctness",
                    "max_num_gen_batches": max_num_gen_batches,
                },
                "adv_estimator": "grpo",
            },
            "actor_rollout_ref": {"actor": {"policy_loss": {"loss_mode": loss_mode}, "loss_agg_mode": "token-mean"}},
        }
    )
    return trainer


@pytest.mark.parametrize("loss_mode", ["ggpo", "agro", "vanilla"])
def test_mixed_reward_refill_admits_every_matched_policy_loss(loss_mode):
    # Reaching the max_num_gen_batches check means the loss_mode whitelist let this arm past.
    trainer = _refill_gate_trainer(loss_mode, max_num_gen_batches=0)
    with pytest.raises(ValueError, match="max_num_gen_batches"):
        trainer._collect_mixed_reward_batch(None, None, 0, {}, {}, False)


def test_mixed_reward_refill_still_rejects_an_unmatched_policy_loss():
    trainer = _refill_gate_trainer("asympo", max_num_gen_batches=32)
    with pytest.raises(ValueError, match="supports only GGPO, AGRO and matched DAPO"):
        trainer._collect_mixed_reward_batch(None, None, 0, {}, {}, False)


def test_dynamic_sampling_fails_when_finite_budget_cannot_fill_batch():
    with pytest.raises(RuntimeError, match="finite generation budget"):
        require_complete_mixed_batch(accepted_groups=255, target_groups=256, generation_rounds=32)


@pytest.mark.parametrize(
    ("length", "expected"),
    [(0, 0.0), (1638, 0.0), (1843, -0.5), (2048, -1.0)],
)
def test_soft_overlong_penalty_matches_approved_boundaries(length, expected):
    assert soft_overlong_penalty(
        length,
        max_response_length=2048,
        buffer_length=410,
        penalty_factor=1.0,
    ) == pytest.approx(expected)


def test_normalized_ggpo_accepts_explicitly_enabled_shaped_rewards():
    log_prob = torch.zeros(8, 2, requires_grad=True)
    ref_log_prob = torch.zeros_like(log_prob)
    response_mask = torch.ones_like(log_prob)
    shaped_rewards = torch.tensor([-0.5, 1.0, 0.0, 0.5, -1.0, 0.2, 0.8, 1.0])
    group_ids = torch.zeros(8, dtype=torch.long)

    loss, _ = compute_ggpo_loss(
        log_prob,
        ref_log_prob,
        shaped_rewards,
        response_mask,
        group_ids,
        raw_correctness=torch.tensor([0, 1, 0, 1, 0, 1, 1, 1]),
        beta=0.001,
        group_size=8,
        global_batch_size=8,
        batch_num_tokens=int(response_mask.sum().item()),
        dp_size=1,
        sequence_log_prob_reduction="sum",
    )
    centered = shaped_rewards - shaped_rewards.mean()
    expected = (torch.logsumexp(centered / 0.001, dim=0) - math.log(8)) / (math.sqrt(7) * 2)
    assert loss.detach().item() == pytest.approx(expected.item())


def test_token_mean_uses_one_optimizer_minibatch_denominator_across_microbatches_and_dp():
    rank0_losses = torch.tensor([[1.0, 2.0], [3.0, 99.0]])
    rank0_mask = torch.tensor([[1, 1], [1, 0]])
    rank1_losses = torch.tensor([[4.0, 5.0], [6.0, 7.0]])
    rank1_mask = torch.ones_like(rank1_losses)
    global_tokens = int(rank0_mask.sum() + rank1_mask.sum())

    rank_losses = []
    for losses, mask in ((rank0_losses, rank0_mask), (rank1_losses, rank1_mask)):
        microbatch_losses = [
            core_algos.agg_loss(
                loss_mat=losses[index : index + 1],
                loss_mask=mask[index : index + 1],
                loss_agg_mode="token-mean",
                batch_num_tokens=global_tokens,
                dp_size=2,
            )
            for index in range(2)
        ]
        rank_losses.append(sum(microbatch_losses))

    ddp_averaged_loss = sum(rank_losses) / 2
    expected = (1 + 2 + 3 + 4 + 5 + 6 + 7) / global_tokens
    assert ddp_averaged_loss == pytest.approx(expected)


def test_interactive_segment_accepts_arbitrary_positive_span_and_rejects_mismatch():
    assert resolve_segment_end(checkpoint_step=20, total_training_steps=800, start_raw="20", end_raw="37") == 37
    with pytest.raises(ValueError, match="boundary mismatch"):
        resolve_segment_end(checkpoint_step=19, total_training_steps=800, start_raw="20", end_raw="37")
    with pytest.raises(ValueError, match="must exceed"):
        resolve_segment_end(checkpoint_step=20, total_training_steps=800, start_raw="20", end_raw="20")








def test_resumed_validation_only_run_validates_and_exits(monkeypatch):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "project",
                "experiment_name": "experiment",
                "logger": [],
                "val_before_train": False,
                "val_only": True,
            },
            "skip": {},
        }
    )
    trainer.total_training_steps = 100
    trainer.train_dataloader = [object()]
    trainer._dump_executor = SimpleNamespace(_shutdown=False)
    trainer.checkpoint_manager = SimpleNamespace(update_weights=MagicMock())
    trainer._load_checkpoint = lambda: setattr(trainer, "global_steps", 20)
    trainer._validate = MagicMock(return_value={"val-core/acc": 0.5})
    trainer._shutdown_dump_executor = MagicMock()
    logger = MagicMock()
    monkeypatch.delenv("INTERACTIVE_SEGMENT_START_STEP", raising=False)
    monkeypatch.delenv("INTERACTIVE_SEGMENT_END_STEP", raising=False)

    with patch("verl.utils.tracking.Tracking", return_value=logger), patch.object(SkipManager, "init"):
        trainer.fit()

    trainer._validate.assert_called_once_with()
    logger.log.assert_called_once_with(data={"val-core/acc": 0.5, "training/global_step": 20}, step=20)
    trainer._shutdown_dump_executor.assert_called_once_with()




def _valid_matched_config(tmp_path: Path, method: str = "dapo", entropy_coeff: float = 0.0):
    model = _pinned_model_stub(tmp_path)
    data = _matched_data_stub(tmp_path)
    expected = expected_scientific_config(
        method,
        model_path=model,
        train_source=data / "train.parquet",
        validation_source=data / "math500.parquet",
        entropy_coeff=entropy_coeff,
    )
    config = OmegaConf.create()
    for path, value in expected.items():
        OmegaConf.update(config, path, value, force_add=True)
    contract = {
        "run_id": "matched-run",
        "method": method,
        "model_revision": MODEL_REVISION,
        "model_path": str(model.resolve()),
        "train_source": str((data / "train.parquet").resolve()),
        "validation_source": str((data / "math500.parquet").resolve()),
        "target_optimizer_steps": 800,
        "entropy_coeff": entropy_coeff,
    }
    OmegaConf.update(config, "trainer.matched_contract", contract, force_add=True)
    return config, contract


@pytest.mark.parametrize(
    ("method", "baseline"),
    [
        ("ggpo", "ggpo-exp-seqsum-beta0.001-qwen3-1.7b-base-dapo-filter"),
        ("dapo", "grpo-kl0-qwen3-1.7b-base-dapo-filter-dapo-cliphigher-token-mean"),
    ],
)
def test_matched_run_display_name_tracks_entropy_coeff(method, baseline):
    assert run_display_name(method, 0) == baseline
    assert run_display_name(method, 0.001) == f"{baseline}-entropy0.001"


@pytest.mark.parametrize("method", ["ggpo", "dapo"])
def test_matched_expected_config_tracks_entropy_coeff(tmp_path, method):
    model = _pinned_model_stub(tmp_path)
    data = _matched_data_stub(tmp_path)
    expected = expected_scientific_config(
        method,
        model_path=model,
        train_source=data / "train.parquet",
        validation_source=data / "math500.parquet",
        entropy_coeff=0.001,
    )
    display_name = f"{run_display_name(method)}-entropy0.001"

    assert expected["actor_rollout_ref.actor.entropy_coeff"] == 0.001
    assert expected["trainer.experiment_name"] == display_name
    assert expected["trainer.wandb_display_name"] == display_name


def test_matched_contract_rejects_resolved_entropy_coeff_drift(tmp_path, monkeypatch):
    config, _ = _valid_matched_config(tmp_path, entropy_coeff=0.001)
    monkeypatch.setenv("WANDB_RUN_ID", "matched-run")
    config.actor_rollout_ref.actor.entropy_coeff = 0.0

    with pytest.raises(ValueError, match="actor_rollout_ref.actor.entropy_coeff"):
        validate_matched_config(config)


def test_matched_contract_rejects_resolved_sampling_drift(tmp_path, monkeypatch):
    config, _ = _valid_matched_config(tmp_path)
    monkeypatch.setenv("WANDB_RUN_ID", "matched-run")
    validate_matched_config(config)

    config.actor_rollout_ref.rollout.top_p = 0.95
    with pytest.raises(ValueError, match="actor_rollout_ref.rollout.top_p"):
        validate_matched_config(config)


def test_matched_checkpoint_persists_replay_payload_before_marker(tmp_path, monkeypatch):
    config, contract = _valid_matched_config(tmp_path)
    monkeypatch.setenv("WANDB_RUN_ID", "matched-run")
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = config
    trainer.global_steps = 20
    trainer.total_training_steps = 800
    trainer._checkpoint_boundary_metrics = {
        "training/global_step": 20,
        "critic/score/mean": 0.1,
        "actor/pg_loss": 0.2,
        "actor/grad_norm": 0.3,
        "actor/entropy": 0.4,
        "val-core/alp_math500/acc/mean@32": 0.5,
        "val-core/alp_math500/acc/best@32/mean": 0.6,
        "val-core/alp_math500/acc/best@32/std": 0.1,
        "timing_s/step": 99,
    }
    folder = tmp_path / "global_step_20"
    folder.mkdir()

    trainer._persist_matched_boundary_metrics(str(folder))

    payload = json.loads((folder / "matched_boundary_metrics.json").read_text())
    assert payload["contract"] == contract
    assert payload["metrics"]["training/global_step"] == 20
    assert "timing_s/step" not in payload["metrics"]
    source = inspect.getsource(RayPPOTrainer._save_checkpoint)
    assert source.index("_checkpoint_fixed_lag_state") < source.index("_atomic_write_json")
    assert source.index("_atomic_write_json") < source.index("_persist_matched_boundary_metrics")
    assert source.index("_persist_matched_boundary_metrics") < source.index(
        "latest_checkpointed_iteration.txt"
    )




def test_fit_retains_validation_at_true_final_training_step():
    source = inspect.getsource(RayPPOTrainer.fit)

    assert "is_final_training_step = self.global_steps >= self.total_training_steps" in source
    assert "is_final_training_step" in source
    assert "self.global_steps % self.config.trainer.test_freq == 0" in source


def test_hybrid_esi_checkpoint_forces_validation_metrics():
    source = inspect.getsource(RayPPOTrainer.fit)

    assert "requires_checkpoint_validation = should_save_checkpoint" in source
    assert '"_checkpoint_requires_validation_metrics"' in source


def test_fit_still_honors_total_epoch_budget():
    fit_source = inspect.getsource(RayPPOTrainer.fit)
    candidate_source = inspect.getsource(RayPPOTrainer._next_training_candidate)
    filter_source = inspect.getsource(RayPPOTrainer._collect_mixed_reward_batch)

    assert "epoch < self.config.trainer.total_epochs" in fit_source
    assert "self._next_training_candidate(" in fit_source
    assert "if epoch >= self.config.trainer.total_epochs" in candidate_source
    assert "dynamic sampling exhausted trainer.total_epochs" in filter_source

    trainer = SimpleNamespace(
        config=OmegaConf.create({"trainer": {"total_epochs": 1}}),
        train_dataloader=[],
    )
    batch, _, epoch = RayPPOTrainer._next_training_candidate(
        trainer, iter(()), epoch=0, require_batch=False
    )
    assert batch is None
    assert epoch == 1

    with pytest.raises(RuntimeError, match="training exhausted trainer.total_epochs"):
        RayPPOTrainer._next_training_candidate(trainer, iter(()), epoch=0, require_batch=True)


def _checkpoint_load_stub(tmp_path: Path, *, dynamic_filter: bool):
    folder = tmp_path / "global_step_2"
    folder.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("2")
    torch.save({"cursor": "exhausted"}, folder / "data.pt")
    if dynamic_filter:
        (folder / "trainer_state.json").write_text(
            '{"schema_version": 1, "global_step": 2, "dynamic_filter_epoch": 0}\n'
        )
    loader = MagicMock()
    loader.__len__.return_value = 2
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "trainer": {
                    "resume_mode": "auto",
                    "resume_from_path": None,
                    "default_local_dir": str(tmp_path),
                    "default_hdfs_dir": None,
                    "del_local_ckpt_after_load": False,
                },
                "algorithm": {"filter_groups": {"enable": dynamic_filter}},
            }
        ),
        train_dataloader=loader,
        actor_rollout_wg=SimpleNamespace(load_checkpoint=MagicMock()),
        use_critic=False,
    )
    return trainer, loader, folder


def test_resume_at_exact_epoch_boundary_starts_the_next_standard_epoch(tmp_path):
    trainer, loader, folder = _checkpoint_load_stub(tmp_path, dynamic_filter=False)

    resolved = RayPPOTrainer._load_checkpoint(trainer)

    assert resolved == str(folder)
    loader.load_state_dict.assert_not_called()


def test_dynamic_filter_resume_restores_candidate_epoch_and_exhausted_cursor(tmp_path):
    trainer, loader, folder = _checkpoint_load_stub(tmp_path, dynamic_filter=True)

    resolved = RayPPOTrainer._load_checkpoint(trainer)

    assert resolved == str(folder)
    assert trainer._restored_dynamic_filter_epoch == 0
    loader.load_state_dict.assert_called_once_with({"cursor": "exhausted"})


def test_dynamic_filter_and_fixed_lag_resume_restore_one_schema_v2_boundary(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    config = OmegaConf.create(
        {
            "trainer": {
                "fixed_policy_lag": 2,
                "fixed_lag_ledger_path": str(tmp_path / "batch-ledger.jsonl"),
                "default_local_dir": str(checkpoint_root),
                "default_hdfs_dir": None,
                "resume_mode": "auto",
                "resume_from_path": None,
                "del_local_ckpt_after_load": False,
                "remove_previous_ckpt_in_save": False,
                "max_actor_ckpt_to_keep": None,
                "max_critic_ckpt_to_keep": None,
            },
            "data": {"train_batch_size": 2},
            "actor_rollout_ref": {
                "actor": {"checkpoint": {"async_save": False}},
                "rollout": {"n": 2},
            },
            "algorithm": {"filter_groups": {"enable": True}},
        }
    )

    producer = RayPPOTrainer.__new__(RayPPOTrainer)
    producer.config = config
    producer.total_training_steps = 40
    producer.global_steps = 0
    producer._init_fixed_lag_state()
    producer._initialize_fixed_lag_ledger()

    def enqueue(slot, behavior, kind):
        batch = DataProto.from_single_dict(
            {
                "rollout_log_probs": torch.zeros((4, 2)),
                "response_mask": torch.ones((4, 2), dtype=torch.bool),
                "uid": np.asarray(["a", "a", "b", "b"], dtype=object),
            }
        )
        producer._tag_fixed_lag_batch(batch, slot, behavior, kind)
        producer._enqueue_fixed_lag_batch(batch)

    enqueue(-1, 0, "warmup")
    enqueue(0, 0, "warmup")
    for step in (1, 2):
        enqueue(step, step - 1, "steady")
        producer._consume_fixed_lag_batch(step)
    producer.global_steps = 2
    producer._current_dynamic_filter_epoch = 3
    producer._ggpo_moving_reference_enabled = False
    producer.use_critic = False
    producer.actor_rollout_wg = SimpleNamespace(
        save_checkpoint=MagicMock(side_effect=lambda path, *_args, **_kwargs: Path(path).mkdir(parents=True))
    )
    producer.train_dataloader = SimpleNamespace(
        state_dict=MagicMock(return_value={"cursor": "after-warmup-and-two-updates"})
    )
    producer._save_checkpoint()

    folder = checkpoint_root / "global_step_2"
    state = json.loads((folder / "trainer_state.json").read_text())
    assert state["schema_version"] == 2
    assert state["dynamic_filter_epoch"] == 3
    assert state["fixed_lag"]["lag"] == 2
    assert (checkpoint_root / "latest_checkpointed_iteration.txt").read_text() == "2"

    restored = RayPPOTrainer.__new__(RayPPOTrainer)
    restored.config = config
    restored.total_training_steps = 40
    restored._ggpo_moving_reference_enabled = False
    restored._init_fixed_lag_state()
    restored.train_dataloader = MagicMock()
    restored.train_dataloader.__len__.return_value = 10
    restored.actor_rollout_wg = SimpleNamespace(load_checkpoint=MagicMock())
    restored.use_critic = False

    resolved = restored._load_checkpoint()

    assert resolved == str(folder)
    assert restored._restored_dynamic_filter_epoch == 3
    restored.train_dataloader.load_state_dict.assert_called_once_with(
        {"cursor": "after-warmup-and-two-updates"}
    )
    assert [entry["logical_generation_step"] for entry in restored._fixed_lag_queue] == [1, 2]
