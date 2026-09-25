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
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.config import _validate_fixed_policy_lag


def _config(tmp_path: Path, *, lag=2, group_size=2, rollout_correction=None):
    return OmegaConf.create(
        {
            "trainer": {
                "fixed_policy_lag": lag,
                "fixed_lag_ledger_path": None if lag is None else str(tmp_path / "batch-ledger.jsonl"),
                "default_local_dir": str(tmp_path / "checkpoints"),
                "use_v1": False,
                "nnodes": 1,
                "ggpo_moving_reference": {"enabled": False, "update_freq": None},
                "total_training_steps": 40,
                "total_epochs": 10,
            },
            "data": {"train_batch_size": 2},
            "actor_rollout_ref": {
                "hybrid_engine": True,
                "rollout": {"n": group_size, "calculate_log_probs": True},
                "actor": {"policy_loss": {"loss_mode": "vanilla"}},
            },
            "algorithm": {
                "rollout_correction": rollout_correction
                or {
                    "rollout_is": None,
                    "rollout_is_threshold": 2.0,
                    "rollout_rs": None,
                    "rollout_rs_threshold": None,
                    "bypass_mode": True,
                    "loss_type": "ppo_clip",
                    "rollout_is_batch_normalize": False,
                },
                "filter_groups": {"enable": False},
            },
        }
    )


def _trainer(tmp_path: Path, *, lag=2, group_size=2, rollout_correction=None, initialize=True):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = _config(
        tmp_path,
        lag=lag,
        group_size=group_size,
        rollout_correction=rollout_correction,
    )
    trainer.total_training_steps = 40
    trainer.global_steps = 0
    trainer._ggpo_moving_reference_enabled = False
    trainer._init_fixed_lag_state()
    if initialize and lag is not None:
        trainer._initialize_fixed_lag_ledger()
    return trainer


def _batch(group_size=2, groups=2, *, sentinel=0.0, uid_order=None):
    response_count = group_size * groups
    if uid_order is None:
        uid_order = [f"group-{group}" for group in range(groups) for _ in range(group_size)]
    batch = DataProto.from_single_dict(
        {
            "rollout_log_probs": torch.full((response_count, 3), sentinel),
            "response_mask": torch.ones((response_count, 3), dtype=torch.bool),
            "token_level_scores": torch.arange(response_count * 3, dtype=torch.float32).reshape(
                response_count, 3
            ),
            "uid": np.asarray(uid_order, dtype=object),
            "acc": np.arange(response_count),
        }
    )
    batch.meta_info["reward_extra_keys"] = ["acc"]
    return batch


def _enqueue(trainer, slot, behavior_version, kind, *, batch=None, sampling_metrics=None):
    batch = _batch(group_size=int(trainer.config.actor_rollout_ref.rollout.n)) if batch is None else batch
    trainer._tag_fixed_lag_batch(
        batch,
        logical_generation_step=slot,
        behavior_policy_version=behavior_version,
        kind=kind,
        sampling_metrics=sampling_metrics,
    )
    trainer._enqueue_fixed_lag_batch(batch)
    return batch


def _checkpoint_at_step_two(tmp_path: Path):
    trainer = _trainer(tmp_path)
    _enqueue(trainer, -1, 0, "warmup")
    _enqueue(trainer, 0, 0, "warmup")
    for step in (1, 2):
        _enqueue(trainer, step, step - 1, "steady")
        trainer._consume_fixed_lag_batch(step)
    trainer.global_steps = 2
    pointer = trainer._checkpoint_fixed_lag_state()
    state = {
        "lag": 2,
        "ledger_path": trainer._fixed_lag_ledger_path,
        "checkpoint_event_seq": pointer,
    }
    return trainer, state


def test_fixed_lag_off_with_nulls_creates_no_ledger_or_metrics(tmp_path):
    trainer = _trainer(tmp_path, lag=None)

    assert trainer._fixed_policy_lag is None
    assert trainer._fixed_lag_ledger_path is None
    assert list(trainer._fixed_lag_queue) == []
    assert not (tmp_path / "batch-ledger.jsonl").exists()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("trainer.fixed_policy_lag", 0, "greater than zero"),
        ("trainer.fixed_policy_lag", -1, "greater than zero"),
        ("trainer.fixed_policy_lag", True, "greater than zero"),
        ("trainer.fixed_lag_ledger_path", None, "nonempty string"),
        ("trainer.fixed_lag_ledger_path", "", "nonempty string"),
        ("trainer.use_v1", True, "V0 trainer"),
        ("trainer.nnodes", 2, "nnodes=1"),
        ("actor_rollout_ref.hybrid_engine", False, "hybrid_engine=true"),
        ("async_training", {}, "fully-async"),
        ("offline_data", {}, "online V0"),
        ("actor_rollout_ref.rollout.calculate_log_probs", False, "calculate_log_probs=true"),
        ("actor_rollout_ref.rollout.n", 1, "greater than one"),
    ],
)
def test_fixed_lag_validation_rejects_invalid_contracts(tmp_path, path, value, message):
    config = _config(tmp_path)
    OmegaConf.update(config, path, value, merge=False, force_add=True)

    with pytest.raises(ValueError, match=message):
        _validate_fixed_policy_lag(config)


def test_fixed_lag_validation_rejects_orphan_ledger_path(tmp_path):
    config = _config(tmp_path, lag=None)
    config.trainer.fixed_lag_ledger_path = str(tmp_path / "orphan.jsonl")
    with pytest.raises(ValueError, match="must be null"):
        _validate_fixed_policy_lag(config)


def test_warmup_fifo_is_slot_exact_single_use_and_retains_k_entries(tmp_path):
    trainer = _trainer(tmp_path, lag=4)
    for slot in range(-3, 1):
        batch = _enqueue(trainer, slot, 0, "warmup", sampling_metrics={"sampling/slot": slot})
        assert batch.meta_info["fixed_lag_behavior_policy_version"] == 0

    _enqueue(trainer, 1, 0, "steady")
    trainer.global_steps = 1
    consumed = trainer._consume_fixed_lag_batch(1)
    assert consumed.meta_info["fixed_lag_generation_step"] == -3
    assert trainer._fixed_lag_consumption_metrics(consumed) == {
        "sampling/slot": -3,
        "actor/fixed_lag_generation_step": -3,
        "actor/fixed_lag_behavior_policy_version": 0,
        "actor/fixed_lag_effective": 4,
        "actor/fixed_lag_policy_version_lag": 0,
    }
    assert len(trainer._fixed_lag_queue) == 4
    with pytest.raises(RuntimeError, match="already enqueued or consumed"):
        trainer._enqueue_fixed_lag_batch(consumed)


def test_checkpoint_20_lag_4_retains_behavior_versions_16_through_19(tmp_path):
    trainer = _trainer(tmp_path, lag=4)
    for slot in range(-3, 1):
        _enqueue(trainer, slot, 0, "warmup")
    for step in range(1, 21):
        _enqueue(trainer, step, step - 1, "steady")
        trainer._consume_fixed_lag_batch(step)

    trainer.global_steps = 20
    pointer = trainer._checkpoint_fixed_lag_state()

    assert [entry["logical_generation_step"] for entry in trainer._fixed_lag_queue] == [17, 18, 19, 20]
    assert [entry["behavior_policy_version"] for entry in trainer._fixed_lag_queue] == [16, 17, 18, 19]
    assert trainer._fixed_lag_checkpoint_events[pointer]["pending_entry_ids"] == [
        entry["entry_id"] for entry in trainer._fixed_lag_queue
    ]


def test_complete_n8_groups_and_payload_survive_atomic_round_trip_in_order(tmp_path):
    trainer = _trainer(tmp_path, lag=2, group_size=8)
    original = _batch(group_size=8, groups=2, sentinel=-7.0)
    _enqueue(trainer, -1, 0, "warmup", batch=original, sampling_metrics={"sampling/generated_groups": 2})

    restored = trainer._consume_fixed_lag_batch(1)

    assert restored.non_tensor_batch["uid"].tolist() == original.non_tensor_batch["uid"].tolist()
    assert restored.non_tensor_batch["acc"].tolist() == original.non_tensor_batch["acc"].tolist()
    assert torch.equal(restored.batch["rollout_log_probs"], original.batch["rollout_log_probs"])
    assert torch.equal(restored.batch["token_level_scores"], original.batch["token_level_scores"])
    assert restored.meta_info["reward_extra_keys"] == ["acc"]
    assert restored.meta_info["fixed_lag_sampling_metrics"] == {"sampling/generated_groups": 2}
    assert not list(Path(trainer._fixed_lag_batch_dir).glob(".fixed_lag_batch.*"))


def test_fixed_lag_rejects_nonfinite_behavior_log_prob_on_response_mask(tmp_path):
    trainer = _trainer(tmp_path)
    batch = _batch()
    batch.batch["rollout_log_probs"][0, 0] = torch.nan

    with pytest.raises(RuntimeError, match="non-finite"):
        trainer._tag_fixed_lag_batch(batch, -1, 0, "warmup")


@pytest.mark.parametrize(
    "uids",
    [
        ["a", "b", "a", "b"],
        ["a", "a", "b"],
        ["a", "a", "b", "b", "a", "a"],
    ],
)
def test_fixed_lag_rejects_interleaved_incomplete_or_duplicate_uid_groups(tmp_path, uids):
    trainer = _trainer(tmp_path)
    batch = _batch(group_size=1, groups=len(uids), uid_order=uids)
    with pytest.raises((ValueError, RuntimeError), match="divisible|contiguous|exactly one group"):
        trainer._tag_fixed_lag_batch(batch, -1, 0, "warmup")


def test_dapo_filtering_and_top_up_run_only_at_production(tmp_path):
    trainer = _trainer(tmp_path)
    produced = _batch()
    trainer._collect_mixed_reward_batch = MagicMock(return_value=(produced, "iterator-after-top-up", 3))
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=MagicMock())
    generation_metrics = {}

    batch, iterator, epoch = trainer._produce_training_batch(
        {}, "iterator", 2, True, generation_metrics, {}, False
    )
    _enqueue(trainer, -1, 0, "warmup", batch=batch)
    consumed = trainer._consume_fixed_lag_batch(1)

    trainer._collect_mixed_reward_batch.assert_called_once()
    trainer.checkpoint_manager.sleep_replicas.assert_called_once()
    assert iterator == "iterator-after-top-up"
    assert epoch == 3
    assert torch.equal(consumed.batch["token_level_scores"], produced.batch["token_level_scores"])
    assert consumed.non_tensor_batch["acc"].tolist() == produced.non_tensor_batch["acc"].tolist()


def test_plain_grpo_uses_stored_behavior_tensor_as_old_log_probs(tmp_path):
    trainer = _trainer(tmp_path)
    batch = _batch(sentinel=-13.0)
    stored_behavior = batch.batch["rollout_log_probs"].clone()

    batch, metrics = trainer._prepare_behavior_log_probs(batch, {})

    assert metrics == {}
    assert torch.equal(batch.batch["old_log_probs"], stored_behavior)
    assert torch.equal(batch.batch["rollout_log_probs"], stored_behavior)


def test_decoupled_token_tis_keeps_stored_behavior_as_denominator(tmp_path):
    correction = {
        "rollout_is": "token",
        "rollout_is_threshold": 2.0,
        "rollout_rs": None,
        "rollout_rs_threshold": None,
        "bypass_mode": False,
        "loss_type": "ppo_clip",
        "rollout_is_batch_normalize": False,
    }
    trainer = _trainer(tmp_path, rollout_correction=correction)
    batch = _batch(sentinel=0.0)
    current_proximal = torch.full_like(batch.batch["rollout_log_probs"], torch.log(torch.tensor(2.0)))
    trainer._compute_old_log_prob = lambda _: DataProto.from_single_dict({"old_log_probs": current_proximal})

    batch, metrics = trainer._prepare_behavior_log_probs(batch, {})

    assert torch.equal(batch.batch["rollout_log_probs"], torch.zeros_like(current_proximal))
    assert torch.equal(batch.batch["old_log_probs"], current_proximal)
    assert torch.allclose(batch.batch["rollout_is_weights"], torch.full_like(current_proximal, 2.0))
    assert metrics


def test_decoupled_sequence_tis_uses_current_learner_and_keeps_behavior(tmp_path):
    correction = {
        "rollout_is": "sequence",
        "rollout_is_threshold": 2.0,
        "rollout_rs": None,
        "rollout_rs_threshold": None,
        "bypass_mode": False,
        "loss_type": "ppo_clip",
        "rollout_is_batch_normalize": False,
    }
    trainer = _trainer(tmp_path, rollout_correction=correction)
    batch = _batch(sentinel=-0.5)
    batch.batch["response_mask"][0, -1] = False
    stored_behavior = batch.batch["rollout_log_probs"].clone()
    current_proximal = torch.tensor(
        [[-0.4, -0.3, 1.0], [-0.2, -0.1, 0.0], [-0.7, -0.6, -0.5], [0.0, 0.1, 0.2]]
    )
    trainer._compute_old_log_prob = lambda _: DataProto.from_single_dict({"old_log_probs": current_proximal})

    batch, metrics = trainer._prepare_behavior_log_probs(batch, {})

    mask = batch.batch["response_mask"]
    expected_sequence = torch.exp(((current_proximal - stored_behavior) * mask).sum(dim=-1)).clamp(max=2.0)
    expected_weights = expected_sequence.unsqueeze(-1).expand_as(current_proximal) * mask
    assert torch.equal(batch.batch["rollout_log_probs"], stored_behavior)
    assert torch.equal(batch.batch["old_log_probs"], current_proximal)
    assert torch.allclose(batch.batch["rollout_is_weights"], expected_weights)
    assert metrics


def test_ggpo_theory_identity_uses_stored_behavior_version(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.global_steps = 5
    batch = _batch()
    batch.meta_info["fixed_lag_behavior_policy_version"] = 2

    identity = trainer._ggpo_theory_policy_identity(batch)

    assert identity["training_policy_version"] == 4
    assert identity["behavior_policy_versions"].tolist() == [2, 2, 2, 2]
    assert identity["behavior_version_stratification"] is False


def test_moving_reference_metric_is_captured_before_reference_scoring_and_checkpoint_update():
    source = inspect.getsource(RayPPOTrainer.fit)
    capture = "ggpo_reference_version_used = int(self._ggpo_reference_version)"
    emission = 'metrics["actor/ggpo_reference_version"] = ggpo_reference_version_used'
    assert source.index(capture) < source.index("self._compute_ref_log_prob(batch)")
    assert source.index(emission) < source.index("self._save_checkpoint()")


def test_moving_reference_resume_fails_when_backing_actor_checkpoint_is_missing(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = _config(tmp_path)
    trainer._restored_ggpo_reference_version = 20
    trainer.ref_policy_wg = SimpleNamespace(load_reference_checkpoint=MagicMock())
    loaded = tmp_path / "global_step_37"
    loaded.mkdir()

    with pytest.raises(RuntimeError, match="backing actor checkpoint"):
        trainer._restore_ggpo_reference(str(loaded))

    backing_actor = tmp_path / "global_step_20" / "actor"
    backing_actor.mkdir(parents=True)
    trainer._restore_ggpo_reference(str(loaded))
    trainer.ref_policy_wg.load_reference_checkpoint.assert_called_once_with(str(backing_actor))
    assert trainer._ggpo_reference_version == 20


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_resume_fails_closed_for_missing_or_corrupt_pending_batch(tmp_path, failure):
    trainer, state = _checkpoint_at_step_two(tmp_path)
    pending_path = Path(trainer._fixed_lag_batch_path(trainer._fixed_lag_queue[0]["batch_path"]))
    if failure == "missing":
        pending_path.unlink()
    else:
        pending_path.write_bytes(b"not a DataProto")
    restored = _trainer(tmp_path, initialize=False)

    with pytest.raises(RuntimeError, match="missing|corrupt or unreadable"):
        restored._restore_fixed_lag_checkpoint(state, 2)


@pytest.mark.parametrize("field", ["lag", "ledger_path", "checkpoint_event_seq"])
def test_resume_fails_closed_for_wrong_lag_path_or_pointer(tmp_path, field):
    _, state = _checkpoint_at_step_two(tmp_path)
    state = dict(state)
    state[field] = {"lag": 3, "ledger_path": str(tmp_path / "wrong.jsonl"), "checkpoint_event_seq": 999}[field]
    restored = _trainer(tmp_path, initialize=False)

    with pytest.raises(RuntimeError, match="lag|ledger path|pointer"):
        restored._restore_fixed_lag_checkpoint(state, 2)


def test_resume_fails_closed_for_wrong_queue_length(tmp_path):
    trainer, state = _checkpoint_at_step_two(tmp_path)
    ledger_path = Path(trainer._fixed_lag_ledger_path)
    records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    checkpoint = records[state["checkpoint_event_seq"] - 1]
    checkpoint["pending_entry_ids"] = checkpoint["pending_entry_ids"][:1]
    ledger_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    restored = _trainer(tmp_path, initialize=False)

    with pytest.raises(RuntimeError, match="exactly k"):
        restored._restore_fixed_lag_checkpoint(state, 2)


def test_ledger_tail_is_ignored_and_resume_does_not_warm_up_again(tmp_path):
    trainer, state = _checkpoint_at_step_two(tmp_path)
    checkpoint_ids = [entry["entry_id"] for entry in trainer._fixed_lag_queue]
    _enqueue(trainer, 3, 2, "steady")
    trainer._consume_fixed_lag_batch(3)
    restored = _trainer(tmp_path, initialize=False)

    restored._restore_fixed_lag_checkpoint(state, 2)

    assert [entry["entry_id"] for entry in restored._fixed_lag_queue] == checkpoint_ids
    assert [entry["logical_generation_step"] for entry in restored._fixed_lag_queue] == [1, 2]
    assert json.loads(Path(restored._fixed_lag_ledger_path).read_text().splitlines()[-1])["record_type"] == "resume"
    fit_source = inspect.getsource(RayPPOTrainer.fit)
    assert "if loaded_checkpoint is None:" in fit_source
    assert "self._warm_up_fixed_lag(" in fit_source


def test_load_checkpoint_requires_trainer_state_for_fixed_lag(tmp_path):
    folder = tmp_path / "checkpoints" / "global_step_2"
    folder.mkdir(parents=True)
    (tmp_path / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("2")
    torch.save({"cursor": 2}, folder / "data.pt")
    trainer = _trainer(tmp_path, initialize=False)
    trainer.config.trainer.resume_mode = "auto"
    trainer.config.trainer.resume_from_path = None
    trainer.config.trainer.default_hdfs_dir = None
    trainer.config.trainer.del_local_ckpt_after_load = False
    trainer.train_dataloader = MagicMock()
    trainer.train_dataloader.__len__.return_value = 10
    trainer.actor_rollout_wg = SimpleNamespace(load_checkpoint=MagicMock())
    trainer.use_critic = False

    with pytest.raises(RuntimeError, match="missing trainer_state.json"):
        trainer._load_checkpoint()


def test_warmup_wakes_replicas_after_every_production(tmp_path):
    # Each production sleeps the rollout replicas; without a wake between
    # warmup productions the second generation runs on freed engine memory
    # (observed as a CUDA illegal memory access on the smoke gate).
    trainer = _trainer(tmp_path)
    trainer.checkpoint_manager = SimpleNamespace(update_weights=MagicMock())
    trainer._next_training_candidate = MagicMock(
        side_effect=lambda iterator, epoch, require_batch=True: ({}, iterator, epoch)
    )
    trainer._produce_training_batch = MagicMock(
        side_effect=lambda *args, **kwargs: (_batch(), "iterator", 0)
    )

    trainer._warm_up_fixed_lag("iterator", 0, mixed_filter_enabled=True)

    assert trainer._produce_training_batch.call_count == trainer._fixed_policy_lag
    assert trainer.checkpoint_manager.update_weights.call_count == trainer._fixed_policy_lag
    for call in trainer.checkpoint_manager.update_weights.call_args_list:
        assert call.args == (0,)
    assert len(trainer._fixed_lag_queue) == trainer._fixed_policy_lag
    assert [entry["logical_generation_step"] for entry in trainer._fixed_lag_queue] == [-1, 0]
