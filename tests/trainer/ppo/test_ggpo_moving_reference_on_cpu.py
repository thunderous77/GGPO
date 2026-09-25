# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.config import validate_policy_loss_trainer_compatibility
from verl.workers.engine_workers import ActorRolloutRefWorker


def _moving_reference_config(*, loss_mode="ggpo", enabled=True, update_freq=4, save_freq=4):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "strategy": "fsdp",
                    "policy_loss": {"loss_mode": loss_mode},
                    "checkpoint": {
                        "save_contents": ["model", "optimizer", "extra"],
                        "load_contents": ["model", "optimizer", "extra"],
                        "async_save": False,
                    },
                },
                "ref": {"strategy": "fsdp"},
                "model": {"lora": {"rank": 0}, "lora_adapter_path": None},
                "rollout": {
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "repetition_penalty": 1.0,
                },
            },
            "trainer": {
                "balance_batch": False,
                "use_v1": False,
                "save_freq": save_freq,
                "resume_mode": "disable",
                "del_local_ckpt_after_load": False,
                "logger": [],
                "ggpo_moving_reference": {"enabled": enabled, "update_freq": update_freq},
                "ggpo_theory_contract": {
                    "run_id": "run",
                    "source_identity": "source",
                    "reference_policy_identity": "initial-ref",
                    "behavior_policy_identity": "behavior",
                },
            },
        }
    )


def test_moving_reference_frequency_must_equal_checkpoint_frequency():
    config = _moving_reference_config(update_freq=3, save_freq=4)

    with pytest.raises(ValueError, match="update_freq must be positive and equal trainer.save_freq"):
        validate_policy_loss_trainer_compatibility(config)


def test_supported_moving_reference_configuration_is_accepted():
    validate_policy_loss_trainer_compatibility(_moving_reference_config())

    update_cadence = _moving_reference_config(update_freq=3, save_freq=4)
    update_cadence.trainer.ggpo_moving_reference.refresh_every_updates = 160
    validate_policy_loss_trainer_compatibility(update_cadence)


def test_supported_offline_moving_reference_configurations_are_accepted():
    training = _moving_reference_config()
    training.offline_data = {"mode": "train", "smoke": False}
    validate_policy_loss_trainer_compatibility(training)

    smoke = _moving_reference_config(save_freq=-1)
    smoke.offline_data = {"mode": "train", "smoke": True}
    validate_policy_loss_trainer_compatibility(smoke)

    denser_checkpoints = _moving_reference_config(update_freq=20, save_freq=5)
    denser_checkpoints.offline_data = {"mode": "train", "smoke": False}
    validate_policy_loss_trainer_compatibility(denser_checkpoints)

    invalid_divisor = _moving_reference_config(update_freq=20, save_freq=6)
    invalid_divisor.offline_data = {"mode": "train", "smoke": False}
    with pytest.raises(ValueError, match="positive multiple"):
        validate_policy_loss_trainer_compatibility(invalid_divisor)


def test_moving_reference_accepts_grpo_with_kl_and_agro():
    grpo = _moving_reference_config(loss_mode="vanilla")
    grpo.actor_rollout_ref.actor.use_kl_loss = True
    validate_policy_loss_trainer_compatibility(grpo)

    agro = _moving_reference_config(loss_mode="agro")
    agro.actor_rollout_ref.actor.policy_loss.sequence_log_prob_reduction = "sum"
    validate_policy_loss_trainer_compatibility(agro)

    grpo.actor_rollout_ref.actor.use_kl_loss = False
    with pytest.raises(ValueError, match="Vanilla moving reference requires.*use_kl_loss=true"):
        validate_policy_loss_trainer_compatibility(grpo)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("trainer.use_v1", True, "V0 trainer"),
        ("async_training", {"rollout_backend": "verl"}, "fully-async"),
        ("offline_data", {"mode": "unsupported"}, "unsupported offline_data.mode"),
        ("actor_rollout_ref.actor.strategy", "megatron", "synchronous FSDP"),
        ("actor_rollout_ref.ref.fsdp_config.forward_only", False, "forward-only reference"),
        ("actor_rollout_ref.model.lora.rank", 8, "does not support LoRA"),
        ("actor_rollout_ref.actor.checkpoint.async_save", True, "synchronous actor checkpoint"),
        ("trainer.del_local_ckpt_after_load", True, "del_local_ckpt_after_load=false"),
    ],
)
def test_moving_reference_rejects_unsupported_paths(path, value, message):
    config = _moving_reference_config()
    if path == "trainer.del_local_ckpt_after_load":
        config.trainer.resume_mode = "auto"
    OmegaConf.update(config, path, value, force_add=True)

    with pytest.raises(ValueError, match=message):
        validate_policy_loss_trainer_compatibility(config)


def test_reference_checkpoint_load_is_model_only_and_restores_configuration():
    original_contents = ["model", "optimizer", "extra"]
    manager = SimpleNamespace(checkpoint_load_contents=original_contents)
    observed = []

    def load_checkpoint(local_path, del_local_after_load):
        observed.append((local_path, list(manager.checkpoint_load_contents), del_local_after_load))

    worker = SimpleNamespace(
        _is_ref=True,
        config=OmegaConf.create({"ref": {"strategy": "fsdp"}}),
        ref=SimpleNamespace(
            engine=SimpleNamespace(checkpoint_manager=manager),
            load_checkpoint=load_checkpoint,
        ),
    )

    ActorRolloutRefWorker.load_reference_checkpoint(worker, "/checkpoints/global_step_4/actor")

    assert observed == [("/checkpoints/global_step_4/actor", ["model"], False)]
    assert manager.checkpoint_load_contents is original_contents


def test_scheduled_checkpoint_updates_reference_version(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": False}}},
            "algorithm": {},
            "trainer": {
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": None,
                "remove_previous_ckpt_in_save": False,
                "max_actor_ckpt_to_keep": None,
                "max_critic_ckpt_to_keep": None,
                "matched_contract": None,
                "ggpo_moving_reference": {"update_freq": 4},
            },
        }
    )
    trainer.global_steps = 4
    trainer.use_critic = False
    trainer._ggpo_moving_reference_enabled = True
    trainer._ggpo_reference_version = 0

    def save_actor(path, *_args, **_kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)

    trainer.actor_rollout_wg = SimpleNamespace(save_checkpoint=MagicMock(side_effect=save_actor))
    trainer.ref_policy_wg = SimpleNamespace(load_reference_checkpoint=MagicMock())
    trainer.train_dataloader = SimpleNamespace(state_dict=MagicMock(return_value={"step": 4}))

    trainer._save_checkpoint()

    expected_actor_path = str(tmp_path / "global_step_4" / "actor")
    trainer.ref_policy_wg.load_reference_checkpoint.assert_called_once_with(expected_actor_path)
    assert trainer._ggpo_reference_version == 4
    state = json.loads((tmp_path / "global_step_4" / "trainer_state.json").read_text())
    assert state["schema_version"] == 2
    assert state["ggpo_reference_version_after_step"] == 4

    # Forced/last-step saves outside the configured cadence do not add reset points.
    trainer.global_steps = 5
    trainer.train_dataloader.state_dict.return_value = {"step": 5}
    trainer._save_checkpoint()
    trainer.ref_policy_wg.load_reference_checkpoint.assert_called_once_with(expected_actor_path)
    assert trainer._ggpo_reference_version == 4
    state = json.loads((tmp_path / "global_step_5" / "trainer_state.json").read_text())
    assert state["ggpo_reference_version_after_step"] == 4

    # An arbitrary segment checkpoint restores the persisted active version;
    # it does not create an unscheduled reset at the segment boundary.
    resumed_folder = str(tmp_path / "global_step_5")
    trainer._restored_ggpo_reference_version = 4
    trainer._restore_ggpo_reference(resumed_folder)
    trainer.ref_policy_wg.load_reference_checkpoint.assert_called_with(expected_actor_path)
    assert trainer._ggpo_reference_version == 4


def test_reference_version_metric_is_emitted_only_when_feature_is_enabled():
    source = inspect.getsource(RayPPOTrainer.fit)
    metric = 'metrics["actor/ggpo_reference_version"] = ggpo_reference_version_used'

    assert source.index(metric) < source.index("if should_save_checkpoint:")
    assert "if getattr(self, \"_ggpo_moving_reference_enabled\", False):" in source


def _update_cadence_trainer(*, mini_batch_size, rollout_n, completed_updates):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": mini_batch_size},
                "rollout": {"n": rollout_n},
            },
            "trainer": {"ggpo_moving_reference": {"refresh_every_updates": 160}},
        }
    )
    trainer._ggpo_optimizer_updates = completed_updates
    trainer._ggpo_reference_version = 0
    return trainer


def _cadence_batch(train_batch_size, rollout_n):
    responses = train_batch_size * rollout_n
    return DataProto.from_single_dict(
        {
            "sample_id": torch.arange(responses),
            "prompt_id": torch.arange(train_batch_size).repeat_interleave(rollout_n),
            "ref_log_prob": torch.zeros(responses, 1),
        }
    )


def test_update_cadence_refreshes_inside_step_and_rescores_remainder():
    trainer = _update_cadence_trainer(mini_batch_size=32, rollout_n=2, completed_updates=128)
    batch = _cadence_batch(train_batch_size=2048, rollout_n=2)
    events = []

    def update_actor(part, update_lr_scheduler=True):
        events.append(
            (
                "update",
                part.batch["prompt_id"].tolist(),
                part.batch["ref_log_prob"].clone(),
                update_lr_scheduler,
            )
        )
        return DataProto.from_single_dict(data={}, meta_info={"metrics": {"actor/loss": [len(part)]}})

    def refresh(update):
        events.append(("refresh", update))
        trainer._ggpo_reference_version = update

    trainer._update_actor = update_actor
    trainer._refresh_ggpo_reference_at_update = refresh
    trainer._compute_ref_log_prob = lambda remainder: DataProto.from_single_dict(
        {"ref_log_prob": torch.ones(len(remainder), 1)}
    )

    output = trainer._update_actor_with_ggpo_refresh(batch)

    assert [event[0] for event in events] == ["update", "refresh", "update"]
    assert events[1] == ("refresh", 160)
    assert len(events[0][1]) == 32 * 32 * 2
    assert events[0][1][-1] == 1023 and events[2][1][0] == 1024
    assert torch.count_nonzero(events[0][2]) == 0
    assert torch.all(events[2][2] == 1)
    assert events[0][3] is False and events[2][3] is True
    assert output.meta_info["metrics"]["actor/loss"] == [2048, 2048]
    assert trainer._ggpo_optimizer_updates == 192


def test_update_cadence_aligned_boundary_matches_step_cadence():
    trainer = _update_cadence_trainer(mini_batch_size=32, rollout_n=1, completed_updates=0)
    batch = _cadence_batch(train_batch_size=256, rollout_n=1)
    events = []

    trainer._update_actor = lambda part, update_lr_scheduler=True: (
        events.append(("update", len(part), update_lr_scheduler))
        or DataProto.from_single_dict(data={}, meta_info={"metrics": {"actor/loss": [len(part)]}})
    )
    trainer._refresh_ggpo_reference_at_update = lambda update: events.append(("refresh", update))
    trainer._compute_ref_log_prob = MagicMock()

    for _ in range(20):
        trainer._update_actor_with_ggpo_refresh(batch)

    legacy = RayPPOTrainer.__new__(RayPPOTrainer)
    legacy.config = OmegaConf.create({"trainer": {"ggpo_moving_reference": {"update_freq": 20}}})
    legacy._ggpo_moving_reference_enabled = True
    legacy._ggpo_reference_version = 0
    legacy.global_steps = 20

    assert events[:-2] == [("update", 256, True)] * 19
    assert events[-2:] == [("update", 256, True), ("refresh", 160)]
    assert trainer._compute_ref_log_prob.call_count == 0
    assert legacy._ggpo_reference_version_after_step() == 20
