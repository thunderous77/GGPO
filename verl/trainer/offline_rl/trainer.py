# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""A rollout-free trainer that deterministically reuses immutable behavior batches."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.offline_rl.contract import (
    PARAMETERIZED_OFFLINE_CONTRACT,
    SEALED_MIXED_FILTERED_OFFLINE_CONTRACT,
    SEALED_OFFLINE_CONTRACT,
    filtered_manifest_round_count,
    filtered_training_round_count,
    validate_checkpoint_dataset_identity,
    validate_parameterized_contract,
    validate_reference_contract,
)
from verl.trainer.offline_rl.dataset import OfflineBatchDataset
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.mixed_reward_filter import raw_correctness_tensor, select_mixed_reward_group_indices
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    compute_advantage,
    maybe_apply_group_reference_advantage_bypass,
)
from verl.trainer.ppo.utils import Role, need_reference_policy
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.debug import marked_timer
from verl.utils.fs import local_mkdir_safe
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking


class OfflineCursorLoader:
    """Checkpointable cursor evidence for one deterministic round schedule."""

    def __init__(self, batch_count: int, *, schedule: str = "one_pass_prefix"):
        self.batch_count = int(batch_count)
        self.schedule = str(schedule)
        self.global_step = 0

    def __len__(self) -> int:
        return self.batch_count

    def set_global_step(self, global_step: int) -> None:
        self.global_step = int(global_step)

    def state_dict(self) -> dict[str, int | str]:
        return {
            "schedule": self.schedule,
            "completed_optimizer_steps": self.global_step,
            "next_batch_index": self.global_step,
            "batch_count": self.batch_count,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("schedule") != self.schedule or int(state["batch_count"]) != self.batch_count:
            raise ValueError("offline dataloader state does not match its deterministic round schedule")
        completed = int(state["completed_optimizer_steps"])
        if completed < 0 or completed > self.batch_count or int(state["next_batch_index"]) != completed:
            raise ValueError("offline dataloader checkpoint has an inconsistent next batch")
        self.global_step = completed


def assert_finite_metrics(metrics: dict[str, object]) -> None:
    """Reject a segment before logging or checkpointing any non-finite metric."""
    invalid: list[str] = []
    for name, value in metrics.items():
        if torch.is_tensor(value):
            if not torch.isfinite(value.detach()).all().item():
                invalid.append(name)
        elif isinstance(value, np.ndarray):
            if not np.isfinite(value).all():
                invalid.append(name)
        elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
            invalid.append(name)
    if invalid:
        raise FloatingPointError(f"offline training produced non-finite metrics: {sorted(invalid)}")


def _offline_ggpo_actor_dispatch_unit(config, *, group_size: int) -> int:
    """Return the response-count unit required by actor DP dispatch and mini-batching."""
    from verl.utils.config import get_trainer_process_on_nodes

    total_trainer_gpus = sum(get_trainer_process_on_nodes(config))
    legacy_sp = int(
        OmegaConf.select(config, "actor_rollout_ref.actor.ulysses_sequence_parallel_size", default=1)
    )
    engine_sp = int(
        OmegaConf.select(
            config,
            "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size",
            default=1,
        )
    )
    sequence_parallel_size = legacy_sp if legacy_sp > 1 else engine_sp
    if sequence_parallel_size <= 0 or total_trainer_gpus % sequence_parallel_size != 0:
        raise ValueError(
            "offline GGPO requires trainer GPU count to be divisible by actor Ulysses SP: "
            f"gpus={total_trainer_gpus}, sp={sequence_parallel_size}"
        )
    dp_size = total_trainer_gpus // sequence_parallel_size

    global_mini_batch_responses = int(config.actor_rollout_ref.actor.ppo_mini_batch_size) * group_size
    # engine_workers.py::TrainingWorker.train_mini_batch asserts mini_batch_size %
    # get_data_parallel_size() == 0 before deriving the per-DP-rank mini-batch.
    if global_mini_batch_responses <= 0 or global_mini_batch_responses % dp_size != 0:
        raise ValueError(
            "offline GGPO actor response mini-batch must be divisible by effective DP size: "
            f"responses={global_mini_batch_responses}, dp={dp_size}"
        )
    per_dp_mini_batch_responses = global_mini_batch_responses // dp_size
    # single_controller/base/decorator.py dispatches through chunk_tensordict,
    # whose assertion requires the dispatched batch length to be divisible by
    # dp_size; the actor iterator additionally asserts that each DP rank's
    # share is divisible by its per-rank mini-batch (tensordict_utils.make_iterator).
    # (final / dp_size) % per_dp_mini == 0  <=>  final % (dp_size * per_dp_mini) == 0,
    # i.e. the batch must hold whole global mini-batches.
    return dp_size * per_dp_mini_batch_responses


def prepare_offline_ggpo_batch(batch: DataProto, config) -> dict[str, float | int]:
    """Filter legacy GGPO data, or strictly verify a prefiltered full round."""
    loss_mode = OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode")
    contract = OmegaConf.select(config, "offline_data.contract", default=SEALED_OFFLINE_CONTRACT)
    mixed_filtered = str(contract) == SEALED_MIXED_FILTERED_OFFLINE_CONTRACT
    if loss_mode != "ggpo" and not mixed_filtered:
        return {}

    rewards = batch.batch["token_level_scores"].sum(dim=-1)
    batch.non_tensor_batch["acc"] = rewards.detach().cpu().numpy()
    raw_correctness = raw_correctness_tensor(batch.non_tensor_batch, expected_responses=len(batch))

    filter_config = OmegaConf.select(config, "algorithm.filter_groups", default=None)
    if not mixed_filtered:
        if not filter_config or not bool(filter_config.get("enable", False)):
            return {}
        if str(filter_config.get("metric")) != "raw_correctness":
            raise ValueError("offline GGPO supports only DAPO filtering by raw_correctness")

    group_size = int(batch.meta_info["responses_per_group"])
    if raw_correctness.numel() % group_size != 0:
        raise ValueError("offline GGPO responses do not form complete filter groups")
    total_groups = raw_correctness.numel() // group_size
    response_indices, filter_stats = select_mixed_reward_group_indices(
        raw_correctness,
        group_size=group_size,
        max_groups=total_groups,
    )
    kept_groups = filter_stats["selected_groups"]
    if mixed_filtered and kept_groups != total_groups:
        raise RuntimeError(
            "sealed_mixedfilt strict guard found a non-mixed raw-correctness group: "
            f"mixed={kept_groups}, total={total_groups}"
        )
    if kept_groups == 0:
        raise RuntimeError("offline GGPO batch has zero mixed raw-correctness groups after filtering")

    dispatch_unit = _offline_ggpo_actor_dispatch_unit(config, group_size=group_size)
    response_multiple = math.lcm(group_size, dispatch_unit)
    final_responses = (response_indices.numel() // response_multiple) * response_multiple
    if final_responses == 0:
        raise RuntimeError(
            "offline GGPO batch has zero responses after truncating mixed groups to the actor dispatch unit"
        )
    truncated_groups = kept_groups - final_responses // group_size
    if mixed_filtered:
        if final_responses != len(batch) or truncated_groups != 0:
            raise RuntimeError(
                "sealed_mixedfilt strict guard forbids actor-dispatch truncation: "
                f"responses={len(batch)}, dispatch_responses={final_responses}, "
                f"truncated_groups={truncated_groups}"
            )
        return {
            "offline/filter_kept_groups": kept_groups,
            "offline/filter_dropped_groups": 0,
            "offline/filter_kept_fraction": 1.0,
            "offline/filter_truncated_groups": 0,
            "offline/filter_final_responses": final_responses,
        }
    response_indices = response_indices[:final_responses]
    filtered = batch.select_idxs(response_indices)
    batch.batch = filtered.batch
    batch.non_tensor_batch = filtered.non_tensor_batch
    batch.meta_info = filtered.meta_info
    dropped_groups = total_groups - kept_groups
    return {
        "offline/filter_kept_groups": kept_groups,
        "offline/filter_dropped_groups": dropped_groups,
        "offline/filter_kept_fraction": kept_groups / total_groups,
        "offline/filter_truncated_groups": truncated_groups,
        "offline/filter_final_responses": final_responses,
    }


def _validate_sealed_offline_training_config(config, *, allowed_modes: tuple[str, ...] = ("train",)) -> None:
    offline = config.offline_data
    validate_reference_contract(config, training=True)
    contract = str(offline.get("contract", SEALED_OFFLINE_CONTRACT))
    mixed_filtered = contract == SEALED_MIXED_FILTERED_OFFLINE_CONTRACT
    if str(offline.mode) not in allowed_modes:
        raise ValueError(f"offline trainer requires offline_data.mode in {allowed_modes}, got {offline.mode!r}")
    if not bool(offline.fully_offline):
        raise ValueError("offline_data.fully_offline must be true")
    if str(offline.old_log_probs_source) != "fixed_behavior":
        raise ValueError("offline training requires fixed behavior old_log_probs for immutable diagnostics")
    if mixed_filtered:
        filtered_manifest_round_count(config)
        expected_rounds = filtered_training_round_count(config)
    else:
        expected_rounds = 100
    if not mixed_filtered and int(offline.expected_batches) != 286:
        raise ValueError("this experiment is approved for exactly 286 fixed global batches")
    if int(offline.prompt_groups_per_batch) != 256 or int(offline.responses_per_group) != 8:
        raise ValueError("offline experiment requires 256 prompt groups x 8 responses per global batch")
    if int(config.actor_rollout_ref.rollout.n) != int(offline.responses_per_group):
        raise ValueError("actor rollout_n metadata must match offline responses_per_group")
    if int(config.trainer.total_training_steps) != expected_rounds:
        if mixed_filtered:
            raise ValueError("filtered offline training steps must equal its configured total round count")
        raise ValueError("this experiment is approved for exactly 100 optimizer steps")
    if int(offline.selection.count) != expected_rounds:
        if mixed_filtered:
            raise ValueError("filtered offline selection count must equal its configured total round count")
        raise ValueError("offline training requires exactly the approved first 100 batches")
    mode = str(offline.mode)
    if mode == "train" and bool(offline.smoke) and config.trainer.resume_mode != "disable":
        raise ValueError("offline smoke must start fresh with resume_mode=disable")
    if mode == "train" and not bool(offline.smoke) and config.trainer.resume_mode != "resume_path":
        raise ValueError("offline training must resume the separately evaluated step-0 or prior segment checkpoint")
    if mode == "initialize" and (bool(offline.smoke) or config.trainer.resume_mode != "disable"):
        raise ValueError("offline step-0 initialization requires resume_mode=disable")
    if bool(config.trainer.get("val_before_train", False)) or int(config.trainer.test_freq) > 0:
        raise ValueError("Math500 inference must run in separate checkpoint-bound jobs, never in offline training")
    if config.critic.enable:
        raise ValueError("offline policy optimization does not use a critic")
    loss_mode = str(config.actor_rollout_ref.actor.policy_loss.loss_mode)
    if loss_mode != "ggpo" and config.algorithm.adv_estimator not in (AdvantageEstimator.GRPO, "grpo"):
        raise ValueError("offline AGRO/ASymPO uses the GRPO outcome advantage family")
    configured_offset = int(offline.wandb_step_offset)
    if configured_offset < 0:
        raise ValueError("offline W&B History offset cannot be negative")
    environment_offset = int(os.environ.get("WANDB_STEP_OFFSET", "0"))
    if environment_offset != configured_offset:
        raise ValueError("WANDB_STEP_OFFSET must match offline_data.wandb_step_offset")


def _validate_parameterized_offline_training_config(
    config,
    *,
    allowed_modes: tuple[str, ...] = ("train",),
) -> None:
    offline = config.offline_data
    validate_parameterized_contract(config, training=True)
    if str(offline.mode) not in allowed_modes:
        raise ValueError(f"offline trainer requires offline_data.mode in {allowed_modes}, got {offline.mode!r}")
    if not bool(offline.fully_offline):
        raise ValueError("offline_data.fully_offline must be true")
    if str(offline.old_log_probs_source) != "fixed_behavior":
        raise ValueError("offline training requires fixed behavior old_log_probs")
    if config.critic.enable:
        raise ValueError("parameterized offline training does not run a critic")
    if bool(config.trainer.get("val_before_train", False)) or int(config.trainer.test_freq) > 0:
        raise ValueError("offline validation must run in a separate checkpoint-bound job")
    mode = str(offline.mode)
    if mode == "train" and bool(offline.smoke) and config.trainer.resume_mode != "disable":
        raise ValueError("offline smoke must start fresh with resume_mode=disable")
    if mode == "initialize" and (bool(offline.smoke) or config.trainer.resume_mode != "disable"):
        raise ValueError("offline step-0 initialization requires resume_mode=disable")
    configured_offset = int(offline.wandb_step_offset)
    if configured_offset < 0:
        raise ValueError("offline W&B History offset cannot be negative")
    if int(os.environ.get("WANDB_STEP_OFFSET", "0")) != configured_offset:
        raise ValueError("WANDB_STEP_OFFSET must match offline_data.wandb_step_offset")


def validate_offline_training_config(config, *, allowed_modes: tuple[str, ...] = ("train",)) -> None:
    """Dispatch to a sealed provenance contract or the separate configurable one."""
    contract = str(config.offline_data.get("contract", SEALED_OFFLINE_CONTRACT))
    if contract in {SEALED_OFFLINE_CONTRACT, SEALED_MIXED_FILTERED_OFFLINE_CONTRACT}:
        _validate_sealed_offline_training_config(config, allowed_modes=allowed_modes)
    elif contract == PARAMETERIZED_OFFLINE_CONTRACT:
        _validate_parameterized_offline_training_config(config, allowed_modes=allowed_modes)
    else:
        raise ValueError(
            f"offline_data.contract must be {SEALED_OFFLINE_CONTRACT!r}, "
            f"{SEALED_MIXED_FILTERED_OFFLINE_CONTRACT!r}, or {PARAMETERIZED_OFFLINE_CONTRACT!r}, "
            f"got {contract!r}"
        )


class OfflineRLTrainer(RayPPOTrainer):
    """Ray actor/reference orchestration with no rollout-side services."""

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        offline = self.config.offline_data
        self.train_dataset = OfflineBatchDataset(
            offline.dataset_root,
            expected_batches=int(offline.expected_batches),
            prompt_groups_per_batch=int(offline.prompt_groups_per_batch),
            responses_per_group=int(offline.responses_per_group),
            selection_method=str(offline.selection.method),
            selection_start=int(offline.selection.start),
            selection_count=int(offline.selection.count),
            expected_config=self.config,
        )
        self.val_dataset = None
        cursor_schedule = (
            "seeded_pool_cycle"
            if str(offline.selection.method) == "seeded_pool_cycle"
            else "one_pass_prefix"
        )
        self.train_dataloader = OfflineCursorLoader(len(self.train_dataset), schedule=cursor_schedule)
        self.val_dataloader = None
        self.total_training_steps = int(self.config.trainer.total_training_steps)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps

    def init_workers(self):
        """Create the actor and any loss-required reference; omit rollout/reward services."""
        validate_offline_training_config(self.config, allowed_modes=("train", "initialize"))
        self.resource_pool_manager.create_resource_pool()
        colocated_reference = need_reference_policy(self.config)
        actor_role = Role.ActorRolloutRef if colocated_reference else Role.Actor
        resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        worker_role = "actor_ref" if colocated_reference else "actor"
        actor_ref_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config.actor_rollout_ref,
            distillation_config=None,
            role=worker_role,
        )
        worker_dict_cls = create_colocated_worker_cls(class_dict={worker_role: actor_ref_cls})
        kwargs = {"device_name": self.device_name}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        worker_group = self.ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls,
            **kwargs,
        )
        all_workers = worker_group.spawn(prefix_set={worker_role})
        self.actor_rollout_wg = all_workers[worker_role]
        self.ref_policy_wg = self.actor_rollout_wg if colocated_reference else None
        self.actor_rollout_wg.init_model()
        self.distillation_config = None
        self.teacher_model_manager = None
        self.reward_loop_manager = None
        self.llm_server_manager = None
        self.async_rollout_manager = None
        self.checkpoint_manager = None

    def _offline_state_path(self, global_step_folder: str | Path) -> Path:
        return Path(global_step_folder) / "offline_state.json"

    def _ggpo_theory_policy_identity(self, batch: DataProto):
        """Bind offline GGPO to its fixed step-zero behavior law."""
        training_version = max(int(self.global_steps) - 1, 0)
        return {
            "actor_update_step": int(self.global_steps),
            "training_policy_version": training_version,
            "behavior_policy_versions": torch.zeros(len(batch), dtype=torch.long),
            "behavior_version_stratification": True,
        }

    def _save_checkpoint(self):
        self.train_dataloader.set_global_step(self.global_steps)
        checkpoint_root = Path(self.config.trainer.default_local_dir)
        folder = checkpoint_root / f"global_step_{self.global_steps}"
        temporary_folder = checkpoint_root / f".global_step_{self.global_steps}.{os.getpid()}.tmp"
        if self.config.trainer.default_hdfs_dir is not None:
            raise ValueError("offline interactive checkpoints must use the bound local checkpoint root")
        if bool(self.config.actor_rollout_ref.actor.checkpoint.async_save):
            raise ValueError("offline interactive checkpoints require synchronous actor saves")
        if folder.exists():
            raise FileExistsError(f"offline checkpoint boundary already exists: {folder}")
        if temporary_folder.exists():
            raise FileExistsError(f"offline checkpoint staging directory already exists: {temporary_folder}")
        wandb_run_id = os.environ.get("WANDB_RUN_ID")
        if not wandb_run_id:
            raise ValueError("offline checkpoints require a nonempty immutable WANDB_RUN_ID")
        reference_version_after_step = self._ggpo_reference_version_after_step()
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep")
        if (
            self._ggpo_moving_reference_enabled
            and self._ggpo_reference_version > 0
            and reference_version_after_step == self._ggpo_reference_version
        ):
            # Denser evaluation checkpoints must not evict the actor snapshot
            # that still backs the active moving reference.
            max_actor_ckpt_to_keep = None

        actor_folder = temporary_folder / "actor"
        self.actor_rollout_wg.save_checkpoint(
            str(actor_folder),
            hdfs_path=None,
            global_step=self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )
        local_mkdir_safe(str(temporary_folder))

        dataloader_path = temporary_folder / "data.pt"
        temporary_dataloader = temporary_folder / f".data.{os.getpid()}.tmp"
        with open(temporary_dataloader, "wb") as stream:
            torch.save(self.train_dataloader.state_dict(), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_dataloader, dataloader_path)

        state = {
            "schema_version": 2,
            "completed_optimizer_steps": self.global_steps,
            "next_batch_index": self.global_steps,
            "dataset_id": str(self.config.offline_data.dataset_id),
            "implementation_commit": str(self.config.offline_data.implementation_commit),
            "loss_mode": str(self.config.actor_rollout_ref.actor.policy_loss.loss_mode),
            "experiment_name": str(self.config.trainer.experiment_name),
            "wandb_run_id": wandb_run_id,
            "wandb_step_offset": int(self.config.offline_data.wandb_step_offset),
            "schedule": self.train_dataloader.schedule,
        }
        if reference_version_after_step is not None:
            state["ggpo_reference_version_after_step"] = reference_version_after_step
        temporary = temporary_folder / f".offline_state.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self._offline_state_path(temporary_folder))
        os.replace(temporary_folder, folder)

        latest_path = checkpoint_root / "latest_checkpointed_iteration.txt"
        temporary_latest = latest_path.with_name(f".{latest_path.name}.{os.getpid()}.tmp")
        with open(temporary_latest, "w", encoding="utf-8") as stream:
            stream.write(f"{self.global_steps}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_latest, latest_path)
        if reference_version_after_step is not None and reference_version_after_step != self._ggpo_reference_version:
            self._update_ggpo_reference(str(folder), reference_version_after_step)

    def _resolve_resume_folder(self) -> str | None:
        mode = self.config.trainer.resume_mode
        if mode == "disable":
            existing = find_latest_ckpt_path(self.config.trainer.default_local_dir)
            if existing is not None:
                raise ValueError(f"fresh offline segment refuses existing checkpoint lineage: {existing}")
            return None
        if mode == "resume_path":
            folder = self.config.trainer.resume_from_path
            if not isinstance(folder, str):
                raise ValueError("resume_path must name one exact global_step checkpoint")
            resolved = Path(folder).resolve()
            step_text = resolved.name.removeprefix("global_step_")
            if not step_text.isdigit() or resolved.name != f"global_step_{int(step_text)}":
                raise ValueError("resume_path must end in the canonical global_step_<step> directory name")
            expected_parent = Path(self.config.trainer.default_local_dir).resolve()
            if resolved.parent != expected_parent:
                raise ValueError("resume_path must be an exact child of the configured checkpoint lineage")
            return str(resolved)
        raise ValueError(f"unsupported resume_mode {mode!r}")

    def _load_checkpoint(self):
        folder = self._resolve_resume_folder()
        if folder is None:
            self.global_steps = 0
            self.train_dataloader.set_global_step(0)
            return
        self.global_steps = int(str(folder).rsplit("global_step_", 1)[1])
        contract = str(self.config.offline_data.get("contract", SEALED_OFFLINE_CONTRACT))
        resume_frequency = (
            int(self.config.trainer.save_freq)
            if contract == SEALED_MIXED_FILTERED_OFFLINE_CONTRACT
            else int(self.config.offline_data.segment_optimizer_steps)
        )
        if self.global_steps < 0 or self.global_steps % resume_frequency != 0:
            raise ValueError("offline resume checkpoint is not an accepted persisted boundary")
        with open(self._offline_state_path(folder), encoding="utf-8") as stream:
            state = json.load(stream)
        expected_next = self.global_steps
        if int(state.get("schema_version", -1)) != 2:
            raise ValueError("offline checkpoint state has an unsupported schema")
        if int(state["completed_optimizer_steps"]) != self.global_steps:
            raise ValueError("offline checkpoint step does not match its directory")
        validate_checkpoint_dataset_identity(
            state,
            dataset_id=str(self.config.offline_data.dataset_id),
        )
        implementation_commit = str(self.config.offline_data.implementation_commit)
        if implementation_commit and state.get("implementation_commit") != implementation_commit:
            raise ValueError("offline checkpoint implementation commit does not match the training configuration")
        if state.get("loss_mode") != str(self.config.actor_rollout_ref.actor.policy_loss.loss_mode):
            raise ValueError("offline checkpoint objective does not match the training configuration")
        if state.get("experiment_name") != str(self.config.trainer.experiment_name):
            raise ValueError("offline checkpoint display identity does not match the training configuration")
        if self._ggpo_moving_reference_enabled:
            reference_version = state.get("ggpo_reference_version_after_step")
            if (
                not isinstance(reference_version, int)
                or isinstance(reference_version, bool)
                or reference_version < 0
                or reference_version > self.global_steps
            ):
                raise ValueError("offline GGPO checkpoint has an invalid moving-reference version")
            self._restored_ggpo_reference_version = reference_version
        if (
            int(state["next_batch_index"]) != expected_next
            or state["schedule"] != self.train_dataloader.schedule
        ):
            raise ValueError("offline checkpoint cursor does not match the deterministic round schedule")
        segment_steps = int(self.config.offline_data.segment_optimizer_steps)
        saved_offset = int(state.get("wandb_step_offset", -1))
        expected_saved_offset = 0 if self.global_steps == 0 else (self.global_steps - 1) // segment_steps
        current_offset = int(self.config.offline_data.wandb_step_offset)
        expected_current_offset = self.global_steps // segment_steps
        if saved_offset != expected_saved_offset or current_offset != expected_current_offset:
            raise ValueError("offline checkpoint and current segment have an inconsistent W&B History offset")
        current_wandb_id = os.environ.get("WANDB_RUN_ID")
        if not current_wandb_id or state.get("wandb_run_id") != current_wandb_id:
            raise ValueError("offline checkpoint W&B identity does not match WANDB_RUN_ID")
        self.actor_rollout_wg.load_checkpoint(
            os.path.join(folder, "actor"),
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )
        dataloader_path = os.path.join(folder, "data.pt")
        loader_state = torch.load(dataloader_path, map_location="cpu", weights_only=False)
        self.train_dataloader.load_state_dict(loader_state)
        if self.train_dataloader.global_step != self.global_steps:
            raise ValueError("offline checkpoint dataloader and optimizer steps disagree")
        if self._ggpo_moving_reference_enabled:
            self._restore_ggpo_reference(folder)

    @staticmethod
    def _offline_reward_metrics(batch: DataProto) -> dict[str, float]:
        """Summarize immutable rewards without an extra policy forward pass."""
        rewards = batch.batch["token_level_scores"].sum(dim=-1)
        group_size = int(batch.meta_info["responses_per_group"])
        group_rewards = rewards.reshape(-1, group_size)
        informative = (group_rewards.sum(-1) > 0) & (group_rewards.sum(-1) < group_size)
        return {
            "offline/reward_mean": float(rewards.mean().item()),
            "offline/informative_group_fraction": float(informative.float().mean().item()),
            "offline/all_zero_group_fraction": float((group_rewards.sum(-1) == 0).float().mean().item()),
            "offline/all_one_group_fraction": float((group_rewards.sum(-1) == group_size).float().mean().item()),
        }

    def fit(self):
        validate_offline_training_config(self.config)
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        self._load_checkpoint()
        start_step = self.global_steps
        segment_steps = int(self.config.offline_data.segment_optimizer_steps)
        contract = str(self.config.offline_data.get("contract", SEALED_OFFLINE_CONTRACT))
        if contract == SEALED_MIXED_FILTERED_OFFLINE_CONTRACT:
            next_segment_boundary = (start_step // segment_steps + 1) * segment_steps
            end_step = min(next_segment_boundary, self.total_training_steps)
        else:
            end_step = min(start_step + segment_steps, self.total_training_steps)
        if start_step >= self.total_training_steps:
            logger.finish()
            raise ValueError(f"checkpoint step {start_step} has already reached the target")
        if end_step - start_step != segment_steps:
            if contract != SEALED_MIXED_FILTERED_OFFLINE_CONTRACT or end_step != self.total_training_steps:
                raise ValueError("offline target must be exactly divisible into 20-step interactive segments")

        progress = tqdm(total=end_step - start_step, desc=f"Offline steps {start_step + 1}-{end_step}")
        try:
            for step in range(start_step + 1, end_step + 1):
                self.global_steps = step
                batch_index = step - 1
                dataset_pass, pool_round_index = self.train_dataset.schedule_entry(batch_index)
                timing_raw: dict[str, float] = {}
                metrics: dict[str, float] = {
                    "offline/batch_index": batch_index,
                    "offline/dataset_pass": dataset_pass,
                    "offline/pool_round_index": pool_round_index,
                    "offline/pool_round_count": self.train_dataset.pool_round_count,
                    "offline/dataset_batch_count": len(self.train_dataset),
                }
                ggpo_reference_version_used = None
                with marked_timer("step", timing_raw):
                    with marked_timer("offline_load", timing_raw, color="yellow"):
                        batch = self.train_dataset.load_batch(batch_index)
                    batch.meta_info.update(
                        {
                            "temperature": self.config.actor_rollout_ref.rollout.temperature,
                            "images_seqlens": [],
                            "responses_per_group": int(self.config.offline_data.responses_per_group),
                        }
                    )

                    metrics.update(self._offline_reward_metrics(batch))
                    metrics.update(prepare_offline_ggpo_batch(batch, self.config))
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()
                    frozen_old_log_probs = batch.batch["old_log_probs"].clone()

                    if self.ref_policy_wg is not None:
                        if self._ggpo_moving_reference_enabled:
                            ggpo_reference_version_used = int(self._ggpo_reference_version)
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            batch = batch.union(self._compute_ref_log_prob(batch))

                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                    if not maybe_apply_group_reference_advantage_bypass(batch, self.config):
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=int(self.config.offline_data.responses_per_group),
                            norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = self._update_actor(batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                    if self._ggpo_moving_reference_enabled:
                        if ggpo_reference_version_used is None:
                            raise RuntimeError("offline GGPO moving reference did not record the learner version")
                        metrics["actor/ggpo_reference_version"] = ggpo_reference_version_used
                    if not torch.equal(batch.batch["old_log_probs"], frozen_old_log_probs):
                        raise RuntimeError("actor update mutated fixed behavior old_log_probs")

                metrics.update(
                    {
                        "training/global_step": step,
                        "training/epoch": dataset_pass,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch,
                        timing_raw=timing_raw,
                        n_gpus=self.resource_pool_manager.get_n_gpus(),
                    )
                )
                assert_finite_metrics(metrics)
                logger.log(data=metrics, step=step)
                if bool(self.config.offline_data.smoke):
                    pass
                else:
                    save_freq = int(self.config.trainer.save_freq)
                    save_final = bool(self.config.trainer.get("save_final_checkpoint", False))
                    if step % save_freq == 0 or (step == self.total_training_steps and save_final):
                        checkpoint_timing: dict[str, float] = {}
                        with marked_timer("save_checkpoint", checkpoint_timing, color="green"):
                            self._save_checkpoint()
                progress.update(1)

            self.train_dataloader.set_global_step(self.global_steps)
            if bool(self.config.offline_data.smoke):
                checkpoint_root = Path(self.config.trainer.default_local_dir)
                entries = list(checkpoint_root.iterdir()) if checkpoint_root.exists() else []
                if self._ggpo_theory_run_receipt is not None:
                    receipt_dir = checkpoint_root / "ggpo_theory"
                    receipt = receipt_dir / "run_receipt.json"
                    forbidden = [entry for entry in entries if entry != receipt_dir]
                    if forbidden or not receipt.is_file() or any(entry != receipt for entry in receipt_dir.iterdir()):
                        raise RuntimeError("offline GGPO smoke created forbidden checkpoint artifacts")
                elif entries:
                    raise RuntimeError("offline smoke created forbidden checkpoint artifacts")
            pprint(
                {
                    "offline_segment_accepted": True,
                    "start_step": start_step,
                    "end_step": end_step,
                    "next_batch_index": end_step,
                    "smoke": bool(self.config.offline_data.smoke),
                }
            )
        finally:
            progress.close()
            logger.finish()
            self._shutdown_dump_executor()

    def initialize_checkpoint(self):
        """Create the immutable step-0 resume point used by precursor Math500 evaluation."""
        validate_offline_training_config(self.config, allowed_modes=("initialize",))
        if self.config.trainer.resume_mode != "disable":
            raise ValueError("step-0 checkpoint initialization must use resume_mode=disable")
        if self._resolve_resume_folder() is not None:
            raise AssertionError("fresh step-0 initialization unexpectedly resolved a checkpoint")
        self.global_steps = 0
        self.train_dataloader.set_global_step(0)
        try:
            self._save_checkpoint()
            pprint(
                {
                    "offline_step_zero_checkpoint_accepted": True,
                    "checkpoint": str(Path(self.config.trainer.default_local_dir) / "global_step_0"),
                }
            )
        finally:
            self._shutdown_dump_executor()
