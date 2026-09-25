# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import asyncio
import logging
import math
import os
import shutil
import time
from collections import Counter
from datetime import datetime
from typing import Any

import ray
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.fully_async_policy.actor_update_padding import (
    PHYSICAL_MINI_BATCH_SIZE_META,
    pad_fully_async_actor_update,
    select_fully_async_real_samples,
)
from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    assemble_batch_from_rollout_samples,
    compute_lagged_log_prob_metrics,
    compute_policy_version_metrics,
    declare_fully_async_step_metrics,
    validate_rollout_group_admission,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.controlled_stop import AdaptiveUpdateBoundaryStop, controlled_stop_reached
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ggpo import GGPO_SURROGATE_CURRENT_LOG_PROBS
from verl.trainer.ppo.ggpo_theory import GGPO_THEORY_CURRENT_LOG_PROBS, seal_ggpo_theory_run_receipt
from verl.trainer.ppo.mixed_reward_filter import (
    raw_correctness_tensor,
    select_mixed_reward_group_indices,
)
from verl.trainer.ppo.ray_trainer import (
    ResourcePoolManager,
    is_ggpo_policy_loss,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.rollout_training_mismatch import MISMATCH_AGGREGATION_META
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import (
    get_fully_async_prompts_per_update,
    get_trainer_process_on_nodes,
    omega_conf_to_dataclass,
)
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

logger = logging.getLogger(__name__)


class TrainingStopException(Exception):
    """Exception raised to signal training should stop"""

    pass


def _resolve_terminal_policy_version(current_param_version: int, local_trigger_step: int) -> tuple[int, int, bool]:
    """Publish partial learner work under a real final policy version."""
    if local_trigger_step > 1:
        return current_param_version + 1, 1, True
    return current_param_version, local_trigger_step, False


def _should_save_fully_async_checkpoint(
    *,
    force: bool,
    current_param_version: int,
    last_ckpt_version: int,
    save_freq: int,
    esi_close_to_expiration: bool,
    save_final_checkpoint: bool = True,
) -> bool:
    """Return whether the current learner state must be checkpointed."""
    # Smoke runs can explicitly suppress the terminal force-save. Periodic
    # checkpointing remains governed independently by ``save_freq`` so a smoke
    # can disable both with save_freq=-1 and save_final_checkpoint=false.
    if force and not save_final_checkpoint:
        return False
    # A terminal force-save must bypass the periodic cadence, but it must not
    # rewrite an already-saved version.  FSDP checkpoint retention registers
    # paths, not versions; registering the same path twice can therefore
    # delete the newly written checkpoint as if it were the previous one.
    if current_param_version == last_ckpt_version:
        return False
    return force or (save_freq > 0 and (current_param_version % save_freq == 0 or esi_close_to_expiration))


def _require_optimizer_progress(completed_optimizer_steps: int) -> None:
    """Fail a smoke or training run that never completed an actor update."""
    if completed_optimizer_steps < 1:
        raise RuntimeError("Fully async training terminated without completing an actor optimizer step.")


def _optimizer_step_limit_reached(completed_optimizer_steps: int, max_optimizer_steps: int | None) -> bool:
    """Return whether a bounded run completed its requested actor updates."""
    return max_optimizer_steps is not None and completed_optimizer_steps >= max_optimizer_steps


def _assert_bounded_smoke_update_health(trainer: Any) -> None:
    """Fail closed at each update boundary for declared no-checkpoint smokes."""
    config = getattr(trainer, "config", None)
    if config is None or OmegaConf.select(config, "async_training.runtime_root", default=None) is None:
        return
    if OmegaConf.select(config, "trainer.save_freq", default=None) != -1 or bool(
        OmegaConf.select(config, "trainer.save_final_checkpoint", default=True)
    ):
        return
    metrics = trainer.metrics
    nonfinite = [
        key
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value))
    ]
    if nonfinite:
        raise FloatingPointError(f"Smoke update emitted non-finite metrics: {sorted(nonfinite)}")
    required = {"actor/optimizer_applied", "actor/grad_norm", "actor/lr"}
    missing = required.difference(metrics)
    if missing:
        raise RuntimeError(f"Smoke update is missing optimizer observability: {sorted(missing)}")
    if float(metrics["actor/optimizer_applied"]) != 1.0:
        raise RuntimeError("Smoke actor optimizer did not apply an update")
    if float(metrics["actor/grad_norm"]) <= 0.0:
        raise RuntimeError("Smoke actor gradient norm is not positive")
    if int(getattr(trainer, "completed_optimizer_steps", 0)) < 1 or int(
        getattr(trainer, "current_param_version", 0)
    ) < 1:
        raise RuntimeError("Smoke actor update did not advance optimizer and policy versions")
    for key in ("fully_async/version/future_rollout_count", "fully_async/version/over_age_count"):
        if float(metrics.get(key, 0.0)) != 0.0:
            raise RuntimeError(f"Smoke consumed an invalid token policy age: {key}={metrics[key]}")
    metrics["fully_async/health/all_update_gates_passed"] = 1.0


def _wall_clock_stop_reached(
    started_at: float,
    stop_after_seconds: float | None,
    deadline_unix_seconds: float | None = None,
) -> bool:
    """Check the controlled deadline only between complete optimizer updates."""
    return controlled_stop_reached(
        started_at=started_at,
        stop_after_seconds=stop_after_seconds,
        deadline_unix_seconds=deadline_unix_seconds,
    )


async def _run_fully_async_step_loop(trainer: Any) -> str:
    """Run learner steps and stop rollout production at the configured bound."""
    started_at = time.monotonic()
    adaptive_stop_config = getattr(trainer, "adaptive_controlled_stop", None)
    adaptive_stop = None
    if adaptive_stop_config is not None:
        adaptive_stop = AdaptiveUpdateBoundaryStop(
            hard_deadline_unix_seconds=adaptive_stop_config["hard_deadline_unix_seconds"],
            initial_optimizer_step_seconds=adaptive_stop_config["initial_optimizer_step_seconds"],
            finalization_seconds=adaptive_stop_config["finalization_seconds"],
            sync_steps=trainer.trigger_parameter_sync_step,
            safety_factor=adaptive_stop_config.get("safety_factor", 1.25),
            history_size=adaptive_stop_config.get("history_size", 8),
        )
    while True:
        adaptive_stop_reached = adaptive_stop is not None and adaptive_stop.should_stop_before_next_update(
            local_trigger_step=trainer.local_trigger_step,
        )
        legacy_stop_reached = adaptive_stop is None and _wall_clock_stop_reached(
            started_at,
            getattr(trainer, "controlled_stop_after_seconds", None),
            getattr(trainer, "controlled_stop_deadline_unix_seconds", None),
        )
        if adaptive_stop_reached or legacy_stop_reached:
            adaptive_detail = ""
            if adaptive_stop is not None:
                adaptive_detail = (
                    f" predicted_sync_window_seconds="
                    f"{adaptive_stop.optimizer_step_estimate_seconds * adaptive_stop.sync_steps:.3f}"
                    f" required_headroom_seconds={adaptive_stop.required_headroom_seconds:.3f}"
                    f" sync_steps={adaptive_stop.sync_steps}."
                )
            print(
                "[FullyAsyncTrainer] Reached update-boundary controlled wall-clock stop after "
                f"{trainer.completed_optimizer_steps} completed optimizer updates. "
                f"Stopping rollout production.{adaptive_detail}"
            )
            await trainer.rollouter.request_stop.remote()
            return "controlled_wall_clock"
        optimizer_step_started_at = time.monotonic()
        try:
            await trainer.fit_step()
        except TrainingStopException as exc:
            if (
                trainer.max_optimizer_steps is not None
                and trainer.completed_optimizer_steps != trainer.max_optimizer_steps
            ):
                raise RuntimeError(
                    "Fully async rollout queue terminated after "
                    f"{trainer.completed_optimizer_steps} actor optimizer steps; "
                    f"expected exactly {trainer.max_optimizer_steps}."
                ) from exc
            print("[FullyAsyncTrainer] Training stopped by queue termination signal")
            return "queue_terminated"
        if adaptive_stop is not None:
            adaptive_stop.observe_optimizer_step(time.monotonic() - optimizer_step_started_at)
        _assert_bounded_smoke_update_health(trainer)
        if trainer.completed_optimizer_steps >= 2 and getattr(trainer, "consumed_cross_version_response", False):
            print(
                "[FullyAsyncTrainer] Dynamic partial-rollout gate passed after "
                f"{trainer.completed_optimizer_steps} complete updates; stopping at update boundary."
            )
            await trainer.rollouter.request_stop.remote()
            return "partial_rollout_verified"
        if _optimizer_step_limit_reached(
            trainer.completed_optimizer_steps,
            trainer.max_optimizer_steps,
        ):
            print(
                "[FullyAsyncTrainer] Reached actor optimizer-step limit: "
                f"{trainer.completed_optimizer_steps}/{trainer.max_optimizer_steps}. "
                "Stopping rollout production. partial_rollout_dynamic_coverage="
                f"{getattr(trainer, 'consumed_cross_version_response', False)}."
            )
            await trainer.rollouter.request_stop.remote()
            return "max_optimizer_steps"


def _finish_fully_async_run(trainer: Any) -> None:
    """Flush terminal metrics and apply the configured checkpoint policy."""
    trainer._fit_log_aggregated_training_metrics(step=trainer.current_param_version)
    trainer._fit_save_checkpoint(force=True)
    trainer.logger.finish()


def _log_fully_async_version_metrics(trainer: Any, *, step: int | None = None) -> None:
    """Commit one merged metrics record for an explicit policy version."""

    aggregated_metrics = trainer.metrics_aggregator.get_aggregated_metrics()
    metrics = {**aggregated_metrics, **trainer._pending_version_metrics}
    if metrics:
        trainer.logger.log(
            data=metrics,
            step=trainer.current_param_version if step is None else step,
        )
    trainer.metrics_aggregator.reset()
    trainer._pending_version_metrics.clear()


def _finalize_fully_async_step_metrics(trainer: Any, *, weights_updated: bool) -> None:
    """Record a completed learner step before logging its policy version."""
    trainer._fit_postprocess_step()
    if weights_updated:
        trainer._fit_log_aggregated_training_metrics()


async def _finalize_fully_async_policy(trainer: Any, *, has_partial_update: bool) -> None:
    """Publish a partial policy once without inventing a terminal validation step."""
    if has_partial_update:
        await trainer._fit_update_weights()


@ray.remote(num_cpus=10)
class FullyAsyncTrainer(SeparateRayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        device_name=None,
    ):
        # ==================== RayPPOTrainer config ====================

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.config = config
        self._ggpo_theory_run_receipt = (
            seal_ggpo_theory_run_receipt(self.config) if is_ggpo_policy_loss(self.config) else None
        )

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        # distillation config needed by _update_actor in ray_trainer.py
        from verl.trainer.distillation.losses import is_distillation_enabled

        if is_distillation_enabled(self.config.get("distillation")):
            self.distillation_config = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.distillation_config = None

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if OmegaConf.select(self.config, "algorithm.use_kl_in_reward", default=False):
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        # ==================== SeparateRayPPOTrainer config ====================
        self.global_steps = 0
        self.epoch = 0
        self._init_dump_executor()
        self.validation_generations_logger = None
        self.max_steps_duration = 0
        self.progress_bar = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # Metrics emitted outside the actor update (weight sync and validation)
        # are buffered here and committed together with the matching training
        # metrics. W&B advances an explicit step after every log call, so two
        # separate commits at the same policy version would drop the latter.
        self._pending_version_metrics = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # ==================== fully async config ====================

        self.message_queue_client = None

        # Statistics
        self.local_trigger_step = 1
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.completed_optimizer_steps = 0
        self.max_optimizer_steps = config.trainer.get("max_optimizer_steps")
        self.controlled_stop_after_seconds = config.trainer.get("controlled_stop_after_seconds")
        self.controlled_stop_deadline_unix_seconds = config.trainer.get("controlled_stop_deadline_unix_seconds")
        self.adaptive_controlled_stop = config.trainer.get("adaptive_controlled_stop")
        self.seen_group_ids: set[Any] = set()
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0
        self.train_role = Role.ActorRollout if config.async_training.use_trainer_do_validate else Role.Actor

        # Each AReaL queue item is one atomic prompt group, not one response.
        self.require_batches = config.async_training.require_batches
        self.required_samples = get_fully_async_prompts_per_update(config)
        total_gpus = sum(get_trainer_process_on_nodes(config)) + config.rollout.nnodes * config.rollout.n_gpus_per_node
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)
        self._step_metric_aggregation = {}

        # Reference to rollouter for parameter synchronization
        self.rollouter = None
        self.rollout_capabilities = None
        self.checkpoint_manager = None
        self.group_admission_counts = Counter()
        self.consumed_cross_version_response = False

        # Hybrid checkpoint manager for trainer-side validation (use_trainer_do_validate)
        # Uses naive backend to sync weights from trainer to hybrid rollout replicas.
        # Initialized in _setup_hybrid_checkpoint_manager_and_sleep() via set_rollouter().
        self.hybrid_checkpoint_manager = None

    async def _setup_checkpoint_manager(self):
        """Setup checkpoint manager after rollouter is initialized"""
        replicas = await self.rollouter.get_replicas.remote()
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, actor_wg=self.actor_wg, replicas=replicas
        )
        print("[FullyAsyncTrainer] Checkpoint manager initialized")

    async def _setup_hybrid_checkpoint_manager(self):
        """Setup hybrid checkpoint manager and perform initial sleep of hybrid replicas.

        When use_trainer_do_validate is enabled:
          1. Creates a CheckpointEngineManager with naive backend for trainer-side
             weight sync to hybrid rollout replicas.
          2. Fetches hybrid replicas from the rollouter's ALM (created during
             rollouter.init_workers()).
          3. Registers them with the hybrid CP manager and calls sleep_replicas()
             to release GPU memory for training.

        Must be called AFTER set_rollouter() so that self.rollouter is available,
        and AFTER rollouter.init_workers() so that hybrid replicas exist.
        This mirrors the colocate pattern in ray_trainer.py:882-889 but fetches
        replicas from the rollouter's ALM via RPC since they live on the rollout side.
        """
        if not self.config.async_training.use_trainer_do_validate:
            return

        # --- Part 1: Create hybrid CheckpointEngineManager with naive backend ---
        print("[FullyAsyncTrainer] Setting up hybrid checkpoint manager (naive backend)")

        # Create hybrid CheckpointEngineManager with naive backend.
        checkpoint_engine_cfg = self.config.actor_rollout_ref.rollout.checkpoint_engine
        original_backend = checkpoint_engine_cfg.backend
        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = "naive"
        checkpoint_engine_config = omega_conf_to_dataclass(checkpoint_engine_cfg)

        self.hybrid_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=[],  # Start empty; will be populated below
        )

        # Restore original backend value
        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = original_backend

        print("[FullyAsyncTrainer] Hybrid checkpoint manager initialized (naive backend)")

        # --- Part 2: Fetch hybrid replicas from rollouter's ALM ---
        print("[FullyAsyncTrainer] Fetching hybrid replicas from rollouter...")
        hybrid_replicas_dict = ray.get(self.rollouter.get_all_hybrid_replicas.remote())
        print(
            f"[FullyAsyncTrainer] Got {len(hybrid_replicas_dict)} hybrid replicas: {list(hybrid_replicas_dict.keys())}"
        )

        if not hybrid_replicas_dict:
            print("[FullyAsyncTrainer] No hybrid replicas found, skipping initial sleep")
            return

        # --- Part 3: Register replicas and perform initial sleep ---
        for resource_id, replica in hybrid_replicas_dict.items():
            self.hybrid_checkpoint_manager.replicas.append(replica)
            print(
                f"[FullyAsyncTrainer] Registered '{resource_id}' "
                f"(mode={getattr(replica, 'rollout_mode', '?')}, "
                f"addr={getattr(replica, '_server_address', '?')})"
            )

        # Step 3: Sleep all hybrid replicas
        print(
            f"[FullyAsyncTrainer] Calling sleep_replicas() on "
            f"{len(self.hybrid_checkpoint_manager.replicas)} replicas..."
        )
        await self.hybrid_checkpoint_manager.sleep_replicas()
        print("[FullyAsyncTrainer] Initial sleep complete, GPU memory now owned by training engine")

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    async def set_rollouter(self, rollouter):
        """Set rollouter reference and initialize all checkpoint managers."""
        self.rollouter = rollouter
        self.rollout_capabilities = await self.rollouter.get_capabilities.remote()
        weight_sync = self.rollout_capabilities.get("weight_sync")
        if weight_sync == "verl_checkpoint_engine":
            await self._setup_checkpoint_manager()
        elif weight_sync == "hf_disk":
            self.checkpoint_manager = None
        else:
            raise ValueError(f"Unsupported async rollout weight-sync capability: {weight_sync!r}")

        if self.config.async_training.use_trainer_do_validate:
            if not self.rollout_capabilities.get("hybrid_validation", False):
                backend = self.rollout_capabilities.get("name", "unknown")
                raise ValueError(f"async rollout backend {backend!r} does not support trainer-side hybrid validation")
            await self._setup_hybrid_checkpoint_manager()

    def set_total_train_steps(self, total_training_steps):
        self.total_train_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

        self.progress_bar = tqdm(total=self.total_train_steps, initial=0, desc="Training Progress")

    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg

    async def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """
        Get samples from message queue and compose gen_batch_output
        Uses a loop to continuously collect samples until enough are gathered

        Returns:
            tuple: (epoch, batch_dict, gen_batch_output)
        """
        print(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue",
            flush=True,
        )

        # Collect samples using a simple loop calling get_sample
        consumer_start = time.time()
        queue_samples = []
        admission_counts = Counter()
        filter_config = self.config.algorithm.get("filter_groups", None)
        filter_enabled = bool(filter_config and filter_config.get("enable", False))
        if filter_enabled and filter_config.get("metric") != "raw_correctness":
            raise ValueError("async mixed-reward filtering must use raw_correctness")
        max_generation_rounds = int(filter_config.get("max_num_gen_batches", 0)) if filter_enabled else 1
        if filter_enabled and max_generation_rounds <= 0:
            raise ValueError("algorithm.filter_groups.max_num_gen_batches must be a finite positive integer")
        # The synchronous collector admits up to train_batch_size prompt groups
        # in each generation round. Queue items are one atomic prompt group, so
        # the equivalent finite budget is rounds * required prompt groups.
        max_candidate_groups = self.required_samples * max_generation_rounds
        candidate_groups = 0
        filter_counts = Counter()
        queue_len = 0
        while len(queue_samples) < self.required_samples:
            # Get a single sample and wait until there is a sample or None is received
            sample, queue_len = await self.message_queue_client.get_sample()

            if sample is None:
                print(
                    f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                    f"Collected {len(queue_samples)}/{self.required_samples} samples"
                )
                break

            rollout_sample = ray.cloudpickle.loads(sample)
            admission = validate_rollout_group_admission(
                rollout_sample,
                expected_group_size=self.config.actor_rollout_ref.rollout.n,
                learner_version=self.current_param_version,
                max_policy_version_lag=self.config.async_training.get("max_policy_version_lag"),
                require_single_policy_version=(
                    self.config.actor_rollout_ref.actor.policy_loss.loss_mode == "agro"
                ),
                seen_group_ids=self.seen_group_ids,
            )
            admission_counts[admission.reason] += 1
            self.group_admission_counts[admission.reason] += 1
            if not admission.accepted:
                print(
                    "[FullyAsyncTrainer] Rejected rollout group before batching: "
                    f"reason={admission.reason}, behavior_version={admission.policy_version}, "
                    f"learner_version={self.current_param_version}",
                    flush=True,
                )
                continue

            candidate_groups += 1
            if filter_enabled:
                raw_correctness = raw_correctness_tensor(
                    {
                        "acc": rollout_sample.full_batch.non_tensor_batch["raw_correctness"],
                    },
                    expected_responses=self.config.actor_rollout_ref.rollout.n,
                )
                selected, stats = select_mixed_reward_group_indices(
                    raw_correctness,
                    group_size=self.config.actor_rollout_ref.rollout.n,
                    max_groups=1,
                )
                filter_counts.update(stats)
                if selected.numel() == 0:
                    if candidate_groups >= max_candidate_groups:
                        raise RuntimeError(
                            "async mixed-reward sampling exhausted its finite generation budget without "
                            f"filling the learner batch: accepted={len(queue_samples)}, "
                            f"required={self.required_samples}, candidate_groups={candidate_groups}, "
                            f"max_generation_rounds={max_generation_rounds}"
                        )
                    continue

            queue_samples.append(rollout_sample)

            if len(queue_samples) % 64 == 0:
                print(
                    f"[FullyAsyncTrainer] Collected {len(queue_samples)}/{self.required_samples} samples. "
                    f"mq_len: {queue_len}"
                )

        consumer_end = time.time()

        if not queue_samples or len(queue_samples) < self.required_samples:
            print("[FullyAsyncTrainer] not enough samples collected after loop")
            return None, None
        total_wait_time = consumer_end - consumer_start

        print(
            f"[FullyAsyncTrainer] Loop collection completed: {len(queue_samples)}/{self.required_samples} samples, "
            f"total wait time: {total_wait_time:.2f} seconds. "
            f"mq_len: {queue_len}"
        )

        # Assemble batch - now working directly with RolloutSample objects
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(queue_samples, self._balance_batch)
        else:
            batch = assemble_batch_from_rollout_samples(queue_samples)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        for reason, count in admission_counts.items():
            batch.meta_info[f"fully_async/admission/{reason}_groups"] = count
        for reason, count in self.group_admission_counts.items():
            batch.meta_info[f"fully_async/admission/cumulative_{reason}_groups"] = count
        if filter_enabled:
            batch.meta_info.update(
                {
                    "sampling/generated_groups": filter_counts["generated_groups"],
                    "sampling/accepted_groups": len(queue_samples),
                    "sampling/rejected_constant_groups": (
                        filter_counts["all_zero_groups"] + filter_counts["all_one_groups"]
                    ),
                    "sampling/raw_all_zero_groups": filter_counts["all_zero_groups"],
                    "sampling/raw_all_one_groups": filter_counts["all_one_groups"],
                    "sampling/generation_rounds": math.ceil(candidate_groups / self.required_samples),
                    "sampling/acceptance_rate": len(queue_samples) / candidate_groups,
                }
            )
        return 0, batch

    def _create_actor_rollout_classes(self):
        # create actor — always use Role.Actor (not ActorRollout) even when
        # use_trainer_do_validate is enabled. Rollout capability on trainer GPUs
        # is handled by ElasticAgentLoopManager's hybrid replicas.
        for role in [self.train_role]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _create_reward_model_class(self):
        # In fully async mode, RM is managed by RewardLoopManager (standalone). Skip worker group creation for RM.
        pass

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.actor_wg = self.all_wg[str(self.train_role)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.
        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.rollouter is None:
            raise ValueError("rollouter not set. Call set_rollouter() first.")

        self.max_steps_duration = 0

        self.global_steps += 1

        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        stop_reason = await _run_fully_async_step_loop(self)

        self.progress_bar.close()
        _require_optimizer_progress(self.completed_optimizer_steps)
        # Publish a partially accumulated learner update before the rollouter
        # terminates.  A published set of weights needs its own real policy
        # version; otherwise the final metrics/checkpoint would either reuse the
        # previous version or invent a tracker-only version.
        self.current_param_version, self.local_trigger_step, has_partial_update = _resolve_terminal_policy_version(
            self.current_param_version,
            self.local_trigger_step,
        )

        await _finalize_fully_async_policy(
            self,
            has_partial_update=has_partial_update,
        )
        # If termination lands between parameter-sync boundaries, flush the
        # partial learner window at the actual published policy version rather
        # than creating a tracker-only future version.
        # Ray may tear down this actor before Python runs Tracking.__del__.
        # Finish explicitly so successful async runs are not marked crashed.
        _finish_fully_async_run(self)
        return {
            "stop_reason": stop_reason,
            "completed_optimizer_steps": self.completed_optimizer_steps,
        }

    async def fit_step(self, batch_dict: dict = None):
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing

        Args:
            batch_dict: Raw data dictionary
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self._step_metric_aggregation = {}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        steps = self.config.global_profiler.steps
        should_profile = steps is not None and (self.current_param_version + 1) in steps
        self._fit_start_profile(should_profiler=should_profile)

        with marked_timer("step", self.timing_raw):
            batch = await self._fit_generate(None)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            self._collect_lagged_mismatch_metrics(batch, self.metrics)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)
            actor_will_update = self.config.trainer.critic_warmup <= self.global_steps
            batch = self._fit_update_actor(batch)
            if actor_will_update:
                self.completed_optimizer_steps += 1
                self.metrics["fully_async/count/completed_optimizer_steps"] = self.completed_optimizer_steps
            # Physical actor-update padding is never rollout data. Restore the
            # exact admitted batch before counters, dumps, and scientific
            # metrics observe it.
            batch = select_fully_async_real_samples(batch)
            self.reward_tensor, self.reward_extra_infos_dict = extract_reward(batch)
            self._fit_update_local_step()
            weights_updated = await self._fit_update_weights()
            self._fit_dump_data(batch)

        await self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_stop_profile(should_profiler=should_profile)
        self._fit_collect_metrics(batch)
        _finalize_fully_async_step_metrics(self, weights_updated=weights_updated)

    async def _fit_generate(self, batch: DataProto = None) -> DataProto | None:
        metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            epoch, batch = await self._get_samples_from_queue()
            if batch is None:
                raise TrainingStopException("Training terminated: queue returned None")
            self._collect_metrics_from_samples(batch, metrics)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return batch

    def _compute_old_log_prob(self, batch: DataProto):
        """
        If algorithm.rollout_correction.bypass_mode is False,
        use model engine and first version model params to re-calculate old_log_prob.

        If trigger_parameter_sync_step == 1, there is only one fit transaction
        before the next published policy version.  Its current parameters are
        already the proximal anchor, so a CPU snapshot would never be restored
        and is pure overhead.

        If local_trigger_step == 1, load the training engine's parameters to the CPU
          and save a copy for subsequent MIS use.

        If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
        then restore the parameters of the current version.
        """
        if self.trigger_parameter_sync_step == 1:
            return super()._compute_old_log_prob(batch)

        if self.local_trigger_step == 1:
            with marked_timer("fully_async_anchor_snapshot_save", self.timing_raw):
                self.actor_rollout_wg.save_model_to_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
        else:
            with marked_timer("fully_async_anchor_snapshot_save", self.timing_raw):
                self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            with marked_timer("fully_async_anchor_restore_base", self.timing_raw):
                self.actor_rollout_wg.restore_model_from_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
            with marked_timer("fully_async_anchor_restore_current", self.timing_raw):
                self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            with marked_timer("fully_async_anchor_snapshot_clear", self.timing_raw):
                self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
        return old_log_prob, old_log_prob_mfu

    def _actor_update_calculate_entropy(self) -> bool:
        """Preserve the pre-existing async entropy contract in the shared dispatcher."""
        actor = self.config.actor_rollout_ref.actor
        calculate_entropy = bool(actor.calculate_entropy)
        if actor.policy_loss.loss_mode != "ggpo":
            # AGRO, ASymPO, and the standard losses retain their previous
            # behavior: a nonzero coefficient requests entropy even when the
            # diagnostic switch is disabled.
            calculate_entropy = calculate_entropy or actor.entropy_coeff != 0.0
        return calculate_entropy

    def _fit_update_actor(self, batch: DataProto) -> DataProto:
        """Run the shared actor dispatcher for either physical geometry."""
        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", self.timing_raw, color="red"):
                actor_output = self._update_actor(batch)
            self.metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
            self._step_metric_aggregation.update(actor_output.meta_info[MISMATCH_AGGREGATION_META])
        return batch

    def _fit_compute_reward(self, batch: DataProto) -> DataProto:
        """Compute reward on real groups, then add actor-only physical padding."""
        batch = super()._fit_compute_reward(batch)
        batch, padding_spec = pad_fully_async_actor_update(batch, self.config)
        if padding_spec.enabled:
            # The inherited advantage stage consumes this cached tensor. Rebind
            # it after padding; dummy rm_scores were explicitly zeroed.
            self.reward_tensor, self.reward_extra_infos_dict = extract_reward(batch)
            self.metrics.update(
                {
                    "fully_async/actor_update_padding/logical_responses": (
                        padding_spec.logical_responses_per_update
                    ),
                    "fully_async/actor_update_padding/physical_responses": (
                        padding_spec.physical_responses_per_update
                    ),
                    "fully_async/actor_update_padding/padding_responses": (
                        padding_spec.padding_responses_per_update
                    ),
                    "fully_async/actor_update_padding/logical_mini_batch_size": (
                        padding_spec.logical_responses_per_mini
                    ),
                    "fully_async/actor_update_padding/physical_mini_batch_size": (
                        padding_spec.physical_responses_per_mini
                    ),
                }
            )
        return batch

    def _fit_compute_log_prob(self, batch: DataProto) -> DataProto:
        """Compute training log-probs plus a read-only current-policy diagnostic.

        The inherited implementation deliberately makes ``old_log_probs`` the
        behavior policy in bypass mode and the proximal anchor in decoupled
        mode. When the inherited path already computes current-policy
        log-probabilities, its capture hook supplies that exact tensor here.
        Bypass mode still needs a separate read-only forward pass. The
        diagnostic is removed before actor update except for the opt-in GGPO
        memory-safe surrogate, which consumes it on the driver first.
        """
        batch = super()._fit_compute_log_prob(batch)
        if "rollout_log_probs" not in batch.batch:
            return batch
        if "_diagnostic_current_log_probs" in batch.batch:
            return batch

        with marked_timer("diagnostic_current_log_prob", self.timing_raw, color="blue"):
            current_policy_output, current_policy_mfu = super()._compute_old_log_prob(batch)
        current_log_probs = current_policy_output.batch["old_log_probs"].detach()
        batch.batch["_diagnostic_current_log_probs"] = current_log_probs
        if is_ggpo_policy_loss(self.config):
            batch.batch[GGPO_THEORY_CURRENT_LOG_PROBS] = current_log_probs
        self.metrics["perf/mfu/diagnostic_actor_infer"] = current_policy_mfu
        return batch

    def _fit_capture_current_log_probs_for_diagnostics(
        self,
        batch: DataProto,
        current_log_probs: torch.Tensor,
        current_log_prob_mfu: float,
    ) -> None:
        """Retain the inherited learner-policy forward for lag diagnostics."""
        current_log_probs = current_log_probs.detach()
        batch.batch["_diagnostic_current_log_probs"] = current_log_probs
        if is_ggpo_policy_loss(self.config):
            batch.batch[GGPO_THEORY_CURRENT_LOG_PROBS] = current_log_probs
        self.metrics["perf/mfu/diagnostic_actor_infer"] = current_log_prob_mfu

    def _ggpo_theory_policy_identity(self, batch: DataProto):
        versions = batch.batch.get("rollout_policy_versions")
        if versions is None:
            raise ValueError("Fully async GGPO Set-A requires per-token rollout_policy_versions")
        return {
            "actor_update_step": int(self.current_param_version) + 1,
            "training_policy_version": int(self.current_param_version),
            "behavior_policy_versions": versions,
            "behavior_version_stratification": True,
        }

    def _fit_update_local_step(self):
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"[FullyAsyncTrainer] global_steps: {self.global_steps} "
            f"local_trigger_step: {self.local_trigger_step} "
            f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
            f"{time_str}"
        )
        if self.local_trigger_step < self.trigger_parameter_sync_step:
            self.local_trigger_step += 1
        else:
            self.current_param_version += 1
            self.local_trigger_step = 1

    def _areal_weight_export_path(self, version: int) -> str:
        """Return an isolated, versioned directory for transient HF exports."""
        configured_root = self.config.async_training.weight_sync.get("export_dir")
        if configured_root is None:
            configured_root = os.path.join(self.config.trainer.default_local_dir, "async_weight_exports")
        export_base = os.path.abspath(os.path.expanduser(configured_root))
        export_path = os.path.abspath(os.path.join(export_base, f"version_{version}"))
        if os.path.dirname(export_path) != export_base:
            raise ValueError(f"Unsafe async weight export path: {export_path}")
        return export_path

    async def _sync_rollout_weights(self) -> dict:
        """Synchronize learner weights using the selected backend capability."""
        mode = self.rollout_capabilities.get("weight_sync")
        if mode == "verl_checkpoint_engine":
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
            return {
                "fully_async/weight_sync/version": self.current_param_version,
                "fully_async/weight_sync/is_hf_disk": 0,
            }
        if mode != "hf_disk":
            raise ValueError(f"Unsupported async rollout weight-sync capability: {mode!r}")

        export_path = self._areal_weight_export_path(self.current_param_version)
        if os.path.exists(export_path):
            shutil.rmtree(export_path)
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        try:
            self.actor_rollout_wg.export_hf_model(export_path)
            hf_path = os.path.join(export_path, "huggingface")
            if not os.path.isdir(hf_path):
                raise RuntimeError(f"Actor HF export did not produce {hf_path}")
            backend_metrics = await self.rollouter.sync_weights.remote(
                version=self.current_param_version,
                hf_path=hf_path,
            )
        finally:
            # AReaL removes the consumed huggingface directory. Remove only the
            # exact per-version parent in case auxiliary export metadata remains.
            shutil.rmtree(export_path, ignore_errors=True)

        return {
            "fully_async/weight_sync/version": self.current_param_version,
            "fully_async/weight_sync/is_hf_disk": 1,
            **(backend_metrics or {}),
        }

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return False

        steps = self.config.global_profiler.steps
        last_profiler_step = self.current_param_version
        supports_profiling = self.rollout_capabilities.get("profiling", False)
        if supports_profiling and steps is not None and last_profiler_step in steps:
            await asyncio.wrap_future(self.rollouter._stop_profiling.remote().future())

        if not self.config.async_training.partial_rollout:
            drain_metrics = await asyncio.wrap_future(
                self.rollouter.pause_and_drain_for_weight_sync.remote().future()
            )
            self._pending_version_metrics.update(drain_metrics)

        with marked_timer("timing_s/param_sync", self.timing_raw):
            sync_metrics = await self._sync_rollout_weights()
        self._pending_version_metrics.update(
            {
                **sync_metrics,
                "timing_s/param_sync": self.timing_raw["timing_s/param_sync"],
            }
        )
        print(
            f"[FullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f} seconds "
            f"self.current_param_version: {self.current_param_version}"
        )

        profiler_step = last_profiler_step + 1

        if supports_profiling and steps is not None and profiler_step in steps:
            await asyncio.wrap_future(self.rollouter._start_profiling.remote().future())

        # Reset staleness in rollouter
        timing_raw = await asyncio.wrap_future(
            self.rollouter.reset_staleness.remote(version=self.current_param_version).future()
        )
        self._pending_version_metrics.update(timing_raw)
        return True

    def _fit_log_aggregated_training_metrics(self, step: int | None = None):
        _log_fully_async_version_metrics(self, step=step)

    async def _fit_validate(self, val_before_train=False):
        if self.local_trigger_step != 1:
            return

        # Check if validation is needed
        need_validate = (
            self.config.trainer.test_freq > 0
            and self.current_param_version % self.config.trainer.test_freq == 0
            and self.current_param_version > 0
        )
        # Skip validation if not needed and not validation before training
        if not need_validate and not val_before_train:
            return
        # Execute validation
        if self.config.async_training.use_trainer_do_validate:
            val_metrics = await self._trainer_side_validate()
        else:
            val_metrics = await self.rollouter.do_validate.remote()
        self._pending_version_metrics.update(val_metrics)

    async def _trainer_side_validate(self):
        """Run trainer-side validation using hybrid rollout replicas."""
        print("[FullyAsyncTrainer] _trainer_side_validate === START ===")
        validate_start = time.time()
        # ================================================================
        # Phase 1: Switch ALL trainer GPUs to ROLLOUT mode
        # ================================================================
        phase_1_start = time.time()
        print("[FullyAsyncTrainer] Phase 1: Switching all GPUs to ROLLOUT mode")
        await self.hybrid_checkpoint_manager.update_weights(global_steps=self.current_param_version)
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        hybrid_replicas_dict = await self.rollouter.get_all_hybrid_replicas.remote()
        hybrid_resource_ids = list(hybrid_replicas_dict.keys())
        await self.rollouter.add_replicas.remote(hybrid_resource_ids)
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()
        print(f"[FullyAsyncTrainer] Phase 1 done ({time.time() - phase_1_start:.2f}s)")

        # ================================================================
        # Phase 2: Run validation via RPC to rollouter
        # ================================================================
        print("[FullyAsyncTrainer] Phase 2: Running validation")
        val_metrics = await self.rollouter.do_validate.remote()

        # ================================================================
        # Phase 3: Switch hybrid GPUs back to TRAIN mode
        # ================================================================
        print("[FullyAsyncTrainer] Phase 3: Switching hybrid GPUs back to TRAIN mode")
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        # Batch remove all hybrid replicas from the load balancer in a single RPC.
        await self.rollouter.remove_replicas.remote(hybrid_resource_ids)
        await self.hybrid_checkpoint_manager.sleep_replicas()
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()

        total_time = time.time() - validate_start
        print(f"[FullyAsyncTrainer] _trainer_side_validate === END === (total: {total_time:.2f}s)")
        return val_metrics

    def _fit_save_checkpoint(self, force=False):
        timing_raw = self.timing_raw
        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        # A terminal force-save bypasses the periodic-save switch, while the
        # helper still de-duplicates an already persisted policy version.
        save_freq = self.config.trainer.save_freq
        if _should_save_fully_async_checkpoint(
            force=force,
            current_param_version=self.current_param_version,
            last_ckpt_version=self.last_ckpt_version,
            save_freq=save_freq,
            esi_close_to_expiration=esi_close_to_expiration,
            save_final_checkpoint=self.config.trainer.get("save_final_checkpoint", True),
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                # sleep replicas to avoid OOM during checkpoint saving
                self._save_checkpoint()
                self.last_ckpt_version = self.current_param_version

    def _fit_postprocess_step(self):
        self.global_steps += 1

        aggregation = declare_fully_async_step_metrics(
            self.metrics,
            prompt_count=self.required_samples,
            declared=self._step_metric_aggregation,
        )
        self.metrics_aggregator.add_step_metrics(metrics=self.metrics, aggregation=aggregation)

        if self.local_trigger_step == 1:
            self.progress_bar.update(1)

    def _save_checkpoint(self):
        # Warning: Currently, to align the training process and metrics of colocate,
        # we use current_param_version instead of global step.
        # This can be logically aligned with the original self.global_steps of colocate
        # and is used for metrics and ckpt. which means that the parameter synchronization
        # from trainer to rollouter will increase by 1 each time.

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )

        print(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    async def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"[FullyAsyncTrainer] Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        print(
            f"[FullyAsyncTrainer] Setting global step to {self.global_steps}, "
            f"current_param_version to {self.current_param_version}"
        )
        print(f"[FullyAsyncTrainer] Resuming from  {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        return self.current_param_version

    def _collect_metrics_from_samples(self, batch, metrics):
        """
        Collect metrics from samples
        """
        response_mask = batch.batch.get("response_mask")
        if response_mask is not None:
            lengths = response_mask.to(torch.float64).sum(dim=-1)
            if lengths.numel():
                metrics.update(
                    {
                        "fully_async/response_length/mean": float(lengths.mean().item()),
                        "fully_async/response_length/p50": float(torch.quantile(lengths, 0.50).item()),
                        "fully_async/response_length/p95": float(torch.quantile(lengths, 0.95).item()),
                        "fully_async/response_length/max": float(lengths.max().item()),
                        "fully_async/response_length/max_length_ratio": float(
                            (lengths >= int(self.config.data.max_response_length)).to(torch.float64).mean().item()
                        ),
                    }
                )
                # AReaL records one wall-clock generation duration on every
                # response in a prompt group.  Count that duration once per
                # uid so the denominator is group-service time, not n-times
                # duplicated response time.  This is intentionally distinct
                # from learner end-to-end throughput (perf/throughput).
                processing_times = batch.non_tensor_batch.get("processing_times")
                group_uids = batch.non_tensor_batch.get("uid")
                if processing_times is not None and group_uids is not None:
                    group_seconds = 0.0
                    seen_uids = set()
                    for uid, seconds in zip(group_uids, processing_times, strict=True):
                        uid_key = str(uid)
                        if uid_key not in seen_uids:
                            seen_uids.add(uid_key)
                            group_seconds += float(seconds)
                    if group_seconds > 0.0:
                        metrics["fully_async/rollout/response_tokens_per_service_second"] = float(
                            lengths.sum().item() / group_seconds
                        )
        ended_with_eos = batch.batch.get("response_ended_with_eos")
        if ended_with_eos is not None and ended_with_eos.numel():
            eos_ratio = ended_with_eos.to(torch.float64).mean().item()
            metrics["fully_async/response/eos_ratio"] = float(eos_ratio)
        correctness = batch.batch.get("verifier_correctness")
        if correctness is not None and correctness.numel():
            accuracy = correctness.to(torch.float64).mean().item()
            metrics["fully_async/verifier/accuracy"] = float(accuracy)
        if hasattr(batch, "meta_info") and batch.meta_info:
            token_versions = batch.batch.get("rollout_policy_versions")
            token_mask = batch.batch.get("response_mask")
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            version_metrics = compute_policy_version_metrics(
                learner_version=self.current_param_version,
                rollout_versions=token_versions.cpu().numpy() if token_versions is not None else trajectory_param_versions,
                response_mask=token_mask.cpu().numpy() if token_versions is not None else None,
                reference_delta=self.config.async_training.get("observability_version_delta_reference", 8),
            )
            self.consumed_cross_version_response = self.consumed_cross_version_response or (
                float(version_metrics.get("fully_async/version/partial_response_ratio", 0.0)) > 0.0
            )
            metrics.update(
                {
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                    **version_metrics,
                }
            )
            self._step_metric_aggregation.update(version_metrics.aggregation)
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s") or key.startswith("sampling/"):
                    metrics[key] = value

    def _collect_lagged_mismatch_metrics(self, batch, metrics):
        required = {"rollout_log_probs", "_diagnostic_current_log_probs", "response_mask"}
        if not required.issubset(batch.batch.keys()):
            return
        try:
            diagnostic_batch = select_fully_async_real_samples(batch)
            lagged_metrics = compute_lagged_log_prob_metrics(
                learner_version=self.current_param_version,
                rollout_versions=diagnostic_batch.batch["rollout_policy_versions"],
                behavior_log_probs=diagnostic_batch.batch["rollout_log_probs"],
                current_log_probs=diagnostic_batch.batch["_diagnostic_current_log_probs"],
                response_mask=diagnostic_batch.batch["response_mask"],
                max_lag=self.config.async_training.get("observability_version_delta_reference", 8),
            )
            metrics.update(lagged_metrics)
            self._step_metric_aggregation.update(lagged_metrics.aggregation)
        finally:
            current_log_probs = batch.batch.pop("_diagnostic_current_log_probs")
            if bool(
                OmegaConf.select(
                    self.config,
                    "actor_rollout_ref.actor.policy_loss.memory_safe_surrogate",
                    default=False,
                )
            ):
                # Keep this detached tensor only until the group-reference actor-update preparation
                # derives scalar weights; it is never dispatched to actor workers.
                batch.batch[GGPO_SURROGATE_CURRENT_LOG_PROBS] = current_log_probs
