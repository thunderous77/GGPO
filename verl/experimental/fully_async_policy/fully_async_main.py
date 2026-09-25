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
import hashlib
import json
import os
import socket
import threading
from pathlib import Path
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.experimental.fully_async_policy.rollout_backend.factory import create_rollout_backend
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.controlled_stop import AdaptiveUpdateBoundaryStop, validate_controlled_stop
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local

_ROLLOUT_RESOURCE_DUPLICATES = {
    "rollout.nnodes": "actor_rollout_ref.rollout.nnodes",
    "rollout.n_gpus_per_node": "actor_rollout_ref.rollout.n_gpus_per_node",
    "rollout.n": "actor_rollout_ref.rollout.n",
}

_ROLLOUT_ENGINE_FIELDS = (
    "seed",
    "dtype",
    "gpu_memory_utilization",
    "max_num_seqs",
    "max_num_batched_tokens",
    "enforce_eager",
)


def _require_equal_config_values(config, canonical_path: str, duplicate_path: str) -> None:
    """Reject duplicate settings instead of silently choosing a last writer."""

    canonical = OmegaConf.select(config, canonical_path)
    duplicate = OmegaConf.select(config, duplicate_path)
    if canonical is None or duplicate is None:
        raise ValueError(
            f"Both {canonical_path} and {duplicate_path} must be explicitly resolved; "
            "fully async configuration never fills a missing duplicate at runtime."
        )
    if canonical != duplicate:
        raise ValueError(
            f"Conflicting duplicate rollout settings: {canonical_path}={canonical!r}, {duplicate_path}={duplicate!r}."
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_smoke_runtime_paths(config, *, backend: str) -> None:
    """Fail closed when a declared smoke runtime can escape its exact root."""

    raw_root = OmegaConf.select(config, "async_training.runtime_root")
    if raw_root is None:
        return
    root = Path(str(raw_root)).expanduser()
    if not root.is_absolute():
        raise ValueError("async_training.runtime_root must be an absolute path")
    if root.is_symlink():
        raise ValueError("async_training.runtime_root must not be a symlink")
    root = root.resolve(strict=False)

    if OmegaConf.select(config, "trainer.save_freq") != -1:
        raise ValueError("A declared smoke runtime requires trainer.save_freq=-1")
    if OmegaConf.select(config, "trainer.save_final_checkpoint") is not False:
        raise ValueError("A declared smoke runtime requires trainer.save_final_checkpoint=false")

    protected_roots = OmegaConf.select(config, "async_training.protected_roots", default=[])
    for raw_protected in protected_roots or []:
        protected = Path(str(raw_protected)).expanduser().resolve(strict=False)
        if _is_relative_to(root, protected) or _is_relative_to(protected, root):
            raise ValueError(f"Smoke runtime root {root} overlaps protected root {protected}")

    managed_paths = {
        "trainer.default_local_dir": OmegaConf.select(config, "trainer.default_local_dir"),
    }
    if backend == "areal":
        managed_paths.update(
            {
                "async_training.weight_sync.export_dir": OmegaConf.select(
                    config, "async_training.weight_sync.export_dir"
                ),
                "async_training.areal.fileroot": OmegaConf.select(config, "async_training.areal.fileroot"),
                "async_training.areal.name_resolve_root": OmegaConf.select(
                    config, "async_training.areal.name_resolve_root"
                ),
            }
        )
    for setting, raw_path in managed_paths.items():
        if raw_path is None:
            raise ValueError(f"A declared smoke runtime requires explicit {setting}")
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{setting} must be absolute for a smoke runtime")
        resolved = path.resolve(strict=False)
        if not _is_relative_to(resolved, root):
            raise ValueError(f"{setting}={resolved} escapes smoke runtime root {root}")


def effective_rollout_contract(config) -> dict[str, object]:
    """Return the backend-independent resolved contract used by smoke gates."""

    template_kwargs = OmegaConf.select(config, "data.apply_chat_template_kwargs", default={})
    if OmegaConf.is_config(template_kwargs):
        template_kwargs = OmegaConf.to_container(template_kwargs, resolve=True)
    template_kwargs = dict(template_kwargs or {})
    rollout = "actor_rollout_ref.rollout"
    return {
        "backend": OmegaConf.select(config, "async_training.rollout_backend", default="verl"),
        "seed": OmegaConf.select(config, f"{rollout}.seed"),
        "sampling": {
            "do_sample": OmegaConf.select(config, f"{rollout}.do_sample"),
            "temperature": OmegaConf.select(config, f"{rollout}.temperature"),
            "top_p": OmegaConf.select(config, f"{rollout}.top_p"),
            "top_k": OmegaConf.select(config, f"{rollout}.top_k"),
        },
        "resources": {
            "nnodes": OmegaConf.select(config, f"{rollout}.nnodes"),
            "n_gpus_per_node": OmegaConf.select(config, f"{rollout}.n_gpus_per_node"),
            "responses_per_prompt": OmegaConf.select(config, f"{rollout}.n"),
            "tensor_parallel_size": OmegaConf.select(config, f"{rollout}.tensor_model_parallel_size"),
            "pipeline_parallel_size": OmegaConf.select(config, f"{rollout}.pipeline_model_parallel_size"),
        },
        "engine": {field: OmegaConf.select(config, f"{rollout}.{field}") for field in _ROLLOUT_ENGINE_FIELDS},
        "preprocessing": {
            "apply_chat_template_kwargs": template_kwargs,
            "filter_overlong_prompts": OmegaConf.select(config, "data.filter_overlong_prompts"),
            "truncation": OmegaConf.select(config, "data.truncation"),
        },
    }


def validate_fully_async_config(config) -> None:
    """Validate backend-independent safety invariants before Ray starts."""
    if OmegaConf.select(config, "actor_rollout_ref.hybrid_engine") is not False:
        raise ValueError(
            "Fully async training uses resource-isolated trainer and rollout workers; "
            "set actor_rollout_ref.hybrid_engine=false."
        )
    if OmegaConf.select(config, "data.train_batch_size") != 0:
        raise ValueError(
            "Fully async training disables the synchronous dataloader; set data.train_batch_size=0 "
            "and control learner update size with async_training.prompts_per_update."
        )
    max_optimizer_steps = OmegaConf.select(config, "trainer.max_optimizer_steps")
    if max_optimizer_steps is not None:
        if isinstance(max_optimizer_steps, bool) or not isinstance(max_optimizer_steps, int) or max_optimizer_steps < 1:
            raise ValueError("trainer.max_optimizer_steps must be a positive integer or null.")
        # This is intentionally a per-process bound.  `load_checkpoint()`
        # restores the policy/version and the optimizer state, while
        # `completed_optimizer_steps` starts at zero for the new process so a
        # bounded interactive segment executes exactly its requested number of
        # fresh actor updates.  Treating this local counter as a lifetime
        # counter made valid checkpoint resumes fail configuration validation.

    controlled_stop_after_seconds = OmegaConf.select(config, "trainer.controlled_stop_after_seconds")
    controlled_stop_deadline_unix_seconds = OmegaConf.select(
        config,
        "trainer.controlled_stop_deadline_unix_seconds",
    )
    validate_controlled_stop(
        controlled_stop_after_seconds,
        controlled_stop_deadline_unix_seconds,
    )
    adaptive_controlled_stop = OmegaConf.select(config, "trainer.adaptive_controlled_stop")
    if adaptive_controlled_stop is not None:
        if controlled_stop_after_seconds is not None or controlled_stop_deadline_unix_seconds is not None:
            raise ValueError("adaptive_controlled_stop and legacy controlled-stop modes are mutually exclusive")
        sync_steps = OmegaConf.select(config, "async_training.trigger_parameter_sync_step")
        controller = AdaptiveUpdateBoundaryStop(
            hard_deadline_unix_seconds=OmegaConf.select(
                adaptive_controlled_stop,
                "hard_deadline_unix_seconds",
            ),
            initial_optimizer_step_seconds=OmegaConf.select(
                adaptive_controlled_stop,
                "initial_optimizer_step_seconds",
            ),
            finalization_seconds=OmegaConf.select(
                adaptive_controlled_stop,
                "finalization_seconds",
            ),
            sync_steps=sync_steps,
            safety_factor=OmegaConf.select(
                adaptive_controlled_stop,
                "safety_factor",
                default=1.25,
            ),
            history_size=OmegaConf.select(
                adaptive_controlled_stop,
                "history_size",
                default=8,
            ),
        )
        if max_optimizer_steps is not None and max_optimizer_steps % controller.sync_steps != 0:
            raise ValueError(
                "trainer.max_optimizer_steps must end on a complete trigger_parameter_sync_step window "
                "when adaptive_controlled_stop is enabled"
            )
    # A controlled stop is likewise scoped to this process.  It is checked at
    # completed-update boundaries and is safe after `load_checkpoint()` has
    # restored the global policy version.

    backend = OmegaConf.select(config, "async_training.rollout_backend", default="verl")
    if backend not in {"verl", "areal"}:
        raise ValueError(f"async_training.rollout_backend must be 'verl' or 'areal', got {backend!r}.")

    if OmegaConf.select(config, "async_training.require_complete_groups") is not True:
        raise ValueError(
            "Fully async GRPO/GGPO requires async_training.require_complete_groups=true; "
            "partial response groups are not valid learner batches."
        )

    for public_path, actor_path in _ROLLOUT_RESOURCE_DUPLICATES.items():
        _require_equal_config_values(config, public_path, actor_path)

    # VERL engine_kwargs and AReaL backend blocks are compatibility overlays,
    # not alternate sources of truth for shared engine fields.
    for engine_name in ("vllm", "sglang"):
        for field in _ROLLOUT_ENGINE_FIELDS:
            duplicate_path = f"actor_rollout_ref.rollout.engine_kwargs.{engine_name}.{field}"
            duplicate = OmegaConf.select(config, duplicate_path)
            if duplicate is not None:
                canonical_path = f"actor_rollout_ref.rollout.{field}"
                canonical = OmegaConf.select(config, canonical_path)
                if duplicate != canonical:
                    raise ValueError(
                        "Conflicting duplicate rollout settings: "
                        f"{canonical_path}={canonical!r}, {duplicate_path}={duplicate!r}."
                    )

    max_policy_version_lag = OmegaConf.select(config, "async_training.max_policy_version_lag")
    if max_policy_version_lag is not None:
        if isinstance(max_policy_version_lag, bool) or not isinstance(max_policy_version_lag, int):
            raise ValueError("async_training.max_policy_version_lag must be a non-negative integer or null.")
        if max_policy_version_lag < 0:
            raise ValueError("async_training.max_policy_version_lag must be non-negative or null.")

    rollout_n = OmegaConf.select(config, "actor_rollout_ref.rollout.n")
    if isinstance(rollout_n, bool) or not isinstance(rollout_n, int) or rollout_n < 1:
        raise ValueError("actor_rollout_ref.rollout.n must be a positive integer.")

    if backend == "areal" and max_policy_version_lag is None:
        raise ValueError("The AReaL async rollout backend requires an explicit async_training.max_policy_version_lag.")
    if (
        backend == "areal"
        and OmegaConf.select(config, "async_training.areal.max_head_offpolicyness") != max_policy_version_lag
    ):
        raise ValueError("async_training.areal.max_head_offpolicyness must equal async_training.max_policy_version_lag")
    if backend == "areal" and OmegaConf.select(config, "async_training.use_trainer_do_validate", default=False):
        raise ValueError(
            "The AReaL async rollout backend does not support trainer-side hybrid validation; "
            "set async_training.use_trainer_do_validate=false to use native AReaL validation."
        )
    if backend == "areal":
        explicit_reward = OmegaConf.select(config, "async_training.areal.reward_fn")
        custom_reward_path = OmegaConf.select(config, "reward.custom_reward_function.path")
        if not explicit_reward and not custom_reward_path:
            reward_source = OmegaConf.select(config, "reward.reward_manager.source")
            reward_name = OmegaConf.select(config, "reward.reward_manager.name")
            if reward_source != "register" or reward_name != "batch":
                raise ValueError(
                    "AReaL rollout without an explicit reward requires "
                    "reward.reward_manager.source=register and reward.reward_manager.name=batch"
                )
        actor_strategy = OmegaConf.select(config, "actor_rollout_ref.actor.strategy")
        if actor_strategy not in {"fsdp", "fsdp2"}:
            raise ValueError(
                "The AReaL async rollout backend currently requires actor.strategy=fsdp or fsdp2 "
                "for transient Hugging Face weight export."
            )
        template_kwargs = OmegaConf.select(config, "data.apply_chat_template_kwargs", default={})
        if OmegaConf.is_config(template_kwargs):
            template_kwargs = OmegaConf.to_container(template_kwargs, resolve=True)
        template_kwargs = dict(template_kwargs or {})
        unsupported_template_kwargs = set(template_kwargs) - {"enable_thinking"}
        if unsupported_template_kwargs:
            raise ValueError(
                f"AReaL cannot reproduce data.apply_chat_template_kwargs fields: {sorted(unsupported_template_kwargs)}"
            )
        common_thinking = bool(template_kwargs.get("enable_thinking", False))
        areal_thinking = OmegaConf.select(config, "async_training.areal.enable_thinking")
        if areal_thinking is None or areal_thinking is not common_thinking:
            raise ValueError(
                "async_training.areal.enable_thinking must equal data.apply_chat_template_kwargs.enable_thinking"
            )
        if OmegaConf.select(config, "data.chat_template_func") is not None:
            raise ValueError(
                "AReaL does not support data.chat_template_func; use the tokenizer's canonical chat template"
            )
        if OmegaConf.select(config, "data.filter_overlong_prompts") is not True:
            raise ValueError("AReaL backend parity requires data.filter_overlong_prompts=true")
        if OmegaConf.select(config, "data.truncation") != "error":
            raise ValueError("AReaL backend parity requires data.truncation=error")

    loss_mode = OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode", default="vanilla")
    if loss_mode in {"ggpo", "agro"}:
        if max_policy_version_lag is None:
            raise ValueError(
                f"Fully async {str(loss_mode).upper()} requires an explicit async_training.max_policy_version_lag."
            )
        if OmegaConf.select(config, "algorithm.use_kl_in_reward", default=False):
            raise ValueError(
                f"{str(loss_mode).upper()} requires algorithm.use_kl_in_reward=false because its policy loss "
                "already embeds "
                "reference-policy regularization and consumes raw verifier scores."
            )
        if OmegaConf.select(config, "actor_rollout_ref.actor.use_kl_loss", default=False):
            raise ValueError(f"{str(loss_mode).upper()} requires actor.use_kl_loss=false")
        correction = OmegaConf.select(config, "algorithm.rollout_correction", default=None)
        if correction is not None and any(
            OmegaConf.select(config, f"algorithm.rollout_correction.{field}", default=None) is not None
            for field in ("rollout_is", "rollout_rs")
        ):
            raise ValueError(f"{str(loss_mode).upper()} forbids rollout IS weights and rejection sampling")
        policy_loss = config.actor_rollout_ref.actor.policy_loss
        if loss_mode == "agro":
            if policy_loss.sequence_log_prob_reduction != "sum":
                raise ValueError("AGRO Eq. (4) requires SeqSum")
    elif loss_mode == "asympo":
        if OmegaConf.select(config, "algorithm.use_kl_in_reward", default=False):
            raise ValueError("ASymPO requires algorithm.use_kl_in_reward=false")
        if OmegaConf.select(config, "actor_rollout_ref.actor.use_kl_loss", default=False):
            raise ValueError("ASymPO requires actor.use_kl_loss=false")
        correction = OmegaConf.select(config, "algorithm.rollout_correction", default=None)
        if correction is not None and (
            OmegaConf.select(config, "algorithm.rollout_correction.rollout_is", default=None) is not None
            or OmegaConf.select(config, "algorithm.rollout_correction.rollout_rs", default=None) is not None
            or bool(OmegaConf.select(config, "algorithm.rollout_correction.bypass_mode", default=False))
        ):
            raise ValueError("ASymPO forbids rollout correction and bypass-mode loss replacement")
        if str(OmegaConf.select(config, "algorithm.adv_estimator")) != "grpo":
            raise ValueError("ASymPO requires algorithm.adv_estimator=grpo")
        if bool(OmegaConf.select(config, "algorithm.norm_adv_by_std_in_grpo", default=True)):
            raise ValueError("ASymPO requires raw-reward mean centering without standard-deviation normalization")

    _validate_smoke_runtime_paths(config, backend=backend)


def describe_fully_async_mode(config) -> str:
    """Return the resolved algorithm/backend mode for logs and smoke gates."""
    algorithm = str(
        OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode", default="vanilla")
    )
    if algorithm == "vanilla":
        algorithm = "grpo"
    backend = OmegaConf.select(config, "async_training.rollout_backend", default="verl")
    if algorithm == "ggpo":
        policy_loss = config.actor_rollout_ref.actor.policy_loss
        return (
            f"algorithm=ggpo backend={backend} beta={policy_loss.beta} "
            f"reduction={policy_loss.sequence_log_prob_reduction} "
            f"group_size={config.actor_rollout_ref.rollout.n}"
        )
    if algorithm == "agro":
        policy_loss = config.actor_rollout_ref.actor.policy_loss
        return (
            f"algorithm=agro backend={backend} beta={policy_loss.beta} "
            f"reduction={policy_loss.sequence_log_prob_reduction} group_size={config.actor_rollout_ref.rollout.n}"
        )
    if algorithm == "asympo":
        return f"algorithm=asympo backend={backend} reduction=response-equal"
    return f"algorithm=grpo backend={backend} advantage_estimator={config.algorithm.adv_estimator}"


@ray.remote(num_cpus=1)
class FullyAsyncTaskRunner:
    """
    Ray remote class for executing distributed PPO training tasks.
    """

    def __init__(self):
        self.running = False
        self.components = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        print("[ASYNC MAIN] Starting fully async PPO training...")
        try:
            self._initialize_components(config)
            self._run_training_loop()
        finally:
            self._shutdown_components()

    def _initialize_components(self, config) -> None:
        print(f"[ASYNC MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        OmegaConf.resolve(config)
        resolved_config = OmegaConf.to_yaml(config, resolve=True, sort_keys=True)
        pprint(OmegaConf.to_container(config, resolve=True))
        runtime_root = OmegaConf.select(config, "async_training.runtime_root", default=None)
        if runtime_root:
            runtime_path = Path(str(runtime_root))
            runtime_path.mkdir(parents=True, exist_ok=True)
            config_path = runtime_path / "resolved-config.yaml"
            config_path.write_text(resolved_config, encoding="utf-8")
            (runtime_path / "resolved-config.sha256").write_text(
                f"{hashlib.sha256(resolved_config.encode('utf-8')).hexdigest()}  resolved-config.yaml\n",
                encoding="utf-8",
            )

        print("[ASYNC MAIN] Initializing model and tokenizer...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        print("[ASYNC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        print("[ASYNC MAIN] Creating FullyAsyncTrainer first (needed for hybrid worker group injection)...")
        self._create_trainer(config)

        print("[ASYNC MAIN] Injecting trainer's worker group into rollouter for hybrid replicas...")
        self._setup_hybrid_worker_group(config)

        print("[ASYNC MAIN] Creating FullyAsyncRollouter...")
        self._create_rollouter(config)

        print("[ASYNC MAIN] Setting up rollouter reference on trainer")
        ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))

        # sync total_train_steps between rollouter and trainer
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"total_train_steps {total_train_steps}")
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        # max_queue_size
        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        print(f"[ASYNC MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(self.components["rollouter"].set_message_queue_client.remote(self.components["message_queue_client"]))
        ray.get(self.components["trainer"].set_message_queue_client.remote(self.components["message_queue_client"]))

        # param_version resume from ckpt or default 0
        initial_param_version = ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

        print("[ASYNC MAIN] Param sync before fit..")
        ray.get(self.components["trainer"]._fit_update_weights.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(self.components["trainer"]._fit_validate.remote(True))
        # Initial weight-sync and optional validation both belong to policy
        # version zero. Commit them together so W&B sees exactly one log call
        # for that explicit step; this also flushes initial sync when validation
        # before training is disabled.
        ray.get(self.components["trainer"]._fit_log_aggregated_training_metrics.remote(step=initial_param_version))

        print("[ASYNC MAIN] All components initialized successfully")

    def _create_rollouter(self, config) -> None:
        print("[ASYNC MAIN] Starting create rollouter...")
        rollouter = create_rollout_backend(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        # Store the actor before initialization so a partially initialized
        # backend can still release Ray/AReaL resources on failure.
        self.components["rollouter"] = rollouter

        capabilities = ray.get(rollouter.get_capabilities.remote())
        print(f"[ASYNC MAIN] Rollout backend capabilities: {capabilities}")

        # set_hybrid_worker_group must be called BEFORE init_workers() so that
        # _init_async_rollout_manager can pass the hybrid WG to ALM.create().
        if "hybrid_worker_group" in self.components:
            if not capabilities.get("hybrid_validation", False):
                raise ValueError(
                    f"rollout backend {capabilities.get('name', 'unknown')!r} "
                    "does not support trainer-side hybrid validation"
                )
            ray.get(rollouter.set_hybrid_worker_group.remote(self.components["hybrid_worker_group"]))
            print("[ASYNC MAIN] Hybrid worker group injected into rollouter")

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        print("[ASYNC MAIN] Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        print("[ASYNC MAIN] Starting create trainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")

    def _setup_hybrid_worker_group(self, config) -> None:
        """
        Extract the trainer's actor_rollout_wg and store it for later injection
        into the rollouter. This WG backs the hybrid rollout replicas
        used during trainer-side validation (use_trainer_do_validate).
        """
        trainer = self.components["trainer"]
        if config.async_training.use_trainer_do_validate:
            trainer_wg = ray.get(trainer.get_actor_wg.remote())
            self.components["hybrid_worker_group"] = trainer_wg
            print(
                f"[ASYNC MAIN] Hybrid worker group extracted from trainer "
                f"(world_size={getattr(trainer_wg, 'world_size', '?')})"
            )
        else:
            print("[ASYNC MAIN] use_trainer_do_validate=False, skipping hybrid worker group setup")

    def _run_training_loop(self):
        self.running = True

        print("[ASYNC MAIN] Starting Rollouter and Trainer...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        futures = [rollouter_future, trainer_future]

        try:
            while futures:
                # Use ray.wait to monitor all futures and return when any one is completed.
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    try:
                        ray.get(future)
                        print("[ASYNC MAIN] One component completed successfully")
                    except Exception as e:
                        print(f"[ASYNC MAIN] Component failed with error: {e}")
                        for remaining_future in remaining_futures:
                            ray.cancel(remaining_future)
                        raise e

                futures = remaining_futures

        except Exception as e:
            print(f"[ASYNC MAIN] Training failed: {e}")
            for future in futures:
                ray.cancel(future)
            raise
        finally:
            self.running = False
            print("[ASYNC MAIN] Training completed or interrupted")

    def _shutdown_components(self) -> None:
        """Best-effort teardown without hiding the original training error."""
        rollouter = self.components.get("rollouter")
        if rollouter is not None:
            try:
                ray.get(rollouter.shutdown.remote())
                print("[ASYNC MAIN] Rollout backend shut down")
            except Exception as exc:
                print(f"[ASYNC MAIN] Rollout backend shutdown warning: {exc}")

        message_queue_client = self.components.get("message_queue_client")
        if message_queue_client is not None:
            try:
                asyncio.run(message_queue_client.clear_queue())
                print("[ASYNC MAIN] Message queue cleared")
            except Exception as exc:
                print(f"[ASYNC MAIN] Message queue cleanup warning: {exc}")


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config

    # Ensure async training config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    from time import time

    start_time = time()
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    validate_fully_async_config(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    print(f"[ASYNC MAIN] Resolved mode: {describe_fully_async_mode(config)}")
    print("[ASYNC MAIN] EFFECTIVE_ROLLOUT_CONTRACT=" + json.dumps(effective_rollout_contract(config), sort_keys=True))
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
