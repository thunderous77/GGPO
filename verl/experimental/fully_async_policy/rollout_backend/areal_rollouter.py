"""Native vendored-AReaL rollout actor for VERL's fully-async trainer."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

import numpy as np
import ray
import torch

from verl.experimental.fully_async_policy.detach_utils import RolloutSample
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouterBase
from verl.protocol import DataProto
from verl.utils.profiler import marked_timer
from verl.utils.skip import SkipManager

from .adapter import areal_trajectory_to_dataproto, build_areal_task_data
from .capabilities import AREAL_CAPABILITIES
from .dispatch import submit_areal_prompt_group
from .factory import config_get
from .reward_bridge import (
    AREAL_BATCH_REWARD_FN,
    AREAL_CUSTOM_REWARD_FN,
    CUSTOM_REWARD_KWARGS_KEY,
    CUSTOM_REWARD_NAME_KEY,
    CUSTOM_REWARD_PATH_KEY,
)
from .scheduler import build_areal_ray_scheduler
from .vendored import load_vendored_areal
from .weight_sync import areal_weight_update_key, validate_transient_hf_export


def _validate_strict_6_2_topology(
    *, live_gpu_counts: list[int], requested_rollout_gpus: int, available_gpus: int, initialized: bool = False
) -> None:
    """Validate the exact two-node 6-rollout/2-learner smoke placement."""
    expected_available = 0 if initialized else 6
    if sorted(live_gpu_counts) != [4, 4] or requested_rollout_gpus != 6 or available_gpus != expected_available:
        phase = "after rollout initialization" if initialized else "after packed learner placement"
        raise RuntimeError(
            "strict 6:2 placement requires two live 4-GPU nodes, six rollout GPUs, "
            f"and {expected_available} GPUs free {phase}; got nodes={sorted(live_gpu_counts)}, "
            f"requested={requested_rollout_gpus}, available={available_gpus}"
        )

# Importing this module is itself backend selection.  Verify the package source
# before any ``areal.*`` import can be resolved from the ambient environment.
load_vendored_areal()

from areal.api import WeightUpdateMeta  # noqa: E402
from areal.api.alloc_mode import ModelAllocation  # noqa: E402
from areal.api.cli_args import (  # noqa: E402
    GenerationHyperparameters,
    InferenceEngineConfig,
    SchedulingSpec,
    SchedulingStrategy,
    SchedulingStrategyType,
    SGLangConfig,
    vLLMConfig,
)
from areal.engine.sglang_remote import RemoteSGLangEngine  # noqa: E402
from areal.engine.vllm_remote import RemotevLLMEngine  # noqa: E402
from areal.infra.rpc.rtensor import RTensor  # noqa: E402
from areal.utils import name_resolve  # noqa: E402


async def _run_areal_disk_update(
    controller: Any,
    meta: WeightUpdateMeta,
    publish_update_key: Callable[[], None],
) -> None:
    """Bridge RolloutController coroutines with blocking name resolution."""

    await controller.pause_generation()
    update_task = asyncio.create_task(controller.update_weights_from_disk(meta))
    try:
        await asyncio.to_thread(publish_update_key)
        await update_task
        await controller.continue_generation()
    except BaseException:
        # A failed rendezvous or load must leave generation paused.  The caller
        # poisons the backend so no partially loaded policy can resume rollout.
        if not update_task.done():
            update_task.cancel()
        await asyncio.gather(update_task, return_exceptions=True)
        raise


@asynccontextmanager
async def _localized_areal_trajectory(trajectory: dict[str, Any]):
    """Materialize a Ray-transport trajectory and release its remote shards."""

    shards_by_node = RTensor.collect_shards(trajectory)
    primary_error: BaseException | None = None
    try:
        localized = await asyncio.to_thread(RTensor.localize, trajectory)
        yield localized
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if shards_by_node:
            cleanup_items = list(shards_by_node.items())
            cleanup_results = await asyncio.gather(
                *(RTensor.clear_node(node_addr, shard_ids) for node_addr, shard_ids in cleanup_items),
                return_exceptions=True,
            )
            cleanup_failures = [
                (node_addr, result)
                for ((node_addr, _), result) in zip(cleanup_items, cleanup_results, strict=True)
                if isinstance(result, BaseException)
            ]
            if cleanup_failures:
                details = "; ".join(
                    f"node {node_addr!r}: {type(error).__name__}: {error}" for node_addr, error in cleanup_failures
                )
                cleanup_message = f"AReaL RTensor cleanup failed on {len(cleanup_failures)} node(s): {details}"
                if primary_error is not None:
                    if hasattr(primary_error, "add_note"):
                        primary_error.add_note(cleanup_message)
                    else:
                        # BaseException.add_note was introduced in Python 3.11;
                        # preserve the primary error and expose the same note
                        # convention on VERL's supported Python 3.10 runtime.
                        notes = list(getattr(primary_error, "__notes__", ()))
                        notes.append(cleanup_message)
                        primary_error.__notes__ = notes
                else:
                    raise RuntimeError(cleanup_message) from cleanup_failures[0][1]


def _areal_consumer_batch_size(required_samples: Any, trigger_parameter_sync_step: Any) -> int:
    """Return complete prompt-group tasks consumed per published policy version."""

    values = {
        "VERL required_samples": required_samples,
        "async_training.trigger_parameter_sync_step": trigger_parameter_sync_step,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive integer")
        if value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return required_samples * trigger_parameter_sync_step


def _areal_generation_capacity(config: Any, *, required_samples: int, staleness_capacity: int) -> int:
    """Keep filtered refill live up to the synchronous generation-round cap."""

    if not config_get(config, "algorithm.filter_groups.enable", False):
        return staleness_capacity
    max_rounds = config_get(config, "algorithm.filter_groups.max_num_gen_batches", None)
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError("algorithm.filter_groups.max_num_gen_batches must be a finite positive integer")
    trigger_steps = config_get(config, "async_training.trigger_parameter_sync_step", 1)
    return max(staleness_capacity, _areal_consumer_batch_size(required_samples, trigger_steps) * max_rounds)


def _areal_staleness_batch_size(config: Any, *, required_samples: int) -> int:
    """Count filtered candidates, not only admitted groups, per learner version."""

    trigger_steps = config_get(config, "async_training.trigger_parameter_sync_step", None)
    admitted_groups = _areal_consumer_batch_size(required_samples, trigger_steps)
    return _areal_generation_capacity(
        config,
        required_samples=required_samples,
        staleness_capacity=admitted_groups,
    )


def _areal_validation_is_enabled(config: Any) -> bool:
    """Return whether the native AReaL validation controller is needed."""

    if config_get(config, "trainer.val_before_train", True) is True:
        return True
    test_freq = config_get(config, "trainer.test_freq", -1)
    return not isinstance(test_freq, bool) and isinstance(test_freq, int) and test_freq > 0


def _areal_validation_concurrency(config: Any) -> int:
    """Resolve the prompt-level validation bound independently of dataset size."""

    value = config_get(
        config,
        "async_training.areal.validation_max_concurrent_rollouts",
        config_get(config, "async_training.areal.max_concurrent_rollouts", 8),
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("async_training.areal.validation_max_concurrent_rollouts must be a positive integer")
    return value


def _areal_validation_group_size(config: Any) -> int:
    """Return the exact number of validation responses generated per prompt."""

    value = config_get(config, "actor_rollout_ref.rollout.val_kwargs.n", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("actor_rollout_ref.rollout.val_kwargs.n must be a positive integer")
    return value


def _extract_areal_validation_rewards(trajectory: dict[str, Any], *, expected_group_size: int) -> list[float]:
    """Copy only scalar rewards out of a localized validation trajectory."""

    rewards = trajectory.get("rewards")
    if not isinstance(rewards, torch.Tensor):
        raise TypeError("AReaL validation trajectory must contain a tensor 'rewards' field")
    flat_rewards = rewards.detach().cpu().reshape(-1)
    if flat_rewards.numel() != expected_group_size:
        raise RuntimeError(
            f"AReaL validation prompt returned {flat_rewards.numel()} rewards, expected {expected_group_size}"
        )
    return [float(value) for value in flat_rewards.tolist()]


async def _collect_areal_validation_batch(
    test_batch: DataProto,
    *,
    samples_per_prompt: int,
    max_concurrency: int,
    generate_rewards: Callable[[DataProto], Awaitable[list[float]]],
) -> tuple[list[str], list[str], list[float]]:
    """Generate a validation batch in bounded chunks and retain only metrics."""

    if samples_per_prompt < 1:
        raise ValueError("AReaL validation samples_per_prompt must be positive")
    if max_concurrency < 1:
        raise ValueError("AReaL validation max_concurrency must be positive")
    if "data_source" not in test_batch.non_tensor_batch:
        test_batch.non_tensor_batch["data_source"] = np.array(["unknown"] * len(test_batch), dtype=object)
    if "uid" not in test_batch.non_tensor_batch:
        test_batch.non_tensor_batch["uid"] = np.array([str(uuid4()) for _ in range(len(test_batch))], dtype=object)

    data_sources: list[str] = []
    sample_uids: list[str] = []
    rewards: list[float] = []

    async def generate_one(index: int) -> tuple[str, str, list[float]]:
        # Slicing keeps at most max_concurrency prompt tensors alive in active
        # tasks. The caller immediately reduces each localized trajectory to
        # Python reward scalars, so full-Math500 validation cannot retain all
        # generated token tensors.
        prompt_batch = test_batch[index : index + 1]
        prompt_rewards = await generate_rewards(prompt_batch)
        if len(prompt_rewards) != samples_per_prompt:
            raise RuntimeError(
                f"AReaL validation prompt returned {len(prompt_rewards)} rewards, expected {samples_per_prompt}"
            )
        return (
            str(test_batch.non_tensor_batch["data_source"][index]),
            str(test_batch.non_tensor_batch["uid"][index]),
            prompt_rewards,
        )

    for start in range(0, len(test_batch), max_concurrency):
        stop = min(start + max_concurrency, len(test_batch))
        chunk_tasks = [
            asyncio.create_task(
                generate_one(index),
                name=f"areal_validation_prompt_{index}",
            )
            for index in range(start, stop)
        ]
        try:
            chunk = await asyncio.gather(*chunk_tasks)
        except BaseException:
            # ``gather`` propagates the first exception without cancelling its
            # other awaitables. Explicitly stop and reap every sibling so its
            # exact AReaL dispatcher task is cancelled before validation or
            # backend teardown can continue.
            for task in chunk_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*chunk_tasks, return_exceptions=True)
            raise
        for data_source, uid, prompt_rewards in chunk:
            data_sources.extend([data_source] * samples_per_prompt)
            sample_uids.extend([uid] * samples_per_prompt)
            rewards.extend(prompt_rewards)

    return data_sources, sample_uids, rewards


def _set_areal_controller_version_after_disk_update(
    controller: Any,
    version: int,
    *,
    initial_weight_sync: bool,
) -> None:
    """Publish a loaded version and bound capacity after an initial resume jump."""

    controller.set_version(version)
    if not initial_weight_sync or version == 0:
        return
    staleness_manager = controller.staleness_manager
    if staleness_manager is None:
        raise RuntimeError("AReaL staleness manager is unavailable after controller initialization")
    staleness_manager.on_version_recovered(version)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    items = getattr(value, "items", None)
    if callable(items):
        return dict(items())
    raise TypeError(f"Expected a mapping-like AReaL config, got {type(value).__name__}")


def _dataclass_options(dataclass_type, *sources: Any) -> dict[str, Any]:
    allowed = {item.name for item in fields(dataclass_type)}
    options: dict[str, Any] = {}
    for source in sources:
        options.update({key: value for key, value in _mapping(source).items() if key in allowed})
    return options


def _merge_engine_option(
    options: dict[str, Any],
    provenance: dict[str, str],
    *,
    key: str,
    value: Any,
    source: str,
) -> None:
    """Merge an engine option while rejecting conflicting duplicate sources."""

    if key in options and options[key] != value:
        raise ValueError(
            f"Conflicting AReaL engine option {key!r}: {provenance[key]}={options[key]!r}, {source}={value!r}"
        )
    options[key] = value
    provenance.setdefault(key, source)


def _merge_engine_mapping(
    options: dict[str, Any],
    provenance: dict[str, str],
    source_options: dict[str, Any],
    *,
    source: str,
    allowed: set[str],
    reject_unknown: bool,
) -> None:
    unknown = set(source_options) - allowed
    if reject_unknown and unknown:
        raise ValueError(f"Unknown {source} engine option(s): {sorted(unknown)}")
    for key, value in source_options.items():
        if key in allowed:
            _merge_engine_option(options, provenance, key=key, value=value, source=f"{source}.{key}")


def _build_areal_vllm_server_args(
    config: Any,
    areal_config: Any,
    *,
    model_path: str,
    max_model_len: int,
    allocation: ModelAllocation,
) -> dict[str, Any]:
    """Build final vLLM args from the shared contract, rejecting drift."""

    allowed = {item.name for item in fields(vLLMConfig)}
    common_keys = {
        "seed",
        "dtype",
        "gpu_memory_utilization",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enforce_eager",
    }
    options: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for key in common_keys:
        value = config_get(config, f"actor_rollout_ref.rollout.{key}", None)
        if value is not None:
            _merge_engine_option(
                options,
                provenance,
                key=key,
                value=value,
                source=f"actor_rollout_ref.rollout.{key}",
            )
    _merge_engine_option(options, provenance, key="model", value=model_path, source="actor_rollout_ref.model.path")
    _merge_engine_option(
        options,
        provenance,
        key="max_model_len",
        value=max_model_len,
        source="prompt_length+response_length",
    )
    areal_options = _mapping(areal_config)
    engine_options = _mapping(areal_options.get("vllm", {}))
    _merge_engine_mapping(
        options,
        provenance,
        areal_options,
        source="async_training.areal",
        allowed=allowed,
        reject_unknown=False,
    )
    _merge_engine_mapping(
        options,
        provenance,
        engine_options,
        source="async_training.areal.vllm",
        allowed=allowed,
        reject_unknown=True,
    )
    runtime_config = vLLMConfig(**options)
    server_args = vLLMConfig.build_args(
        runtime_config,
        tp_size=allocation.parallel.tp_size,
        pp_size=allocation.parallel.pp_size,
    )
    for key in common_keys:
        expected = config_get(config, f"actor_rollout_ref.rollout.{key}", None)
        if expected is not None and server_args.get(key) != expected:
            raise RuntimeError(
                f"Final AReaL vLLM args changed canonical {key}: expected {expected!r}, got {server_args.get(key)!r}"
            )
    return server_args


def _resolve_areal_reward(config: Any) -> tuple[str, dict[str, Any]]:
    """Resolve an explicit reward or a supported VERL reward bridge."""

    explicit_reward = config_get(config, "async_training.areal.reward_fn", None)
    if explicit_reward:
        return str(explicit_reward), {}

    custom_path = config_get(config, "reward.custom_reward_function.path", None)
    if custom_path:
        custom_name = config_get(config, "reward.custom_reward_function.name", None)
        if not custom_name:
            raise ValueError("reward.custom_reward_function.name is required when path is configured")
        reward_kwargs = _mapping(config_get(config, "reward.custom_reward_function.reward_kwargs", {}))
        return AREAL_CUSTOM_REWARD_FN, {
            CUSTOM_REWARD_PATH_KEY: str(custom_path),
            CUSTOM_REWARD_NAME_KEY: str(custom_name),
            CUSTOM_REWARD_KWARGS_KEY: reward_kwargs,
        }

    reward_manager_source = config_get(config, "reward.reward_manager.source", None)
    reward_manager_name = config_get(config, "reward.reward_manager.name", None)
    if reward_manager_source == "register" and reward_manager_name == "batch":
        return AREAL_BATCH_REWARD_FN, {}

    raise ValueError(
        "AReaL rollout without an explicit reward supports only VERL's "
        "registered batch reward manager; expected "
        "reward.reward_manager.source=register and reward.reward_manager.name=batch, "
        f"got source={reward_manager_source!r}, name={reward_manager_name!r}. "
        "Configure reward.custom_reward_function.path or async_training.areal.reward_fn "
        "to use another reward implementation."
    )


def _resolve_areal_reward_executor_config(config: Any) -> dict[str, float | int]:
    """Resolve bounded reward-process settings for an AReaL rollout worker."""
    timeout = config_get(config, "async_training.areal.reward_timeout_seconds", 45.0)
    max_workers = config_get(config, "async_training.areal.reward_max_workers", 1)
    max_retries = config_get(config, "async_training.areal.reward_max_retries", 0)
    if isinstance(timeout, bool) or not isinstance(timeout, int | float) or timeout <= 0:
        raise ValueError("AReaL reward_timeout_seconds must be positive")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("AReaL reward_max_workers must be a positive integer")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("AReaL reward_max_retries must be a non-negative integer")
    return {
        "reward_timeout_seconds": float(timeout),
        "reward_max_workers": max_workers,
        "reward_max_retries": max_retries,
    }


def _build_areal_generation_config(
    config: Any,
    *,
    prompt_length: int,
    response_length: int,
    validation: bool = False,
) -> GenerationHyperparameters:
    """Map the canonical VERL sampling contract to AReaL generation args."""

    prefix = "actor_rollout_ref.rollout"
    val_prefix = f"{prefix}.val_kwargs"

    def selected(name: str, default: Any) -> Any:
        if validation:
            return config_get(config, f"{val_prefix}.{name}", config_get(config, f"{prefix}.{name}", default))
        return config_get(config, f"{prefix}.{name}", default)

    top_k = int(selected("top_k", -1))
    if top_k <= 0:
        top_k = int(1e8)
    temperature = float(selected("temperature", 1.0))
    do_sample = bool(selected("do_sample", not validation))
    return GenerationHyperparameters(
        n_samples=1,
        max_new_tokens=response_length,
        max_tokens=prompt_length + response_length,
        greedy=not do_sample or temperature <= 0,
        temperature=max(temperature, 1e-6),
        top_p=float(selected("top_p", 1.0)),
        top_k=top_k,
        ignore_eos=bool(config_get(config, f"{prefix}.ignore_eos", False)),
    )


def _canonical_enable_thinking(config: Any) -> bool:
    return bool(
        config_get(
            config,
            "data.apply_chat_template_kwargs.enable_thinking",
            False,
        )
    )


def _validate_areal_allocation(allocation: ModelAllocation, *, configured_rollout_gpus: int, backend: str) -> None:
    if allocation.parallel.world_size != configured_rollout_gpus:
        raise ValueError(
            f"AReaL backend {backend!r} requests {allocation.parallel.world_size} GPUs, "
            f"but rollout config reserves {configured_rollout_gpus}; the effective "
            "allocation must consume every reserved rollout GPU"
        )


@ray.remote(num_cpus=10, max_concurrency=100)
class AReaLFullyAsyncRollouter(FullyAsyncRollouterBase):
    """Queue-compatible rollouter backed by AReaL's native controller/workflow."""

    def __init__(self, config, tokenizer, processor=None, device_name=None):
        super().__init__(config=config, tokenizer=tokenizer, processor=processor, device_name=device_name)
        self.areal_config = config_get(config, "async_training.areal", {})
        self.group_size = int(config_get(config, "actor_rollout_ref.rollout.n", 1))
        self.prompt_length = int(
            config_get(
                config,
                "actor_rollout_ref.rollout.prompt_length",
                config_get(config, "data.max_prompt_length", 512),
            )
        )
        self.response_length = int(
            config_get(
                config,
                "actor_rollout_ref.rollout.response_length",
                config_get(config, "data.max_response_length", 512),
            )
        )
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if self.group_size < 1:
            raise ValueError(f"AReaL rollout group size must be positive, got {self.group_size}")
        if self.pad_token_id is None:
            raise ValueError("AReaL rollout requires a tokenizer pad_token_id or eos_token_id")

        self.areal_scheduler = None
        self.areal_controller = None
        self.areal_validation_controller = None
        self.areal_inference_config = None
        self.areal_validation_inference_config = None
        self.areal_experiment_config = None
        self.areal_workflow = str(
            config_get(config, "async_training.areal.workflow", "areal.workflow.rlvr.RLVRWorkflow")
        )
        self.areal_reward_fn, self.areal_reward_metadata = _resolve_areal_reward(config)
        self.areal_workflow_kwargs = None
        self.areal_validation_workflow_kwargs = None
        self._validation_lock = None
        self._areal_shutdown = False
        self._weight_sync_failed = False
        self._initial_weight_sync_complete = False

    def get_capabilities(self) -> dict:
        return AREAL_CAPABILITIES.as_dict()

    def _allocation(self, backend: str) -> ModelAllocation:
        allocation = ModelAllocation.from_str(backend)
        configured_rollout_gpus = int(config_get(self.config, "rollout.nnodes", 1)) * int(
            config_get(self.config, "rollout.n_gpus_per_node", 1)
        )
        _validate_areal_allocation(
            allocation,
            configured_rollout_gpus=configured_rollout_gpus,
            backend=backend,
        )
        return allocation

    def _default_backend(self) -> str:
        engine = str(config_get(self.config, "async_training.areal.engine", "vllm")).lower()
        if engine not in {"vllm", "sglang"}:
            raise ValueError(f"AReaL engine must be 'vllm' or 'sglang', got {engine!r}")
        total_gpus = int(config_get(self.config, "rollout.nnodes", 1)) * int(
            config_get(self.config, "rollout.n_gpus_per_node", 1)
        )
        tp_size = int(config_get(self.config, "actor_rollout_ref.rollout.tensor_model_parallel_size", 1))
        if total_gpus < 1 or tp_size < 1 or total_gpus % tp_size:
            raise ValueError(
                f"Cannot derive AReaL allocation from total_gpus={total_gpus}, tensor_parallel_size={tp_size}"
            )
        return f"{engine}:d{total_gpus // tp_size}p1t{tp_size}"

    def _build_inference_config(self) -> tuple[InferenceEngineConfig, ModelAllocation]:
        backend = config_get(self.config, "async_training.areal.backend", None)
        if backend is None:
            backend = self._default_backend()
        backend = str(backend)
        allocation = self._allocation(backend)
        experiment_name = str(config_get(self.config, "trainer.project_name", "verl"))
        trial_name = str(config_get(self.config, "trainer.experiment_name", "fully_async"))
        worker_cpu = int(config_get(self.config, "async_training.areal.worker_cpu", 4))
        worker_mem_gb = int(config_get(self.config, "async_training.areal.worker_mem_gb", 32))
        max_concurrent_rollouts = int(config_get(self.config, "async_training.areal.max_concurrent_rollouts", 8))
        queue_size = int(config_get(self.config, "async_training.areal.queue_size", 128))
        consumer_batch_size = _areal_staleness_batch_size(self.config, required_samples=self.required_samples)
        setup_timeout = float(config_get(self.config, "async_training.areal.setup_timeout", 600.0))
        request_timeout = float(config_get(self.config, "async_training.areal.request_timeout", 3600.0))
        request_retries = int(config_get(self.config, "async_training.areal.request_retries", 3))
        max_head_offpolicyness = int(config_get(self.config, "async_training.areal.max_head_offpolicyness", 0))
        if worker_cpu < 1 or worker_mem_gb < 1:
            raise ValueError("AReaL worker_cpu and worker_mem_gb must both be positive")
        if max_concurrent_rollouts < 1 or queue_size < 1:
            raise ValueError("AReaL max_concurrent_rollouts and queue_size must both be positive")
        if setup_timeout <= 0 or request_timeout <= 0:
            raise ValueError("AReaL setup_timeout and request_timeout must both be positive")
        if request_retries < 0 or max_head_offpolicyness < 0:
            raise ValueError("AReaL request_retries and max_head_offpolicyness must be non-negative")

        placement_strategy = str(config_get(self.config, "async_training.areal.ray_placement_strategy", "separate"))
        instance_size = allocation.parallel.tp_size * allocation.parallel.pp_size
        gpus_per_node = int(config_get(self.config, "rollout.n_gpus_per_node", 1))
        if placement_strategy == "separate" and instance_size > gpus_per_node:
            raise ValueError(
                f"AReaL instance size {instance_size} exceeds rollout.n_gpus_per_node={gpus_per_node}; "
                "set async_training.areal.ray_placement_strategy=deferred for a multi-node engine"
            )
        worker_spec = SchedulingSpec(
            cpu=worker_cpu,
            gpu=1,
            mem=worker_mem_gb,
            cmd="python -m areal.infra.rpc.rpc_server",
            ray_placement_strategy=placement_strategy,
        )
        inference_config = InferenceEngineConfig(
            experiment_name=experiment_name,
            trial_name=trial_name,
            fileroot=str(config_get(self.config, "async_training.areal.fileroot", "./areal_runs")),
            max_concurrent_rollouts=max_concurrent_rollouts,
            queue_size=queue_size,
            # AReaL counts all generated prompt-group tasks. Filtering may
            # consume up to the synchronous generation-round cap before VERL
            # admits one learner batch, so use the candidate count per version.
            consumer_batch_size=consumer_batch_size,
            max_head_offpolicyness=max_head_offpolicyness,
            tokenizer_path=str(config_get(self.config, "actor_rollout_ref.model.path", "")),
            setup_timeout=setup_timeout,
            request_timeout=request_timeout,
            request_retries=request_retries,
            scheduling_spec=(worker_spec,),
            backend=backend,
        )
        return inference_config, allocation

    def _build_server(self, inference_config: InferenceEngineConfig, allocation: ModelAllocation):
        model_path = str(config_get(self.config, "actor_rollout_ref.model.path", ""))
        if not model_path:
            raise ValueError("actor_rollout_ref.model.path is required for the AReaL rollout backend")

        areal_options = _mapping(self.areal_config)
        engine_options = _mapping(areal_options.get(allocation.backend, {}))
        tp_size = allocation.parallel.tp_size
        pp_size = allocation.parallel.pp_size
        max_model_len = self.prompt_length + self.response_length

        if allocation.backend == "vllm":
            server_args = _build_areal_vllm_server_args(
                self.config,
                self.areal_config,
                model_path=model_path,
                max_model_len=max_model_len,
                allocation=allocation,
            )
            controller = RemotevLLMEngine.as_controller(inference_config, self.areal_scheduler)
        elif allocation.backend == "sglang":
            options = _dataclass_options(SGLangConfig, areal_options, engine_options)
            options.update(model_path=model_path)
            options.setdefault("context_length", max_model_len)
            if "gpu_memory_utilization" in areal_options and "mem_fraction_static" not in options:
                options["mem_fraction_static"] = areal_options["gpu_memory_utilization"]
            if "max_num_seqs" in areal_options and "max_running_requests" not in options:
                options["max_running_requests"] = areal_options["max_num_seqs"]
            runtime_config = SGLangConfig(**options)
            server_args = SGLangConfig.build_args(runtime_config, tp_size=tp_size, base_gpu_id=0, pp_size=pp_size)
            controller = RemoteSGLangEngine.as_controller(inference_config, self.areal_scheduler)
        else:
            raise ValueError(
                f"AReaL async rollout supports only vllm and sglang inference backends, got {allocation.backend!r}"
            )
        return controller, server_args

    def _build_workflow_kwargs(self) -> dict[str, Any]:
        generation_config = _build_areal_generation_config(
            self.config,
            prompt_length=self.prompt_length,
            response_length=self.response_length,
        )
        return {
            "reward_fn": self.areal_reward_fn,
            "gconfig": generation_config,
            "tokenizer": str(config_get(self.config, "actor_rollout_ref.model.path", "")),
            "enable_thinking": _canonical_enable_thinking(self.config),
            **_resolve_areal_reward_executor_config(self.config),
        }

    def _build_validation_inference_config(
        self,
        inference_config: InferenceEngineConfig,
        *,
        training_role: str,
    ) -> InferenceEngineConfig:
        """Build an eval dispatcher that shares, but never owns, rollout servers."""

        validation_config = deepcopy(inference_config)
        concurrency = _areal_validation_concurrency(self.config)
        validation_config.max_concurrent_rollouts = concurrency
        validation_config.queue_size = concurrency
        validation_config.consumer_batch_size = concurrency
        # Match native AReaL evaluation: validation is not constrained by the
        # training dispatcher's finite accepted-sample/version budget.
        validation_config.max_head_offpolicyness = int(1e12)
        validation_config.dump_to_file = False
        validation_config.enable_rollout_tracing = False
        validation_config.scheduling_strategy = SchedulingStrategy(
            type=SchedulingStrategyType.colocation,
            target=training_role,
        )
        for spec in validation_config.scheduling_spec:
            spec.gpu = 0
        return validation_config

    def _build_validation_workflow_kwargs(self) -> dict[str, Any]:
        """Use VERL's val_kwargs while retaining the same reward bridge."""
        generation_config = _build_areal_generation_config(
            self.config,
            prompt_length=self.prompt_length,
            response_length=self.response_length,
            validation=True,
        )
        return {
            "reward_fn": self.areal_reward_fn,
            "gconfig": generation_config,
            "tokenizer": str(config_get(self.config, "actor_rollout_ref.model.path", "")),
            "enable_thinking": _canonical_enable_thinking(self.config),
            **_resolve_areal_reward_executor_config(self.config),
        }

    async def init_workers(self):
        """Launch AReaL inference workers through the already-running Ray cluster."""

        if not ray.is_initialized():
            raise RuntimeError(
                "AReaL rollout requires an existing Ray cluster; this adapter intentionally never calls ray.init()"
            )
        self._init_async_objects()
        self._weight_sync_lock = asyncio.Lock()
        self._validation_lock = asyncio.Lock()
        inference_config, allocation = self._build_inference_config()
        experiment_config, scheduler = build_areal_ray_scheduler(self.config, inference_config)
        requested_gpus = allocation.parallel.world_size
        cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
        if requested_gpus > cluster_gpus:
            raise RuntimeError(
                f"AReaL rollout requests {requested_gpus} Ray GPUs, but the current cluster exposes only {cluster_gpus}"
            )
        strict_topology = bool(config_get(self.config, "async_training.areal.strict_6_2_topology", False))
        if strict_topology:
            live_gpu_counts = sorted(
                int(node.get("Resources", {}).get("GPU", 0))
                for node in ray.nodes()
                if node.get("Alive")
            )
            available_gpus = int(ray.available_resources().get("GPU", 0))
            _validate_strict_6_2_topology(
                live_gpu_counts=live_gpu_counts,
                requested_rollout_gpus=requested_gpus,
                available_gpus=available_gpus,
            )

        # RayScheduler schedules dedicated child workers/placement groups in the
        # existing cluster.  It does not use this CPU actor's CUDA visibility.
        self.areal_scheduler = scheduler
        self.areal_inference_config = inference_config
        self.areal_experiment_config = experiment_config
        name_resolve.reconfigure(experiment_config.cluster.name_resolve)
        self.areal_workflow_kwargs = self._build_workflow_kwargs()
        controller, server_args = self._build_server(inference_config, allocation)
        print("[AREAL] EFFECTIVE_SERVER_ARGS=" + json.dumps(server_args, sort_keys=True, default=str))
        print(
            "[AREAL] EFFECTIVE_GENERATION_CONFIG="
            + json.dumps(
                asdict(self.areal_workflow_kwargs["gconfig"]),
                sort_keys=True,
                default=str,
            )
        )
        training_role = str(config_get(self.config, "async_training.areal.role", "areal_rollout"))
        validation_controller = None
        validation_config = None
        try:
            await asyncio.to_thread(
                controller.initialize,
                role=training_role,
                server_args=server_args,
            )
            if strict_topology:
                _validate_strict_6_2_topology(
                    live_gpu_counts=live_gpu_counts,
                    requested_rollout_gpus=requested_gpus,
                    available_gpus=int(ray.available_resources().get("GPU", 0)),
                    initialized=True,
                )
            if _areal_validation_is_enabled(self.config):
                validation_config = self._build_validation_inference_config(
                    inference_config,
                    training_role=training_role,
                )
                self.areal_validation_workflow_kwargs = self._build_validation_workflow_kwargs()
                validation_controller, validation_server_args = self._build_server(
                    validation_config,
                    allocation,
                )
                await asyncio.to_thread(
                    validation_controller.initialize,
                    role=f"{training_role}_validation",
                    server_args=validation_server_args,
                    server_infos=controller.server_infos,
                )
        except BaseException:
            if validation_controller is not None:
                try:
                    await asyncio.to_thread(validation_controller.destroy)
                except BaseException:
                    pass
            try:
                await asyncio.to_thread(controller.destroy)
            except BaseException:
                pass
            try:
                await asyncio.to_thread(self.areal_scheduler.delete_workers)
            except BaseException:
                pass
            try:
                await asyncio.to_thread(name_resolve.reset)
            except BaseException:
                pass
            self.areal_scheduler = None
            self.areal_experiment_config = None
            raise

        self.areal_controller = controller
        self.areal_validation_controller = validation_controller
        self.areal_validation_inference_config = validation_config
        self.async_rollout_manager = controller
        SkipManager.init(self.config)

    async def set_max_required_samples(self):
        async with self.lock:
            staleness_capacity = int(
                self.required_samples
                * (self.staleness_threshold + 1)
                * self.config.async_training.trigger_parameter_sync_step
            )
            self.max_required_samples = _areal_generation_capacity(
                self.config,
                required_samples=self.required_samples,
                staleness_capacity=staleness_capacity,
            )
            self.total_train_steps = int(
                self.total_rollout_steps
                / (self.required_samples * self.config.async_training.trigger_parameter_sync_step)
            )
            configured_concurrency = int(config_get(self.config, "async_training.areal.max_concurrent_rollouts", 8))
            if configured_concurrency < 1:
                raise ValueError("async_training.areal.max_concurrent_rollouts must be positive")
            self.max_concurrent_samples = min(configured_concurrency, self.max_required_samples)
            self.max_queue_size = self.max_required_samples

    def get_replicas(self):
        return []

    def get_all_hybrid_replicas(self) -> dict:
        return {}

    def get_hybrid_replicas_info(self) -> list[dict]:
        return []

    def get_num_active_replicas(self) -> int:
        if self.areal_controller is None:
            return 0
        return len(self.areal_controller.workers)

    async def add_replicas(self, resource_ids: list[str]) -> int:
        raise RuntimeError("AReaL rollout does not support VERL hybrid replica scaling")

    async def remove_replicas(self, resource_ids: list[str]) -> int:
        raise RuntimeError("AReaL rollout does not support VERL hybrid replica scaling")

    async def get_hybrid_statistics(self) -> dict:
        return await self.get_statistics()

    async def _start_profiling(self):
        return None

    async def _stop_profiling(self):
        return None

    async def _generate_validation_rewards(self, prompt_batch: DataProto) -> list[float]:
        if self.areal_validation_controller is None:
            raise RuntimeError("AReaL validation controller is not initialized")
        if self.areal_validation_workflow_kwargs is None:
            raise RuntimeError("AReaL validation workflow is not initialized")

        task_data = build_areal_task_data(prompt_batch)
        task_data.update(self.areal_reward_metadata)
        samples_per_prompt = _areal_validation_group_size(self.config)
        request_timeout = float(config_get(self.config, "async_training.areal.request_timeout", 3600.0))
        group_retries = int(
            config_get(
                self.config,
                "async_training.areal.group_retries",
                config_get(self.config, "async_training.areal.request_retries", 3),
            )
        )
        trajectory = await submit_areal_prompt_group(
            self.areal_validation_controller,
            data=task_data,
            workflow=self.areal_workflow,
            workflow_kwargs=self.areal_validation_workflow_kwargs,
            group_size=samples_per_prompt,
            timeout=request_timeout,
            retries=group_retries,
            is_eval=True,
        )
        if trajectory is None:
            raise RuntimeError("AReaL validation dropped a prompt group without rewards")
        async with _localized_areal_trajectory(trajectory) as localized_trajectory:
            return _extract_areal_validation_rewards(
                localized_trajectory,
                expected_group_size=samples_per_prompt,
            )

    async def _validate_areal(self) -> dict[str, float]:
        if self.areal_controller is None or self.areal_validation_controller is None:
            raise RuntimeError("AReaL validation controller is not initialized")

        # Both controller clients address the same inference servers. Align the
        # eval dispatcher's local version/capacity bookkeeping with the policy
        # version already published by the training controller.
        training_version = int(await asyncio.to_thread(self.areal_controller.get_version))
        validation_version = int(await asyncio.to_thread(self.areal_validation_controller.get_version))
        if validation_version != training_version:
            await asyncio.to_thread(
                self.areal_validation_controller.set_version,
                training_version,
            )

        samples_per_prompt = _areal_validation_group_size(self.config)
        max_concurrency = _areal_validation_concurrency(self.config)
        all_data_sources: list[str] = []
        all_sample_uids: list[str] = []
        all_rewards: list[float] = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            data_sources, sample_uids, rewards = await _collect_areal_validation_batch(
                test_batch,
                samples_per_prompt=samples_per_prompt,
                max_concurrency=max_concurrency,
                generate_rewards=self._generate_validation_rewards,
            )
            all_data_sources.extend(data_sources)
            all_sample_uids.extend(sample_uids)
            all_rewards.extend(rewards)

        if not all_rewards:
            raise RuntimeError("AReaL validation dataloader produced no prompt rewards")
        return self._val_metrics_update(
            np.asarray(all_data_sources, dtype=object),
            all_sample_uids,
            # VERL's registered batch reward manager always publishes ``acc``
            # equal to the scalar score. Preserve that canonical validation
            # schema when the same scorer is reached through AReaL.
            {"reward": all_rewards, "acc": all_rewards},
            [],
        )

    async def do_validate(self):
        """Run native AReaL validation without VERL hybrid rollout replicas."""

        if self._validation_lock is None:
            raise RuntimeError("AReaL validation was requested before worker initialization")
        async with self._validation_lock:
            # Keep all responses in one validation pass on one published policy
            # even if an unexpected concurrent sync RPC reaches this actor.
            async with self._weight_sync_lock:
                timing_raw: dict[str, float] = {}
                with marked_timer("rollouter/validate_time", timing_raw, color="green"):
                    val_metrics = await self._validate_areal()
        return timing_raw | val_metrics

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        if self.areal_controller is None:
            raise RuntimeError("AReaL controller is not initialized")

        task_data = build_areal_task_data(rollout_sample.full_batch)
        task_data.update(self.areal_reward_metadata)
        request_timeout = float(config_get(self.config, "async_training.areal.request_timeout", 3600.0))
        group_retries = int(
            config_get(
                self.config,
                "async_training.areal.group_retries",
                config_get(self.config, "async_training.areal.request_retries", 3),
            )
        )
        if group_retries < 0:
            raise ValueError(f"async_training.areal.group_retries must be non-negative, got {group_retries}")
        started = time.monotonic()
        trajectory = await submit_areal_prompt_group(
            self.areal_controller,
            data=task_data,
            workflow=self.areal_workflow,
            workflow_kwargs=self.areal_workflow_kwargs,
            group_size=self.group_size,
            timeout=request_timeout,
            retries=group_retries,
        )

        if trajectory is None:
            self.dropped_stale_samples += 1
            self.processed_sample_count += 1
            return

        async with _localized_areal_trajectory(trajectory) as localized_trajectory:
            rollout_sample.full_batch = areal_trajectory_to_dataproto(
                localized_trajectory,
                rollout_sample.full_batch,
                group_size=self.group_size,
                prompt_length=self.prompt_length,
                response_length=self.response_length,
                pad_token_id=self.pad_token_id,
                sample_id=rollout_sample.sample_id,
                generation_seconds=time.monotonic() - started,
                incorrect_reward=float(config_get(self.config, "async_training.areal.incorrect_reward", -5.0)),
                correct_reward=float(config_get(self.config, "async_training.areal.correct_reward", 5.0)),
                eos_token_id=self.tokenizer.eos_token_id,
                overlong_buffer_length=(
                    int(config_get(self.config, "reward.reward_kwargs.overlong_buffer_cfg.len", 0))
                    if config_get(self.config, "reward.reward_kwargs.overlong_buffer_cfg.enable", False)
                    else None
                ),
                overlong_penalty_factor=float(
                    config_get(self.config, "reward.reward_kwargs.overlong_buffer_cfg.penalty_factor", 1.0)
                ),
            )
        if len(rollout_sample.full_batch) != self.group_size:
            raise RuntimeError(
                f"AReaL queue item must contain exactly {self.group_size} trajectories, "
                f"got {len(rollout_sample.full_batch)}"
            )
        rollout_sample.rollout_status = await self.get_statistics()
        success = await self.message_queue_client.put_sample(sample=ray.cloudpickle.dumps(rollout_sample))
        if success:
            self.total_generated_samples += 1
        else:
            self.dropped_stale_samples += 1
        self.processed_sample_count += 1

    async def sync_weights(self, version: int, hf_path: str) -> dict[str, float]:
        """Pause, atomically publish, and load one disposable HF export."""

        if self.areal_controller is None or self.areal_inference_config is None:
            raise RuntimeError("AReaL controller is not initialized")
        if self._weight_sync_failed:
            raise RuntimeError("A previous AReaL weight update failed; restart the backend before continuing")
        version = int(version)
        export_path = validate_transient_hf_export(self.config, version=version, hf_path=hf_path)
        source_model = Path(str(config_get(self.config, "actor_rollout_ref.model.path", ""))).expanduser()
        if source_model.exists() and source_model.resolve() == export_path:
            raise ValueError("Refusing to pass the persistent source model to AReaL's destructive disk loader")

        async with self._weight_sync_lock:
            old_version = int(self.areal_controller.get_version())
            if version < old_version:
                raise ValueError(f"Refusing to downgrade AReaL weights from version {old_version} to {version}")
            initial_weight_sync = not self._initial_weight_sync_complete
            started = time.monotonic()
            await asyncio.to_thread(self.areal_controller.pause)
            try:
                meta = WeightUpdateMeta(
                    type="disk",
                    path=str(export_path),
                    clear_checkpoint_after_load=False,
                    version=version,
                )
                update_key = areal_weight_update_key(
                    self.areal_inference_config.experiment_name,
                    self.areal_inference_config.trial_name,
                    old_version,
                )
                await _run_areal_disk_update(
                    self.areal_controller,
                    meta,
                    lambda: name_resolve.add(
                        update_key,
                        str(datetime.now().timestamp()),
                        keepalive_ttl=120,
                        replace=True,
                    ),
                )
                await asyncio.to_thread(
                    _set_areal_controller_version_after_disk_update,
                    self.areal_controller,
                    version,
                    initial_weight_sync=initial_weight_sync,
                )
                await asyncio.to_thread(self.areal_controller.resume)
                self._initial_weight_sync_complete = True
            except BaseException:
                # Fail closed: resuming after a partial load could mix behavior
                # policies within a GGPO prompt group.
                self._weight_sync_failed = True
                raise

        return {
            "fully_async/areal/weight_sync_seconds": time.monotonic() - started,
            "fully_async/areal/old_version": float(old_version),
            "fully_async/areal/new_version": float(version),
        }

    async def reset_staleness(self, version: int | None = None):
        if version is not None and self.areal_controller is not None:
            controller_version = int(self.areal_controller.get_version())
            if controller_version != int(version):
                raise RuntimeError(
                    f"AReaL controller version {controller_version} does not match learner version {version}"
                )
        return await super().reset_staleness(version=version)

    async def shutdown(self):
        """Idempotently stop producer tasks and release all AReaL Ray workers."""

        if self._areal_shutdown:
            return
        self._areal_shutdown = True
        await self.request_stop()

        controller = self.areal_controller
        validation_controller = self.areal_validation_controller
        scheduler = self.areal_scheduler
        self.areal_controller = None
        self.areal_validation_controller = None
        self.async_rollout_manager = None
        self.areal_scheduler = None
        self.areal_validation_inference_config = None
        self.areal_experiment_config = None
        try:
            if validation_controller is not None:
                await asyncio.to_thread(validation_controller.destroy)
        finally:
            try:
                if controller is not None:
                    await asyncio.to_thread(controller.destroy)
            finally:
                try:
                    # Also clean partially initialized roles/placement groups.
                    # This is harmless after both controllers removed roles.
                    if scheduler is not None:
                        await asyncio.to_thread(scheduler.delete_workers)
                finally:
                    await asyncio.to_thread(name_resolve.reset)
