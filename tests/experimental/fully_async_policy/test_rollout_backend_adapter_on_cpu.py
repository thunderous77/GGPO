# Copyright 2026 OpenAI
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
import copy
import inspect
import logging
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import areal.infra.controller.rollout_controller as areal_rollout_controller_module
import areal.infra.platforms as areal_platforms
import numpy as np
import pytest
import ray
import torch
from areal.api import AsyncRewardWrapper
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import InferenceEngineConfig, SchedulingSpec, vLLMConfig
from areal.infra.controller.rollout_controller import (
    RolloutController,
    _ranked_server_args,
)
from areal.infra.rpc.rtensor import (
    RTensor,
    TensorShardInfo,
    clear_fetch_buffer,
    fetch_buffer_stats,
    set_backend,
)
from areal.infra.staleness_manager import StalenessManager
from areal.infra.utils import ray_placement_group as areal_ray_placement_group
from areal.infra.workflow_executor import BatchTaskDispatcher
from areal.workflow.rlvr import RLVRWorkflow, decode_reward_completion
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouterBase
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.detach_utils import RolloutSample
from verl.experimental.fully_async_policy.rollout_backend.adapter import (
    REQUIRE_COMPLETE_GROUP_KEY,
    areal_trajectory_to_dataproto,
    build_areal_task_data,
)
from verl.experimental.fully_async_policy.rollout_backend.areal_rollouter import (
    _areal_consumer_batch_size,
    _areal_generation_capacity,
    _areal_staleness_batch_size,
    _areal_validation_is_enabled,
    _build_areal_generation_config,
    _build_areal_vllm_server_args,
    _canonical_enable_thinking,
    _collect_areal_validation_batch,
    _extract_areal_validation_rewards,
    _localized_areal_trajectory,
    _resolve_areal_reward,
    _resolve_areal_reward_executor_config,
    _run_areal_disk_update,
    _set_areal_controller_version_after_disk_update,
    _validate_areal_allocation,
)
from verl.experimental.fully_async_policy.rollout_backend.capabilities import (
    AREAL_CAPABILITIES,
    VERL_CAPABILITIES,
)
from verl.experimental.fully_async_policy.rollout_backend.dispatch import (
    extract_task_trajectory,
    submit_areal_prompt_group,
)
from verl.experimental.fully_async_policy.rollout_backend.factory import (
    config_get,
    resolve_rollout_backend_name,
)
from verl.experimental.fully_async_policy.rollout_backend.reward_bridge import (
    AREAL_BATCH_REWARD_FN,
    AREAL_CUSTOM_REWARD_FN,
    CUSTOM_REWARD_KWARGS_KEY,
    CUSTOM_REWARD_NAME_KEY,
    CUSTOM_REWARD_PATH_KEY,
    areal_batch_reward_fn,
    areal_custom_reward_fn,
)
from verl.experimental.fully_async_policy.rollout_backend.scheduler import build_areal_ray_scheduler
from verl.experimental.fully_async_policy.rollout_backend.vendored import (
    load_vendored_areal,
    vendored_areal_root,
)
from verl.experimental.fully_async_policy.rollout_backend.weight_sync import (
    areal_weight_update_key,
    validate_transient_hf_export,
)
from verl.protocol import DataProto
from verl.utils import reward_score
from verl.utils.reward_score import math_verify


def test_areal_staleness_capacity_matches_one_published_policy_version():
    assert _areal_consumer_batch_size(2, 4) == 8
    assert _areal_consumer_batch_size(2, 1) == 2


def test_reference_async_batch_geometry_is_prompt_group_atomic():
    actor_mini_batch_prompt_groups = 32
    require_batches = 8
    responses_per_prompt = 8
    required_prompt_groups = actor_mini_batch_prompt_groups * require_batches

    assert required_prompt_groups == 256
    assert required_prompt_groups * responses_per_prompt == 2048
    assert _areal_consumer_batch_size(required_prompt_groups, 1) == 256


def test_areal_filter_refill_capacity_uses_synchronous_generation_cap():
    config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": {"enable": True, "max_num_gen_batches": 32}},
            "async_training": {"trigger_parameter_sync_step": 1},
        }
    )

    assert _areal_generation_capacity(config, required_samples=256, staleness_capacity=281) == 8192
    assert _areal_staleness_batch_size(config, required_samples=256) == 8192


def test_areal_unfiltered_capacity_preserves_async_staleness_bound():
    config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": {"enable": False}},
            "async_training": {"trigger_parameter_sync_step": 1},
        }
    )

    assert _areal_generation_capacity(config, required_samples=256, staleness_capacity=281) == 281
    assert _areal_staleness_batch_size(config, required_samples=256) == 256


@pytest.mark.parametrize(
    ("val_before_train", "test_freq", "expected"),
    [
        (True, -1, True),
        (False, 1, True),
        (False, -1, False),
    ],
)
def test_native_areal_validation_controller_is_created_when_needed(
    val_before_train,
    test_freq,
    expected,
):
    config = {
        "trainer": {
            "val_before_train": val_before_train,
            "test_freq": test_freq,
        }
    }

    assert _areal_validation_is_enabled(config) is expected


@pytest.mark.parametrize(
    ("required_samples", "trigger_parameter_sync_step", "error_type"),
    [
        (0, 1, ValueError),
        (-1, 1, ValueError),
        (1, 0, ValueError),
        (1, -1, ValueError),
        (True, 1, TypeError),
        (1, True, TypeError),
        (1.0, 1, TypeError),
        (1, 1.0, TypeError),
    ],
)
def test_areal_staleness_capacity_rejects_invalid_batch_inputs(
    required_samples,
    trigger_parameter_sync_step,
    error_type,
):
    with pytest.raises(error_type, match="positive integer"):
        _areal_consumer_batch_size(required_samples, trigger_parameter_sync_step)


class _VersionedController:
    def __init__(self, *, consumer_batch_size: int, max_staleness: int):
        self.version = 0
        self.staleness_manager = StalenessManager(
            version_provider=self,
            max_concurrent_rollouts=128,
            consumer_batch_size=consumer_batch_size,
            max_staleness=max_staleness,
        )

    def get_version(self):
        return self.version

    def set_version(self, version):
        self.version = version


def _accept_areal_tasks(controller: _VersionedController, count: int) -> None:
    for _ in range(count):
        controller.staleness_manager.on_rollout_enqueued()
        controller.staleness_manager.on_rollout_submitted()
        controller.staleness_manager.on_rollout_accepted()


def test_initial_checkpoint_version_jump_rebases_areal_accepted_count():
    consumer_batch_size = _areal_consumer_batch_size(2, 4)
    controller = _VersionedController(consumer_batch_size=consumer_batch_size, max_staleness=0)

    _set_areal_controller_version_after_disk_update(
        controller,
        5,
        initial_weight_sync=True,
    )

    assert controller.staleness_manager.get_stats().accepted == 5 * consumer_batch_size
    assert controller.staleness_manager.get_capacity() == consumer_batch_size


def test_normal_version_update_does_not_rebase_prefetched_areal_tasks():
    consumer_batch_size = _areal_consumer_batch_size(2, 4)
    controller = _VersionedController(consumer_batch_size=consumer_batch_size, max_staleness=1)
    _set_areal_controller_version_after_disk_update(
        controller,
        0,
        initial_weight_sync=True,
    )
    _accept_areal_tasks(controller, 2 * consumer_batch_size)

    _set_areal_controller_version_after_disk_update(
        controller,
        1,
        initial_weight_sync=False,
    )

    assert controller.staleness_manager.get_stats().accepted == 2 * consumer_batch_size
    assert controller.staleness_manager.get_capacity() == consumer_batch_size


def test_initial_checkpoint_version_jump_requires_initialized_staleness_manager():
    controller = SimpleNamespace(staleness_manager=None, set_version=lambda version: None)

    with pytest.raises(RuntimeError, match="staleness manager"):
        _set_areal_controller_version_after_disk_update(
            controller,
            3,
            initial_weight_sync=True,
        )


def _object_array(*values):
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def _source_group() -> DataProto:
    messages = [{"role": "user", "content": "What is 1 + 1?"}]
    return DataProto.from_dict(
        tensors={"source_row": torch.tensor([[0], [1]])},
        non_tensors={
            "raw_prompt": _object_array(messages, copy.deepcopy(messages)),
            "reward_model": _object_array(
                {"ground_truth": "2", "style": "rule"},
                {"ground_truth": "2", "style": "rule"},
            ),
            "data_source": np.array(["math", "math"], dtype=object),
        },
    )


def _complete_trajectory() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(
            [
                [11, 12, 21, 22, 0],
                [13, 23, 24, 25, 0],
            ],
            dtype=torch.int32,
        ),
        "loss_mask": torch.tensor(
            [
                [0, 0, 1, 1, 0],
                [0, 1, 1, 1, 0],
            ],
            dtype=torch.int32,
        ),
        "logprobs": torch.tensor(
            [
                [0.0, 0.0, -0.1, -0.2, 0.0],
                [0.0, -0.3, -0.4, -0.5, 0.0],
            ]
        ),
        "versions": torch.tensor(
            [
                [-1, -1, 3, 4, 0],
                [-1, 4, 4, 4, 0],
            ],
            dtype=torch.int32,
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
            ],
            dtype=torch.bool,
        ),
        "rewards": torch.tensor([1.0, 0.0]),
    }


def test_factory_resolution_is_lazy_and_fails_closed_for_unknown_backends():
    config = {"async_training": {"rollout_backend": "A-ReaL"}}
    assert config_get(config, "async_training.rollout_backend") == "A-ReaL"
    assert resolve_rollout_backend_name(config) == "areal"
    assert resolve_rollout_backend_name("native") == "verl"

    with pytest.raises(ValueError, match="Unknown fully-async rollout backend"):
        resolve_rollout_backend_name("silent-fallback")


def test_capabilities_make_backend_limitations_explicit():
    assert VERL_CAPABILITIES.weight_sync == "verl_checkpoint_engine"
    assert VERL_CAPABILITIES.hybrid_validation is True
    assert AREAL_CAPABILITIES.weight_sync == "hf_disk"
    assert AREAL_CAPABILITIES.hybrid_validation is False
    assert AREAL_CAPABILITIES.atomic_prompt_groups is True


def test_vendored_source_is_resolved_inside_this_repository():
    root = vendored_areal_root()
    module = load_vendored_areal()
    assert root.name == "AReaL"
    assert root.parent.name == "third_party"
    assert (root / "areal" / "__init__.py").is_file()
    assert (root / "areal") in Path(module.__file__).resolve().parents


def test_areal_ray_scheduler_detects_cluster_gpu_from_cpu_controller(monkeypatch):
    monkeypatch.setattr(areal_platforms, "is_npu_available", False)
    monkeypatch.setattr(areal_ray_placement_group.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(areal_ray_placement_group.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        areal_ray_placement_group.ray,
        "cluster_resources",
        lambda: {"CPU": 32.0, "GPU": 2.0},
    )

    assert areal_ray_placement_group.ray_resource_type() == "GPU"
    assert areal_ray_placement_group._bundle_spec(cpu=4, gpu=1, mem=32) == {
        "CPU": 4,
        "GPU": 1.0,
        "memory": 32 * 1024**3,
    }
    assert areal_ray_placement_group._actor_resource_spec(cpu=4, gpu=1, mem=32) == {
        "num_cpus": 4,
        "num_gpus": 1.0,
        "memory": 32 * 1024**3,
    }


def test_areal_vllm_bridge_preserves_shared_max_num_batched_tokens():
    config = {
        "actor_rollout_ref": {
            "rollout": {
                "seed": 42,
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.6,
                "max_num_seqs": 256,
                "max_num_batched_tokens": 8192,
                "enforce_eager": True,
            }
        }
    }
    areal_config = {
        "seed": 42,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.6,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "enforce_eager": True,
    }
    allocation = ModelAllocation.from_str("vllm:d4p1t1")

    server_args = _build_areal_vllm_server_args(
        config,
        areal_config,
        model_path="/models/qwen3-1.7b-base",
        max_model_len=4096,
        allocation=allocation,
    )

    assert server_args["max_num_batched_tokens"] == 8192
    assert server_args["seed"] == 42
    assert server_args["dtype"] == "bfloat16"
    assert server_args["gpu_memory_utilization"] == 0.6
    assert server_args["max_num_seqs"] == 256
    assert server_args["enforce_eager"] is True
    assert server_args["tensor_parallel_size"] == 1
    assert server_args["pipeline_parallel_size"] == 1
    server_command = vLLMConfig.build_cmd_from_args(server_args)
    token_budget_flag = server_command.index("--max-num-batched-tokens")
    assert server_command[token_budget_flag + 1] == "8192"


@pytest.mark.parametrize(
    "areal_config",
    [
        {"seed": 7},
        {"vllm": {"seed": 7}},
        {"gpu_memory_utilization": 0.5},
    ],
)
def test_areal_vllm_conflicting_duplicate_engine_settings_fail_closed(areal_config):
    config = {
        "actor_rollout_ref": {
            "rollout": {
                "seed": 42,
                "gpu_memory_utilization": 0.6,
                "max_num_seqs": 256,
                "max_num_batched_tokens": 8192,
                "enforce_eager": True,
                "dtype": "bfloat16",
            }
        }
    }
    with pytest.raises(ValueError, match="Conflicting AReaL engine option"):
        _build_areal_vllm_server_args(
            config,
            areal_config,
            model_path="/models/qwen3-1.7b-base",
            max_model_len=4096,
            allocation=ModelAllocation.from_str("vllm:d4p1t1"),
        )


def test_areal_vllm_unknown_nested_engine_setting_fails_closed():
    with pytest.raises(ValueError, match="Unknown async_training.areal.vllm"):
        _build_areal_vllm_server_args(
            {"actor_rollout_ref": {"rollout": {"seed": 42}}},
            {"vllm": {"typo_seed": 42}},
            model_path="/models/qwen3-1.7b-base",
            max_model_len=4096,
            allocation=ModelAllocation.from_str("vllm:d4p1t1"),
        )


def test_common_sampling_and_thinking_map_to_areal_effective_args():
    config = {
        "actor_rollout_ref": {
            "rollout": {
                "do_sample": False,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "ignore_eos": False,
            }
        },
        "data": {"apply_chat_template_kwargs": {"enable_thinking": True}},
        "async_training": {"areal": {"enable_thinking": True}},
    }
    generation = _build_areal_generation_config(config, prompt_length=2048, response_length=2048)
    assert generation.greedy is True
    assert generation.max_tokens == 4096
    assert generation.max_new_tokens == 2048
    assert generation.top_k == int(1e8)
    assert generation.ignore_eos is False
    assert _canonical_enable_thinking(config) is True


def test_areal_allocation_must_consume_every_reserved_gpu():
    _validate_areal_allocation(
        ModelAllocation.from_str("vllm:d4p1t1"),
        configured_rollout_gpus=4,
        backend="vllm:d4p1t1",
    )
    with pytest.raises(ValueError, match="consume every reserved rollout GPU"):
        _validate_areal_allocation(
            ModelAllocation.from_str("vllm:d2p1t1"),
            configured_rollout_gpus=4,
            backend="vllm:d2p1t1",
        )


def test_areal_vllm_d4_controller_requests_four_gpus(monkeypatch):
    captured = {}
    config = InferenceEngineConfig(
        backend="vllm:d4p1t1",
        consumer_batch_size=8,
        max_concurrent_rollouts=8,
        scheduling_spec=(
            SchedulingSpec(
                cpu=4,
                gpu=1,
                mem=32,
                cmd="python -m areal.infra.rpc.rpc_server",
                ray_placement_strategy="separate",
            ),
        ),
    )
    controller = RolloutController(
        inf_engine=SimpleNamespace,
        config=config,
        scheduler=SimpleNamespace(),
    )

    def capture_job(async_fn, job, server_args, *args, **kwargs):
        del async_fn, server_args, args, kwargs
        captured["job"] = job

    monkeypatch.setattr(areal_rollout_controller_module, "run_async_task", capture_job)
    monkeypatch.setattr(
        areal_rollout_controller_module.BatchTaskDispatcher,
        "initialize",
        lambda self, logger: None,
    )
    monkeypatch.setattr(controller, "_start_callback_server", lambda: None)

    controller.initialize(role="rollout", server_args={"max_num_batched_tokens": 8192})

    job = captured["job"]
    assert controller.rollout_alloc.parallel.world_size == 4
    assert job.replicas == 4
    assert len(job.tasks) == 4
    assert [task.gpu for task in job.tasks] == [1, 1, 1, 1]
    assert sum(task.gpu for task in job.tasks) == 4


@pytest.mark.parametrize(
    ("backend", "executor_backend", "seed_key", "expected_launch_concurrency"),
    [
        pytest.param("vllm:d4p1t1", "mp", "seed", 1, id="vllm-mp-serialized"),
        pytest.param("vllm:d4p1t1", "uni", "seed", 4, id="vllm-uni-concurrent"),
        pytest.param("sglang:d4p1t1", None, "random_seed", 4, id="sglang-concurrent"),
    ],
)
def test_areal_launch_server_derives_and_reports_per_rank_seeds(
    monkeypatch, backend, executor_backend, seed_key, expected_launch_concurrency
):
    reported_engine_seeds = []

    def capture_engine_seed_log(message, *args, **kwargs):
        del kwargs
        if message == "AREAL_ENGINE_SEEDS=%s":
            reported_engine_seeds.append((message, args[0]))

    monkeypatch.setattr(
        areal_rollout_controller_module.logger,
        "info",
        capture_engine_seed_log,
    )

    class FakeScheduler:
        def __init__(self):
            self.workers = [SimpleNamespace(id=f"worker-{rank}") for rank in range(4)]
            self.launches = {}
            self.active_launches = 0
            self.max_active_launches = 0
            self.active_initializations = 0
            self.max_active_initializations = 0

        def create_workers(self, job):
            del job
            return [worker.id for worker in self.workers]

        def get_workers(self, role):
            assert role == "rollout"
            return self.workers

        async def create_engine(self, **kwargs):
            del kwargs

        async def async_call_engine(self, *, method, engine_name, **kwargs):
            if method == "launch_server":
                self.active_launches += 1
                self.max_active_launches = max(self.max_active_launches, self.active_launches)
                try:
                    await asyncio.sleep(0)
                    self.launches[engine_name] = kwargs["server_args"]
                    return SimpleNamespace(host="127.0.0.1", port=8000)
                finally:
                    self.active_launches -= 1
            assert method == "initialize"
            self.active_initializations += 1
            self.max_active_initializations = max(self.max_active_initializations, self.active_initializations)
            try:
                await asyncio.sleep(0)
            finally:
                self.active_initializations -= 1

    scheduler = FakeScheduler()
    controller = RolloutController(
        inf_engine=SimpleNamespace,
        config=SimpleNamespace(backend=backend),
        scheduler=scheduler,
    )
    controller._worker_role = "rollout"
    server_args = {seed_key: 42, "max_num_batched_tokens": 8192}
    if executor_backend is not None:
        server_args["distributed_executor_backend"] = executor_backend

    asyncio.run(
        controller._async_initialize(
            job=SimpleNamespace(role="rollout"),
            server_args=server_args,
        )
    )

    expected_server_args = {seed_key: 42, "max_num_batched_tokens": 8192}
    if executor_backend is not None:
        expected_server_args["distributed_executor_backend"] = executor_backend
    assert server_args == expected_server_args
    assert [scheduler.launches[f"rollout/{rank}"][seed_key] for rank in range(4)] == [42, 43, 44, 45]
    assert all(scheduler.launches[f"rollout/{rank}"]["max_num_batched_tokens"] == 8192 for rank in range(4))
    assert scheduler.max_active_launches == expected_launch_concurrency
    assert scheduler.max_active_initializations == 4
    assert reported_engine_seeds == [("AREAL_ENGINE_SEEDS=%s", "42,43,44,45")]


@pytest.mark.parametrize("seed_key", ["seed", "random_seed"])
def test_ranked_server_args_rejects_non_integer_seed(seed_key):
    with pytest.raises(TypeError, match=rf"^{seed_key} must be an integer"):
        _ranked_server_args({seed_key: "42"}, rank=0)


def test_ranked_server_args_rejects_ambiguous_seed_fields():
    with pytest.raises(ValueError, match="not both"):
        _ranked_server_args({"seed": 42, "random_seed": 42}, rank=0)


def test_areal_ray_scheduler_rejects_gpu_request_on_cpu_only_cluster(monkeypatch):
    monkeypatch.setattr(areal_platforms, "is_npu_available", False)
    monkeypatch.setattr(areal_ray_placement_group.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(areal_ray_placement_group.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        areal_ray_placement_group.ray,
        "cluster_resources",
        lambda: {"CPU": 32.0},
    )

    assert areal_ray_placement_group.ray_resource_type() == "CPU"
    with pytest.raises(ValueError, match="Current detected device is CPU"):
        areal_ray_placement_group._bundle_spec(cpu=4, gpu=1, mem=32)


def test_ray_scheduler_has_experiment_context_and_vendored_worker_pythonpath(tmp_path):
    config = {
        "async_training": {"areal": {"setup_timeout": 17.0}},
        "rollout": {"nnodes": 2, "n_gpus_per_node": 4},
    }
    inference_config = InferenceEngineConfig(
        experiment_name="ggpo",
        trial_name="areal-smoke",
        fileroot=str(tmp_path / "areal-runs"),
        tokenizer_path="/models/test",
        scheduling_spec=(
            SchedulingSpec(
                cpu=1,
                gpu=1,
                mem=1,
                cmd="python -m areal.infra.rpc.rpc_server",
                ray_placement_strategy="separate",
            ),
        ),
        backend="vllm:d1",
    )

    experiment_config, scheduler = build_areal_ray_scheduler(config, inference_config)

    assert scheduler.exp_config is experiment_config
    assert scheduler.startup_timeout == 17.0
    assert experiment_config.cluster.cluster_name == "ray"
    assert experiment_config.cluster.n_nodes == 2
    assert experiment_config.cluster.n_gpus_per_node == 4
    worker_env = scheduler._build_env_vars(inference_config.scheduling_spec[0])
    assert str(vendored_areal_root()) in worker_env["PYTHONPATH"].split(os.pathsep)


def test_exact_areal_task_wrapper_and_atomic_group_submission():
    class FakeDispatcher:
        def __init__(self):
            self.waited = []

        def wait_for_task(self, task_id, timeout, raise_timeout):
            self.waited.append((task_id, timeout, raise_timeout))
            return SimpleNamespace(task_id=task_id, trajectory={"group": "complete"})

        def cancel_task(self, task_id):
            raise AssertionError(f"unexpected cancellation for task {task_id}")

    class FakeController:
        def __init__(self):
            self.dispatcher = FakeDispatcher()
            self.submissions = []

        def submit(self, **kwargs):
            self.submissions.append(kwargs)
            return 41

    controller = FakeController()
    trajectory = asyncio.run(
        submit_areal_prompt_group(
            controller,
            data={"messages": []},
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs={"reward_fn": "test.reward"},
            group_size=4,
            timeout=9.0,
            retries=0,
        )
    )

    assert trajectory == {"group": "complete"}
    assert controller.submissions[0]["group_size"] == 4
    assert controller.submissions[0]["is_eval"] is False
    assert controller.dispatcher.waited == [(41, 9.0, True)]

    asyncio.run(
        submit_areal_prompt_group(
            controller,
            data={"messages": []},
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs={"reward_fn": "test.reward"},
            group_size=3,
            timeout=9.0,
            retries=0,
            is_eval=True,
        )
    )
    assert controller.submissions[1]["is_eval"] is True

    with pytest.raises(RuntimeError, match="returned task 42"):
        extract_task_trajectory(
            SimpleNamespace(task_id=42, trajectory={}),
            expected_task_id=41,
        )
    with pytest.raises(TypeError, match="task result wrapper"):
        extract_task_trajectory({"trajectory": {}}, expected_task_id=41)


def test_cancelled_areal_submission_cancels_the_exact_backend_task():
    wait_started = threading.Event()
    wait_released = threading.Event()

    class FakeDispatcher:
        def __init__(self):
            self.cancelled = []

        def wait_for_task(self, task_id, timeout, raise_timeout):
            del timeout, raise_timeout
            wait_started.set()
            if not wait_released.wait(timeout=1.0):
                raise TimeoutError(f"test did not cancel AReaL task {task_id}")
            return SimpleNamespace(task_id=task_id, trajectory=None)

        def cancel_task(self, task_id):
            self.cancelled.append(task_id)
            wait_released.set()

    class FakeController:
        def __init__(self):
            self.dispatcher = FakeDispatcher()

        def submit(self, **kwargs):
            del kwargs
            return 73

    controller = FakeController()

    async def cancel_submission():
        submission = asyncio.create_task(
            submit_areal_prompt_group(
                controller,
                data={"messages": []},
                workflow="areal.workflow.rlvr.RLVRWorkflow",
                workflow_kwargs={"reward_fn": "test.reward"},
                group_size=4,
                timeout=9.0,
                retries=0,
            )
        )
        started = await asyncio.to_thread(wait_started.wait, 1.0)
        assert started
        submission.cancel()
        with pytest.raises(asyncio.CancelledError):
            await submission

    asyncio.run(cancel_submission())

    assert controller.dispatcher.cancelled == [73]
    assert wait_released.is_set()


def test_cancelled_areal_submit_waits_for_task_id_then_cancels_exact_task():
    submit_started = threading.Event()
    submit_released = threading.Event()

    class FakeDispatcher:
        def __init__(self):
            self.cancelled = []

        def wait_for_task(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("cancelled submit must not begin waiting for a result")

        def cancel_task(self, task_id):
            self.cancelled.append(task_id)

    class FakeController:
        def __init__(self):
            self.dispatcher = FakeDispatcher()

        def submit(self, **kwargs):
            del kwargs
            submit_started.set()
            if not submit_released.wait(timeout=1.0):
                raise TimeoutError("test did not release blocked AReaL submit")
            return 91

    controller = FakeController()

    async def cancel_blocked_submit():
        submission = asyncio.create_task(
            submit_areal_prompt_group(
                controller,
                data={"messages": []},
                workflow="areal.workflow.rlvr.RLVRWorkflow",
                workflow_kwargs={"reward_fn": "test.reward"},
                group_size=4,
                timeout=9.0,
                retries=0,
            )
        )
        started = await asyncio.to_thread(submit_started.wait, 1.0)
        assert started
        submission.cancel()
        await asyncio.sleep(0)
        assert not submission.done()
        submit_released.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(submission, timeout=1.0)

    try:
        asyncio.run(cancel_blocked_submit())
    finally:
        submit_released.set()

    assert controller.dispatcher.cancelled == [91]


def test_real_areal_dispatcher_pending_cancellation_unblocks_exact_task_waiter():
    class VersionProvider:
        @staticmethod
        def get_version():
            return 0

    def task_factory(_task_input):
        async def unexpected_run():
            raise AssertionError("paused dispatcher must keep this task pending")

        return unexpected_run

    dispatcher = BatchTaskDispatcher(
        max_queue_size=2,
        task_factory=task_factory,
        staleness_manager=StalenessManager(
            version_provider=VersionProvider(),
            max_concurrent_rollouts=1,
            consumer_batch_size=1,
            max_staleness=0,
        ),
    )
    dispatcher.initialize(logging.getLogger("areal-dispatch-pending-cancel-test"))
    dispatcher.pause()
    waiter_errors = []
    wait_entered = threading.Event()
    condition_wait = dispatcher._result_cv.wait

    def instrumented_condition_wait(timeout=None):
        wait_entered.set()
        return condition_wait(timeout=timeout)

    dispatcher._result_cv.wait = instrumented_condition_wait

    def wait_for_cancelled_task():
        try:
            dispatcher.wait_for_task(16, timeout=30.0)
        except BaseException as error:
            waiter_errors.append(error)

    try:
        dispatcher.submit_task_input(SimpleNamespace(task_id=16))
        with dispatcher._input_cv:
            assert [task.task_id for task in dispatcher._pending_inputs] == [16]
        waiter = threading.Thread(target=wait_for_cancelled_task)
        waiter.start()
        assert wait_entered.wait(timeout=1.0)
        assert dispatcher.cancel_task(16)
        waiter.join(timeout=1.0)
        assert not waiter.is_alive()
        with dispatcher._result_cv:
            assert 16 not in dispatcher._active_task_ids
        with dispatcher._input_cv:
            assert not dispatcher._pending_inputs
    finally:
        dispatcher.destroy()

    assert len(waiter_errors) == 1
    assert isinstance(waiter_errors[0], RuntimeError)
    assert "Task 16 was cancelled while waiting" in str(waiter_errors[0])


def test_real_areal_dispatcher_running_cancellation_unblocks_before_late_result():
    class VersionProvider:
        @staticmethod
        def get_version():
            return 0

    started = threading.Event()

    def task_factory(_task_input):
        async def run_until_cancelled():
            started.set()
            await asyncio.Event().wait()

        return run_until_cancelled

    dispatcher = BatchTaskDispatcher(
        max_queue_size=2,
        task_factory=task_factory,
        staleness_manager=StalenessManager(
            version_provider=VersionProvider(),
            max_concurrent_rollouts=1,
            consumer_batch_size=1,
            max_staleness=0,
        ),
    )
    allow_result_fetch = threading.Event()
    runner_wait = dispatcher.runner.wait

    def gated_runner_wait(*args, **kwargs):
        if not allow_result_fetch.wait(timeout=0.01):
            raise TimeoutError
        return runner_wait(*args, **kwargs)

    dispatcher.runner.wait = gated_runner_wait
    dispatcher.initialize(logging.getLogger("areal-dispatch-cancel-test"))
    waiter_errors = []
    wait_entered = threading.Event()
    condition_wait = dispatcher._result_cv.wait

    def instrumented_condition_wait(timeout=None):
        wait_entered.set()
        return condition_wait(timeout=timeout)

    dispatcher._result_cv.wait = instrumented_condition_wait

    def wait_for_cancelled_task():
        try:
            dispatcher.wait_for_task(17, timeout=30.0)
        except BaseException as error:
            waiter_errors.append(error)

    try:
        dispatcher.submit_task_input(SimpleNamespace(task_id=17))
        assert started.wait(timeout=1.0)
        waiter = threading.Thread(target=wait_for_cancelled_task)
        waiter.start()
        assert wait_entered.wait(timeout=1.0)
        assert dispatcher.cancel_task(17)
        waiter.join(timeout=1.0)
        assert not waiter.is_alive()
        with dispatcher._result_cv:
            assert 17 not in dispatcher._active_task_ids
            assert 17 in dispatcher._discard_result_task_ids
            assert 17 not in dispatcher._pending_results
        allow_result_fetch.set()
        with dispatcher._result_cv:
            assert dispatcher._result_cv.wait_for(
                lambda: 17 not in dispatcher._discard_result_task_ids,
                timeout=1.0,
            )
            assert 17 not in dispatcher._pending_results
    finally:
        allow_result_fetch.set()
        dispatcher.destroy()

    assert len(waiter_errors) == 1
    assert isinstance(waiter_errors[0], RuntimeError)
    assert "Task 17 was cancelled while waiting" in str(waiter_errors[0])


def test_full_math500_validation_is_bounded_and_keeps_only_reward_metrics():
    prompt_count = 500
    batch = DataProto.from_dict(
        tensors={"prompts": torch.arange(prompt_count).reshape(prompt_count, 1)},
        non_tensors={
            "data_source": np.array(["math"] * prompt_count, dtype=object),
            "raw_prompt": _object_array(
                *([{"role": "user", "content": f"problem-{index}"}] for index in range(prompt_count))
            ),
        },
    )
    active = 0
    max_active = 0

    async def generate_rewards(prompt_batch):
        nonlocal active, max_active
        assert len(prompt_batch) == 1
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
        return [0.0, 0.5, 1.0]

    data_sources, sample_uids, rewards = asyncio.run(
        _collect_areal_validation_batch(
            batch,
            samples_per_prompt=3,
            max_concurrency=7,
            generate_rewards=generate_rewards,
        )
    )

    assert max_active == 7
    assert len(data_sources) == len(sample_uids) == len(rewards) == prompt_count * 3
    assert len(set(sample_uids)) == prompt_count
    metrics = FullyAsyncRollouterBase._val_metrics_update(
        SimpleNamespace(),
        np.asarray(data_sources[:6], dtype=object),
        sample_uids[:6],
        {"reward": rewards[:6], "acc": rewards[:6]},
        [],
    )
    assert metrics["val-core/math/acc/mean@3"] == pytest.approx(0.5)
    assert metrics["val-aux/math/reward/mean@3"] == pytest.approx(0.5)
    assert not any(key.startswith("val-core/math/reward/") for key in metrics)


def test_areal_validation_failure_cancels_and_reaps_sibling_prompts():
    prompt_count = 3
    batch = DataProto.from_dict(
        tensors={"prompts": torch.arange(prompt_count).reshape(prompt_count, 1)},
        non_tensors={
            "data_source": np.array(["math"] * prompt_count, dtype=object),
            "uid": np.array([f"prompt-{index}" for index in range(prompt_count)], dtype=object),
        },
    )
    started_count = 0
    all_started = asyncio.Event()
    never_finish = asyncio.Event()
    cancelled_siblings = []

    async def generate_rewards(prompt_batch):
        nonlocal started_count
        prompt_index = int(prompt_batch.batch["prompts"].item())
        started_count += 1
        if started_count == prompt_count:
            all_started.set()
        try:
            await all_started.wait()
            if prompt_index == 0:
                raise RuntimeError("validation prompt failed")
            await never_finish.wait()
        except asyncio.CancelledError:
            cancelled_siblings.append(prompt_index)
            await asyncio.sleep(0)
            raise
        return [1.0]

    with pytest.raises(RuntimeError, match="validation prompt failed"):
        asyncio.run(
            _collect_areal_validation_batch(
                batch,
                samples_per_prompt=1,
                max_concurrency=prompt_count,
                generate_rewards=generate_rewards,
            )
        )

    assert sorted(cancelled_siblings) == [1, 2]


def test_areal_validation_reward_extraction_drops_sequence_tensors():
    trajectory = _complete_trajectory()

    assert _extract_areal_validation_rewards(trajectory, expected_group_size=2) == [1.0, 0.0]

    with pytest.raises(RuntimeError, match="expected 3"):
        _extract_areal_validation_rewards(trajectory, expected_group_size=3)


def test_initial_disk_sync_rendezvous_uses_controller_version_zero():
    update_key = areal_weight_update_key("ggpo", "areal-smoke", 0)

    assert update_key.endswith("/ggpo/areal-smoke/update_weights_from_disk/0")


def test_areal_controller_weight_sync_contract_is_async():
    assert inspect.iscoroutinefunction(RolloutController.pause_generation)
    assert inspect.iscoroutinefunction(RolloutController.update_weights_from_disk)
    assert inspect.iscoroutinefunction(RolloutController.continue_generation)


def test_areal_reward_bridge_uses_unified_verl_custom_reward(tmp_path):
    reward_path = tmp_path / "custom_reward.py"
    reward_path.write_text(
        "def compute_score(data_source, solution_str, ground_truth, extra_info, scale=1.0):\n"
        "    assert data_source == 'math'\n"
        "    assert solution_str == 'answer=2'\n"
        "    assert ground_truth == '2'\n"
        "    assert extra_info == {'split': 'smoke'}\n"
        "    return {'score': scale, 'acc': 1.0}\n",
        encoding="utf-8",
    )
    config = {
        "async_training": {"areal": {"reward_fn": None}},
        "reward": {
            "custom_reward_function": {
                "path": str(reward_path),
                "name": "compute_score",
                "reward_kwargs": {"scale": 0.75},
            }
        },
    }

    reward_fn, metadata = _resolve_areal_reward(config)

    assert reward_fn == AREAL_CUSTOM_REWARD_FN
    assert metadata == {
        CUSTOM_REWARD_PATH_KEY: str(reward_path),
        CUSTOM_REWARD_NAME_KEY: "compute_score",
        CUSTOM_REWARD_KWARGS_KEY: {"scale": 0.75},
    }
    assert (
        areal_custom_reward_fn(
            "prompt",
            "answer=2",
            [1],
            [2],
            answer="2",
            # RLVR forwards the full VERL row after passing the decoded prompt
            # positionally.  This duplicate-shaped field must stay metadata.
            prompt=[{"role": "user", "content": "What is 1 + 1?"}],
            data_source="math",
            extra_info={"split": "smoke"},
            **metadata,
        )
        == 0.75
    )


def test_areal_reward_bridge_resolves_registered_batch_without_custom_path():
    config = {
        "async_training": {"areal": {"reward_fn": None}},
        "reward": {
            "custom_reward_function": {"path": None, "name": "compute_score"},
            "reward_manager": {"source": "register", "name": "batch"},
        },
    }

    assert _resolve_areal_reward(config) == (AREAL_BATCH_REWARD_FN, {})


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        (r"The final answer is \boxed{2}.", 1.0),
        (r"The final answer is \boxed{3}.", 0.0),
        ("The final answer is 2.", 0.0),
    ],
)
def test_areal_batch_reward_bridge_matches_verl_boxed_math_semantics(monkeypatch, completion, expected):
    def deterministic_boxed_score(model_output, ground_truth, timeout_score):
        del timeout_score
        return float(math_verify.extract_boxed_answer(model_output) == ground_truth)

    monkeypatch.setattr(math_verify, "_compute_boxed_score", deterministic_boxed_score)
    direct_score = reward_score.batch_compute_score(
        data_sources=["alp_guru_rl92k_math"],
        solution_strs=[completion],
        ground_truths=["2"],
        extra_infos=[{"split": "train"}],
    )[0]

    bridged_score = areal_batch_reward_fn(
        "prompt",
        completion,
        [1],
        [2],
        answer="2",
        data_source="alp_guru_rl92k_math",
        extra_info={"split": "train"},
    )

    assert direct_score == expected
    assert bridged_score == direct_score


def test_areal_reward_decode_matches_verl_special_token_policy():
    class RecordingTokenizer:
        eos_token = "<eos>"

        def __init__(self):
            self.calls = []

        def decode(self, token_ids, *, skip_special_tokens):
            self.calls.append((list(token_ids), skip_special_tokens))
            return r"The final answer is \boxed{2}."

    tokenizer = RecordingTokenizer()
    verifier_input = decode_reward_completion(tokenizer, [11, 12, 2, 0])

    assert verifier_input == r"The final answer is \boxed{2}."
    assert tokenizer.calls == [([11, 12, 2, 0], True)]


@pytest.mark.parametrize(
    ("source", "name"),
    [
        ("register", "naive"),
        ("importlib", "batch"),
        (None, "batch"),
    ],
)
def test_areal_reward_resolution_fails_closed_for_unsupported_manager(source, name):
    config = {
        "async_training": {"areal": {"reward_fn": None}},
        "reward": {
            "custom_reward_function": {"path": None, "name": "compute_score"},
            "reward_manager": {"source": source, "name": name},
        },
    }

    with pytest.raises(ValueError, match="registered batch reward manager"):
        _resolve_areal_reward(config)


def test_explicit_areal_reward_overrides_verl_reward_bridge():
    config = {
        "async_training": {"areal": {"reward_fn": "custom.areal.reward"}},
        "reward": {"custom_reward_function": {"path": "/unused.py", "name": "unused"}},
    }

    assert _resolve_areal_reward(config) == ("custom.areal.reward", {})


def test_areal_reward_executor_is_bounded_and_forwarded(monkeypatch):
    assert _resolve_areal_reward_executor_config({"async_training": {"areal": {}}}) == {
        "reward_timeout_seconds": 45.0,
        "reward_max_workers": 1,
        "reward_max_retries": 0,
    }
    with pytest.raises(ValueError, match="reward_max_workers"):
        _resolve_areal_reward_executor_config({"async_training": {"areal": {"reward_max_workers": 0}}})

    captured = {}

    class FakeRewardWrapper:
        def __init__(self, reward_fn, **kwargs):
            captured.update(kwargs)

    class FakeGenerationConfig:
        def new_with_stop_and_pad_token_ids(self, tokenizer):
            return self

    monkeypatch.setattr("areal.workflow.rlvr.AsyncRewardWrapper", FakeRewardWrapper)
    RLVRWorkflow(
        reward_fn=lambda **_: 0.0,
        gconfig=FakeGenerationConfig(),
        tokenizer=object(),
        reward_timeout_seconds=45.0,
        reward_max_workers=1,
        reward_max_retries=0,
    )
    assert captured == {
        "timeout_seconds": 45.0,
        "max_workers": 1,
        "max_retries": 0,
    }


def _thread_reward(delay: float, value: float) -> float:
    time.sleep(delay)
    return value


def _raise_thread_timeout() -> float:
    raise TimeoutError("inner reward timeout")


def _thread_backed_reward_wrapper(*, timeout_seconds: float, executor=None, reward_fn=_thread_reward):
    wrapper = AsyncRewardWrapper.__new__(AsyncRewardWrapper)
    wrapper.reward_fn = reward_fn
    wrapper.timeout_seconds = timeout_seconds
    wrapper.max_workers = 1
    wrapper.max_retries = 0
    wrapper._executor_key = 100_001
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=1)
    AsyncRewardWrapper._executors[wrapper._executor_key] = executor
    return wrapper, executor


def test_areal_reward_timeout_excludes_shared_executor_queue_time():
    wrapper, executor = _thread_backed_reward_wrapper(timeout_seconds=0.15)
    wrappers = [wrapper]
    for _ in range(3):
        shared, _ = _thread_backed_reward_wrapper(timeout_seconds=0.15, executor=executor)
        wrappers.append(shared)

    async def run():
        return await asyncio.gather(*(shared(0.05, float(index)) for index, shared in enumerate(wrappers)))

    try:
        limiter = wrapper._get_limiter(wrapper._executor_key, wrapper.max_workers)
        assert all(shared._get_limiter(shared._executor_key, shared.max_workers) is limiter for shared in wrappers)
        assert asyncio.run(run()) == [0.0, 1.0, 2.0, 3.0]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        AsyncRewardWrapper._executors.pop(wrapper._executor_key, None)
        AsyncRewardWrapper._limiters.clear()


def test_areal_reward_limiter_is_global_across_event_loops():
    wrapper, executor = _thread_backed_reward_wrapper(timeout_seconds=0.15)
    wrappers = [wrapper]
    for _ in range(3):
        shared, _ = _thread_backed_reward_wrapper(timeout_seconds=0.15, executor=executor)
        wrappers.append(shared)
    barrier = threading.Barrier(len(wrappers))
    results = {}
    errors = []

    def call_from_own_loop(index, shared):
        try:
            barrier.wait(timeout=1.0)
            results[index] = asyncio.run(shared(0.05, float(index)))
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=call_from_own_loop, args=(index, shared)) for index, shared in enumerate(wrappers)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)
        assert all(not thread.is_alive() for thread in threads)
        assert not errors
        assert results == {index: float(index) for index in range(len(wrappers))}
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        AsyncRewardWrapper._executors.pop(wrapper._executor_key, None)
        AsyncRewardWrapper._limiters.clear()


def test_areal_reward_function_timeout_is_not_misclassified(monkeypatch):
    wrapper, executor = _thread_backed_reward_wrapper(timeout_seconds=1.0, reward_fn=_raise_thread_timeout)
    executor_before = AsyncRewardWrapper._executors[wrapper._executor_key]
    monkeypatch.setattr(
        wrapper,
        "_replace_timed_out_executor",
        lambda *args: pytest.fail(f"business TimeoutError must not retire pool: {args}"),
    )
    try:
        with pytest.raises(TimeoutError, match="^inner reward timeout$"):
            asyncio.run(wrapper())
        assert AsyncRewardWrapper._executors[wrapper._executor_key] is executor_before
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        AsyncRewardWrapper._executors.pop(wrapper._executor_key, None)
        AsyncRewardWrapper._limiters.clear()


def test_areal_reward_timeout_replaces_pool_and_fails_closed(monkeypatch):
    wrapper, executor = _thread_backed_reward_wrapper(timeout_seconds=0.01)
    replacements = []

    def record_replacement(executor_key, max_workers, timed_out):
        replacements.append((executor_key, max_workers, timed_out))
        return object()

    monkeypatch.setattr(wrapper, "_replace_timed_out_executor", record_replacement)

    async def run():
        return await wrapper(0.05, 1.0)

    try:
        with pytest.raises(TimeoutError, match="Reward computation exceeded"):
            asyncio.run(run())
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        AsyncRewardWrapper._executors.pop(wrapper._executor_key, None)
        AsyncRewardWrapper._limiters.clear()

    assert replacements == [(wrapper._executor_key, 1, executor)]


def test_areal_disk_sync_awaits_controller_coroutines_and_rendezvous():
    events = []
    update_started = threading.Event()
    rendezvous_published = threading.Event()

    class FakeController:
        async def pause_generation(self):
            events.append("pause_generation")

        async def update_weights_from_disk(self, meta):
            events.append(("update_weights_from_disk:start", meta))
            update_started.set()
            while not rendezvous_published.is_set():
                await asyncio.sleep(0)
            events.append(("update_weights_from_disk:done", meta))

        async def continue_generation(self):
            events.append("continue_generation")

    meta = SimpleNamespace(version=0)

    def publish_update_key():
        assert update_started.wait(timeout=1.0)
        events.append("publish_update_key")
        rendezvous_published.set()

    asyncio.run(_run_areal_disk_update(FakeController(), meta, publish_update_key))

    assert events == [
        "pause_generation",
        ("update_weights_from_disk:start", meta),
        "publish_update_key",
        ("update_weights_from_disk:done", meta),
        "continue_generation",
    ]


def test_verl_prompt_group_maps_to_one_strict_areal_rlvr_task():
    task = build_areal_task_data(_source_group())

    assert task["messages"] == [{"role": "user", "content": "What is 1 + 1?"}]
    assert task["answer"] == "2"
    assert task["data_source"] == "math"
    assert task[REQUIRE_COMPLETE_GROUP_KEY] is True


def test_complete_areal_group_converts_to_one_padded_verl_queue_item():
    converted = areal_trajectory_to_dataproto(
        _complete_trajectory(),
        _source_group(),
        group_size=2,
        prompt_length=3,
        response_length=4,
        pad_token_id=0,
        sample_id="sample-7",
        generation_seconds=1.25,
    )

    assert len(converted) == 2
    assert torch.equal(converted.batch["prompts"], torch.tensor([[0, 11, 12], [0, 0, 13]]))
    assert torch.equal(converted.batch["responses"], torch.tensor([[21, 22, 0, 0], [23, 24, 25, 0]]))
    assert torch.equal(converted.batch["response_mask"], torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]))
    assert torch.allclose(
        converted.batch["rollout_log_probs"],
        torch.tensor([[-0.1, -0.2, 0.0, 0.0], [-0.3, -0.4, -0.5, 0.0]]),
    )
    assert torch.equal(converted.batch["rollout_policy_versions"], torch.tensor([[3, 4, -1, -1], [4, 4, 4, -1]]))
    assert torch.equal(converted.batch["verifier_correctness"], torch.tensor([1.0, 0.0]))
    assert torch.equal(converted.batch["rm_scores"], torch.tensor([[0.0, 5.0, 0.0, 0.0], [0.0, 0.0, -5.0, 0.0]]))
    assert converted.non_tensor_batch["uid"].tolist() == ["uid_sample-7", "uid_sample-7"]
    assert converted.non_tensor_batch["min_global_steps"].tolist() == [3, 4]
    assert converted.non_tensor_batch["max_global_steps"].tolist() == [4, 4]
    assert converted.meta_info["metrics"] == [
        {"generate_sequences": 1.25, "tool_calls": 0.0},
        {"generate_sequences": 1.25, "tool_calls": 0.0},
    ]


def test_areal_adapter_reuses_dapo_penalty_and_preserves_raw_reward_metadata():
    converted = areal_trajectory_to_dataproto(
        _complete_trajectory(),
        _source_group(),
        group_size=2,
        prompt_length=3,
        response_length=4,
        pad_token_id=0,
        sample_id="dapo-metadata",
        incorrect_reward=0.0,
        correct_reward=1.0,
        overlong_buffer_length=2,
        overlong_penalty_factor=1.0,
    )

    # Valid response lengths are 2 and 3. With max=4 and buffer=2, the
    # synchronous DAPO helper returns penalties 0 and -0.5 respectively.
    assert torch.equal(converted.batch["rm_scores"], torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -0.5, 0.0]]))
    assert converted.non_tensor_batch["raw_correctness"].tolist() == [1.0, 0.0]
    assert converted.non_tensor_batch["acc"].tolist() == [1.0, 0.0]
    assert converted.non_tensor_batch["overlong_reward"].tolist() == [0.0, -0.5]
    assert converted.non_tensor_batch["overlong"].tolist() == [False, True]
    assert converted.meta_info["reward_extra_keys"] == [
        "acc",
        "raw_correctness",
        "overlong_reward",
        "overlong",
    ]
    assert torch.equal(converted.batch["response_mask"], torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]))
    assert torch.equal(converted.batch["rollout_policy_versions"], torch.tensor([[3, 4, -1, -1], [4, 4, 4, -1]]))


def test_async_queue_filters_constant_groups_and_refills_without_losing_group_metadata():
    def make_sample(sample_id, rewards):
        trajectory = _complete_trajectory()
        trajectory["rewards"] = torch.tensor(rewards, dtype=torch.float32)
        batch = areal_trajectory_to_dataproto(
            trajectory,
            _source_group(),
            group_size=2,
            prompt_length=3,
            response_length=4,
            pad_token_id=0,
            sample_id=sample_id,
            incorrect_reward=0.0,
            correct_reward=1.0,
            overlong_buffer_length=2,
        )
        return RolloutSample(full_batch=batch, sample_id=sample_id, epoch=0, rollout_status={})

    serialized = [
        ray.cloudpickle.dumps(make_sample("all-zero", [0, 0])),
        ray.cloudpickle.dumps(make_sample("all-one", [1, 1])),
        ray.cloudpickle.dumps(make_sample("mixed-a", [0, 1])),
        ray.cloudpickle.dumps(make_sample("mixed-b", [1, 0])),
    ]

    class Queue:
        async def get_sample(self):
            if serialized:
                return serialized.pop(0), len(serialized)
            return None, 0

    trainer_class = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = trainer_class.__new__(trainer_class)
    trainer.required_samples = 2
    trainer.message_queue_client = Queue()
    trainer.current_param_version = 4
    trainer.seen_group_ids = set()
    trainer.group_admission_counts = Counter()
    trainer.tokenizer = None
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "filter_groups": {
                    "enable": True,
                    "metric": "raw_correctness",
                    "max_num_gen_batches": 2,
                }
            },
            "actor_rollout_ref": {
                "actor": {"policy_loss": {"loss_mode": "ggpo"}},
                "rollout": {"n": 2, "multi_turn": {"enable": False}},
            },
            "async_training": {"max_policy_version_lag": 8},
            "trainer": {"balance_batch": False},
        }
    )

    epoch, batch = asyncio.run(trainer._get_samples_from_queue())

    assert epoch == 0
    assert len(batch) == 4
    assert batch.non_tensor_batch["uid"].tolist() == [
        "uid_mixed-a",
        "uid_mixed-a",
        "uid_mixed-b",
        "uid_mixed-b",
    ]
    assert batch.non_tensor_batch["raw_correctness"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert torch.equal(
        batch.batch["rollout_policy_versions"],
        torch.tensor(
            [[3, 4, -1, -1], [4, 4, 4, -1], [3, 4, -1, -1], [4, 4, 4, -1]]
        ),
    )
    assert torch.equal(
        batch.batch["response_mask"],
        torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0]]),
    )
    assert batch.meta_info["sampling/generated_groups"] == 4
    assert batch.meta_info["sampling/accepted_groups"] == 2
    assert batch.meta_info["sampling/raw_all_zero_groups"] == 1
    assert batch.meta_info["sampling/raw_all_one_groups"] == 1
    assert batch.meta_info["sampling/generation_rounds"] == 2


def test_ray_transport_trajectory_is_localized_and_released_before_conversion():
    tensors = _complete_trajectory()
    deleted = []

    class FakeRTensorBackend:
        def fetch(self, shards):
            return [tensors[shard.shard_id] for shard in shards]

        def store(self, tensor):
            raise AssertionError("unexpected RTensor store")

        async def delete(self, node_addr, shard_ids):
            deleted.append((node_addr, list(shard_ids)))

    remote_trajectory = {
        key: RTensor(
            shard=TensorShardInfo(shard_id=key, node_addr=""),
            data=torch.empty_like(value, device="meta"),
        )
        for key, value in tensors.items()
    }

    async def convert():
        async with _localized_areal_trajectory(remote_trajectory) as localized:
            assert all(isinstance(value, torch.Tensor) for value in localized.values())
            assert fetch_buffer_stats()["num_entries"] == len(remote_trajectory)
            converted = areal_trajectory_to_dataproto(
                localized,
                _source_group(),
                group_size=2,
                prompt_length=3,
                response_length=4,
                pad_token_id=0,
                sample_id="sample-ray",
            )
        return converted, fetch_buffer_stats()["num_entries"]

    clear_fetch_buffer()
    set_backend(FakeRTensorBackend())
    try:
        converted, remaining_fetch_entries = asyncio.run(convert())
    finally:
        set_backend(None)
        clear_fetch_buffer()

    assert len(converted) == 2
    assert deleted == [("", list(remote_trajectory))]
    assert remaining_fetch_entries == 0


def test_ray_transport_trajectory_is_released_when_adapter_conversion_fails():
    tensors = _complete_trajectory()
    deleted = []

    class FakeRTensorBackend:
        def fetch(self, shards):
            return [tensors[shard.shard_id] for shard in shards]

        def store(self, tensor):
            raise AssertionError("unexpected RTensor store")

        async def delete(self, node_addr, shard_ids):
            deleted.append((node_addr, list(shard_ids)))

    remote_trajectory = {
        key: RTensor(
            shard=TensorShardInfo(shard_id=key, node_addr=""),
            data=torch.empty_like(value, device="meta"),
        )
        for key, value in tensors.items()
    }

    async def convert_invalid_trajectory():
        async with _localized_areal_trajectory(remote_trajectory) as localized:
            localized["loss_mask"] = torch.zeros_like(localized["loss_mask"])
            areal_trajectory_to_dataproto(
                localized,
                _source_group(),
                group_size=2,
                prompt_length=3,
                response_length=4,
                pad_token_id=0,
                sample_id="sample-invalid",
            )

    clear_fetch_buffer()
    set_backend(FakeRTensorBackend())
    try:
        with pytest.raises(ValueError, match="contains no trainable response tokens"):
            asyncio.run(convert_invalid_trajectory())
        remaining_fetch_entries = fetch_buffer_stats()["num_entries"]
    finally:
        set_backend(None)
        clear_fetch_buffer()

    assert deleted == [("", list(remote_trajectory))]
    assert remaining_fetch_entries == 0


def test_rtensor_cleanup_failure_preserves_primary_error_and_fails_closed_without_one():
    tensors = _complete_trajectory()
    delete_calls = []

    class FailingDeleteBackend:
        def fetch(self, shards):
            return [tensors[str(shard.shard_id).split(":", 1)[1]] for shard in shards]

        def store(self, tensor):
            raise AssertionError("unexpected RTensor store")

        async def delete(self, node_addr, shard_ids):
            delete_calls.append((node_addr, list(shard_ids)))
            raise RuntimeError(f"delete failed for {node_addr}")

    def remote_trajectory():
        return {
            key: RTensor(
                shard=TensorShardInfo(
                    shard_id=f"{('node-a' if index % 2 == 0 else 'node-b')}:{key}",
                    node_addr="node-a" if index % 2 == 0 else "node-b",
                ),
                data=torch.empty_like(value, device="meta"),
            )
            for index, (key, value) in enumerate(tensors.items())
        }

    async def fail_with_primary_error():
        async with _localized_areal_trajectory(remote_trajectory()):
            raise ValueError("primary adapter failure")

    async def succeed_body():
        async with _localized_areal_trajectory(remote_trajectory()) as localized:
            assert all(isinstance(value, torch.Tensor) for value in localized.values())

    clear_fetch_buffer()
    set_backend(FailingDeleteBackend())
    try:
        with pytest.raises(ValueError, match="primary adapter failure") as primary_error:
            asyncio.run(fail_with_primary_error())
        assert len(delete_calls) == 2
        assert {node_addr for node_addr, _ in delete_calls} == {"node-a", "node-b"}
        cleanup_notes = getattr(primary_error.value, "__notes__", [])
        assert len(cleanup_notes) == 1
        assert "AReaL RTensor cleanup failed on 2 node(s)" in cleanup_notes[0]
        assert "delete failed for node-a" in cleanup_notes[0]
        assert "delete failed for node-b" in cleanup_notes[0]

        delete_calls.clear()
        with pytest.raises(RuntimeError, match="AReaL RTensor cleanup failed on 2 node") as cleanup_error:
            asyncio.run(succeed_body())
        assert isinstance(cleanup_error.value.__cause__, RuntimeError)
        assert len(delete_calls) == 2
        assert {node_addr for node_addr, _ in delete_calls} == {"node-a", "node-b"}
    finally:
        set_backend(None)
        clear_fetch_buffer()


@pytest.mark.parametrize(
    "non_tensor_input_ids",
    [
        pytest.param(lambda value: value.tolist(), id="list"),
        pytest.param(lambda value: value.numpy(), id="ndarray"),
    ],
)
def test_areal_adapter_rejects_non_tensor_sequence_fields(non_tensor_input_ids):
    trajectory = _complete_trajectory()
    trajectory["input_ids"] = non_tensor_input_ids(trajectory["input_ids"])

    with pytest.raises(TypeError, match="field 'input_ids' must be a tensor"):
        areal_trajectory_to_dataproto(
            trajectory,
            _source_group(),
            group_size=2,
            prompt_length=3,
            response_length=4,
            pad_token_id=0,
            sample_id="sample-non-tensor",
        )


def test_incomplete_areal_group_is_rejected_before_queue_publication():
    incomplete = {key: value[:1] for key, value in _complete_trajectory().items()}

    with pytest.raises(ValueError, match="Incomplete AReaL prompt group"):
        areal_trajectory_to_dataproto(
            incomplete,
            _source_group(),
            group_size=2,
            prompt_length=3,
            response_length=4,
            pad_token_id=0,
            sample_id="sample-8",
        )


def test_response_policy_versions_must_be_initialized():
    trajectory = _complete_trajectory()
    trajectory["versions"][0, 2] = -1

    with pytest.raises(ValueError, match="unset policy version"):
        areal_trajectory_to_dataproto(
            trajectory,
            _source_group(),
            group_size=2,
            prompt_length=3,
            response_length=4,
            pad_token_id=0,
            sample_id="sample-9",
        )


def test_destructive_weight_loader_accepts_only_exact_transient_version_path(tmp_path):
    config = {
        "async_training": {"weight_sync": {"export_dir": str(tmp_path / "exports")}},
        "trainer": {"default_local_dir": str(tmp_path / "checkpoints")},
    }
    valid = tmp_path / "exports" / "version_7" / "huggingface"
    valid.mkdir(parents=True)
    (valid / "config.json").write_text("{}", encoding="utf-8")

    assert validate_transient_hf_export(config, version=7, hf_path=str(valid)) == valid.resolve()

    with pytest.raises(ValueError, match="exact transient export path"):
        validate_transient_hf_export(config, version=8, hf_path=str(valid))
    with pytest.raises(ValueError, match="exact transient export path"):
        validate_transient_hf_export(config, version=7, hf_path=str(tmp_path / "source-model"))
