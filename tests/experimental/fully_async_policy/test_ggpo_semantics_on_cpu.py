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
import inspect
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.agent_loop import agent_loop as agent_loop_module
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopWorker,
    AgentLoopMetrics,
    AgentLoopOutput,
    IncompleteRolloutError,
    require_complete_rollout_output,
)
from verl.experimental.fully_async_policy import fully_async_trainer
from verl.experimental.fully_async_policy.actor_update_padding import (
    DUMMY_SAFE_RESPONSE_MASK,
    REAL_SAMPLE_MASK,
    get_actor_update_padding_spec,
    pad_fully_async_actor_update,
    select_fully_async_real_samples,
)
from verl.experimental.fully_async_policy.detach_utils import MetricsAggregator, declare_fully_async_step_metrics
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouterBase
from verl.experimental.fully_async_policy.fully_async_trainer import (
    TrainingStopException,
    _assert_bounded_smoke_update_health,
    _finalize_fully_async_policy,
    _finalize_fully_async_step_metrics,
    _finish_fully_async_run,
    _log_fully_async_version_metrics,
    _optimizer_step_limit_reached,
    _require_optimizer_progress,
    _resolve_terminal_policy_version,
    _run_fully_async_step_loop,
    _should_save_fully_async_checkpoint,
)
from verl.experimental.separation import ray_trainer as separation_ray_trainer
from verl.trainer import controlled_stop
from verl.trainer.controlled_stop import AdaptiveUpdateBoundaryStop
from verl.trainer.ppo import ray_trainer as ppo_ray_trainer
from verl.trainer.ppo import rollout_corr_helper
from verl.trainer.ppo.rollout_training_mismatch import (
    INTERNAL_MISMATCH_STATS_KEY,
    MISMATCH_AGGREGATION_META,
    compute_mismatch_stats,
    finalize_mismatch_metrics,
)
from verl.utils import config as config_utils
from verl.utils import tensordict_utils as tu


def _agent_loop_output(*, stop_reason="completed", response_ids=None, response_logprobs=None):
    response_ids = [11, 12] if response_ids is None else response_ids
    return AgentLoopOutput(
        prompt_ids=[1],
        response_ids=response_ids,
        response_mask=[1] * len(response_ids),
        response_logprobs=response_logprobs,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_stop_reason": stop_reason},
    )


def _smoke_health_trainer(metrics, *, save_freq=-1, save_final_checkpoint=False):
    return SimpleNamespace(
        config=OmegaConf.create(
            {
                "async_training": {"runtime_root": "/runtime/smoke"},
                "trainer": {"save_freq": save_freq, "save_final_checkpoint": save_final_checkpoint},
            }
        ),
        metrics=dict(metrics),
        completed_optimizer_steps=1,
        current_param_version=1,
    )


def test_bounded_smoke_health_accepts_native_flat_optimizer_contract():
    trainer = _smoke_health_trainer(
        {"actor/optimizer_applied": 1.0, "actor/grad_norm": 0.5, "actor/lr": 1.0e-6}
    )

    _assert_bounded_smoke_update_health(trainer)

    assert trainer.metrics["fully_async/health/all_update_gates_passed"] == 1.0


def test_bounded_smoke_health_rejects_unapplied_native_optimizer_update():
    trainer = _smoke_health_trainer(
        {"actor/optimizer_applied": 0.0, "actor/grad_norm": 0.5, "actor/lr": 1.0e-6}
    )

    with pytest.raises(RuntimeError, match="did not apply"):
        _assert_bounded_smoke_update_health(trainer)


def test_optimizer_health_gate_is_not_applied_to_checkpointed_formal_run():
    trainer = _smoke_health_trainer({}, save_freq=20, save_final_checkpoint=True)

    _assert_bounded_smoke_update_health(trainer)

    assert trainer.metrics == {}


def test_complete_rollout_with_matching_logprobs_is_admitted():
    require_complete_rollout_output(
        _agent_loop_output(response_logprobs=[-0.1, -0.2]),
        require_logprobs=True,
        reject_aborted=True,
    )


@pytest.mark.parametrize(
    ("output", "match"),
    [
        (_agent_loop_output(stop_reason="aborted", response_ids=[], response_logprobs=None), "was aborted"),
        (_agent_loop_output(response_logprobs=None), "response_logprobs=None"),
        (_agent_loop_output(response_logprobs=[-0.1]), "lengths differ"),
    ],
)
def test_incomplete_rollout_fails_closed(output, match):
    with pytest.raises(IncompleteRolloutError, match=match):
        require_complete_rollout_output(output, require_logprobs=True, reject_aborted=True)


def test_partial_mode_may_admit_aborted_output_only_when_logprobs_are_complete():
    require_complete_rollout_output(
        _agent_loop_output(stop_reason="aborted", response_logprobs=[-0.1, -0.2]),
        require_logprobs=True,
        reject_aborted=False,
    )


def test_agent_loop_worker_rejects_aborted_training_rollout_before_postprocess(monkeypatch):
    class FakeAgentLoop:
        async def run(self, sampling_params, **kwargs):
            return _agent_loop_output(stop_reason="aborted", response_ids=[], response_logprobs=None)

    monkeypatch.setitem(agent_loop_module._agent_loop_registry, "incomplete-test", {"_target_": "unused"})
    monkeypatch.setattr(agent_loop_module.hydra.utils, "instantiate", lambda **kwargs: FakeAgentLoop())

    worker = object.__new__(AgentLoopWorker)
    worker.config = OmegaConf.create(
        {
            "async_training": {"partial_rollout": False},
            "data": {},
        }
    )
    worker.llm_client = None
    worker.tokenizer = None
    worker.processor = None
    worker.dataset_cls = None
    worker.tools = []

    with pytest.raises(IncompleteRolloutError, match="was aborted"):
        asyncio.run(
            worker._run_agent_loop(
                {"logprobs": 1},
                {"step": 0, "sample_index": 0, "rollout_n": 0, "validate": False},
                agent_name="incomplete-test",
                trace=False,
            )
        )


def test_non_partial_weight_sync_pause_waits_for_complete_inflight_group():
    async def scenario():
        rollouter = object.__new__(FullyAsyncRollouterBase)
        rollouter.paused = False
        rollouter._weight_sync_pause = False
        rollouter.active_tasks = set()
        rollouter._init_async_objects()

        task_started = asyncio.Event()
        release_task = asyncio.Event()

        async def in_flight_group():
            task_started.set()
            await release_task.wait()
            return "complete-group"

        task = asyncio.create_task(in_flight_group())
        rollouter.active_tasks.add(task)
        rollouter._active_tasks_drained.clear()
        task.add_done_callback(rollouter._signal_if_active_tasks_drained)
        await task_started.wait()

        drain = asyncio.create_task(rollouter.pause_and_drain_for_weight_sync())
        await asyncio.sleep(0)
        assert rollouter.paused
        assert rollouter._weight_sync_pause
        assert not rollouter._resume_event.is_set()
        assert not drain.done()

        release_task.set()
        metrics = await asyncio.wait_for(drain, timeout=1)

        assert not rollouter.active_tasks
        assert metrics["fully_async/weight_sync/drain_active_at_request"] == 1
        assert metrics["fully_async/weight_sync/drain_completed_tasks"] == 1
        # reset_staleness owns the matching resume after the weights are safe.
        assert rollouter.paused
        assert not rollouter._resume_event.is_set()

    asyncio.run(scenario())


def test_processor_never_holds_state_lock_while_waiting_for_generation_task():
    async def scenario():
        class Sample:
            def __init__(self, sample_id):
                self.sample_id = sample_id

        rollouter = object.__new__(FullyAsyncRollouterBase)
        rollouter.paused = False
        rollouter._weight_sync_pause = False
        rollouter.max_concurrent_samples = 1
        rollouter.staleness_samples = 0
        rollouter.active_tasks = set()
        rollouter.pending_queue = asyncio.Queue()
        rollouter._init_async_objects()

        first_started = asyncio.Event()
        release_first = asyncio.Event()
        completed = []

        async def should_pause():
            return False

        async def process(sample):
            if sample.sample_id == "first":
                first_started.set()
                await release_first.wait()
                # Incomplete-group accounting uses this same lock.  Waiting
                # for the task while holding it would deadlock here.
                async with rollouter.lock:
                    completed.append(sample.sample_id)
            else:
                completed.append(sample.sample_id)

        rollouter._should_pause_generation = should_pause
        rollouter._process_single_sample_streaming = process
        await rollouter.pending_queue.put(Sample("first"))
        await rollouter.pending_queue.put(Sample("second"))
        await rollouter.pending_queue.put(None)

        processor = asyncio.create_task(rollouter._processor_worker())
        await first_started.wait()
        release_first.set()
        await asyncio.wait_for(processor, timeout=1)

        assert completed == ["first", "second"]
        assert not rollouter.active_tasks

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("partial_rollout", "expected_calls"),
    [
        (False, ["drain", "sync", "reset"]),
        (True, ["sync", "reset"]),
    ],
)
def test_trainer_drains_before_sync_only_for_non_partial_rollout(monkeypatch, partial_rollout, expected_calls):
    class ImmediateRemoteMethod:
        def __init__(self, callback):
            self.callback = callback

        def remote(self, *args, **kwargs):
            result = self.callback(*args, **kwargs)
            future = asyncio.get_running_loop().create_future()
            future.set_result(result)

            class Ref:
                def future(self):
                    return future

            return Ref()

    calls = []

    class Rollouter:
        pause_and_drain_for_weight_sync = ImmediateRemoteMethod(
            lambda: calls.append("drain")
            or {
                "fully_async/weight_sync/drain_wait_s": 0.25,
                "fully_async/weight_sync/drain_active_at_request": 3.0,
                "fully_async/weight_sync/drain_completed_tasks": 3.0,
            }
        )
        reset_staleness = ImmediateRemoteMethod(lambda **_kwargs: calls.append("reset") or {})

    trainer_class = fully_async_trainer.FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_class)
    trainer.config = OmegaConf.create(
        {
            "async_training": {"partial_rollout": partial_rollout},
            "global_profiler": {"steps": None},
        }
    )
    trainer.rollout_capabilities = {"profiling": False}
    trainer.rollouter = Rollouter()
    trainer.local_trigger_step = 1
    trainer.current_param_version = 1
    trainer.timing_raw = {}
    trainer._pending_version_metrics = {}

    async def sync_weights():
        calls.append("sync")
        return {"fully_async/weight_sync/version": 1}

    monkeypatch.setattr(trainer, "_sync_rollout_weights", sync_weights)

    assert asyncio.run(trainer._fit_update_weights())
    assert calls == expected_calls
    if partial_rollout:
        assert "fully_async/weight_sync/drain_wait_s" not in trainer._pending_version_metrics
    else:
        assert trainer._pending_version_metrics["fully_async/weight_sync/drain_wait_s"] == pytest.approx(0.25)


def test_fully_async_trainer_seals_ggpo_theory_receipt_during_initialization(monkeypatch):
    receipt = {"digest": "unit-test-receipt"}
    monkeypatch.setattr(fully_async_trainer, "is_ggpo_policy_loss", lambda _config: True)
    monkeypatch.setattr(fully_async_trainer, "seal_ggpo_theory_run_receipt", lambda _config: receipt)

    trainer_class = fully_async_trainer.FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_class)
    config = OmegaConf.create({"actor_rollout_ref": {"hybrid_engine": True}})

    with pytest.raises(AssertionError):
        trainer_class.__init__(
            trainer,
            config=config,
            tokenizer=None,
            role_worker_mapping={},
            resource_pool_manager=None,
        )

    assert trainer._ggpo_theory_run_receipt is receipt


def _advantage_trainer(loss_mode: str):
    trainer_class = fully_async_trainer.FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_class)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "grpo",
                "gamma": 1.0,
                "lam": 1.0,
                "norm_adv_by_std_in_grpo": True,
                "rollout_correction": {"bypass_mode": False},
                "use_kl_in_reward": False,
            },
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {"loss_mode": loss_mode},
                    "loss_agg_mode": "token-mean",
                    "loss_scale_factor": None,
                },
                "rollout": {"n": 2},
            },
        }
    )
    trainer.metrics = {}
    trainer.timing_raw = {}
    trainer.reward_tensor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    trainer.reward_extra_infos_dict = {}
    return trainer


def _fully_async_logprob_trainer(loss_mode: str):
    return _advantage_trainer(loss_mode)


def _rollout_batch():
    return DataProto.from_single_dict(
        {
            "rollout_log_probs": torch.zeros(2, 2),
            "response_mask": torch.ones(2, 2),
            "responses": torch.ones(2, 2, dtype=torch.long),
        }
    )


def _fully_async_padding_config(*, ulysses_sequence_parallel_size: int = 2, groups_per_mini: int = 1):
    return OmegaConf.create(
        {
            "async_training": {
                "require_batches": 4,
                "actor_update_padding": {
                    "enabled": True,
                    "groups_per_mini": groups_per_mini,
                },
            },
            "data": {"train_batch_size": 0, "val_batch_size": None},
            "trainer": {
                "balance_batch": False,
                "n_gpus_per_node": 3,
                "nnodes": 2,
                "use_v1": False,
                "ggpo_theory_contract": {
                    "run_id": "unit-test-run",
                    "source_identity": "unit-test-source",
                    "reference_policy_identity": "unit-test-reference",
                    "behavior_policy_identity": "unit-test-behavior",
                },
            },
            "actor_rollout_ref": {
                "actor": {
                    "_target_": "unused.ActorConfig",
                    "policy_loss": {
                        "loss_mode": "ggpo",
                    },
                    "ppo_mini_batch_size": 32,
                    "shuffle": False,
                    "strategy": "fsdp",
                    "fsdp_config": {
                        "ulysses_sequence_parallel_size": ulysses_sequence_parallel_size,
                    },
                    "use_dynamic_bsz": False,
                    "use_kl_loss": False,
                },
                "model": {"use_remove_padding": True},
                "rollout": {
                    "log_prob_micro_batch_size": None,
                    "log_prob_micro_batch_size_per_gpu": 1,
                    "n": 8,
                    "name": "vllm",
                    "temperature": 1.0,
                    "val_kwargs": {"do_sample": False},
                },
            },
        }
    )


def _fully_async_padding_batch():
    group_size = 8
    logical_groups = 32 * 4
    logical_responses = logical_groups * group_size
    uid = np.repeat(np.asarray([f"group-{index}" for index in range(logical_groups)], dtype=object), group_size)
    response_values = torch.arange(logical_responses * 2, dtype=torch.long).reshape(logical_responses, 2)
    batch = DataProto.from_single_dict(
        {
            "responses": response_values,
            "attention_mask": torch.ones(logical_responses, 3, dtype=torch.long),
            "response_mask": torch.ones(logical_responses, 2, dtype=torch.long),
            "token_level_scores": response_values.float(),
            "advantages": response_values.float() / 10,
            "returns": response_values.float() / 20,
            "uid": uid,
        }
    )
    batch.meta_info = {
        "global_token_num": [3] * logical_responses,
        "round_trip_sentinel": "preserve-me",
        "trajectory_param_versions": np.arange(logical_groups, dtype=np.int64),
    }
    return batch


def test_fully_async_actor_padding_interleaves_complete_groups_and_round_trips_real_data():
    config = _fully_async_padding_config()
    original = _fully_async_padding_batch()
    original_uids = original.non_tensor_batch["uid"].copy()
    original_tensors = {name: value.clone() for name, value in original.batch.items()}
    original_global_tokens = list(original.meta_info["global_token_num"])
    original_versions = original.meta_info["trajectory_param_versions"].copy()

    padded, spec = pad_fully_async_actor_update(original, config)

    assert spec == get_actor_update_padding_spec(config)
    assert spec.logical_responses_per_mini == 256
    assert spec.physical_responses_per_mini == 264
    assert spec.logical_responses_per_update == 1024
    assert spec.physical_responses_per_update == 1056
    assert spec.padding_responses_per_update == 32
    assert len(padded) == 1056

    for mini_index in range(4):
        start = mini_index * 264
        mini = padded[start : start + 264]
        real_mask = mini.batch[REAL_SAMPLE_MASK]
        torch.testing.assert_close(
            real_mask,
            torch.tensor([True] * 256 + [False] * 8),
        )
        assert int(real_mask.sum().item()) == 256
        assert int((~real_mask).sum().item()) == 8
        assert len(set(mini.non_tensor_batch["uid"][-8:].tolist())) == 1
        assert mini.non_tensor_batch["uid"][-1].startswith("__fully_async_actor_padding__")
        torch.testing.assert_close(mini.batch["response_mask"][-8:], torch.zeros(8, 2, dtype=torch.long))
        torch.testing.assert_close(
            mini.batch[DUMMY_SAFE_RESPONSE_MASK][-8:],
            torch.ones(8, 2, dtype=torch.long),
        )
        torch.testing.assert_close(mini.batch["token_level_scores"][-8:], torch.zeros(8, 2))
        torch.testing.assert_close(mini.batch["advantages"][-8:], torch.zeros(8, 2))
        torch.testing.assert_close(mini.batch["returns"][-8:], torch.zeros(8, 2))

    # Padding is actor-local: it neither mutates its input nor survives the
    # explicit real-sample projection used by counters and diagnostics.
    assert REAL_SAMPLE_MASK not in original.batch
    np.testing.assert_array_equal(original.non_tensor_batch["uid"], original_uids)
    for name, expected in original_tensors.items():
        torch.testing.assert_close(original.batch[name], expected)
    assert original.meta_info["global_token_num"] == original_global_tokens
    assert original.meta_info["round_trip_sentinel"] == "preserve-me"
    np.testing.assert_array_equal(original.meta_info["trajectory_param_versions"], original_versions)

    # Dispatch first shards the complete 1056-response update over effective
    # DP=3, then each rank iterates four local 88-response physical minis.
    # Exercise that exact order instead of independently slicing global minis.
    expected_real_counts = (
        (88, 88, 80, 88),
        (88, 80, 88, 88),
        (80, 88, 88, 80),
    )
    dummy_only_micro_counts = {4: 0, 8: 0}
    for rank, rank_expected in enumerate(expected_real_counts):
        rank_batch = padded[rank * 352 : (rank + 1) * 352]
        assert len(rank_batch) == 352
        observed = []
        for local_mini_index in range(4):
            local_mini = rank_batch[local_mini_index * 88 : (local_mini_index + 1) * 88]
            observed.append(int(local_mini.batch[REAL_SAMPLE_MASK].sum().item()))
            for group_start in range(0, 88, 8):
                group_uids = local_mini.non_tensor_batch["uid"][group_start : group_start + 8]
                assert len(set(group_uids.tolist())) == 1
            for micro_size in dummy_only_micro_counts:
                for micro_start in range(0, 88, micro_size):
                    micro = local_mini[micro_start : micro_start + micro_size]
                    if not micro.batch[REAL_SAMPLE_MASK].any():
                        dummy_only_micro_counts[micro_size] += 1
        assert tuple(observed) == rank_expected
    assert dummy_only_micro_counts == {4: 8, 8: 4}

    restored = select_fully_async_real_samples(padded)
    assert len(restored) == 1024
    assert REAL_SAMPLE_MASK not in restored.batch
    assert DUMMY_SAFE_RESPONSE_MASK not in restored.batch
    np.testing.assert_array_equal(restored.non_tensor_batch["uid"], original_uids)
    for name, expected in original_tensors.items():
        torch.testing.assert_close(restored.batch[name], expected)
    assert restored.meta_info["global_token_num"] == original_global_tokens
    assert restored.meta_info["round_trip_sentinel"] == "preserve-me"
    np.testing.assert_array_equal(restored.meta_info["trajectory_param_versions"], original_versions)


def test_fully_async_actor_padding_supports_multiple_complete_groups_per_mini():
    config = _fully_async_padding_config(ulysses_sequence_parallel_size=1, groups_per_mini=3)
    config.trainer.n_gpus_per_node = 4
    config.trainer.process_on_nodes = [4, 1]
    original = _fully_async_padding_batch()

    padded, spec = pad_fully_async_actor_update(original, config)

    assert spec.logical_responses_per_mini == 256
    assert spec.physical_responses_per_mini == 280
    assert spec.logical_responses_per_update == 1024
    assert spec.physical_responses_per_update == 1120
    assert spec.padding_responses_per_update == 96
    assert len(padded) == 1120

    for mini_index in range(4):
        start = mini_index * 280
        mini = padded[start : start + 280]
        torch.testing.assert_close(
            mini.batch[REAL_SAMPLE_MASK],
            torch.tensor([True] * 256 + [False] * 24),
        )
        padding_uids = mini.non_tensor_batch["uid"][-24:]
        assert len(set(padding_uids.tolist())) == 3
        for padding_start in range(0, 24, 8):
            assert len(set(padding_uids[padding_start : padding_start + 8].tolist())) == 1
        torch.testing.assert_close(mini.batch["response_mask"][-24:], torch.zeros(24, 2, dtype=torch.long))
        torch.testing.assert_close(
            mini.batch[DUMMY_SAFE_RESPONSE_MASK][-24:],
            torch.ones(24, 2, dtype=torch.long),
        )

    restored = select_fully_async_real_samples(padded)
    assert len(restored) == len(original)
    np.testing.assert_array_equal(restored.non_tensor_batch["uid"], original.non_tensor_batch["uid"])
    for name, expected in original.batch.items():
        torch.testing.assert_close(restored.batch[name], expected)
    assert restored.meta_info["global_token_num"] == original.meta_info["global_token_num"]
    assert restored.meta_info["round_trip_sentinel"] == "preserve-me"
    np.testing.assert_array_equal(
        restored.meta_info["trajectory_param_versions"],
        original.meta_info["trajectory_param_versions"],
    )


def test_fully_async_ggpo_reuses_rollout_log_probs_without_replacing_loss_mode(monkeypatch):
    async_trainer = _advantage_trainer("ggpo")
    trainer = object.__new__(separation_ray_trainer.SeparateRayPPOTrainer)
    trainer.config = async_trainer.config
    trainer.metrics = async_trainer.metrics
    trainer.timing_raw = async_trainer.timing_raw
    trainer.config.algorithm.rollout_correction.bypass_mode = True

    def unexpected_call(*args, **kwargs):
        pytest.fail("GGPO must not enter the GRPO bypass-mode or old-log-prob recomputation paths")

    monkeypatch.setattr(trainer, "_compute_old_log_prob", unexpected_call)
    monkeypatch.setattr(rollout_corr_helper, "apply_bypass_mode", unexpected_call)

    batch = trainer._fit_compute_log_prob(_rollout_batch())

    assert trainer.config.actor_rollout_ref.actor.policy_loss.loss_mode == "ggpo"
    torch.testing.assert_close(batch.batch["old_log_probs"], batch.batch["rollout_log_probs"])


def test_fully_async_ggpo_collects_current_actor_diagnostics_without_changing_loss(monkeypatch):
    trainer = _advantage_trainer("ggpo")
    current_log_probs = torch.full((2, 2), -0.25)
    entropys = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def fake_compute_old_log_prob(batch):
        return (
            DataProto.from_single_dict(
                {
                    "old_log_probs": current_log_probs,
                    "entropys": entropys,
                }
            ),
            0.125,
        )

    def unexpected_bypass(*args, **kwargs):
        pytest.fail("GGPO diagnostics must not enable or apply PPO rollout correction")

    monkeypatch.setattr(trainer, "_compute_old_log_prob", fake_compute_old_log_prob)
    monkeypatch.setattr(rollout_corr_helper, "apply_bypass_mode", unexpected_bypass)

    batch = trainer._fit_compute_log_prob(_rollout_batch())

    assert trainer.config.actor_rollout_ref.actor.policy_loss.loss_mode == "ggpo"
    torch.testing.assert_close(batch.batch["old_log_probs"], batch.batch["rollout_log_probs"])
    assert trainer.metrics["actor/entropy"] == pytest.approx(2.5)
    assert trainer.metrics["perf/mfu/actor_infer"] == pytest.approx(0.125)
    assert trainer.metrics["rollout_corr/kl"] == pytest.approx(0.25)
    assert trainer.metrics["training/rollout_probs_diff_valid"] == 1
    assert trainer.metrics["training/rollout_probs_diff_mean"] > 0


def test_separation_asympo_does_not_require_behavior_log_probs(monkeypatch):
    async_trainer = _advantage_trainer("asympo")
    trainer = object.__new__(separation_ray_trainer.SeparateRayPPOTrainer)
    trainer.config = async_trainer.config
    trainer.metrics = async_trainer.metrics
    trainer.timing_raw = async_trainer.timing_raw

    def unexpected_call(*args, **kwargs):
        pytest.fail("ASymPO must not compute a PPO proximal anchor")

    monkeypatch.setattr(trainer, "_compute_old_log_prob", unexpected_call)
    batch = DataProto.from_single_dict(
        {
            "response_mask": torch.ones(2, 2),
            "responses": torch.ones(2, 2, dtype=torch.long),
        }
    )

    batch = trainer._fit_compute_log_prob(batch)

    assert "rollout_log_probs" not in batch.batch
    assert "old_log_probs" not in batch.batch


@pytest.mark.parametrize("with_rollout_log_probs", [False, True])
def test_fully_async_asympo_actor_update_mismatch_is_optional(monkeypatch, with_rollout_log_probs):
    trainer = _fully_async_logprob_trainer("asympo")
    actor_config = trainer.config.actor_rollout_ref.actor
    actor_config.ppo_mini_batch_size = 1
    actor_config.ppo_epochs = 1
    actor_config.data_loader_seed = 42
    actor_config.shuffle = False
    trainer.config.actor_rollout_ref.rollout.multi_turn = {"enable": False}
    trainer.config.actor_rollout_ref.rollout.temperature = 1.0
    observed = {}

    class WorkerGroup:
        def update_actor(self, batch_td):
            observed["collect_mismatch"] = tu.get(batch_td, "collect_rollout_training_mismatch")
            metrics = {"mfu": 0.5}
            if observed["collect_mismatch"]:
                metrics[INTERNAL_MISMATCH_STATS_KEY] = compute_mismatch_stats(
                    current_log_probs=torch.zeros(2, 2),
                    rollout_log_probs=batch_td["rollout_log_probs"],
                    response_mask=batch_td["response_mask"],
                )
            return tu.get_tensordict({}, non_tensor_dict={"metrics": metrics})

    trainer.actor_rollout_wg = WorkerGroup()
    monkeypatch.setattr(ppo_ray_trainer, "left_right_2_no_padding", lambda batch_td: batch_td)
    tensors = {
        "advantages": torch.ones(2, 2),
        "response_mask": torch.ones(2, 2),
    }
    if with_rollout_log_probs:
        tensors["rollout_log_probs"] = torch.zeros(2, 2)

    output = trainer._update_actor(DataProto.from_single_dict(tensors))

    assert observed["collect_mismatch"] is with_rollout_log_probs
    output_metrics = output.meta_info["metrics"]
    assert output_metrics["perf/mfu/actor"] == pytest.approx(0.5)
    mismatch_keys = [key for key in output_metrics if key.startswith("rollout_training_mismatch/")]
    assert bool(mismatch_keys) is with_rollout_log_probs
    assert (
        "rollout_training_mismatch/raw_is/ess_normalized" in output_metrics
    ) is with_rollout_log_probs


def test_fully_async_ggpo_bypass_fallback_supplies_theory_current_log_probs(monkeypatch):
    trainer = _fully_async_logprob_trainer("ggpo")
    trainer.config.algorithm.rollout_correction.bypass_mode = True
    current_log_probs = torch.full((2, 2), -0.25, requires_grad=True)

    monkeypatch.setattr(
        separation_ray_trainer.SeparateRayPPOTrainer,
        "_compute_old_log_prob",
        lambda _self, _batch: (
            DataProto.from_single_dict(
                {
                    "old_log_probs": current_log_probs,
                    "entropys": torch.ones(2, 2),
                }
            ),
            0.125,
        ),
    )

    batch = trainer._fit_compute_log_prob(_rollout_batch())

    theory_log_probs = batch.batch[fully_async_trainer.GGPO_THEORY_CURRENT_LOG_PROBS]
    assert not theory_log_probs.requires_grad
    torch.testing.assert_close(theory_log_probs, current_log_probs)
    torch.testing.assert_close(batch.batch["old_log_probs"], batch.batch["rollout_log_probs"])


@pytest.mark.parametrize("padded", [False, True])
def test_fully_async_actor_records_ggpo_theory_exactly_once_for_each_geometry(padded):
    trainer_class = fully_async_trainer.FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_class)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(critic_warmup=0))
    trainer.global_steps = 0
    trainer.timing_raw = {}
    trainer.metrics = {}
    trainer._step_metric_aggregation = {}
    events = []
    trainer._record_ggpo_theory_metrics = lambda _batch, _metrics: events.append("record")
    trainer._update_actor = lambda _batch: SimpleNamespace(
        meta_info={"metrics": {"actor/pg_loss": [1.25]}, MISMATCH_AGGREGATION_META: {}}
    )
    batch = SimpleNamespace(meta_info={})
    if padded:
        batch.meta_info[fully_async_trainer.PHYSICAL_MINI_BATCH_SIZE_META] = 264

    result = trainer._fit_update_actor(batch)

    assert result is batch
    assert events == []
    assert trainer.metrics["actor/pg_loss"] == pytest.approx(1.25)


def test_padded_async_actor_uses_the_canonical_base_dispatcher():
    trainer_class = fully_async_trainer.FullyAsyncTrainer.__ray_metadata__.modified_class

    assert "_update_actor" not in trainer_class.__dict__
    source = inspect.getsource(ppo_ray_trainer.RayPPOTrainer._update_actor)
    assert '"collect_rollout_training_mismatch": collect_rollout_training_mismatch' in source
    assert "PHYSICAL_MINI_BATCH_SIZE_META" in source


@pytest.mark.parametrize(
    ("loss_mode", "expected_old_log_probs"),
    [
        ("vanilla", "current"),
        ("ggpo", "rollout"),
    ],
)
def test_fully_async_reuses_inherited_current_log_probs_for_lag_diagnostics(
    monkeypatch,
    loss_mode,
    expected_old_log_probs,
):
    trainer = _fully_async_logprob_trainer(loss_mode)
    current_log_probs = torch.full((2, 2), -0.25, requires_grad=True)
    entropys = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    calls = 0

    def fake_compute_old_log_prob(batch):
        nonlocal calls
        calls += 1
        return (
            DataProto.from_single_dict(
                {
                    "old_log_probs": current_log_probs,
                    "entropys": entropys,
                }
            ),
            0.125,
        )

    monkeypatch.setattr(trainer, "_compute_old_log_prob", fake_compute_old_log_prob)

    batch = trainer._fit_compute_log_prob(_rollout_batch())

    assert calls == 1
    assert not batch.batch["_diagnostic_current_log_probs"].requires_grad
    torch.testing.assert_close(batch.batch["_diagnostic_current_log_probs"], current_log_probs)
    if expected_old_log_probs == "current":
        torch.testing.assert_close(batch.batch["old_log_probs"], current_log_probs)
        assert fully_async_trainer.GGPO_THEORY_CURRENT_LOG_PROBS not in batch.batch
    else:
        torch.testing.assert_close(batch.batch["old_log_probs"], batch.batch["rollout_log_probs"])
        torch.testing.assert_close(
            batch.batch[fully_async_trainer.GGPO_THEORY_CURRENT_LOG_PROBS],
            current_log_probs,
        )
    assert trainer.metrics["perf/mfu/diagnostic_actor_infer"] == pytest.approx(0.125)


def test_single_fit_policy_version_skips_unused_cpu_anchor_snapshot(monkeypatch):
    trainer = _fully_async_logprob_trainer("vanilla")
    trainer.trigger_parameter_sync_step = 1
    trainer.local_trigger_step = 1

    class WorkerGroup:
        def save_model_to_cpu(self, _version):
            raise AssertionError("single-fit policy versions must not save an unused anchor")

        def restore_model_from_cpu(self, _version):
            raise AssertionError("single-fit policy versions must not restore an anchor")

        def clear_cpu_model(self, _version):
            raise AssertionError("single-fit policy versions must not clear a nonexistent anchor")

    trainer.actor_rollout_wg = WorkerGroup()
    expected = (object(), 0.25)
    monkeypatch.setattr(
        separation_ray_trainer.SeparateRayPPOTrainer,
        "_compute_old_log_prob",
        lambda _self, _batch: expected,
    )

    assert trainer._compute_old_log_prob(_rollout_batch()) == expected
    assert not any(key.startswith("fully_async_anchor_") for key in trainer.timing_raw)


def test_multi_fit_policy_version_preserves_and_times_anchor_routing(monkeypatch):
    trainer = _fully_async_logprob_trainer("vanilla")
    trainer.trigger_parameter_sync_step = 2
    trainer.local_trigger_step = 2
    calls = []

    class WorkerGroup:
        def save_model_to_cpu(self, version):
            calls.append(("save", version))

        def restore_model_from_cpu(self, version):
            calls.append(("restore", version))

        def clear_cpu_model(self, version):
            calls.append(("clear", version))

    trainer.actor_rollout_wg = WorkerGroup()
    expected = (object(), 0.25)
    monkeypatch.setattr(
        separation_ray_trainer.SeparateRayPPOTrainer,
        "_compute_old_log_prob",
        lambda _self, _batch: expected,
    )

    assert trainer._compute_old_log_prob(_rollout_batch()) == expected
    assert calls == [
        ("save", 2),
        ("restore", 1),
        ("restore", 2),
        ("clear", 2),
    ]
    assert set(trainer.timing_raw) == {
        "fully_async_anchor_snapshot_save",
        "fully_async_anchor_restore_base",
        "fully_async_anchor_restore_current",
        "fully_async_anchor_snapshot_clear",
    }


def test_fully_async_ggpo_bypasses_advantage_and_grpo_rollout_correction(monkeypatch):
    trainer = _advantage_trainer("ggpo")

    def unexpected_call(*args, **kwargs):
        pytest.fail("GGPO must not call GRPO advantage or rollout-correction code")

    monkeypatch.setattr(separation_ray_trainer, "compute_advantage", unexpected_call)
    monkeypatch.setattr(rollout_corr_helper, "compute_rollout_correction_and_add_to_batch", unexpected_call)

    batch = trainer._fit_compute_advantage(_rollout_batch())

    torch.testing.assert_close(batch.batch["advantages"], torch.zeros_like(trainer.reward_tensor))
    torch.testing.assert_close(batch.batch["returns"], torch.zeros_like(trainer.reward_tensor))
    torch.testing.assert_close(batch.batch["token_level_rewards"], trainer.reward_tensor)


def test_fully_async_grpo_keeps_rollout_correction_and_advantage(monkeypatch):
    trainer = _advantage_trainer("vanilla")
    calls = []

    def fake_rollout_correction(batch, rollout_corr_config):
        calls.append(("rollout_correction", rollout_corr_config.bypass_mode))
        return batch, {"rollout_corr/test": 1.0}

    def fake_compute_advantage(batch, **kwargs):
        calls.append(("advantage", kwargs["adv_estimator"]))
        batch.batch["advantages"] = torch.ones_like(batch.batch["token_level_rewards"])
        batch.batch["returns"] = torch.ones_like(batch.batch["token_level_rewards"])
        return batch

    monkeypatch.setattr(
        rollout_corr_helper,
        "compute_rollout_correction_and_add_to_batch",
        fake_rollout_correction,
    )
    monkeypatch.setattr(separation_ray_trainer, "compute_advantage", fake_compute_advantage)

    batch = trainer._fit_compute_advantage(_rollout_batch())

    assert calls == [("rollout_correction", False), ("advantage", "grpo")]
    assert trainer.metrics["rollout_corr/test"] == 1.0
    torch.testing.assert_close(batch.batch["advantages"], torch.ones_like(trainer.reward_tensor))


def test_fully_async_validation_uses_effective_prompt_batch_without_mutating_loader(monkeypatch):
    config = OmegaConf.create(
        {
            "async_training": {"require_batches": 3},
            "data": {"train_batch_size": 0, "val_batch_size": None},
                "trainer": {
                    "balance_batch": False,
                    "n_gpus_per_node": 4,
                    "nnodes": 1,
                    "use_v1": False,
                    "ggpo_theory_contract": {
                        "run_id": "unit-test-run",
                        "source_identity": "unit-test-source",
                        "reference_policy_identity": "unit-test-reference",
                        "behavior_policy_identity": "unit-test-behavior",
                    },
                },
            "actor_rollout_ref": {
                "actor": {
                    "_target_": "unused.ActorConfig",
                    "policy_loss": {"loss_mode": "ggpo"},
                    "ppo_mini_batch_size": 16,
                    "strategy": "fsdp",
                    "use_dynamic_bsz": False,
                    "use_kl_loss": False,
                },
                "model": {},
                "rollout": {
                    "log_prob_micro_batch_size": None,
                    "log_prob_micro_batch_size_per_gpu": 1,
                    "n": 4,
                    "name": "vllm",
                    "temperature": 1.0,
                    "val_kwargs": {"do_sample": False},
                },
            },
        }
    )
    validated_batch_sizes = []

    class ActorConfigSpy:
        def validate(self, n_gpus, train_batch_size, model_config):
            validated_batch_sizes.append((n_gpus, train_batch_size, model_config))

    monkeypatch.setattr(config_utils, "omega_conf_to_dataclass", lambda config: ActorConfigSpy())

    config_utils.validate_config(config, use_reference_policy=False, use_critic=False)

    assert validated_batch_sizes == [(4, 48, {})]
    assert config_utils.get_train_batch_size_for_validation(config) == 48
    assert config.data.train_batch_size == 0


def test_fully_async_prompt_budget_accepts_equal_explicit_value_and_rejects_conflict():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"actor": {"ppo_mini_batch_size": 16}},
            "async_training": {"require_batches": 3, "prompts_per_update": 48},
        }
    )

    assert config_utils.get_fully_async_prompts_per_update(config) == 48
    config.async_training.prompts_per_update = 47
    with pytest.raises(ValueError, match=r"prompts_per_update must equal.*ppo_mini_batch_size"):
        config_utils.get_fully_async_prompts_per_update(config)

    padding_config = _fully_async_padding_config()
    padding_config.async_training.prompts_per_update = 127
    with pytest.raises(ValueError, match=r"prompts_per_update must equal.*ppo_mini_batch_size"):
        get_actor_update_padding_spec(padding_config)


def test_fully_async_fsdp_batch_validation_accepts_world6_sp2_dp3_padding(monkeypatch):
    config = _fully_async_padding_config(ulysses_sequence_parallel_size=2)
    validated_batch_sizes = []

    class ActorConfigSpy:
        def validate(self, n_gpus, train_batch_size, model_config):
            validated_batch_sizes.append((n_gpus, train_batch_size, model_config))

    monkeypatch.setattr(config_utils, "omega_conf_to_dataclass", lambda config: ActorConfigSpy())

    config_utils.validate_config(config, use_reference_policy=False, use_critic=False)

    assert len(validated_batch_sizes) == 1
    assert validated_batch_sizes[0][:2] == (6, 128)
    assert validated_batch_sizes[0][2].use_remove_padding is True
    spec = get_actor_update_padding_spec(config)
    assert spec.logical_responses_per_mini == 256
    assert spec.physical_responses_per_mini == 264
    assert spec.physical_responses_per_mini // 3 == 88
    assert config.data.train_batch_size == 0


def test_fully_async_fsdp_batch_validation_accepts_world5_sp1_dp5_padding(monkeypatch):
    config = _fully_async_padding_config(ulysses_sequence_parallel_size=1, groups_per_mini=3)
    config.trainer.n_gpus_per_node = 4
    config.trainer.process_on_nodes = [4, 1]
    validated_batch_sizes = []

    class ActorConfigSpy:
        def validate(self, n_gpus, train_batch_size, model_config):
            validated_batch_sizes.append((n_gpus, train_batch_size, model_config))

    monkeypatch.setattr(config_utils, "omega_conf_to_dataclass", lambda config: ActorConfigSpy())

    config_utils.validate_config(config, use_reference_policy=False, use_critic=False)

    assert len(validated_batch_sizes) == 1
    assert validated_batch_sizes[0][:2] == (5, 128)
    assert validated_batch_sizes[0][2].use_remove_padding is True
    spec = get_actor_update_padding_spec(config)
    assert spec.logical_responses_per_mini == 256
    assert spec.physical_responses_per_mini == 280
    assert spec.physical_responses_per_mini // 5 == 56
    assert config.data.train_batch_size == 0


def test_fully_async_fsdp_batch_validation_rejects_padding_that_splits_groups_across_dp_ranks(monkeypatch):
    config = _fully_async_padding_config(ulysses_sequence_parallel_size=1)
    config.trainer.n_gpus_per_node = 4
    config.trainer.nnodes = 2
    monkeypatch.setattr(
        config_utils,
        "omega_conf_to_dataclass",
        lambda config: pytest.fail("actor validation must not run after group-splitting DP geometry"),
    )

    with pytest.raises(
        ValueError,
        match=r"264 responses must be divisible by effective_dp \(8\) \* group_size \(8\)",
    ):
        config_utils.validate_config(config, use_reference_policy=False, use_critic=False)


@pytest.mark.parametrize("use_dynamic_bsz", [False, True])
def test_fully_async_fsdp_batch_validation_rejects_unpadded_mini_that_splits_groups(
    monkeypatch, use_dynamic_bsz
):
    config = _fully_async_padding_config(ulysses_sequence_parallel_size=1)
    config.async_training.actor_update_padding.enabled = False
    config.async_training.require_batches = 8
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 33
    config.actor_rollout_ref.actor.use_dynamic_bsz = use_dynamic_bsz
    config.trainer.n_gpus_per_node = 4
    config.trainer.nnodes = 2
    monkeypatch.setattr(
        config_utils,
        "omega_conf_to_dataclass",
        lambda config: pytest.fail("actor validation must not run after group-splitting DP geometry"),
    )

    with pytest.raises(
        ValueError,
        match=r"264 responses must be divisible by effective_dp \(8\) \* group_size \(8\)",
    ):
        config_utils.validate_config(config, use_reference_policy=False, use_critic=False)


def test_fully_async_fsdp_batch_validation_rejects_world6_sp2_without_padding(monkeypatch):
    config = _fully_async_padding_config(ulysses_sequence_parallel_size=2)
    config.async_training.actor_update_padding.enabled = False
    monkeypatch.setattr(
        config_utils,
        "omega_conf_to_dataclass",
        lambda config: pytest.fail("actor validation must not run after invalid DP3 batch geometry"),
    )

    with pytest.raises(
        ValueError,
        match=r"256 responses must be divisible by effective_dp \(3\) \* group_size \(8\)",
    ):
        config_utils.validate_config(config, use_reference_policy=False, use_critic=False)


def test_fully_async_fsdp_batch_validation_rejects_nondivisible_ulysses_world(monkeypatch):
    config = _fully_async_padding_config(ulysses_sequence_parallel_size=4)

    monkeypatch.setattr(
        config_utils,
        "omega_conf_to_dataclass",
        lambda config: pytest.fail("actor validation must not run after an invalid SP topology"),
    )

    with pytest.raises(
        ValueError,
        match=r"n_gpus \(6\) must be divisible by the actor FSDP Ulysses sequence-parallel size \(4\)",
    ):
        config_utils.validate_config(config, use_reference_policy=False, use_critic=False)


def test_fully_async_actor_padding_rejects_rollout_is_batch_normalization(monkeypatch):
    config = _fully_async_padding_config()
    config.algorithm.rollout_correction = {"rollout_is_batch_normalize": True}
    monkeypatch.setattr(
        config_utils,
        "omega_conf_to_dataclass",
        lambda config: pytest.fail("actor validation must not run after unsafe IS normalization"),
    )

    with pytest.raises(
        ValueError,
        match="does not support algorithm.rollout_correction.rollout_is_batch_normalize=true",
    ):
        config_utils.validate_config(config, use_reference_policy=False, use_critic=False)


def test_terminal_partial_update_gets_a_real_policy_version():
    assert _resolve_terminal_policy_version(7, 3) == (8, 1, True)
    assert _resolve_terminal_policy_version(7, 1) == (7, 1, False)


def test_fully_async_run_without_an_actor_optimizer_step_fails_closed():
    with pytest.raises(RuntimeError, match="without completing an actor optimizer step"):
        _require_optimizer_progress(0)
    _require_optimizer_progress(1)


def test_bounded_fully_async_loop_stops_rollout_after_exact_actor_update_count():
    events = []

    class RequestStop:
        def remote(self):
            async def stop():
                events.append("rollout_stopped")

            return stop()

    class Trainer:
        completed_optimizer_steps = 0
        max_optimizer_steps = 3
        rollouter = type("Rollouter", (), {"request_stop": RequestStop()})()

        async def fit_step(self):
            self.completed_optimizer_steps += 1
            events.append(f"update_{self.completed_optimizer_steps}")

    trainer = Trainer()
    stop_reason = asyncio.run(_run_fully_async_step_loop(trainer))

    assert stop_reason == "max_optimizer_steps"
    assert trainer.completed_optimizer_steps == 3
    assert events == ["update_1", "update_2", "update_3", "rollout_stopped"]
    assert _optimizer_step_limit_reached(3, 3)
    assert not _optimizer_step_limit_reached(2, 3)
    assert not _optimizer_step_limit_reached(100, None)


def test_unbounded_fully_async_loop_stops_at_update_boundary_wall_clock(monkeypatch):
    events = []
    deadline_checks = iter([False, True])
    monkeypatch.setattr(
        fully_async_trainer,
        "_wall_clock_stop_reached",
        lambda _started_at, _stop_after_seconds, _deadline_unix_seconds: next(deadline_checks),
    )

    class RequestStop:
        def remote(self):
            async def stop():
                events.append("rollout_stopped")

            return stop()

    class Trainer:
        completed_optimizer_steps = 0
        max_optimizer_steps = None
        controlled_stop_after_seconds = 5
        controlled_stop_deadline_unix_seconds = None
        rollouter = type("Rollouter", (), {"request_stop": RequestStop()})()

        async def fit_step(self):
            self.completed_optimizer_steps += 1
            events.append("update")

    trainer = Trainer()
    assert asyncio.run(_run_fully_async_step_loop(trainer)) == "controlled_wall_clock"
    assert trainer.completed_optimizer_steps == 1
    assert events == ["update", "rollout_stopped"]


def test_fully_async_absolute_deadline_uses_unix_epoch(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(time, "time", lambda: 106.0)

    assert fully_async_trainer._wall_clock_stop_reached(
        10.0,
        None,
        105.0,
    )
    assert not fully_async_trainer._wall_clock_stop_reached(
        10.0,
        None,
        107.0,
    )


def test_adaptive_stop_scales_headroom_with_sync_steps_and_runtime_history():
    sync_two = AdaptiveUpdateBoundaryStop(
        hard_deadline_unix_seconds=10_000,
        initial_optimizer_step_seconds=100,
        finalization_seconds=20,
        sync_steps=2,
        safety_factor=1.25,
    )
    sync_four = AdaptiveUpdateBoundaryStop(
        hard_deadline_unix_seconds=10_000,
        initial_optimizer_step_seconds=100,
        finalization_seconds=20,
        sync_steps=4,
        safety_factor=1.25,
    )

    assert sync_two.required_headroom_seconds == 275
    assert sync_four.required_headroom_seconds == 525
    assert not sync_two.should_stop_before_next_update(local_trigger_step=1, now_unix_seconds=9_724)
    assert sync_two.should_stop_before_next_update(local_trigger_step=1, now_unix_seconds=9_725)
    assert sync_four.should_stop_before_next_update(local_trigger_step=1, now_unix_seconds=9_475)

    sync_four.observe_optimizer_step(140)
    assert sync_four.optimizer_step_estimate_seconds == 140
    assert sync_four.required_headroom_seconds == 725

    sync_four.observe_optimizer_step(50)
    assert sync_four.optimizer_step_estimate_seconds == 140


def test_adaptive_stop_never_interrupts_an_active_sync_window():
    controller = AdaptiveUpdateBoundaryStop(
        hard_deadline_unix_seconds=10_000,
        initial_optimizer_step_seconds=100,
        finalization_seconds=20,
        sync_steps=4,
    )

    for local_trigger_step in (2, 3, 4):
        assert not controller.should_stop_before_next_update(
            local_trigger_step=local_trigger_step,
            now_unix_seconds=10_100,
        )


def test_fully_async_loop_adaptive_stop_finishes_complete_sync_window(monkeypatch):
    monotonic_values = iter([0, 1, 11, 12, 22])
    monkeypatch.setattr(
        fully_async_trainer,
        "time",
        SimpleNamespace(monotonic=lambda: next(monotonic_values)),
    )
    monkeypatch.setattr(
        controlled_stop,
        "time",
        SimpleNamespace(time=lambda: 1_000),
    )
    events = []

    class RequestStop:
        def remote(self):
            async def stop():
                events.append("rollout_stopped")

            return stop()

    class Trainer:
        completed_optimizer_steps = 0
        max_optimizer_steps = None
        controlled_stop_after_seconds = None
        controlled_stop_deadline_unix_seconds = None
        adaptive_controlled_stop = {
            "hard_deadline_unix_seconds": 1_019,
            "initial_optimizer_step_seconds": 5,
            "finalization_seconds": 1,
            "safety_factor": 1,
            "history_size": 4,
        }
        trigger_parameter_sync_step = 2
        local_trigger_step = 1
        rollouter = type("Rollouter", (), {"request_stop": RequestStop()})()

        async def fit_step(self):
            self.completed_optimizer_steps += 1
            events.append(f"update_{self.completed_optimizer_steps}")
            if self.local_trigger_step == self.trigger_parameter_sync_step:
                self.local_trigger_step = 1
            else:
                self.local_trigger_step += 1

    trainer = Trainer()
    assert asyncio.run(_run_fully_async_step_loop(trainer)) == "controlled_wall_clock"
    assert trainer.completed_optimizer_steps == 2
    assert events == ["update_1", "update_2", "rollout_stopped"]


def test_unbounded_fully_async_loop_preserves_queue_driven_termination():
    class UnexpectedStop:
        def remote(self):
            pytest.fail("unbounded queue termination must not request rollout stop")

    class Trainer:
        completed_optimizer_steps = 0
        max_optimizer_steps = None
        rollouter = type("Rollouter", (), {"request_stop": UnexpectedStop()})()

        async def fit_step(self):
            raise TrainingStopException

    assert asyncio.run(_run_fully_async_step_loop(Trainer())) == "queue_terminated"


def test_bounded_fully_async_loop_fails_if_queue_ends_before_exact_update_count():
    class Trainer:
        completed_optimizer_steps = 2
        max_optimizer_steps = 3

        async def fit_step(self):
            raise TrainingStopException

    with pytest.raises(
        RuntimeError,
        match=r"queue terminated after 2 actor optimizer steps; expected exactly 3",
    ):
        asyncio.run(_run_fully_async_step_loop(Trainer()))


def test_bounded_run_finalization_flushes_wandb_metrics_and_keeps_checkpoint_policy_hook():
    events = []

    class Logger:
        def finish(self):
            events.append(("finish",))

    class Trainer:
        current_param_version = 3
        logger = Logger()

        def _fit_log_aggregated_training_metrics(self, *, step):
            events.append(("metrics", step))

        def _fit_save_checkpoint(self, *, force):
            events.append(("checkpoint_policy", force))

    _finish_fully_async_run(Trainer())

    assert events == [("metrics", 3), ("checkpoint_policy", True), ("finish",)]


def test_completed_optimizer_step_metric_reports_the_terminal_count():
    aggregator = MetricsAggregator(total_gpus=1)
    for completed_steps in (1, 2, 3):
        metrics = {"fully_async/count/completed_optimizer_steps": completed_steps}
        aggregator.add_step_metrics(
            metrics,
            aggregation=declare_fully_async_step_metrics(metrics, prompt_count=1),
        )

    assert aggregator.get_aggregated_metrics()["fully_async/count/completed_optimizer_steps"] == 3


def test_window_metadata_weights_means_and_omits_unmergeable_quantiles():
    aggregator = MetricsAggregator(total_gpus=1)
    for prompt_count, score, p95 in ((1, 2.0, 10.0), (3, 6.0, 20.0)):
        metrics = {"critic/score/mean": score, "custom/latency_p95": p95}
        aggregator.add_step_metrics(
            metrics,
            aggregation=declare_fully_async_step_metrics(metrics, prompt_count=prompt_count),
        )

    merged = aggregator.get_aggregated_metrics()
    assert merged["critic/score/mean"] == pytest.approx(5.0)
    assert "custom/latency_p95" not in merged


def test_canonical_mismatch_window_recomputes_std_quantiles_and_raw_is_from_samples():
    mask_a = torch.ones((1, 2), dtype=torch.bool)
    mask_b = torch.tensor([[1, 0], [1, 0]], dtype=torch.bool)
    first_stats = compute_mismatch_stats(torch.tensor([[1.0, 1.0]]), torch.zeros((1, 2)), mask_a)
    second_stats = compute_mismatch_stats(
        torch.tensor([[3.0, 0.0], [5.0, 0.0]]), torch.zeros((2, 2)), mask_b
    )
    first = finalize_mismatch_metrics(first_stats)
    second = finalize_mismatch_metrics(second_stats)
    expected = finalize_mismatch_metrics([first_stats, second_stats])
    aggregator = MetricsAggregator(total_gpus=1)
    aggregator.add_step_metrics(first, aggregation=first.aggregation)
    aggregator.add_step_metrics(second, aggregation=second.aggregation)

    merged = aggregator.get_aggregated_metrics()
    for name, value in expected.items():
        assert merged[name] == pytest.approx(value), name


def test_synced_training_metrics_log_at_the_same_completed_optimizer_step():
    history = []

    class Trainer:
        def __init__(self):
            self.current_param_version = 0
            self.metrics = {}
            self.metrics_aggregator = MetricsAggregator(total_gpus=1)

        def _fit_postprocess_step(self):
            self.metrics_aggregator.add_step_metrics(
                self.metrics,
                aggregation=declare_fully_async_step_metrics(self.metrics, prompt_count=1),
            )

        def _fit_log_aggregated_training_metrics(self):
            aggregated_metrics = self.metrics_aggregator.get_aggregated_metrics()
            if aggregated_metrics:
                history.append((self.current_param_version, aggregated_metrics))
            self.metrics_aggregator.reset()

    trainer = Trainer()
    expected = [
        (1, 0.125, 0.5),
        (2, 0.25, 1.5),
        (3, 0.375, 2.5),
    ]
    for completed_steps, reward, actor_loss in expected:
        trainer.current_param_version = completed_steps
        trainer.metrics = {
            "fully_async/count/completed_optimizer_steps": completed_steps,
            "critic/score/mean": reward,
            "actor/pg_loss": actor_loss,
        }
        _finalize_fully_async_step_metrics(trainer, weights_updated=True)

    assert [step for step, _ in history] == [1, 2, 3]
    for (logged_step, metrics), (completed_steps, reward, actor_loss) in zip(history, expected, strict=True):
        assert logged_step == completed_steps
        assert metrics["fully_async/count/completed_optimizer_steps"] == completed_steps
        assert metrics["critic/score/mean"] == reward
        assert metrics["actor/pg_loss"] == actor_loss


def test_weight_sync_validation_and_training_metrics_commit_once_per_policy_version():
    history = []

    class Logger:
        def log(self, *, data, step):
            history.append((step, data))

    trainer = type("Trainer", (), {})()
    trainer.current_param_version = 0
    trainer.logger = Logger()
    trainer.metrics_aggregator = MetricsAggregator(total_gpus=1)
    trainer._pending_version_metrics = {
        "fully_async/weight_sync/version": 0,
        "timing_s/param_sync": 1.0,
        "val-core/math/reward/mean@1": 0.75,
    }

    _log_fully_async_version_metrics(trainer)
    for step in (1, 2, 3):
        trainer.current_param_version = step
        metrics = {
            "fully_async/count/completed_optimizer_steps": step,
            "critic/score/mean": step / 10,
            "actor/pg_loss": step / 20,
        }
        trainer.metrics_aggregator.add_step_metrics(
            metrics,
            aggregation=declare_fully_async_step_metrics(metrics, prompt_count=1),
        )
        trainer._pending_version_metrics = {
            "fully_async/weight_sync/version": step,
            "timing_s/param_sync": float(step + 1),
        }
        _log_fully_async_version_metrics(trainer)
        # A second flush at the same explicit step is a no-op, rather than a
        # second W&B commit that would advance and then reject same-step data.
        _log_fully_async_version_metrics(trainer)

    assert [step for step, _ in history] == [0, 1, 2, 3]
    assert history[0][1] == {
        "fully_async/weight_sync/version": 0,
        "timing_s/param_sync": 1.0,
        "val-core/math/reward/mean@1": 0.75,
    }
    for step, metrics in history[1:]:
        assert metrics["fully_async/count/completed_optimizer_steps"] == step
        assert metrics["critic/score/mean"] == pytest.approx(step / 10)
        assert metrics["actor/pg_loss"] == pytest.approx(step / 20)
        assert metrics["fully_async/weight_sync/version"] == step


def test_terminal_policy_does_not_force_off_cadence_validation():
    events = []

    class Trainer:
        config = OmegaConf.create({"trainer": {"test_freq": 20}})
        current_param_version = 3

        async def _fit_update_weights(self):
            events.append("unexpected_sync")
            return True

        def _fit_log_aggregated_training_metrics(self):
            events.append("unexpected_log")

        async def _fit_validate(self, val_before_train=False):
            events.append(("validate", val_before_train))

    asyncio.run(
        _finalize_fully_async_policy(
            Trainer(),
            has_partial_update=False,
        )
    )

    assert events == []


def test_terminal_partial_policy_syncs_once_and_defers_its_single_metrics_commit():
    events = []

    class Trainer:
        config = OmegaConf.create({"trainer": {"test_freq": 20}})
        current_param_version = 4

        async def _fit_update_weights(self):
            events.append("sync")
            return True

        def _fit_log_aggregated_training_metrics(self):
            events.append("metrics")

        async def _fit_validate(self, val_before_train=False):
            events.append(("validate", val_before_train))

    asyncio.run(
        _finalize_fully_async_policy(
            Trainer(),
            has_partial_update=True,
        )
    )

    assert events == ["sync"]


def test_terminal_force_checkpoint_bypasses_frequency_but_not_version_gate():
    assert not _should_save_fully_async_checkpoint(
        force=True,
        current_param_version=6,
        last_ckpt_version=6,
        save_freq=-1,
        esi_close_to_expiration=False,
    )
    assert _should_save_fully_async_checkpoint(
        force=True,
        current_param_version=6,
        last_ckpt_version=5,
        save_freq=-1,
        esi_close_to_expiration=False,
    )
    assert not _should_save_fully_async_checkpoint(
        force=True,
        current_param_version=6,
        last_ckpt_version=5,
        save_freq=-1,
        esi_close_to_expiration=False,
        save_final_checkpoint=False,
    )
    assert _should_save_fully_async_checkpoint(
        force=False,
        current_param_version=6,
        last_ckpt_version=5,
        save_freq=1,
        esi_close_to_expiration=False,
        save_final_checkpoint=False,
    )
    assert not _should_save_fully_async_checkpoint(
        force=False,
        current_param_version=6,
        last_ckpt_version=6,
        save_freq=1,
        esi_close_to_expiration=False,
    )
