import asyncio
from types import SimpleNamespace

import pytest

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncRollouterBase,
)
from verl.experimental.separation import ray_trainer as separation_trainer
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer


def test_fully_async_rollouter_uses_manager_handle_policy():
    rollouter = FullyAsyncRollouterBase.__new__(FullyAsyncRollouterBase)
    rollouter.reward_loop_manager = SimpleNamespace(
        reward_loop_worker_handles=None,
        reward_loop_workers=["batch-worker-without-compute-score"],
    )

    assert rollouter._select_reward_loop_worker_handles(enable_agent_reward_loop=True) is None


def test_fully_async_rollouter_keeps_per_sample_reward_handles():
    rollouter = FullyAsyncRollouterBase.__new__(FullyAsyncRollouterBase)
    rollouter.reward_loop_manager = SimpleNamespace(
        reward_loop_worker_handles=["per-sample-worker"],
        reward_loop_workers=["per-sample-worker"],
    )

    assert rollouter._select_reward_loop_worker_handles(enable_agent_reward_loop=True) == ["per-sample-worker"]
    assert rollouter._select_reward_loop_worker_handles(enable_agent_reward_loop=False) is None


def test_fully_async_rollouter_scores_one_complete_deferred_group():
    rollouter = FullyAsyncRollouterBase.__new__(FullyAsyncRollouterBase)
    combined = SimpleNamespace(batch={"rm_scores": object()})
    reward_batch = object()

    class Group:
        def __init__(self):
            self.batch = {}

        def __len__(self):
            return 8

        def union(self, reward_batch):
            assert reward_batch is expected_reward_batch
            return combined

    expected_reward_batch = reward_batch
    group = Group()
    observed = []

    def compute_reward(batch):
        observed.append(batch)
        assert len(batch) == 8
        return reward_batch

    rollouter.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=8)))
    rollouter.reward_loop_manager = SimpleNamespace(compute_rm_score=compute_reward)

    result = asyncio.run(rollouter._compute_deferred_group_reward(group))

    assert observed == [group]
    assert result is combined


def test_fully_async_rollouter_rejects_incomplete_deferred_group():
    rollouter = FullyAsyncRollouterBase.__new__(FullyAsyncRollouterBase)
    rollouter.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=8)))
    rollouter.reward_loop_manager = SimpleNamespace(
        compute_rm_score=lambda batch: pytest.fail("incomplete group must not be scored")
    )

    class IncompleteGroup:
        def __init__(self):
            self.batch = {}

        def __len__(self):
            return 7

    with pytest.raises(RuntimeError, match="expected 8 responses, got 7"):
        asyncio.run(rollouter._compute_deferred_group_reward(IncompleteGroup()))


def test_fully_async_rollouter_rejects_reward_without_scores():
    rollouter = FullyAsyncRollouterBase.__new__(FullyAsyncRollouterBase)

    class CompleteGroup:
        def __init__(self):
            self.batch = {}

        def __len__(self):
            return 8

        def union(self, reward_batch):
            return self

    rollouter.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=8)))
    rollouter.reward_loop_manager = SimpleNamespace(compute_rm_score=lambda batch: object())

    with pytest.raises(RuntimeError, match="complete-group reward function returned no rm_scores"):
        asyncio.run(rollouter._compute_deferred_group_reward(CompleteGroup()))


def test_fully_async_rollouter_keeps_existing_group_reward():
    rollouter = FullyAsyncRollouterBase.__new__(FullyAsyncRollouterBase)
    group = SimpleNamespace(batch={"rm_scores": object()})
    rollouter._compute_reward_colocate = lambda batch: pytest.fail("reward must not be recomputed")

    assert asyncio.run(rollouter._compute_deferred_group_reward(group)) is group


def test_validation_batch_generation_attaches_deferred_reward_once(monkeypatch):
    generated = SimpleNamespace(batch={})
    combined = SimpleNamespace(batch={"rm_scores": object()})
    reward_batch = object()
    observed = []

    async def generate_without_reward(self, prompts):
        del self
        assert prompts == "validation-prompts"
        return generated

    def union(batch_reward):
        assert batch_reward is reward_batch
        return combined

    def compute_reward(batch):
        observed.append(batch)
        return reward_batch

    generated.union = union
    monkeypatch.setattr(AgentLoopManager, "generate_sequences", generate_without_reward)
    manager = FullyAsyncAgentLoopManager.__new__(FullyAsyncAgentLoopManager)
    manager.deferred_batch_reward_fn = compute_reward

    async def run():
        return await manager.generate_sequences("validation-prompts")

    assert asyncio.run(run()) is combined
    assert observed == [generated]


def test_validation_batch_generation_keeps_existing_reward(monkeypatch):
    generated = SimpleNamespace(batch={"rm_scores": object()})

    async def generate_with_reward(self, prompts):
        del self, prompts
        return generated

    monkeypatch.setattr(AgentLoopManager, "generate_sequences", generate_with_reward)
    manager = FullyAsyncAgentLoopManager.__new__(FullyAsyncAgentLoopManager)
    manager.deferred_batch_reward_fn = lambda batch: pytest.fail(
        f"existing validation reward must not be recomputed: {batch}"
    )

    async def run():
        return await manager.generate_sequences("validation-prompts")

    assert asyncio.run(run()) is generated


def test_fully_async_validation_never_enters_checkpoint_manager_reward_branch():
    rollouter = FullyAsyncRollouterBase.__new__(FullyAsyncRollouterBase)
    enriched = SimpleNamespace(batch={"rm_scores": object()})

    assert not hasattr(rollouter, "checkpoint_manager")
    assert rollouter._needs_colocated_reward(enriched) is False

    with pytest.raises(RuntimeError, match="colocated reward/checkpoint-manager path is unsupported"):
        rollouter._needs_colocated_reward(SimpleNamespace(batch={}))


def test_validation_batch_generation_fails_closed_without_reward_owner(monkeypatch):
    generated = SimpleNamespace(batch={})

    async def generate_without_reward(self, prompts):
        del self, prompts
        return generated

    monkeypatch.setattr(AgentLoopManager, "generate_sequences", generate_without_reward)
    manager = FullyAsyncAgentLoopManager.__new__(FullyAsyncAgentLoopManager)
    manager.deferred_batch_reward_fn = None

    async def run():
        return await manager.generate_sequences("validation-prompts")

    with pytest.raises(RuntimeError, match="no deferred batch reward function"):
        asyncio.run(run())


def test_validation_batch_generation_fails_closed_when_reward_has_no_scores(monkeypatch):
    generated = SimpleNamespace(batch={})
    combined = SimpleNamespace(batch={})

    async def generate_without_reward(self, prompts):
        del self, prompts
        return generated

    generated.union = lambda batch_reward: combined
    monkeypatch.setattr(AgentLoopManager, "generate_sequences", generate_without_reward)
    manager = FullyAsyncAgentLoopManager.__new__(FullyAsyncAgentLoopManager)
    manager.deferred_batch_reward_fn = lambda batch: object()

    async def run():
        return await manager.generate_sequences("validation-prompts")

    with pytest.raises(RuntimeError, match="validation reward function returned no rm_scores"):
        asyncio.run(run())


def test_separate_trainer_fails_clearly_if_deferred_reward_has_no_owner(monkeypatch):
    trainer = SeparateRayPPOTrainer.__new__(SeparateRayPPOTrainer)
    trainer.timing_raw = {}
    trainer.reward_loop_manager = None
    batch = SimpleNamespace(batch={})
    monkeypatch.setattr(separation_trainer, "extract_reward", lambda batch: pytest.fail("unreachable"))

    with pytest.raises(RuntimeError, match="missing rm_scores.*no reward_loop_manager"):
        trainer._fit_compute_reward(batch)
