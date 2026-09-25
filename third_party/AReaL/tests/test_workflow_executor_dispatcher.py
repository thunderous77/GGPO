import asyncio
import logging
import time
from dataclasses import dataclass

import pytest

from areal.api import RolloutStat
from areal.infra.async_task_runner import AsyncTaskRunner
from areal.infra.staleness_manager import StalenessManager
from areal.infra.workflow_executor import BatchTaskDispatcher


@dataclass
class _TaskInput:
    task_id: int


class _NoCapacityStalenessManager:
    def get_pending_limit(self) -> int:
        return 0

    def get_capacity(self) -> int:
        return 0

    def get_stats(self) -> RolloutStat:
        return RolloutStat(accepted=8, enqueued=0, running=0, rejected=3)

    def on_rollout_enqueued(self) -> None:
        raise AssertionError("no task should be enqueued without capacity")


def _input_generator():
    task_id = 0
    while True:
        yield _TaskInput(task_id=task_id)
        task_id += 1


def _wait_until(predicate, timeout: float = 2.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


class _VersionProvider:
    def get_version(self) -> int:
        return 0


async def _return_none():
    return None


async def _return_value():
    return "ok"


async def _sleep_forever():
    await asyncio.Event().wait()


def test_active_submit_and_wait_fails_fast_when_no_progress(monkeypatch):
    dispatcher = BatchTaskDispatcher(
        max_queue_size=4,
        task_factory=lambda _: (lambda: None),
        staleness_manager=_NoCapacityStalenessManager(),
    )
    monkeypatch.setenv("AREAL_PREPARE_BATCH_STALL_TIMEOUT", "0.001")
    monkeypatch.setattr(dispatcher, "wait_results", lambda *_, **__: [])

    with pytest.raises(RuntimeError, match="prepare_batch made no progress") as exc:
        dispatcher.active_submit_and_wait(_input_generator(), batch_size=2)

    message = str(exc.value)
    assert "staleness_capacity=0" in message
    assert "accepted=8" in message
    assert "running=0" in message


def test_rejected_result_releases_staleness_slot():
    manager = StalenessManager(
        version_provider=_VersionProvider(),
        max_concurrent_rollouts=1,
        consumer_batch_size=1,
        max_staleness=0,
    )
    dispatcher = BatchTaskDispatcher(
        max_queue_size=4,
        task_factory=lambda _: _return_none,
        staleness_manager=manager,
    )
    dispatcher.initialize(logger=logging.getLogger("test"))
    try:
        dispatcher.submit_task_input(_TaskInput(task_id=1))
        assert dispatcher.wait_results(1, timeout=2.0) == [None]

        stats = manager.get_stats()
        assert stats.enqueued == 0
        assert stats.running == 0
        assert stats.accepted == 0
        assert stats.rejected == 1
        assert manager.get_capacity() == 1
    finally:
        dispatcher.destroy()


def test_accepted_result_releases_staleness_slot():
    manager = StalenessManager(
        version_provider=_VersionProvider(),
        max_concurrent_rollouts=1,
        consumer_batch_size=1,
        max_staleness=0,
    )
    dispatcher = BatchTaskDispatcher(
        max_queue_size=4,
        task_factory=lambda _: _return_value,
        staleness_manager=manager,
    )
    dispatcher.initialize(logger=logging.getLogger("test"))
    try:
        dispatcher.submit_task_input(_TaskInput(task_id=1))
        assert dispatcher.wait_results(1, timeout=2.0) == ["ok"]

        stats = manager.get_stats()
        assert stats.enqueued == 0
        assert stats.running == 0
        assert stats.accepted == 1
        assert stats.rejected == 0
        assert manager.get_capacity() == 1
    finally:
        dispatcher.destroy()


def test_async_task_runner_cancel_running_task_returns_none():
    runner = AsyncTaskRunner(max_queue_size=4)
    runner.initialize(logger=logging.getLogger("test"))
    try:
        runner.submit(_sleep_forever, task_id=1)
        assert runner.cancel_task(1)
        result = runner.wait(1, timeout=2.0, with_timing=True)[0]
        assert result.task_id == 1
        assert result.data is None
    finally:
        runner.destroy()


def test_cancelled_dispatcher_task_releases_slot_for_replacement():
    manager = StalenessManager(
        version_provider=_VersionProvider(),
        max_concurrent_rollouts=1,
        consumer_batch_size=1,
        max_staleness=0,
    )
    task_fns = {1: _sleep_forever, 2: _return_value}
    dispatcher = BatchTaskDispatcher(
        max_queue_size=4,
        task_factory=lambda task_input: task_fns[task_input.task_id],
        staleness_manager=manager,
    )
    dispatcher.initialize(logger=logging.getLogger("test"))
    try:
        dispatcher.submit_task_input(_TaskInput(task_id=1))
        _wait_until(lambda: manager.get_stats().running == 1)
        assert dispatcher.cancel_task(1)
        _wait_until(lambda: manager.get_stats().rejected == 1)

        dispatcher.submit_task_input(_TaskInput(task_id=2))
        assert dispatcher.wait_results(1, timeout=2.0) == ["ok"]

        stats = manager.get_stats()
        assert stats.enqueued == 0
        assert stats.running == 0
        assert stats.accepted == 1
        assert stats.rejected == 1
    finally:
        dispatcher.destroy()
