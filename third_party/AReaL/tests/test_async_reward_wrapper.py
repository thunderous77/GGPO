# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace

import pytest

from tests.reward_api_worker_fixtures import (
    add_reward as _add_reward,
)
from tests.reward_api_worker_fixtures import (
    crash_worker as _crash_worker,
)
from tests.reward_api_worker_fixtures import (
    marker_slow_reward as _marker_slow_reward,
)
from tests.reward_api_worker_fixtures import (
    raise_timeout_error as _raise_timeout_error,
)
from tests.reward_api_worker_fixtures import (
    raise_value_error as _raise_value_error,
)
from tests.reward_api_worker_fixtures import (
    slow_reward as _slow_reward,
)

from areal.api import reward_api as reward_api_module
from areal.api.reward_api import AsyncRewardWrapper


@pytest.fixture(autouse=True)
def _isolate_executor_state():
    """Shut down and clear all shared executors before and after each test."""
    AsyncRewardWrapper._atexit_shutdown_all()
    AsyncRewardWrapper._poisoned_executor_keys.clear()
    yield
    AsyncRewardWrapper._atexit_shutdown_all()
    AsyncRewardWrapper._poisoned_executor_keys.clear()


class TestAsyncRewardWrapperBasic:
    @pytest.mark.asyncio
    async def test_returns_correct_result(self):
        wrapper = AsyncRewardWrapper(_add_reward, max_workers=1, max_retries=0)
        result = await wrapper(3.0, 4.0)
        assert result == 7.0

    @pytest.mark.asyncio
    async def test_multiple_calls_return_correct_results(self):
        wrapper = AsyncRewardWrapper(_add_reward, max_workers=2, max_retries=0)
        results = [await wrapper(float(i), 1.0) for i in range(5)]
        assert results == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_shared_executor_for_same_max_workers(self):
        w1 = AsyncRewardWrapper(_add_reward, max_workers=2)
        w2 = AsyncRewardWrapper(_add_reward, max_workers=2)
        assert w1._executor_key == w2._executor_key
        assert AsyncRewardWrapper._executors.get(2) is not None
        assert len(AsyncRewardWrapper._executors) == 1

    def test_different_executor_for_different_max_workers(self):
        AsyncRewardWrapper(_add_reward, max_workers=1)
        AsyncRewardWrapper(_add_reward, max_workers=2)
        assert len(AsyncRewardWrapper._executors) == 2


class TestAsyncRewardWrapperTimeout:
    @pytest.mark.asyncio
    async def test_timeout_raises_after_retries(self):
        wrapper = AsyncRewardWrapper(
            _slow_reward, timeout_seconds=0.1, max_workers=1, max_retries=1
        )
        executor_before = AsyncRewardWrapper._executors[1]
        with pytest.raises(TimeoutError, match="Reward computation exceeded"):
            await wrapper(10.0)
        assert AsyncRewardWrapper._executors[1] is not executor_before
        fast_wrapper = AsyncRewardWrapper(
            _add_reward, timeout_seconds=5, max_workers=1, max_retries=0
        )
        assert await fast_wrapper(3.0, 4.0) == 7.0

    @pytest.mark.asyncio
    async def test_timeout_no_retries_raises(self):
        wrapper = AsyncRewardWrapper(
            _slow_reward, timeout_seconds=0.1, max_workers=1, max_retries=0
        )
        executor_before = AsyncRewardWrapper._executors[1]
        with pytest.raises(TimeoutError, match="Reward computation exceeded"):
            await wrapper(10.0)
        assert AsyncRewardWrapper._executors[1] is not executor_before

    @pytest.mark.asyncio
    async def test_active_timeout_kills_worker_and_replacement_is_usable(
        self, tmp_path
    ):
        fast_wrapper = AsyncRewardWrapper(
            _add_reward, timeout_seconds=5, max_workers=1, max_retries=0
        )
        assert await fast_wrapper(1.0, 1.0) == 2.0

        started = tmp_path / "started"
        finished = tmp_path / "finished"
        slow_wrapper = AsyncRewardWrapper(
            _marker_slow_reward,
            timeout_seconds=0.5,
            max_workers=1,
            max_retries=0,
        )
        executor_before = AsyncRewardWrapper._executors[1]
        with pytest.raises(TimeoutError, match="Reward computation exceeded"):
            await slow_wrapper(str(started), str(finished), 1.0)

        assert started.is_file()
        assert AsyncRewardWrapper._executors[1] is not executor_before
        await asyncio.sleep(1.1)
        assert not finished.exists()
        assert await fast_wrapper(2.0, 3.0) == 5.0

    @pytest.mark.asyncio
    async def test_cancellation_kills_active_worker_and_replacement_is_usable(
        self, tmp_path
    ):
        fast_wrapper = AsyncRewardWrapper(
            _add_reward, timeout_seconds=5, max_workers=1, max_retries=0
        )
        assert await fast_wrapper(1.0, 1.0) == 2.0

        started = tmp_path / "cancel-started"
        finished = tmp_path / "cancel-finished"
        slow_wrapper = AsyncRewardWrapper(
            _marker_slow_reward,
            timeout_seconds=30,
            max_workers=1,
            max_retries=0,
        )
        executor_before = AsyncRewardWrapper._executors[1]
        task = asyncio.create_task(slow_wrapper(str(started), str(finished), 1.0))

        async def wait_until_started():
            while not started.exists():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_until_started(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert AsyncRewardWrapper._executors[1] is not executor_before
        await asyncio.sleep(1.1)
        assert not finished.exists()
        assert await fast_wrapper(2.0, 3.0) == 5.0

    @pytest.mark.asyncio
    async def test_timeout_retires_entire_two_worker_pool_once(self, tmp_path):
        fast_wrapper = AsyncRewardWrapper(
            _add_reward, timeout_seconds=5, max_workers=2, max_retries=0
        )
        assert await asyncio.gather(fast_wrapper(1.0, 1.0), fast_wrapper(2.0, 2.0)) == [
            2.0,
            4.0,
        ]

        short_started = tmp_path / "short-started"
        short_finished = tmp_path / "short-finished"
        sibling_started = tmp_path / "sibling-started"
        sibling_finished = tmp_path / "sibling-finished"
        short_wrapper = AsyncRewardWrapper(
            _marker_slow_reward,
            timeout_seconds=0.5,
            max_workers=2,
            max_retries=0,
        )
        sibling_wrapper = AsyncRewardWrapper(
            _marker_slow_reward,
            timeout_seconds=5,
            max_workers=2,
            max_retries=0,
        )
        executor_before = AsyncRewardWrapper._executors[2]
        old_processes = list(executor_before._processes.values())
        short_task = asyncio.create_task(
            short_wrapper(str(short_started), str(short_finished), 2.0)
        )
        sibling_task = asyncio.create_task(
            sibling_wrapper(str(sibling_started), str(sibling_finished), 2.0)
        )

        async def wait_until_both_started():
            while not (short_started.exists() and sibling_started.exists()):
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_until_both_started(), timeout=5)
        with pytest.raises(TimeoutError, match="Reward computation exceeded"):
            await short_task
        with pytest.raises(BrokenProcessPool):
            await sibling_task

        replacement = AsyncRewardWrapper._executors[2]
        assert replacement is not executor_before
        assert 2 not in AsyncRewardWrapper._poisoned_executor_keys
        assert all(not process.is_alive() for process in old_processes)
        assert await fast_wrapper(3.0, 4.0) == 7.0

    def test_manager_join_refreshes_stale_survivor_before_poison(self, monkeypatch):
        events = []

        class StaleUntilManagerProcess:
            pid = 12345

            def __init__(self):
                self.alive = True

            def terminate(self):
                events.append("process_terminate")

            def kill(self):
                events.append("process_kill")

            def join(self, timeout=None):
                events.append("process_join")

            def is_alive(self):
                events.append("process_is_alive")
                return self.alive

        process = StaleUntilManagerProcess()

        class ReapingManager:
            def __init__(self):
                self.alive = True

            def join(self, timeout=None):
                events.append("manager_join")
                process.alive = False
                self.alive = False

            def is_alive(self):
                events.append("manager_is_alive")
                return self.alive

        replacement = SimpleNamespace(
            shutdown=lambda **kwargs: events.append("replacement_shutdown")
        )
        monkeypatch.setattr(
            reward_api_module,
            "ProcessPoolExecutor",
            lambda **kwargs: replacement,
        )
        timed_out = SimpleNamespace(
            _processes={0: process},
            _executor_manager_thread=ReapingManager(),
            shutdown=lambda **kwargs: events.append("executor_shutdown"),
        )
        AsyncRewardWrapper._executors[1] = timed_out

        assert (
            AsyncRewardWrapper._replace_timed_out_executor(1, 1, timed_out)
            is replacement
        )
        assert AsyncRewardWrapper._executors[1] is replacement
        assert 1 not in AsyncRewardWrapper._poisoned_executor_keys
        manager_join_index = events.index("manager_join")
        final_join_index = len(events) - 1 - events[::-1].index("process_join")
        final_alive_index = len(events) - 1 - events[::-1].index("process_is_alive")
        assert manager_join_index < final_join_index < final_alive_index

    def test_unclean_retirement_poison_prevents_new_pool(self):
        class StubbornProcess:
            pid = 12345

            def terminate(self):
                pass

            def kill(self):
                pass

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return True

        poisoned = SimpleNamespace(
            _processes={0: StubbornProcess()},
            _executor_manager_thread=None,
        )
        AsyncRewardWrapper._executors[1] = poisoned

        assert AsyncRewardWrapper._replace_timed_out_executor(1, 1, poisoned) is None
        assert 1 in AsyncRewardWrapper._poisoned_executor_keys
        with pytest.raises(RuntimeError, match="poisoned after an unclean retirement"):
            AsyncRewardWrapper(_add_reward, max_workers=1)


class TestAsyncRewardWrapperBrokenPool:
    @pytest.mark.asyncio
    async def test_crash_raises_broken_process_pool(self):
        wrapper = AsyncRewardWrapper(
            _crash_worker, max_workers=1, max_retries=0, timeout_seconds=5
        )
        with pytest.raises(BrokenProcessPool):
            await wrapper()

    @pytest.mark.asyncio
    async def test_recreation_replaces_executor(self):
        wrapper = AsyncRewardWrapper(
            _crash_worker, max_workers=1, max_retries=1, timeout_seconds=5
        )
        executor_before = AsyncRewardWrapper._executors.get(1)
        assert executor_before is not None

        with pytest.raises(BrokenProcessPool):
            await wrapper()

        executor_after = AsyncRewardWrapper._executors.get(1)
        assert executor_after is not executor_before


class TestAsyncRewardWrapperExceptionHandling:
    @pytest.mark.asyncio
    async def test_reward_fn_exception_propagates(self):
        wrapper = AsyncRewardWrapper(_raise_value_error, max_workers=1, max_retries=0)
        with pytest.raises(ValueError, match="intentional error"):
            await wrapper()

    @pytest.mark.asyncio
    async def test_reward_fn_exception_retries_then_raises(self):
        wrapper = AsyncRewardWrapper(_raise_value_error, max_workers=1, max_retries=2)
        with pytest.raises(ValueError, match="intentional error"):
            await wrapper()

    @pytest.mark.asyncio
    async def test_reward_fn_timeout_error_is_not_outer_deadline(self):
        wrapper = AsyncRewardWrapper(
            _raise_timeout_error,
            timeout_seconds=5,
            max_workers=1,
            max_retries=0,
        )
        executor_before = AsyncRewardWrapper._executors[1]
        with pytest.raises(TimeoutError, match="^inner reward timeout$"):
            await wrapper()
        assert AsyncRewardWrapper._executors[1] is executor_before

    @pytest.mark.asyncio
    async def test_shutdown_then_call_raises_runtime_error(self):
        wrapper = AsyncRewardWrapper(_add_reward, max_workers=1, max_retries=0)
        AsyncRewardWrapper._atexit_shutdown_all()
        with pytest.raises(RuntimeError, match="has been shut down"):
            await wrapper(1.0, 2.0)


class TestAsyncRewardWrapperAtexitCleanup:
    def test_atexit_clears_all_executors(self):
        AsyncRewardWrapper(_add_reward, max_workers=1)
        AsyncRewardWrapper(_add_reward, max_workers=2)
        assert len(AsyncRewardWrapper._executors) == 2

        AsyncRewardWrapper._atexit_shutdown_all()
        assert len(AsyncRewardWrapper._executors) == 0

    def test_atexit_is_idempotent(self):
        AsyncRewardWrapper(_add_reward, max_workers=1)
        AsyncRewardWrapper._atexit_shutdown_all()
        AsyncRewardWrapper._atexit_shutdown_all()
        assert len(AsyncRewardWrapper._executors) == 0


class TestRecreateExecutorRaceSafety:
    def test_recreate_skips_when_already_replaced(self):
        AsyncRewardWrapper(_add_reward, max_workers=1)
        original = AsyncRewardWrapper._executors[1]

        # Simulate a concurrent thread having already replaced the executor
        replacement = ProcessPoolExecutor(max_workers=1)
        AsyncRewardWrapper._executors[1] = replacement

        result = AsyncRewardWrapper._recreate_executor(1, 1, original)
        assert result is replacement

        replacement.shutdown(wait=False)

    def test_recreate_replaces_when_identity_matches(self):
        AsyncRewardWrapper(_add_reward, max_workers=1)
        original = AsyncRewardWrapper._executors[1]

        result = AsyncRewardWrapper._recreate_executor(1, 1, original)
        assert result is not None
        assert result is not original
        assert AsyncRewardWrapper._executors[1] is result
