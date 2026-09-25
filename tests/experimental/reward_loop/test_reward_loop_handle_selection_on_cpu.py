from types import SimpleNamespace

from verl.experimental.reward_loop.reward_loop import RewardLoopManager


def _manager(reward_manager_name, *, reward_model_enable=False, enable_resource_pool=False):
    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.config = SimpleNamespace(
        reward=SimpleNamespace(
            reward_manager=SimpleNamespace(name=reward_manager_name),
            reward_model=SimpleNamespace(
                enable=reward_model_enable,
                enable_resource_pool=enable_resource_pool,
            ),
        )
    )
    manager.reward_loop_workers = ["worker-handle"]
    return manager


def test_batch_rule_based_reward_uses_colocated_batch_path():
    manager = _manager("batch")

    assert manager.reward_loop_worker_handles is None


def test_naive_rule_based_reward_keeps_async_worker_handles():
    manager = _manager("naive")

    assert manager.reward_loop_worker_handles == ["worker-handle"]


def test_reward_model_with_extra_resource_pool_keeps_async_worker_handles():
    manager = _manager("batch", reward_model_enable=True, enable_resource_pool=True)

    assert manager.reward_loop_worker_handles == ["worker-handle"]
