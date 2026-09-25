from threading import Lock
from types import SimpleNamespace

from areal.infra.controller.rollout_controller import RolloutController


def test_export_stats_includes_local_queue_state():
    """Queue counters are exported through the existing learner-step path."""
    queue_state = {
        "staleness_capacity": 4,
        "pending_limit": 32,
        "pending_inputs": 3,
        "pending_results": 2,
        "active_task_ids": 1,
        "runner_input_queue": 5,
        "runner_output_queue": 6,
        "runner_paused": False,
        "accepted": 17,
        "enqueued": 14,
        "running": 1,
        "rejected": 3,
    }
    controller = RolloutController.__new__(RolloutController)
    controller._dispatcher = SimpleNamespace(
        _active_wait_debug_state=lambda: dict(queue_state)
    )
    controller._worker_role = "rollout"
    controller._version = 9
    controller._version_lock = Lock()
    controller._collective_rpc = lambda **kwargs: [
        {"reward": 0.5, "reward__count": 2}
    ]

    metrics = controller.export_stats()

    assert metrics["reward"] == 0.5
    assert metrics["rollout/queue/controller_version"] == 9.0
    for key, value in queue_state.items():
        assert metrics[f"rollout/queue/{key}"] == float(value)
