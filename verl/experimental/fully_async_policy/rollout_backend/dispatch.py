"""Exact task/group dispatch helpers for the AReaL rollout controller."""

from __future__ import annotations

import asyncio
from typing import Any


def extract_task_trajectory(result: Any, *, expected_task_id: int) -> dict[str, Any] | None:
    """Unwrap AReaL's private result wrapper without losing task identity."""

    if result is None:
        return None
    if not hasattr(result, "task_id") or not hasattr(result, "trajectory"):
        raise TypeError(
            "AReaL dispatcher must return its task result wrapper with both 'task_id' and 'trajectory' fields"
        )
    returned_task_id = int(result.task_id)
    if returned_task_id != int(expected_task_id):
        raise RuntimeError(
            f"AReaL dispatcher returned task {returned_task_id} while waiting for task {expected_task_id}"
        )
    trajectory = result.trajectory
    if trajectory is not None and not isinstance(trajectory, dict):
        raise TypeError(
            f"AReaL task {expected_task_id} trajectory must be a dict or None, got {type(trajectory).__name__}"
        )
    return trajectory


async def submit_areal_prompt_group(
    controller: Any,
    *,
    data: dict[str, Any],
    workflow: str,
    workflow_kwargs: dict[str, Any],
    group_size: int,
    timeout: float,
    retries: int,
    is_eval: bool = False,
) -> dict[str, Any] | None:
    """Submit one prompt as one indivisible group and wait for its exact task."""

    if group_size < 1:
        raise ValueError(f"AReaL group_size must be positive, got {group_size}")
    if timeout <= 0:
        raise ValueError(f"AReaL task timeout must be positive, got {timeout}")
    if retries < 0:
        raise ValueError(f"AReaL retries must be non-negative, got {retries}")

    for attempt in range(retries + 1):
        submit_task = asyncio.create_task(
            asyncio.to_thread(
                controller.submit,
                data=data,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                group_size=group_size,
                is_eval=is_eval,
            ),
            name="areal_submit_prompt_group",
        )
        try:
            task_id = await asyncio.shield(submit_task)
        except asyncio.CancelledError as cancelled:
            # ``controller.submit`` can finish in its thread after this
            # coroutine is cancelled. Keep the submit alive long enough to
            # recover its exact task ID, then cancel that backend task rather
            # than leaving an unowned rollout behind.
            try:
                task_id = await asyncio.shield(submit_task)
            except BaseException:
                raise cancelled from None
            await asyncio.shield(asyncio.to_thread(controller.dispatcher.cancel_task, task_id))
            raise
        try:
            result = await asyncio.to_thread(
                controller.dispatcher.wait_for_task,
                task_id,
                timeout,
                True,
            )
        except TimeoutError:
            await asyncio.to_thread(controller.dispatcher.cancel_task, task_id)
            raise
        except asyncio.CancelledError:
            # Cancelling ``to_thread(wait_for_task, ...)`` only cancels this
            # await; its worker thread and the submitted AReaL task can keep
            # running. Cancel the exact backend task before propagating the
            # producer shutdown to the caller. Shield cleanup so the original
            # cancellation cannot interrupt the dispatcher cancellation RPC.
            await asyncio.shield(asyncio.to_thread(controller.dispatcher.cancel_task, task_id))
            raise

        trajectory = extract_task_trajectory(result, expected_task_id=task_id)
        if trajectory is not None:
            return trajectory
        if attempt < retries:
            await asyncio.sleep(0)
    return None
