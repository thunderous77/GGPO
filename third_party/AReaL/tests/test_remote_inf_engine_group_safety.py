import logging
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from areal.api import ModelRequest, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import HttpGenerationResult, HttpRequest
from areal.infra import remote_inf_engine
from areal.infra.remote_inf_engine import (
    REQUIRE_COMPLETE_GROUP_KEY,
    GroupedRolloutWorkflow,
    RemoteInfEngine,
)


class _Backend:
    def __init__(self) -> None:
        self.request_versions: list[int] = []

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool, version: int
    ) -> HttpRequest:
        self.request_versions.append(version)
        return HttpRequest(endpoint="/generate", payload={})

    def parse_generation_response(
        self, response: dict[str, Any]
    ) -> HttpGenerationResult:
        request_idx = response["request_idx"]
        return HttpGenerationResult(
            output_tokens=[10 + request_idx],
            output_logprobs=[-0.1],
            stop_reason="abort" if request_idx == 0 else "stop",
        )


class _SequenceWorkflow(RolloutWorkflow):
    def __init__(self, results: list[dict[str, torch.Tensor] | None | BaseException]):
        self.results = results
        self.next_result = 0
        self.seen_data: list[dict[str, Any]] = []

    async def arun_episode(
        self, engine: Any, data: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        self.seen_data.append(data)
        result = self.results[self.next_result]
        self.next_result += 1
        if isinstance(result, BaseException):
            raise result
        return result


def _trajectory(token: int) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[token]], dtype=torch.long),
        "attention_mask": torch.ones((1, 1), dtype=torch.bool),
    }


@pytest.mark.asyncio
async def test_agenerate_stamps_tokens_with_each_request_version(monkeypatch):
    """Tokens retain the version captured before their HTTP request was sent."""
    # Arrange
    backend = _Backend()
    config = SimpleNamespace(
        request_retries=0,
        request_timeout=1,
        return_routed_experts=False,
        routing_strategy="round_robin",
        use_lora=False,
    )
    engine = RemoteInfEngine(config=config, backend=backend)
    engine.addresses = ["inference-server"]
    engine._workflow_executor = SimpleNamespace(is_paused=lambda: False)
    engine.set_version(4)

    async def get_session():
        return object()

    request_count = 0

    async def request_with_version_change(**kwargs):
        nonlocal request_count
        request_idx = request_count
        request_count += 1
        engine.set_version(7 if request_idx == 0 else 9)
        return {"request_idx": request_idx}

    monkeypatch.setattr(
        remote_inf_engine.workflow_context,
        "get_aiohttp_session",
        get_session,
    )
    monkeypatch.setattr(
        remote_inf_engine,
        "arequest_with_retry",
        request_with_version_change,
    )
    req = ModelRequest(
        input_ids=[1],
        gconfig=GenerationHyperparameters(max_new_tokens=2, max_tokens=3),
    )

    # Act
    response = await engine.agenerate(req)

    # Assert
    assert backend.request_versions == [4, 7]
    assert response.output_versions == [4, 7]
    assert engine.get_version() == 9


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError("generation failed")])
async def test_grouped_rollout_strict_mode_rejects_incomplete_group(failure):
    """Strict per-task mode rejects a group containing None or an exception."""
    # Arrange
    workflow = _SequenceWorkflow([_trajectory(1), failure, _trajectory(3)])
    grouped = GroupedRolloutWorkflow(
        workflow=workflow,
        group_size=3,
        logger=logging.getLogger("GroupedRolloutWorkflowTest"),
    )
    data = {"prompt": "question", REQUIRE_COMPLETE_GROUP_KEY: True}

    # Act
    result = await grouped.arun_episode(engine=object(), data=data)

    # Assert
    assert result is None
    assert len(workflow.seen_data) == 3
    assert all(REQUIRE_COMPLETE_GROUP_KEY not in item for item in workflow.seen_data)
    assert data[REQUIRE_COMPLETE_GROUP_KEY] is True


@pytest.mark.asyncio
async def test_grouped_rollout_strict_mode_keeps_complete_group():
    """Strict per-task mode still concatenates a fully successful group."""
    # Arrange
    workflow = _SequenceWorkflow([_trajectory(1), _trajectory(2), _trajectory(3)])
    grouped = GroupedRolloutWorkflow(
        workflow=workflow,
        group_size=3,
        logger=logging.getLogger("GroupedRolloutWorkflowTest"),
    )

    # Act
    result = await grouped.arun_episode(
        engine=object(),
        data={"prompt": "question", REQUIRE_COMPLETE_GROUP_KEY: True},
    )

    # Assert
    assert result is not None
    assert result["input_ids"].tolist() == [[1], [2], [3]]


@pytest.mark.asyncio
async def test_grouped_rollout_default_mode_salvages_partial_group():
    """The default AReaL behavior continues to keep successful samples."""
    # Arrange
    workflow = _SequenceWorkflow(
        [_trajectory(1), RuntimeError("generation failed"), _trajectory(3)]
    )
    grouped = GroupedRolloutWorkflow(
        workflow=workflow,
        group_size=3,
        logger=logging.getLogger("GroupedRolloutWorkflowTest"),
    )

    # Act
    result = await grouped.arun_episode(
        engine=object(),
        data={"prompt": "question"},
    )

    # Assert
    assert result is not None
    assert result["input_ids"].tolist() == [[1], [3]]
