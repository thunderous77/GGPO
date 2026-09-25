import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

pytest.importorskip("vllm")
pytestmark = pytest.mark.vllm

from areal.engine.vllm_ext.areal_vllm_server import (
    _call_build_app_compat,
    _pause_generation_compat,
    _resume_generation_compat,
)


def test_build_app_compat_omits_removed_vllm_arguments():
    def current_build_app(args):
        return {"args": args}

    assert _call_build_app_compat(
        current_build_app,
        "config",
        supported_tasks=("generate",),
        model_config="model",
    ) == {"args": "config"}


def test_build_app_compat_forwards_legacy_vllm_arguments():
    def legacy_build_app(args, supported_tasks=None, model_config=None):
        return args, supported_tasks, model_config

    assert _call_build_app_compat(
        legacy_build_app,
        "config",
        supported_tasks=("generate",),
        model_config="model",
    ) == ("config", ("generate",), "model")


def test_pause_resume_generation_compat_uses_legacy_api():
    llm = SimpleNamespace(
        pause_generation=AsyncMock(),
        resume_generation=AsyncMock(),
    )

    asyncio.run(_pause_generation_compat(llm))
    asyncio.run(_resume_generation_compat(llm))

    llm.pause_generation.assert_awaited_once_with(
        wait_for_inflight_requests=False,
        clear_cache=True,
    )
    llm.resume_generation.assert_awaited_once_with()


def test_pause_resume_generation_compat_uses_vllm_011_api():
    llm = SimpleNamespace(
        output_processor=SimpleNamespace(
            request_states={"request-a": object(), "request-b": object()}
        ),
        abort=AsyncMock(),
        reset_prefix_cache=AsyncMock(),
    )

    asyncio.run(_pause_generation_compat(llm))
    asyncio.run(_resume_generation_compat(llm))

    llm.abort.assert_awaited_once_with(["request-a", "request-b"])
    llm.reset_prefix_cache.assert_awaited_once_with()


def test_pause_generation_compat_handles_no_active_requests():
    llm = SimpleNamespace(
        output_processor=SimpleNamespace(request_states={}),
        abort=AsyncMock(),
        reset_prefix_cache=AsyncMock(),
    )

    asyncio.run(_pause_generation_compat(llm))

    llm.abort.assert_not_awaited()
    llm.reset_prefix_cache.assert_awaited_once_with()
