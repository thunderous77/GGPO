# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ray")
pytest.importorskip("vllm")

from verl.trainer.ppo.ray_trainer import compute_spec_decode_metrics
from verl.workers.rollout.vllm_rollout import vllm_async_server


class _FakeMtpEngine:
    """Minimal model of the MTP failure mode seen in hybrid sleep."""

    def __init__(self):
        self.mtp_drafter_available = True
        self.sleep_levels_that_discard_mtp_drafter = {2}

    async def sleep(self, level: int):
        if level in self.sleep_levels_that_discard_mtp_drafter:
            self.mtp_drafter_available = False

    async def reset_encoder_cache(self):
        pass

    def sync_actor_weights(self):
        pass

    def generate_spec_decode_stats(self):
        num_draft_tokens = 3
        num_accepted_tokens = num_draft_tokens if self.mtp_drafter_available else 0
        num_verify_steps = 1
        return num_draft_tokens, num_accepted_tokens, num_verify_steps


class _Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def _make_minimal_vllm_server(skip_tokenizer_init: bool):
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = _Config(
        engine_kwargs={},
        limit_images=None,
        cudagraph_capture_sizes=None,
        dtype="bfloat16",
        load_format="dummy",
        skip_tokenizer_init=skip_tokenizer_init,
        enable_chunked_prefill=True,
        max_model_len=32,
        max_num_seqs=1,
        max_num_batched_tokens=32,
        enable_prefix_caching=True,
        enable_sleep_mode=True,
        logprobs_mode="raw_logprobs",
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        disable_log_stats=True,
        tensor_model_parallel_size=1,
        seed=1,
        scheduling_policy="fcfs",
        prometheus=_Config(enable=False, served_model_name=None),
        mtp=_Config(enable=False, enable_rollout=False),
        data_parallel_size=1,
        expert_parallel_size=1,
        enable_rollout_routing_replay=False,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        response_length=16,
        quantization=None,
        qat={},
    )
    server.model_config = _Config(
        local_path="Qwen/Qwen3-1.7B-Base",
        trust_remote_code=True,
        lora={},
        lora_rank=0,
    )
    server.profiler_controller = _Config(config=None, tool_config=None)
    server.replica_rank = 0
    server.node_rank = 0
    server.gpus_per_node = 1
    server.nnodes = 1
    server.workers = []
    server._master_address = "127.0.0.1"
    server._master_port = 12345
    server._dp_rpc_port = 12346
    server._dp_master_port = 12347
    return server


def test_mtp_hybrid_sleep_keeps_drafter_available_for_nonzero_acceptance(monkeypatch):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)

    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(mtp=SimpleNamespace(enable=True, enable_rollout=True))
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    server.engine = _FakeMtpEngine()

    asyncio.run(server._sleep_hybrid())
    server.engine.sync_actor_weights()
    drafts, accepts, verifies = server.engine.generate_spec_decode_stats()
    metrics = compute_spec_decode_metrics(
        spec_drafts=np.array([drafts]),
        spec_accepts=np.array([accepts]),
        spec_verifies=np.array([verifies]),
    )

    assert metrics["rollout/spec_accept_rate"] > 0.0
    assert metrics["rollout/spec_accept_length"] > 1.0


@pytest.mark.parametrize("skip_tokenizer_init", [True, False])
def test_vllm_server_honors_skip_tokenizer_init_config(monkeypatch, skip_tokenizer_init):
    captured = {}

    async def fake_run_server(self, args):
        captured["skip_tokenizer_init"] = args.skip_tokenizer_init

    monkeypatch.setattr(vllm_async_server, "build_vllm_profiler_args", lambda *args, **kwargs: {})
    monkeypatch.setattr(vllm_async_server.vLLMHttpServer, "run_server", fake_run_server)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    server = _make_minimal_vllm_server(skip_tokenizer_init)
    asyncio.run(server.launch_server())

    assert captured["skip_tokenizer_init"] is skip_tokenizer_init
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    assert "PYTORCH_ALLOC_CONF" not in os.environ
