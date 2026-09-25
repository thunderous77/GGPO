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

"""CPU-only contract tests for transient Hugging Face actor exports."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from verl.workers.engine.base import BaseEngine
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker


class _UnsupportedEngine(BaseEngine):
    @property
    def is_param_offload_enabled(self) -> bool:
        return False

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return False


def _make_fsdp_engine_stub(*, previous_global_step=17):
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.checkpoint_manager = SimpleNamespace(
        checkpoint_save_contents=["model", "optimizer", "extra"],
        previous_global_step=previous_global_step,
        previous_saved_paths=["/formal/checkpoint-1", "/formal/checkpoint-2"],
    )
    return engine


def test_unsupported_engine_rejects_transient_hf_export():
    with pytest.raises(NotImplementedError, match="does not support transient Hugging Face model export"):
        _UnsupportedEngine().export_hf_model("/transient/export")


def test_fsdp_export_selects_only_hf_model_and_restores_manager_state():
    engine = _make_fsdp_engine_stub()
    manager = engine.checkpoint_manager
    original_save_contents = manager.checkpoint_save_contents
    original_saved_paths = manager.previous_saved_paths
    calls = []

    def _fake_save_checkpoint(**kwargs):
        assert manager.checkpoint_save_contents == ["hf_model"]
        assert manager.previous_saved_paths == []
        calls.append(kwargs)
        manager.previous_global_step = kwargs["global_step"]
        manager.previous_saved_paths.append(kwargs["local_path"])

    engine.save_checkpoint = _fake_save_checkpoint

    FSDPEngine.export_hf_model(engine, "/transient/export")

    assert calls == [
        {
            "local_path": "/transient/export",
            "hdfs_path": None,
            "global_step": 17,
            "max_ckpt_to_keep": None,
        }
    ]
    assert manager.checkpoint_save_contents is original_save_contents
    assert manager.previous_global_step == 17
    assert manager.previous_saved_paths is original_saved_paths
    assert manager.previous_saved_paths == ["/formal/checkpoint-1", "/formal/checkpoint-2"]


def test_fsdp_export_restores_manager_state_after_failure():
    engine = _make_fsdp_engine_stub(previous_global_step=None)
    manager = engine.checkpoint_manager
    original_save_contents = manager.checkpoint_save_contents
    original_saved_paths = manager.previous_saved_paths

    def _failing_save_checkpoint(**kwargs):
        assert kwargs["global_step"] == 0
        assert manager.checkpoint_save_contents == ["hf_model"]
        manager.previous_global_step = 0
        manager.previous_saved_paths.append(kwargs["local_path"])
        raise RuntimeError("export failed")

    engine.save_checkpoint = _failing_save_checkpoint

    with pytest.raises(RuntimeError, match="export failed"):
        FSDPEngine.export_hf_model(engine, "/transient/export")

    assert manager.checkpoint_save_contents is original_save_contents
    assert manager.previous_global_step is None
    assert manager.previous_saved_paths is original_saved_paths
    assert manager.previous_saved_paths == ["/formal/checkpoint-1", "/formal/checkpoint-2"]


def test_worker_layers_delegate_actor_hf_export():
    training_worker = TrainingWorker.__new__(TrainingWorker)
    training_worker.engine = MagicMock()
    training_worker.engine.export_hf_model.return_value = "engine-result"

    result = TrainingWorker.export_hf_model(training_worker, "/transient/export")

    assert result == "engine-result"
    training_worker.engine.export_hf_model.assert_called_once_with("/transient/export")

    actor_worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    actor_worker.role = "actor"
    actor_worker.actor = MagicMock()
    actor_worker.actor.export_hf_model.return_value = "actor-result"

    result = ActorRolloutRefWorker.export_hf_model(actor_worker, "/transient/export")

    assert result == "actor-result"
    actor_worker.actor.export_hf_model.assert_called_once_with("/transient/export")


def test_actor_worker_rejects_hf_export_without_actor_role():
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.role = "ref"
    worker.actor = None

    with pytest.raises(AssertionError, match="only supports actor role"):
        ActorRolloutRefWorker.export_hf_model(worker, "/transient/export")
