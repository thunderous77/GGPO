from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.utils import fsdp_utils
from verl.utils.fsdp_utils import (
    fsdp1_sharded_load_from_cpu,
    fsdp1_sharded_save_to_cpu,
    fsdp2_sharded_load_from_cpu,
    fsdp2_sharded_save_to_cpu,
)


def _worker(strategy: str, module: torch.nn.Module | None = None) -> DetachActorWorker:
    worker = object.__new__(DetachActorWorker)
    worker.config = OmegaConf.create({"actor": {"strategy": strategy}})
    worker._strategy_handlers = None
    if module is not None:
        worker.actor = SimpleNamespace(engine=SimpleNamespace(module=module))
    return worker


def test_fsdp1_and_fsdp2_strategies_select_distinct_snapshot_handlers():
    fsdp1_worker = _worker("fsdp")
    fsdp2_worker = _worker("fsdp2")
    veomni_worker = _worker("veomni")

    assert fsdp1_worker.copy_handler is fsdp1_sharded_save_to_cpu
    assert fsdp1_worker.restore_handler is fsdp1_sharded_load_from_cpu
    assert fsdp2_worker.copy_handler is fsdp2_sharded_save_to_cpu
    assert fsdp2_worker.restore_handler is fsdp2_sharded_load_from_cpu
    assert veomni_worker.copy_handler is fsdp2_sharded_save_to_cpu
    assert veomni_worker.restore_handler is fsdp2_sharded_load_from_cpu


def test_fsdp1_snapshot_rejects_a_plain_module():
    with pytest.raises(TypeError, match="FSDP1 root module"):
        fsdp1_sharded_save_to_cpu(torch.nn.Linear(3, 2))


def test_fsdp1_worker_uses_the_two_argument_local_shard_contract(monkeypatch):
    module = torch.nn.Linear(1, 1)
    current_shard = torch.arange(4, dtype=torch.float32)
    expected = current_shard.clone()
    monkeypatch.setattr(fsdp_utils, "_validated_fsdp1_local_shards", lambda _: {"flat": current_shard})
    worker = _worker("fsdp", module)

    worker.save_model_to_cpu(1)
    assert worker.cpu_saved_models[1]["flat"].data_ptr() != current_shard.data_ptr()

    current_shard.add_(10)
    worker.restore_model_from_cpu(1)
    torch.testing.assert_close(current_shard, expected)


def test_fsdp1_restore_fails_closed_on_snapshot_model_mismatch():
    module = torch.nn.Linear(3, 2)
    expected_shard = torch.ones(4)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(fsdp_utils, "_validated_fsdp1_local_shards", lambda _: {"flat": expected_shard})
    snapshot = {"unexpected": torch.ones(4)}

    try:
        with pytest.raises(ValueError, match="parameter names do not match"):
            fsdp1_sharded_load_from_cpu(module, snapshot)
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize(
    ("snapshot", "error_type", "message"),
    [
        ({"flat": "not-a-tensor"}, TypeError, "is not a tensor"),
        ({"flat": torch.ones(3)}, ValueError, "does not match the model"),
        ({"flat": torch.ones(4, dtype=torch.float64)}, ValueError, "does not match the model"),
        ({"flat": torch.ones(4, device="meta")}, ValueError, "is not on CPU"),
    ],
)
def test_fsdp1_restore_validates_the_complete_snapshot_before_copy(monkeypatch, snapshot, error_type, message):
    current_shard = torch.zeros(4)
    monkeypatch.setattr(fsdp_utils, "_validated_fsdp1_local_shards", lambda _: {"flat": current_shard})

    with pytest.raises(error_type, match=message):
        fsdp1_sharded_load_from_cpu(torch.nn.Linear(1, 1), snapshot)

    torch.testing.assert_close(current_shard, torch.zeros(4))


def test_unsupported_strategy_fails_closed():
    worker = _worker("not-a-strategy")

    with pytest.raises(NotImplementedError, match="Unsupported strategy"):
        _ = worker.copy_handler
