import asyncio
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import ray
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.reward_loop.reward_loop import BatchRewardLoopWorker
from verl.experimental.reward_loop.reward_manager.batch import BatchRewardManager
from verl.utils.reward_score import math_verify

_TEST_DIR = str(Path(__file__).parent)


def _init_ray_for_test():
    pythonpath = f"{_TEST_DIR}:{os.environ.get('PYTHONPATH', '')}"
    os.environ["PYTHONPATH"] = pythonpath
    ray.init(
        num_cpus=1,
        num_gpus=0,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
        runtime_env={"env_vars": {"PYTHONPATH": pythonpath}},
    )


class _BoxedTokenizer:
    eos_token = None

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return rf"\boxed{{{int(ids[0])}}}"


def _batch_for_boxed_answer(answer: int, ground_truth: str = "6"):
    batch = TensorDict(
        {
            "prompts": torch.tensor([[101]]),
            "responses": torch.tensor([[answer, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        },
        batch_size=[1],
    )
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "reward_model": np.array([{"ground_truth": ground_truth}], dtype=object),
            "data_source": np.array(["alp_guru_rl92k_math"], dtype=object),
            "extra_info": np.array([{}], dtype=object),
        },
    )


def test_batch_reward_manager_scores_boxed_math_on_main_thread(monkeypatch):
    def fake_compute_boxed_score(model_output, ground_truth, timeout_score):
        del timeout_score
        assert threading.current_thread() is threading.main_thread()
        return 1.0 if model_output == r"\boxed{6}" and ground_truth == "6" else 0.0

    monkeypatch.setattr(math_verify, "_compute_boxed_score", fake_compute_boxed_score)
    manager = BatchRewardManager(config=SimpleNamespace(), tokenizer=_BoxedTokenizer(), compute_score=None)

    outputs = asyncio.run(manager.run_batch(_batch_for_boxed_answer(6)))

    assert outputs == [{"reward_score": 1.0, "reward_extra_info": {"acc": 1.0, "valid_response_length": 1}}]


class _RecordingBatchRewardManager:
    def run_batch_sync(self, data):
        assert len(data) == 1
        assert threading.current_thread() is threading.main_thread()
        return [
            {
                "reward_score": 1.0,
                "reward_extra_info": {
                    "acc": 1.0,
                    "valid_response_length": 1,
                    "thread": threading.current_thread().name,
                },
            }
        ]


class _ProbeBatchRewardLoopWorker(BatchRewardLoopWorker):
    def __init__(self):
        self.reward_manager = _RecordingBatchRewardManager()


class _RealBatchRewardLoopWorker(BatchRewardLoopWorker):
    def __init__(self):
        self.reward_manager = BatchRewardManager(config=SimpleNamespace(), tokenizer=_BoxedTokenizer(), compute_score=None)


def test_batch_reward_loop_worker_scores_on_ray_main_thread():
    ray.shutdown()
    _init_ray_for_test()
    try:
        worker = ray.remote(_ProbeBatchRewardLoopWorker).remote()
        outputs = ray.get(worker.compute_score_batch.remote(_batch_for_boxed_answer(6)))
    finally:
        ray.shutdown()

    assert outputs == [
        {
            "reward_score": 1.0,
            "reward_extra_info": {"acc": 1.0, "valid_response_length": 1, "thread": "MainThread"},
        }
    ]


def test_batch_reward_loop_worker_scores_real_math_verify_in_ray_main_thread():
    ray.shutdown()
    _init_ray_for_test()
    try:
        worker = ray.remote(_RealBatchRewardLoopWorker).remote()
        outputs = ray.get(worker.compute_score_batch.remote(_batch_for_boxed_answer(6)))
    finally:
        ray.shutdown()

    assert outputs == [{"reward_score": 1.0, "reward_extra_info": {"acc": 1.0, "valid_response_length": 1}}]
