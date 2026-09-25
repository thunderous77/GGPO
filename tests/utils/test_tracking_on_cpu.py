# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import sys
import types
from unittest.mock import MagicMock, call, patch

import pytest

from verl.utils.tracking import Tracking, ValidationGenerationsLogger


def test_tracking_resumes_wandb_run_from_environment(monkeypatch):
    mock_wandb = MagicMock()
    monkeypatch.setenv("WANDB_ENTITY", "mismatch")
    monkeypatch.setenv("WANDB_RUN_ID", "abc123")
    monkeypatch.delenv("WANDB_RESUME", raising=False)

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        tracking = Tracking("project", "experiment", default_backend=["wandb"], config={"trainer": {}})

    init_kwargs = mock_wandb.init.call_args.kwargs
    assert init_kwargs["project"] == "project"
    assert init_kwargs["name"] == "experiment"
    assert init_kwargs["entity"] == "mismatch"
    assert init_kwargs["id"] == "abc123"
    assert init_kwargs["resume"] == "allow"
    del tracking


def test_tracking_respects_explicit_wandb_resume_policy(monkeypatch):
    mock_wandb = MagicMock()
    monkeypatch.setenv("WANDB_RUN_ID", "abc123")
    monkeypatch.setenv("WANDB_RESUME", "must")

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        tracking = Tracking("project", "experiment", default_backend=["wandb"], config={"trainer": {}})

    init_kwargs = mock_wandb.init.call_args.kwargs
    assert init_kwargs["id"] == "abc123"
    assert init_kwargs["resume"] == "must"
    del tracking


@pytest.mark.parametrize(
    "config",
    [
        {"trainer": {}, "hybrid_data": {"historical_groups": 192, "on_policy_groups": 64}},
        {
            "trainer": {},
            "algorithm": {
                "filter_groups": {"enable": True, "metric": "raw_correctness", "max_num_gen_batches": 32}
            },
        },
    ],
)
def test_tracking_binds_canonical_validation_to_global_step(config):
    mock_wandb = MagicMock()

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        tracking = Tracking(
            "project",
            "experiment",
            default_backend=["wandb"],
            config=config,
        )

    assert mock_wandb.define_metric.call_args_list == [
        call("training/global_step"),
        call("val-core/*", step_metric="training/global_step"),
    ]
    del tracking


def test_tracking_canonicalizes_ggpo_wandb_name(monkeypatch):
    mock_wandb = MagicMock()
    config = {
        "actor_rollout_ref": {
            "actor": {
                "policy_loss": {
                    "loss_mode": "ggpo",
                    "sequence_log_prob_reduction": "mean",
                    "beta": 0.1,
                },
                "use_kl_loss": False,
            },
            "model": {"path": "Qwen/Qwen3-1.7B-Base"},
        },
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
            "rollout_correction": {"bypass_mode": False},
        },
        "data": {"train_files": ["/data/alp/train.parquet"]},
        "trainer": {"total_training_steps": 20},
    }

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        tracking = Tracking(
            "Gibbs-Gap Policy Optimization",
            "manual-name-is-ignored",
            default_backend=["wandb"],
            config=config,
        )

    assert mock_wandb.init.call_args.kwargs["name"] == ("ggpo-exp-seqmean-beta0.1-qwen3-1.7b-base-maxstep20")
    del tracking


def test_tracking_offsets_resumed_wandb_history_step(monkeypatch):
    mock_wandb = MagicMock()
    monkeypatch.setenv("WANDB_RUN_ID", "abc123")
    monkeypatch.setenv("WANDB_RESUME", "allow")
    monkeypatch.setenv("WANDB_STEP_OFFSET", "10")

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        tracking = Tracking("project", "experiment", default_backend=["wandb"], config={"trainer": {}})
        tracking.log({"training/global_step": 1, "critic/score/mean": 0.1}, step=1)

    mock_wandb.log.assert_called_once_with(
        data={"training/global_step": 1, "critic/score/mean": 0.1},
        step=11,
    )
    del tracking


def test_tracking_rejects_non_integer_wandb_step_offset(monkeypatch):
    mock_wandb = MagicMock()
    monkeypatch.setenv("WANDB_STEP_OFFSET", "not-an-integer")

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        try:
            Tracking("project", "experiment", default_backend=["wandb"], config={"trainer": {}})
        except ValueError as exc:
            assert str(exc) == "WANDB_STEP_OFFSET must be an integer"
        else:
            raise AssertionError("expected a non-integer W&B step offset to fail")


def test_validation_generations_logger_logs_trackio_traces():
    mock_trackio = MagicMock()
    mock_trackio.context_vars = types.SimpleNamespace(current_run=MagicMock())
    mock_trackio.context_vars.current_run.get.return_value = None
    mock_trackio.Trace.side_effect = lambda messages, metadata=None: {
        "_type": "trackio.trace",
        "messages": messages,
        "metadata": metadata or {},
    }

    with patch.dict(sys.modules, {"trackio": mock_trackio}):
        ValidationGenerationsLogger().log(
            ["trackio"],
            samples=[["question", "answer", 0.5]],
            step=7,
        )

    mock_trackio.Trace.assert_called_once()
    trace_kwargs = mock_trackio.Trace.call_args.kwargs
    assert trace_kwargs["messages"] == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    assert trace_kwargs["metadata"]["source"] == "validation_generations"
    assert trace_kwargs["metadata"]["score"] == 0.5
    mock_trackio.log.assert_called_once()
    assert mock_trackio.log.call_args.kwargs["step"] == 7
