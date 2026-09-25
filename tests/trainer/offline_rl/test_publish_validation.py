# Copyright 2026 OpenAI

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verl.trainer.offline_rl.publish_validation import (
    VALIDATION_METRICS,
    _wait_for_publication,
    load_evaluation_receipt,
    training_history_offset,
    validation_internal_step,
)


def test_validation_and_training_history_steps_are_strictly_monotonic():
    validation_steps = [validation_internal_step(step) for step in range(0, 101, 20)]
    assert validation_steps == [0, 21, 42, 63, 84, 105]
    assert [training_history_offset(step) for step in range(0, 100, 20)] == [0, 1, 2, 3, 4]
    complete = [0]
    for resume in range(0, 100, 20):
        offset = training_history_offset(resume)
        complete.extend(step + offset for step in range(resume + 1, resume + 21))
        complete.append(validation_internal_step(resume + 20))
    assert complete == list(range(106))


def test_fifty_round_cut_reserves_the_row_after_its_training_row():
    # A run that stops at round 50 ends its history at 52 (round 50 + three
    # preceding validation rows would be 53, and 0/20/40 contribute three).
    assert validation_internal_step(50) == 53
    assert validation_internal_step(40) == 42


@pytest.mark.parametrize("step", [-20, 1, 120])
def test_invalid_validation_boundary_rejected(step):
    with pytest.raises(ValueError, match="validation step"):
        validation_internal_step(step)


def test_evaluation_receipt_requires_exact_parent_and_three_finite_metrics(tmp_path: Path):
    checkpoint = tmp_path / "global_step_20"
    checkpoint.mkdir()
    receipt = tmp_path / "evaluation.json"
    metrics = {key: 0.25 for key in VALIDATION_METRICS}
    metrics["val-aux/alp_math500/valid_response_length/mean@32"] = 123.0
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "accepted",
                "optimizer_step": 20,
                "user_step": 20,
                "wandb_step_offset": 1,
                "wandb_run_id": "run12345",
                "experiment_name": "agro-seqsum-beta0.01-qwen3-1.7b-base-offline-maxstep100",
                "checkpoint": str(checkpoint),
                "metrics": metrics,
            }
        )
    )
    _, step, accepted = load_evaluation_receipt(
        receipt,
        expected_run_id="run12345",
        expected_name="agro-seqsum-beta0.01-qwen3-1.7b-base-offline-maxstep100",
    )
    assert step == 20
    assert accepted == {key: 0.25 for key in VALIDATION_METRICS}


def test_evaluation_receipt_rejects_nonfinite_required_metric(tmp_path: Path):
    checkpoint = tmp_path / "global_step_0"
    checkpoint.mkdir()
    receipt = tmp_path / "evaluation.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "accepted",
                "optimizer_step": 0,
                "user_step": 0,
                "wandb_step_offset": 0,
                "wandb_run_id": "run12345",
                "experiment_name": "asympo-core-qwen3-1.7b-base-offline-maxstep100",
                "checkpoint": str(checkpoint),
                "metrics": {**{key: 0.5 for key in VALIDATION_METRICS}, VALIDATION_METRICS[0]: float("nan")},
            }
        )
    )
    with pytest.raises(ValueError, match="not finite"):
        load_evaluation_receipt(
            receipt,
            expected_run_id="run12345",
            expected_name="asympo-core-qwen3-1.7b-base-offline-maxstep100",
        )


class SummarySubDictLike:
    """Match W&B SummarySubDict's mapping protocol without inheriting Mapping."""

    def __init__(self, data):
        self._data = data

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)


INDEX_ENTRY = {"internal_step": 0, "checkpoint": "/ckpt/global_step_0"}
# The public API flattens the nested index into dotted keys (what a fresh readback sees);
# the live SDK Summary keeps it nested. Both must be recognised.
INDEX_SHAPES = {
    "flattened": {
        "_offline_validation_index.0.internal_step": 0,
        "_offline_validation_index.0.checkpoint": "/ckpt/global_step_0",
    },
    "nested": {"_offline_validation_index": SummarySubDictLike({"0": SummarySubDictLike(INDEX_ENTRY)})},
}


@pytest.mark.parametrize("index_shape", sorted(INDEX_SHAPES))
def test_publication_readback_uses_supported_scan_history_signature(index_shape):
    metrics = {key: 0.25 for key in VALIDATION_METRICS}

    class Remote:
        id = "run12345"
        name = "agro"
        entity = "test-entity"
        project = "test-project"
        lastHistoryStep = 0
        summary = {**metrics, **INDEX_SHAPES[index_shape]}

        def scan_history(self, *, keys, min_step, max_step, page_size):
            assert keys == ["_step", "training/global_step", *VALIDATION_METRICS]
            assert (min_step, max_step, page_size) == (0, 1, 10)
            return [{"_step": 0, "training/global_step": 0, **metrics}]

    class Api:
        def run(self, locator):
            assert locator == "test-entity/test-project/run12345"
            return Remote()

    class Wandb:
        @staticmethod
        def Api(timeout):
            assert timeout == 60
            return Api()

    readback = _wait_for_publication(
        Wandb,
        locator="test-entity/test-project/run12345",
        identity=("run12345", "test-entity", "test-project"),
        internal_step=0,
        optimizer_step=0,
        metrics=metrics,
        timeout_seconds=1,
    )
    assert readback["last_history_step"] == 0
    assert readback["summary_index_step"] == INDEX_ENTRY


def test_publication_readback_surfaces_nontransient_api_errors():
    class Wandb:
        @staticmethod
        def Api(timeout):
            assert timeout == 60
            raise TypeError("unsupported API signature")

    with pytest.raises(TypeError, match="unsupported API signature"):
        _wait_for_publication(
            Wandb,
            locator="test-entity/test-project/run12345",
            identity=("run12345", "test-entity", "test-project"),
            internal_step=0,
            optimizer_step=0,
            metrics={key: 0.25 for key in VALIDATION_METRICS},
            timeout_seconds=60,
        )
