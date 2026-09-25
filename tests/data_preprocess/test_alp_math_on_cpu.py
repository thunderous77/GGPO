import json
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from examples.data_preprocess import alp_math


def test_resolve_revision_uses_dataset_info_api_without_repo_type():
    class FakeHfApi:
        def __init__(self):
            self.calls = []

        def dataset_info(self, repo_id):
            self.calls.append(repo_id)
            return SimpleNamespace(sha="dataset-sha")

    api = FakeHfApi()

    assert alp_math.resolve_revision("repo/id", hf_api=api) == "dataset-sha"
    assert api.calls == ["repo/id"]


def test_openr1_filter_uses_exclusive_point_two_point_eight_bounds():
    assert alp_math.keep_openr1_row({"scores": [0.2, 0.2]}, min_score=0.2, max_score=0.8) is False
    assert alp_math.keep_openr1_row({"scores": [0.8, 0.8]}, min_score=0.2, max_score=0.8) is False
    assert alp_math.keep_openr1_row({"scores": [0.4, 0.6]}, min_score=0.2, max_score=0.8) is True


def test_openr1_conversion_strips_chat_template_and_adds_alp_prompt():
    row = {
        "prompt": [
            {
                "role": "user",
                "content": (
                    "<|im_start|>system\nsys<|im_end|>\n"
                    "<|im_start|>user\nWhat is 1+1?<|im_end|>\n"
                    "<|im_start|>assistant"
                ),
            }
        ],
        "answer": "2",
        "scores": [1, 0, 1],
    }

    record = alp_math.convert_openr1_row(row, index=3, revision="rev-openr1")

    assert record["data_source"] == "alp_openr1_filtered"
    assert record["prompt"] == [
        {"role": "system", "content": alp_math.SYSTEM_PROMPT},
        {"role": "user", "content": f"What is 1+1? {alp_math.USER_PROMPT_SUFFIX}"},
    ]
    assert record["reward_model"] == {"style": "rule", "ground_truth": "2"}
    assert record["extra_info"]["avg_score"] == 2 / 3
    assert record["extra_info"]["source_revision"] == "rev-openr1"


def test_guru_conversion_replaces_old_suffix_and_uses_alp_data_source():
    row = {
        "prompt": [{"role": "user", "content": f"Solve x. {alp_math.OLD_GURU_SUFFIX}"}],
        "reward_model": {"ground_truth": "x"},
    }

    record = alp_math.convert_guru_row(
        row,
        index=1,
        revision="rev-guru",
        split="train",
        data_source="alp_guru_rl92k_math",
        source_file=alp_math.GURU_TRAIN_FILE,
    )

    assert record["data_source"] == "alp_guru_rl92k_math"
    assert record["prompt"][0] == {"role": "system", "content": alp_math.SYSTEM_PROMPT}
    assert record["prompt"][1]["content"] == f"Solve x. {alp_math.USER_PROMPT_SUFFIX}"
    assert record["prompt"][1]["content"].count(alp_math.USER_PROMPT_SUFFIX) == 1
    assert record["extra_info"]["source_file"] == alp_math.GURU_TRAIN_FILE


def test_guru_conversion_accepts_numpy_prompt_arrays_from_parquet():
    row = {
        "prompt": np.array([{"role": "user", "content": "A parquet-loaded problem."}], dtype=object),
        "answer": "42",
    }

    record = alp_math.convert_guru_row(
        row,
        index=2,
        revision="rev-guru",
        split="train",
        data_source="alp_guru_rl92k_math",
        source_file=alp_math.GURU_TRAIN_FILE,
    )

    assert record["prompt"][1]["content"] == f"A parquet-loaded problem. {alp_math.USER_PROMPT_SUFFIX}"
    assert "{'role':" not in record["prompt"][1]["content"]


def test_merge_train_records_keeps_guru_before_duplicate_openr1():
    guru = alp_math.convert_guru_row(
        {"prompt": [{"role": "user", "content": "Duplicate problem"}], "answer": "guru"},
        index=0,
        revision="rev-guru",
        split="train",
        data_source="alp_guru_rl92k_math",
        source_file=alp_math.GURU_TRAIN_FILE,
    )
    openr1_duplicate = alp_math.convert_openr1_row(
        {"prompt": "Duplicate problem", "answer": "openr1", "scores": [0.5]},
        index=0,
        revision="rev-openr1",
    )
    openr1_unique = alp_math.convert_openr1_row(
        {"prompt": "Unique problem", "answer": "openr1-unique", "scores": [0.5]},
        index=1,
        revision="rev-openr1",
    )

    merged = alp_math.merge_train_records([guru], [openr1_duplicate, openr1_unique])

    assert [record["reward_model"]["ground_truth"] for record in merged] == ["guru", "openr1-unique"]
    assert [record["data_source"] for record in merged] == ["alp_guru_rl92k_math", "alp_openr1_filtered"]


def test_write_outputs_writes_expected_parquets_and_manifest(tmp_path):
    train = [
        alp_math.convert_openr1_row({"prompt": "Q1", "answer": "A1", "scores": [0.5]}, 0, "rev-openr1"),
    ]
    math500 = [
        alp_math.convert_guru_row(
            {"prompt": [{"role": "user", "content": "Q2"}], "answer": "A2"},
            index=0,
            revision="rev-guru",
            split="math500",
            data_source="alp_math500",
            source_file=alp_math.GURU_MATH500_FILE,
        )
    ]

    manifest = alp_math.write_outputs(train, math500, tmp_path, metadata={"dedup_keep": "guru first"})

    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    assert manifest["splits"]["train"]["row_count"] == 1
    assert manifest["splits"]["math500"]["row_count"] == 1
    assert pq.read_table(tmp_path / "train.parquet").to_pylist() == train
    assert pq.read_table(tmp_path / "math500.parquet").to_pylist() == math500


def test_build_outputs_passes_pinned_evaluation_revisions(monkeypatch, tmp_path):
    downloads = []

    def load_dataset(repo_id, *, split, revision, cache_dir):
        downloads.append((repo_id, split, revision))
        return [{"problem": repo_id, "answer": "42", "scores": [0.5]}]

    monkeypatch.setattr(alp_math, "load_dataset", load_dataset)
    monkeypatch.setattr(alp_math, "_load_guru_file", lambda *args, **kwargs: [])
    alp_math.build_outputs(tmp_path, openr1_revision="rev-openr1", guru_revision="rev-guru")

    assert downloads == [(alp_math.OPENR1_REPO, "train", "rev-openr1")] + [
        (repo_id, "test", alp_math.EVAL_REVISIONS[name]) for name, repo_id in alp_math.EVAL_DATASETS.items()
    ]
    for name in alp_math.EVAL_DATASETS:
        assert pq.read_table(tmp_path / f"{name}.parquet").num_rows == 1
