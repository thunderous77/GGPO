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

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

OPENR1_REPO = "weqweasdas/from_default_filtered_openr1_with_scores"
GURU_REPO = "LLM360/guru-RL-92k"
GURU_TRAIN_FILE = "train/math__combined_54.4k.parquet"
GURU_MATH500_FILE = "online_eval/math__math_500.parquet"
EVAL_DATASETS = {
    "math500": "HuggingFaceH4/MATH-500",
    "aime24": "math-ai/aime24",
    "aime25": "math-ai/aime25",
    "olympiadbench": "math-ai/olympiadbench",
    "minerva_math": "math-ai/minervamath",
}
EVAL_REVISIONS = {
    "math500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "aime24": "83a7f387baaa524a8bda0022eac0541582297103",
    "aime25": "563bb8404243c5f09de6ec262f2db674fe5bce9b",
    "olympiadbench": "4faaf1e6ec17d11a4218a9bf4c049ecaf954dd84",
    "minerva_math": "ee46ddc498933b1977577953250ca5c66be64f96",
}

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
USER_PROMPT_SUFFIX = "Let's think step by step and output the final answer within \\boxed{}."
OLD_GURU_SUFFIX = "Please output the final answer within \\boxed{}."
OUTPUT_COLUMNS = ("data_source", "prompt", "ability", "reward_model", "extra_info")


def resolve_revision(repo_id: str, hf_api: HfApi | None = None) -> str:
    api = HfApi() if hf_api is None else hf_api
    revision = api.dataset_info(repo_id).sha
    if not revision:
        raise ValueError(f"Could not resolve an immutable revision for dataset {repo_id!r}")
    return revision


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, dict | list | tuple):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    return value


def _as_messages(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, Mapping):
        return [dict(prompt)]
    if hasattr(prompt, "tolist") and not isinstance(prompt, str):
        prompt = prompt.tolist()
    if isinstance(prompt, list | tuple):
        messages = []
        for message in prompt:
            if isinstance(message, Mapping):
                messages.append(dict(message))
        if messages:
            return messages
    return [{"role": "user", "content": str(prompt).strip()}]


def strip_openr1_chat_template(prompt: Any) -> str:
    messages = _as_messages(prompt)
    content = str(messages[0].get("content", "") if messages else prompt)
    content = re.sub(r"<\|im_start\|>system\n.*?<\|im_end\|>\n", "", content, flags=re.DOTALL)
    content = re.sub(r"<\|im_start\|>user\n", "", content)
    content = re.sub(r"<\|im_end\|>\n<\|im_start\|>assistant.*$", "", content, flags=re.DOTALL)
    content = re.sub(r"<\|im_end\|>", "", content)
    return content.strip()


def _strip_suffixes(content: str) -> str:
    stripped = str(content).strip()
    for suffix in (USER_PROMPT_SUFFIX, OLD_GURU_SUFFIX):
        while stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)].strip()
    return stripped


def alp_prompt(problem: Any) -> list[dict[str, str]]:
    problem_text = _strip_suffixes(str(problem))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{problem_text} {USER_PROMPT_SUFFIX}".strip()},
    ]


def extract_user_content(prompt: Any) -> str:
    for message in _as_messages(prompt):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def _normalize_answer(answer: Any) -> str:
    if isinstance(answer, list | tuple):
        return ", ".join(str(item).strip() for item in answer)
    return str(answer).strip()


def _ground_truth(row: Mapping[str, Any]) -> str:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, Mapping) and reward_model.get("ground_truth") is not None:
        return _normalize_answer(reward_model["ground_truth"])
    for key in ("answer", "ground_truth", "final_answer", "solution"):
        if row.get(key) is not None:
            return _normalize_answer(row[key])
    raise KeyError("Could not find a ground-truth answer column")


def average_score(row: Mapping[str, Any]) -> float | None:
    scores = row.get("scores")
    if scores is None or len(scores) == 0:
        return None
    return float(sum(scores) / len(scores))


def keep_openr1_row(row: Mapping[str, Any], min_score: float = 0.2, max_score: float = 0.8) -> bool:
    avg = average_score(row)
    return avg is not None and min_score < avg < max_score


def convert_openr1_row(
    row: Mapping[str, Any],
    index: int,
    revision: str,
    min_score: float = 0.2,
    max_score: float = 0.8,
) -> dict[str, Any]:
    avg = average_score(row)
    if avg is None:
        raise ValueError("OpenR1 row has no scores")

    prompt = strip_openr1_chat_template(row.get("prompt", row.get("question", "")))
    answer = _ground_truth(row)
    return {
        "data_source": "alp_openr1_filtered",
        "prompt": alp_prompt(prompt),
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": index,
            "split": "train",
            "source_dataset": OPENR1_REPO,
            "source_revision": revision,
            "scores": _to_builtin(row.get("scores", [])),
            "avg_score": avg,
            "score_filter_min": min_score,
            "score_filter_max": max_score,
            "original_prompt": prompt,
            "original_answer": answer,
        },
    }


def convert_guru_row(
    row: Mapping[str, Any],
    index: int,
    revision: str,
    split: str,
    data_source: str,
    source_file: str,
) -> dict[str, Any]:
    messages = _as_messages(row.get("prompt", row.get("question", row.get("problem", ""))))
    problem = extract_user_content(messages) or str(row.get("question", row.get("problem", ""))).strip()
    answer = _ground_truth(row)

    extra_info = dict(_to_builtin(row.get("extra_info", {})) or {})
    extra_info.update(
        {
            "index": index,
            "split": split,
            "source_dataset": GURU_REPO,
            "source_file": source_file,
            "source_revision": revision,
        }
    )

    return {
        "data_source": data_source,
        "prompt": alp_prompt(problem),
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": extra_info,
    }


def convert_openr1_rows(
    rows: Iterable[Mapping[str, Any]],
    revision: str,
    min_score: float = 0.2,
    max_score: float = 0.8,
) -> list[dict[str, Any]]:
    records = []
    for index, row in enumerate(rows):
        row = _to_builtin(row)
        if keep_openr1_row(row, min_score=min_score, max_score=max_score):
            records.append(
                convert_openr1_row(row, len(records), revision=revision, min_score=min_score, max_score=max_score)
            )
    return records


def convert_guru_rows(
    rows: Iterable[Mapping[str, Any]],
    revision: str,
    split: str,
    data_source: str,
    source_file: str,
) -> list[dict[str, Any]]:
    return [
        convert_guru_row(
            _to_builtin(row), index, revision=revision, split=split, data_source=data_source, source_file=source_file
        )
        for index, row in enumerate(rows)
    ]


def merge_train_records(
    guru_records: Sequence[Mapping[str, Any]],
    openr1_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_user_prompts: set[str] = set()
    for record in list(guru_records) + list(openr1_records):
        user_content = extract_user_content(record["prompt"])
        if user_content in seen_user_prompts:
            continue
        seen_user_prompts.add(user_content)
        merged.append({column: record[column] for column in OUTPUT_COLUMNS})
    return merged


def _ordered_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{column: record[column] for column in OUTPUT_COLUMNS} for record in records]


def _write_parquet(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    table = pa.Table.from_pylist(_ordered_records(records))
    pq.write_table(table, path, compression="zstd", use_dictionary=False)


def write_outputs(
    train_records: Sequence[Mapping[str, Any]],
    math500_records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.parquet"
    math500_path = output_path / "math500.parquet"
    _write_parquet(train_records, train_path)
    _write_parquet(math500_records, math500_path)

    manifest = {
        "recipe": "alp_single_turn_math",
        "metadata": dict(metadata),
        "splits": {
            "train": {
                "row_count": len(train_records),
            },
            "math500": {
                "row_count": len(math500_records),
            },
        },
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _load_guru_file(filename: str, revision: str, cache_dir: str | None = None) -> list[dict[str, Any]]:
    path = hf_hub_download(
        repo_id=GURU_REPO,
        repo_type="dataset",
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
    )
    return pd.read_parquet(path).to_dict("records")


def build_outputs(
    output_dir: str | Path,
    openr1_revision: str,
    guru_revision: str,
    min_score: float = 0.2,
    max_score: float = 0.8,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    openr1_rows = load_dataset(OPENR1_REPO, split="train", revision=openr1_revision, cache_dir=cache_dir)
    openr1_records = convert_openr1_rows(
        openr1_rows,
        revision=openr1_revision,
        min_score=min_score,
        max_score=max_score,
    )

    guru_train_rows = _load_guru_file(GURU_TRAIN_FILE, revision=guru_revision, cache_dir=cache_dir)
    guru_train_records = convert_guru_rows(
        guru_train_rows,
        revision=guru_revision,
        split="train",
        data_source="alp_guru_rl92k_math",
        source_file=GURU_TRAIN_FILE,
    )

    eval_records = {}
    for name, repo_id in EVAL_DATASETS.items():
        rows = load_dataset(repo_id, split="test", revision=EVAL_REVISIONS[name], cache_dir=cache_dir)
        eval_records[name] = [convert_eval_row(row, index, name) for index, row in enumerate(rows)]

    train_records = merge_train_records(guru_train_records, openr1_records)
    metadata = {
        "openr1_repo": OPENR1_REPO,
        "openr1_revision": openr1_revision,
        "openr1_score_filter": f"({min_score}, {max_score})",
        "openr1_filtered_count": len(openr1_records),
        "guru_repo": GURU_REPO,
        "guru_revision": guru_revision,
        "guru_train_file": GURU_TRAIN_FILE,
        "guru_train_count": len(guru_train_records),
        "evaluation_datasets": EVAL_DATASETS,
        "dedup_key": "user prompt content",
        "dedup_keep": "guru first, then unique openr1",
    }
    result = write_outputs(train_records, eval_records.pop("math500"), output_dir=output_dir, metadata=metadata)
    for name, records in eval_records.items():
        _write_parquet(records, Path(output_dir) / f"{name}.parquet")
        print(f"{name}: {len(records)} problems")
    return result


def convert_eval_row(row: Mapping[str, Any], index: int, name: str) -> dict[str, Any]:
    answer = _ground_truth(row)
    if answer.startswith("\\boxed{") and answer.endswith("}"):
        answer = answer[len("\\boxed{"):-1]
    return {
        "data_source": f"alp_{name}",
        "prompt": alp_prompt(row.get("problem", row.get("question"))),
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {"index": index, "split": "test", "source_dataset": EVAL_DATASETS[name]},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ALP-aligned OpenR1 + Guru math data for VERL.")
    parser.add_argument("--output-dir", required=True, help="Directory for training and five evaluation parquets.")
    parser.add_argument("--openr1-revision", required=True, help="OpenR1 revision used in the paper (Appendix B).")
    parser.add_argument("--guru-revision", required=True, help="Guru revision used in the paper (Appendix B).")
    parser.add_argument("--min-score", type=float, default=0.2, help="OpenR1 lower average-score bound, exclusive.")
    parser.add_argument("--max-score", type=float, default=0.8, help="OpenR1 upper average-score bound, exclusive.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_outputs(
        output_dir=args.output_dir,
        openr1_revision=args.openr1_revision,
        guru_revision=args.guru_revision,
        min_score=args.min_score,
        max_score=args.max_score,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
