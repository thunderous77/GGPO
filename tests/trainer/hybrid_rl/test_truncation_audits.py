import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_script(name):
    path = Path("diagnostics/truncation_reward_audit") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_audit_distinguishes_boundary_eos_from_truncation():
    audit = _load_script("offline_audit")

    assert audit.is_truncated(2048, 2048, [1, 2, 3], 99)
    assert not audit.is_truncated(2048, 2048, [1, 2, 99], 99)
    assert not audit.is_truncated(2047, 2048, [1, 2, 3], 99)


def _row(step, prompt, response):
    return {
        "checkpoint_step": step,
        "prompt_index": prompt,
        "response_index": response,
        "truncated": False,
        "at_token_limit": False,
        "reward": 0.0,
        "response_tokens": 8,
        "finish_reason": "stop",
        "post_box_tail_tokens": None,
    }


def test_aggregate_requires_exact_domain_rank_and_checkpoint(tmp_path, monkeypatch):
    aggregate = _load_script("aggregate_results")
    for rank in range(2):
        rows = [_row(20, prompt, response) for prompt in range(4) if prompt % 2 == rank for response in range(2)]
        (tmp_path / f"part-rank-{rank:02d}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_results.py",
            "--input-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--expected-parts",
            "2",
            "--expected-prompts",
            "4",
            "--responses-per-prompt",
            "2",
            "--expected-checkpoint-step",
            "20",
        ],
    )

    aggregate.main()
    assert json.loads(output.read_text())["checkpoint_step"] == 20

    path = tmp_path / "part-rank-01.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["checkpoint_step"] = 19
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="checkpoint step drifted"):
        aggregate.main()
