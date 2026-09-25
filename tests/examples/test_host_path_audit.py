from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "scripts" / "audit_host_paths.sh"

# This file sits inside the audit's own scope, so a literal host path written
# here would trip the very check it tests. Build the samples from fragments.
LEAKS = (
    "/ho" + "me/alice/scratch/run",
    "/u" + "/bob/ckpt/step_20",
    "/user" + "s/carol/data/",
    "/work" + "/nvme/proj/alp_math/train.parquet",
    "cond" + "a/envs/rl312/bin/python",
    "gh-" + "login",
    "gh" + "x4",
)

CLEAN = (
    '${TRAIN_SOURCE_PATH:?Set TRAIN_SOURCE_PATH to the ALP train parquet}',
    'conda activate "${CONDA_ENV_NAME:?Set CONDA_ENV_NAME}"',
    "sha256:" + "b" * 64,
    "hf:Qwen/Qwen3-1.7B-Base@" + "e" * 40,
    "${HOME}/data",
    "/tmp/run",
    "hf:Qwen/Qwen3-1.7B-Base@${GGPO_MODEL_REVISION}",
    '${TRAIN_SOURCE_PATH:?Set TRAIN_SOURCE_PATH}',
)


def _pattern() -> str:
    out = subprocess.run(
        ["bash", str(AUDIT), "--print-pattern"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_index_is_free_of_host_specific_values():
    result = subprocess.run(["bash", str(AUDIT)], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, f"host-path audit reported leaks:\n{result.stdout}{result.stderr}"


def test_audit_does_not_match_its_own_documentation():
    # The regression this guards: AGENTS.md is inside the audit scope, so
    # embedding the pattern there made the audit match itself and report a hit
    # on a clean index.
    assert _pattern() not in (REPO_ROOT / "AGENTS.md").read_text()


def test_pattern_detects_leaked_host_values():
    pattern = re.compile(_pattern())
    for sample in LEAKS:
        assert pattern.search(sample), f"audit pattern missed a leaked host value: {sample}"


def test_pattern_accepts_parameterized_and_host_independent_values():
    pattern = re.compile(_pattern())
    for sample in CLEAN:
        assert not pattern.search(sample), f"audit pattern falsely flagged: {sample}"
