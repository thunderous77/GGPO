import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_MATH = REPO_ROOT / "examples" / "ggpo" / "run_math.sh"


def _dry_run(extra_env: dict[str, str] | None = None) -> str:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "WANDB_MODE": "disabled",
            "MODEL_PATH": "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "DATA_ROOT": "/tmp/alp_math",
            "ALGORITHM": "grpo",
        }
    )
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        ["bash", str(RUN_MATH)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


def test_run_math_initializes_rollout_tokenizer_by_default():
    output = _dry_run()

    assert "OVERRIDE actor_rollout_ref.rollout.skip_tokenizer_init=False" in output


def test_run_math_allows_explicit_rollout_tokenizer_skip_override():
    output = _dry_run({"ROLLOUT_SKIP_TOKENIZER_INIT": "True"})

    assert "OVERRIDE actor_rollout_ref.rollout.skip_tokenizer_init=True" in output
