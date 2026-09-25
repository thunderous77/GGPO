# GGPO paper experiments

See the [project README](../../README.md) for the method and key results. [REPRODUCTION.md](REPRODUCTION.md) records the software, hardware, numerical precision, and full experiment commands.

- `run_math.sh`: GGPO, AGRO, GRPO, and GRPO+TIS; training with periodic behavior synchronization, fixed-pool collection, or offline training.
- `run_qwen3_*_{ggpo,grpo}.sh`: thin model/method wrappers around `run_math.sh`.
- `offline_rl/run.sh`: convenience wrapper for `MODE=offline`.
- `evaluate.sh`: all five paper benchmarks with 32 samples per problem.
- `summarize_eval.py`: per-benchmark and problem-weighted mean@32/pass@K from evaluation JSONL.
- `math_verifier.py`: binary correctness reward.

Launch from an activated conda environment. Set `DATA_ROOT` at runtime; set `POOL_DIR` for collection/offline training, or `MODEL_PATH` and `EVAL_DIR` for final evaluation. `DRY_RUN=1` prints the configuration without starting workers.
