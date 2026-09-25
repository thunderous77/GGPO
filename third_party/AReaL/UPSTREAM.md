# AReaL source provenance

This directory contains a vendored and modified copy of AReaL. It is part of
this Git repository; it is not a submodule or an independently maintained Git
checkout.

## Imported source

- Import date: 2026-07-20
- Source tree: `KL-in-LLM-RL/AReaL`
- Source repository revision:
  `642b163d9e163a464042c4aa2044c1699473731a`
- Version declared by the imported AReaL package: `1.0.4`
- Upstream project: <https://github.com/areal-project/AReaL>

The source repository did not retain an AReaL-specific upstream Git revision,
so the source repository revision above, together with the worktree changes
listed below, is the authoritative provenance for this import.

## Imported worktree changes

The import includes the source tree's uncommitted worktree changes. These are
project modifications relative to the source repository revision and must not
be represented as unmodified upstream AReaL code.

Modified tracked files:

- `areal/dataset/__init__.py`
- `areal/engine/vllm_ext/areal_vllm_server.py`
- `areal/infra/controller/rollout_controller.py`
- `areal/trainer/ppo/actor.py`
- `areal/utils/hf_utils.py`

Added worktree files:

- `areal/dataset/verl_math.py`
- `examples/math/qwen3_verl_math_rl.py`
- `tests/test_areal_vllm_server_compat.py`
- `tests/test_hf_utils_offline.py`
- `tests/test_ppo_actor_staleness.py`
- `tests/test_rollout_controller_queue_stats.py`
- `tests/test_verl_math_dataset.py`

## Subsequent integration changes

This repository additionally modifies:

- `areal/infra/remote_inf_engine.py` for complete-group rollout handling and
  stable behavior-policy version attribution during asynchronous generation.
- `areal/infra/utils/ray_placement_group.py` so a CPU-only Ray controller can
  schedule accelerator child workers from the cluster resource inventory.

Those files and each modified tracked file listed above carry a prominent
file-level modification notice immediately after the SPDX license identifier.

The import copied all tracked files and all non-ignored worktree files under
`AReaL/`. It excluded Git metadata and ignored/generated material such as
caches, logs, training outputs, checkpoints, and other runtime state.

## License and modifications

AReaL is licensed under the Apache License 2.0. Preserve `LICENSE`, `LEGAL.md`,
existing copyright and attribution notices, and this provenance record when
redistributing the vendored source.

This project has modified the imported work. The worktree changes above and
subsequent integration changes in this repository are not necessarily endorsed
by the upstream AReaL project.
