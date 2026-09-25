"""Group admission helpers for dynamic sampling with raw binary correctness."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def raw_correctness_tensor(reward_extra_info: dict[str, Any], *, expected_responses: int) -> torch.Tensor:
    """Return the unshaped 0/1 correctness values supplied by the reward manager."""
    if "acc" not in reward_extra_info:
        raise ValueError("dynamic mixed-group sampling requires reward extra info key 'acc'")
    raw = torch.as_tensor(np.asarray(reward_extra_info["acc"]), dtype=torch.float32).reshape(-1)
    if raw.numel() != expected_responses:
        raise ValueError(
            f"raw correctness must contain one value per response: expected={expected_responses}, got={raw.numel()}"
        )
    if not torch.isfinite(raw).all() or not torch.logical_or(raw == 0, raw == 1).all():
        raise ValueError("raw correctness rewards must be finite and exactly binary (0 or 1)")
    return raw


def select_mixed_reward_group_indices(
    raw_correctness: torch.Tensor, *, group_size: int, max_groups: int
) -> tuple[torch.Tensor, dict[str, int]]:
    """Select complete mixed-correctness groups in input order."""
    if raw_correctness.ndim != 1:
        raise ValueError("raw_correctness must be one-dimensional")
    if group_size < 2 or max_groups < 0:
        raise ValueError("group_size must be >= 2 and max_groups must be nonnegative")
    if raw_correctness.numel() % group_size != 0:
        raise ValueError("candidate responses do not form complete groups")
    if not torch.isfinite(raw_correctness).all():
        raise ValueError("raw correctness rewards must be finite")
    if not torch.logical_or(raw_correctness == 0, raw_correctness == 1).all():
        raise ValueError("raw correctness rewards must be exactly binary (0 or 1)")

    grouped = raw_correctness.reshape(-1, group_size)
    group_sums = grouped.sum(dim=-1)
    mixed = (group_sums > 0) & (group_sums < group_size)
    mixed_indices = torch.nonzero(mixed, as_tuple=False).flatten()
    selected_group_indices = mixed_indices[:max_groups]
    offsets = torch.arange(group_size, device=raw_correctness.device)
    response_indices = (selected_group_indices[:, None] * group_size + offsets[None, :]).reshape(-1)
    stats = {
        "generated_groups": int(grouped.shape[0]),
        "mixed_groups": int(mixed.sum().item()),
        "all_zero_groups": int((group_sums == 0).sum().item()),
        "all_one_groups": int((group_sums == group_size).sum().item()),
        "selected_groups": int(selected_group_indices.numel()),
        "discarded_mixed_groups": int(mixed.sum().item() - selected_group_indices.numel()),
    }
    return response_indices, stats


def require_complete_mixed_batch(*, accepted_groups: int, target_groups: int, generation_rounds: int) -> None:
    """Fail closed when the finite refill budget cannot fill the learner batch."""
    if accepted_groups != target_groups:
        raise RuntimeError(
            "dynamic sampling exhausted its finite generation budget without filling the learner batch: "
            f"accepted={accepted_groups}, required={target_groups}, rounds={generation_rounds}"
        )
