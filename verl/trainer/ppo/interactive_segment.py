"""Validation for bounded local-resume training segments."""

from __future__ import annotations


def resolve_segment_end(
    *, checkpoint_step: int, total_training_steps: int, start_raw: str | None, end_raw: str | None
) -> int:
    """Resolve an arbitrary positive segment boundary without changing scientific step cadence."""
    if (start_raw is None) != (end_raw is None):
        raise ValueError("interactive segment start/end must be supplied together")
    if start_raw is None:
        return total_training_steps

    start = int(start_raw)
    end = int(end_raw)
    if checkpoint_step != start:
        raise ValueError(f"interactive resume boundary mismatch: checkpoint={checkpoint_step}, requested={start}")
    if end <= start:
        raise ValueError(f"interactive segment end must exceed start: start={start}, end={end}")
    if end > total_training_steps:
        raise ValueError(f"interactive segment end exceeds total_training_steps={total_training_steps}: end={end}")
    return end
