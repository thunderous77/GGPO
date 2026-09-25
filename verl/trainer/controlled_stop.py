# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Update-boundary wall-clock control for synchronous validation runs."""

import math
import time
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any


def _finite_number(name: str, value: Any, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    lower_bound_ok = value >= 0 if allow_zero else value > 0
    if not lower_bound_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return float(value)


def validate_controlled_stop(
    stop_after_seconds: float | None,
    deadline_unix_seconds: float | None,
) -> None:
    """Validate the two mutually exclusive controlled-stop modes."""

    configured = {
        "controlled_stop_after_seconds": stop_after_seconds,
        "controlled_stop_deadline_unix_seconds": deadline_unix_seconds,
    }
    for name, value in configured.items():
        if value is None:
            continue
        _finite_number(name, value)
    if stop_after_seconds is not None and deadline_unix_seconds is not None:
        raise ValueError(
            "controlled_stop_after_seconds and controlled_stop_deadline_unix_seconds are mutually exclusive"
        )


@dataclass
class AdaptiveUpdateBoundaryStop:
    """Admit only sync windows that are predicted to finish before a hard deadline.

    ``initial_optimizer_step_seconds`` and ``finalization_seconds`` are measured
    by the launcher from a full-shape smoke or prior lineage evidence. Runtime
    observations update the initial per-update estimate through a bounded
    rolling maximum. Multiplying by ``sync_steps`` makes the reserve follow the
    externally published learner-step granularity automatically.
    """

    hard_deadline_unix_seconds: float
    initial_optimizer_step_seconds: float
    finalization_seconds: float
    sync_steps: int
    safety_factor: float = 1.25
    history_size: int = 8
    _optimizer_step_seconds: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.hard_deadline_unix_seconds = _finite_number(
            "hard_deadline_unix_seconds",
            self.hard_deadline_unix_seconds,
        )
        self.initial_optimizer_step_seconds = _finite_number(
            "initial_optimizer_step_seconds",
            self.initial_optimizer_step_seconds,
        )
        self.finalization_seconds = _finite_number(
            "finalization_seconds",
            self.finalization_seconds,
            allow_zero=True,
        )
        self.safety_factor = _finite_number("safety_factor", self.safety_factor)
        if self.safety_factor < 1:
            raise ValueError("safety_factor must be at least 1")
        if isinstance(self.sync_steps, bool) or not isinstance(self.sync_steps, int) or self.sync_steps < 1:
            raise ValueError("sync_steps must be a positive integer")
        if isinstance(self.history_size, bool) or not isinstance(self.history_size, int) or self.history_size < 1:
            raise ValueError("history_size must be a positive integer")
        self._optimizer_step_seconds = deque(maxlen=self.history_size)
        self._optimizer_step_seconds.append(self.initial_optimizer_step_seconds)

    @property
    def optimizer_step_estimate_seconds(self) -> float:
        return max(self._optimizer_step_seconds)

    @property
    def required_headroom_seconds(self) -> float:
        work_seconds = self.optimizer_step_estimate_seconds * self.sync_steps
        return self.safety_factor * (work_seconds + self.finalization_seconds)

    def observe_optimizer_step(self, elapsed_seconds: float) -> None:
        self._optimizer_step_seconds.append(
            _finite_number("optimizer step elapsed_seconds", elapsed_seconds)
        )

    def should_stop_before_next_update(
        self,
        *,
        local_trigger_step: int,
        now_unix_seconds: float | None = None,
    ) -> bool:
        """Stop only before a complete parameter-sync window begins.

        Once a window starts, all of its optimizer steps are admitted. This
        prevents a larger ``sync_steps`` value from producing a partial
        externally logged learner step at a controlled wall-clock boundary.
        """

        if (
            isinstance(local_trigger_step, bool)
            or not isinstance(local_trigger_step, int)
            or not 1 <= local_trigger_step <= self.sync_steps
        ):
            raise ValueError("local_trigger_step must be within the configured sync window")
        if local_trigger_step != 1:
            return False
        now = time.time() if now_unix_seconds is None else _finite_number(
            "now_unix_seconds",
            now_unix_seconds,
        )
        return self.hard_deadline_unix_seconds - now <= self.required_headroom_seconds


def controlled_stop_reached(
    *,
    started_at: float,
    stop_after_seconds: float | None,
    deadline_unix_seconds: float | None,
) -> bool:
    """Return whether the configured relative or absolute deadline has passed."""

    if deadline_unix_seconds is not None:
        return time.time() >= deadline_unix_seconds
    return stop_after_seconds is not None and time.monotonic() - started_at >= stop_after_seconds


class ControlledStopDataLoader:
    """Stop yielding new updates after a relative or absolute deadline.

    The wrapped trainer completes the batch already yielded.  The next
    dataloader request observes the deadline and ends iteration, so no actor
    optimizer update is interrupted. ``deadline_unix_seconds`` lets a launcher
    include setup time by passing the Unix epoch for ``job_start + budget``.
    The relative mode remains measured from construction for compatibility.
    """

    def __init__(
        self,
        dataloader: Iterable,
        stop_after_seconds: float | None = None,
        *,
        deadline_unix_seconds: float | None = None,
    ):
        validate_controlled_stop(stop_after_seconds, deadline_unix_seconds)
        if stop_after_seconds is None and deadline_unix_seconds is None:
            raise ValueError("one controlled-stop deadline must be configured")
        self.dataloader = dataloader
        self.started_at = time.monotonic()
        self.stop_after_seconds = None if stop_after_seconds is None else float(stop_after_seconds)
        self.deadline_unix_seconds = None if deadline_unix_seconds is None else float(deadline_unix_seconds)
        self.controlled_stop_reached = False

    def __len__(self) -> int:
        return len(self.dataloader)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[Any]:
        if self.controlled_stop_reached:
            return
        for batch in self.dataloader:
            if controlled_stop_reached(
                started_at=self.started_at,
                stop_after_seconds=self.stop_after_seconds,
                deadline_unix_seconds=self.deadline_unix_seconds,
            ):
                self.controlled_stop_reached = True
                print(
                    "[ControlledStop] Reached update-boundary wall-clock stop; "
                    "no new synchronous optimizer update will start.",
                    flush=True,
                )
                return
            yield batch
