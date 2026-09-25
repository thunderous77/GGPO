# SPDX-License-Identifier: Apache-2.0

"""Lightweight, importable worker callables for RewardAPI spawn tests."""

import os
import time
from pathlib import Path


def add_reward(a: float, b: float) -> float:
    return a + b


def slow_reward(seconds: float) -> float:
    time.sleep(seconds)
    return 1.0


def crash_worker() -> float:
    os._exit(1)


def raise_value_error() -> float:
    raise ValueError("intentional error from reward fn")


def raise_timeout_error() -> float:
    raise TimeoutError("inner reward timeout")


def marker_slow_reward(started_path: str, finished_path: str, seconds: float) -> float:
    Path(started_path).write_text("started\n", encoding="utf-8")
    time.sleep(seconds)
    Path(finished_path).write_text("finished\n", encoding="utf-8")
    return 1.0
