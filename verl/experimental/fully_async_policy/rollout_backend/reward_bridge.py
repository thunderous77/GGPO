"""Reward compatibility bridge from AReaL RLVR to VERL custom rewards."""

from __future__ import annotations

import inspect
import math
from functools import lru_cache
from typing import Any

from verl.utils.import_utils import load_extern_object
from verl.utils.reward_score import batch_compute_score

CUSTOM_REWARD_PATH_KEY = "__verl_custom_reward_path__"
CUSTOM_REWARD_NAME_KEY = "__verl_custom_reward_name__"
CUSTOM_REWARD_KWARGS_KEY = "__verl_custom_reward_kwargs__"
AREAL_CUSTOM_REWARD_FN = "verl.experimental.fully_async_policy.rollout_backend.reward_bridge.areal_custom_reward_fn"
AREAL_BATCH_REWARD_FN = "verl.experimental.fully_async_policy.rollout_backend.reward_bridge.areal_batch_reward_fn"


@lru_cache(maxsize=32)
def _load_custom_reward(path: str, name: str):
    return load_extern_object(module_path=path, object_name=name)


def _normalize_reward(result: Any, *, label: str, function_label: str) -> float:
    if inspect.isawaitable(result):
        raise TypeError(f"AReaL async rollout currently requires a synchronous {function_label}")
    if isinstance(result, dict):
        if "score" not in result:
            raise ValueError(f"{label} dictionaries must contain a 'score' field")
        result = result["score"]

    score = float(result)
    if not math.isfinite(score):
        raise ValueError(f"{label} must be finite, got {score}")
    return score


def areal_batch_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    /,
    answer: Any,
    *,
    data_source: Any = None,
    extra_info: Any = None,
    **task_data: Any,
) -> float:
    """Score one AReaL sample with VERL's registered batch scorer.

    AReaL invokes rewards one completion at a time.  Wrapping that completion
    in singleton lists preserves the exact boxed-math path used by VERL's
    ``BatchRewardManager`` without selecting AReaL's independent math scorer.
    """

    del prompt, prompt_ids, completion_ids, task_data
    results = batch_compute_score(
        data_sources=[data_source],
        solution_strs=[completions],
        ground_truths=[answer],
        extra_infos=[extra_info if extra_info is not None else {}],
    )
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("VERL batch reward bridge expected exactly one score")
    return _normalize_reward(
        results[0],
        label="VERL batch reward",
        function_label="VERL batch reward function",
    )


def areal_custom_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    /,
    answer: Any,
    *,
    data_source: Any = None,
    extra_info: Any = None,
    **task_data: Any,
) -> float:
    """Call VERL's configured custom reward with its native argument names.

    AReaL supplies the decoded generation inputs positionally and forwards the
    original dataset row as keyword metadata.  VERL rows can themselves contain
    fields such as ``prompt``, so the generated inputs must remain
    positional-only to avoid duplicate-keyword binding failures.
    """

    del prompt, prompt_ids, completion_ids
    path = task_data.pop(CUSTOM_REWARD_PATH_KEY, None)
    name = task_data.pop(CUSTOM_REWARD_NAME_KEY, None)
    reward_kwargs = task_data.pop(CUSTOM_REWARD_KWARGS_KEY, {})
    if not path or not name:
        raise ValueError("AReaL custom reward bridge requires a module path and function name")
    if reward_kwargs is None:
        reward_kwargs = {}
    if not isinstance(reward_kwargs, dict):
        raise TypeError("reward.custom_reward_function.reward_kwargs must be a mapping")

    reward_fn = _load_custom_reward(str(path), str(name))
    call_kwargs = {
        "data_source": data_source,
        "solution_str": completions,
        "ground_truth": answer,
        "extra_info": extra_info if extra_info is not None else {},
    }
    # Match VERL's get_custom_reward_fn: configured kwargs take precedence.
    call_kwargs.update(reward_kwargs)
    result = reward_fn(**call_kwargs)
    return _normalize_reward(
        result,
        label="Custom reward",
        function_label="custom reward function",
    )
