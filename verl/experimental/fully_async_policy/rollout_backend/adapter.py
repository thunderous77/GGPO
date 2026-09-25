"""Pure trajectory and task adapters between vendored AReaL and VERL."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from verl.experimental.reward_loop.reward_manager.dapo import soft_overlong_penalty
from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask

REQUIRE_COMPLETE_GROUP_KEY = "__verl_require_complete_group__"


def _python_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _python_value(value.item())
        return [_python_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return copy.deepcopy(value)


def build_areal_task_data(source_batch: DataProto) -> dict[str, Any]:
    """Collapse VERL's repeated prompt group to one AReaL workflow input."""

    if len(source_batch) < 1:
        raise ValueError("Cannot submit an empty VERL prompt batch to AReaL")

    task_data = {
        key: _python_value(np.asarray(values, dtype=object)[0])
        for key, values in source_batch.non_tensor_batch.items()
    }
    if "messages" not in task_data:
        if "raw_prompt" not in task_data:
            raise ValueError("AReaL RLVR input requires 'messages' or VERL 'raw_prompt'")
        task_data["messages"] = copy.deepcopy(task_data["raw_prompt"])

    reward_model = task_data.get("reward_model")
    if "answer" not in task_data and isinstance(reward_model, dict) and "ground_truth" in reward_model:
        task_data["answer"] = copy.deepcopy(reward_model["ground_truth"])

    task_data[REQUIRE_COMPLETE_GROUP_KEY] = True
    return task_data


def _validate_trajectory(trajectory: dict[str, Any], group_size: int) -> dict[str, torch.Tensor]:
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if not isinstance(trajectory, dict):
        raise TypeError(f"AReaL trajectory must be a dict, got {type(trajectory).__name__}")

    sequence_keys = ("input_ids", "loss_mask", "logprobs", "versions", "attention_mask")
    missing = [key for key in (*sequence_keys, "rewards") if key not in trajectory]
    if missing:
        raise ValueError(f"AReaL trajectory is missing required fields: {', '.join(missing)}")

    tensors: dict[str, torch.Tensor] = {}
    sequence_shape = None
    for key in sequence_keys:
        value = trajectory[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"AReaL trajectory field {key!r} must be a tensor")
        if value.ndim != 2:
            raise ValueError(f"AReaL trajectory field {key!r} must have shape [group, sequence]")
        if value.shape[0] != group_size:
            raise ValueError(
                f"Incomplete AReaL prompt group: expected {group_size} rows, got {value.shape[0]} in {key!r}"
            )
        if sequence_shape is None:
            sequence_shape = value.shape
        elif value.shape != sequence_shape:
            raise ValueError(
                f"AReaL sequence fields must share a shape; {key!r} has {tuple(value.shape)}, "
                f"expected {tuple(sequence_shape)}"
            )
        tensors[key] = value.detach().cpu()

    rewards = trajectory["rewards"]
    if not isinstance(rewards, torch.Tensor) or rewards.numel() != group_size:
        shape = tuple(rewards.shape) if isinstance(rewards, torch.Tensor) else type(rewards).__name__
        raise ValueError(f"AReaL rewards must contain one scalar per group row; got {shape}")
    tensors["rewards"] = rewards.detach().cpu().reshape(group_size).to(torch.float32)
    return tensors


def _copy_group_non_tensors(source_batch: DataProto, group_size: int) -> dict[str, np.ndarray]:
    copied: dict[str, np.ndarray] = {}
    for key, values in source_batch.non_tensor_batch.items():
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.shape[0] == group_size:
            copied[key] = array.copy()
        elif array.shape[0] == 1:
            copied[key] = np.repeat(array, group_size, axis=0)
        else:
            raise ValueError(
                f"Source non-tensor field {key!r} has {array.shape[0]} rows; expected 1 or {group_size}"
            )
    return copied


def areal_trajectory_to_dataproto(
    trajectory: dict[str, Any],
    source_batch: DataProto,
    *,
    group_size: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
    sample_id: str,
    generation_seconds: float = 0.0,
    incorrect_reward: float = -5.0,
    correct_reward: float = 5.0,
    eos_token_id: int | None = None,
    overlong_buffer_length: int | None = None,
    overlong_penalty_factor: float = 1.0,
) -> DataProto:
    """Convert one complete native AReaL prompt group into a VERL queue item."""

    if prompt_length < 1 or response_length < 1:
        raise ValueError("prompt_length and response_length must both be positive")
    if pad_token_id is None:
        raise ValueError("A tokenizer pad_token_id (or eos_token_id fallback) is required")
    if overlong_buffer_length is not None and not 0 < overlong_buffer_length <= response_length:
        raise ValueError("overlong_buffer_length must be positive and no larger than response_length")

    tensors = _validate_trajectory(trajectory, group_size)
    prompts = torch.full((group_size, prompt_length), int(pad_token_id), dtype=torch.long)
    responses = torch.full((group_size, response_length), int(pad_token_id), dtype=torch.long)
    prompt_mask = torch.zeros((group_size, prompt_length), dtype=torch.long)
    response_mask = torch.zeros((group_size, response_length), dtype=torch.long)
    rollout_log_probs = torch.zeros((group_size, response_length), dtype=torch.float32)
    rollout_policy_versions = torch.full((group_size, response_length), -1, dtype=torch.long)
    rm_scores = torch.zeros((group_size, response_length), dtype=torch.float32)
    verifier_correctness = torch.zeros((group_size,), dtype=torch.float32)
    response_ended_with_eos = torch.zeros((group_size,), dtype=torch.bool)
    min_versions = np.empty(group_size, dtype=np.int64)
    max_versions = np.empty(group_size, dtype=np.int64)
    raw_correctness = np.empty(group_size, dtype=np.float32)
    overlong_rewards = np.zeros(group_size, dtype=np.float32)

    for row in range(group_size):
        attention = tensors["attention_mask"][row].bool()
        trained = tensors["loss_mask"][row].bool()
        prompt_selector = attention & ~trained
        response_selector = attention & trained

        prompt_tokens = tensors["input_ids"][row][prompt_selector].to(torch.long)
        response_tokens = tensors["input_ids"][row][response_selector].to(torch.long)
        response_log_probs = tensors["logprobs"][row][response_selector]
        raw_response_versions = tensors["versions"][row][response_selector]
        if raw_response_versions.is_floating_point() and (
            not torch.isfinite(raw_response_versions).all()
            or not torch.equal(raw_response_versions, raw_response_versions.trunc())
        ):
            raise ValueError(f"AReaL response row {row} contains a non-integral policy version")
        response_versions = raw_response_versions.to(torch.long)

        if response_tokens.numel() == 0:
            raise ValueError(f"AReaL group row {row} contains no trainable response tokens")
        if prompt_tokens.numel() > prompt_length:
            raise ValueError(
                f"AReaL prompt row {row} has {prompt_tokens.numel()} tokens, exceeding configured {prompt_length}"
            )
        if response_tokens.numel() > response_length:
            raise ValueError(
                f"AReaL response row {row} has {response_tokens.numel()} tokens, exceeding configured {response_length}"
            )
        if torch.any(response_versions < 0):
            raise ValueError(f"AReaL response row {row} contains an unset policy version")
        if not torch.isfinite(response_log_probs).all():
            raise ValueError(f"AReaL response row {row} contains a non-finite behavior log-probability")
        correctness = float(tensors["rewards"][row].item())
        if correctness not in (0.0, 1.0):
            raise ValueError(f"AReaL verifier correctness must be binary 0/1, got {correctness}")

        prompt_count = prompt_tokens.numel()
        response_count = response_tokens.numel()
        prompts[row, prompt_length - prompt_count :] = prompt_tokens
        prompt_mask[row, prompt_length - prompt_count :] = 1
        responses[row, :response_count] = response_tokens
        response_mask[row, :response_count] = 1
        rollout_log_probs[row, :response_count] = response_log_probs
        rollout_policy_versions[row, :response_count] = response_versions
        verifier_correctness[row] = correctness
        raw_correctness[row] = correctness
        response_ended_with_eos[row] = eos_token_id is not None and int(response_tokens[-1].item()) == int(eos_token_id)
        shaped_reward = correct_reward if correctness == 1.0 else incorrect_reward
        if overlong_buffer_length is not None:
            overlong_rewards[row] = soft_overlong_penalty(
                response_count,
                max_response_length=response_length,
                buffer_length=overlong_buffer_length,
                penalty_factor=overlong_penalty_factor,
            )
            shaped_reward += float(overlong_rewards[row])
        rm_scores[row, response_count - 1] = shaped_reward
        min_versions[row] = int(response_versions.min().item())
        max_versions[row] = int(response_versions.max().item())

    input_ids = torch.cat((prompts, responses), dim=-1)
    attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)
    position_ids = compute_position_id_with_mask(attention_mask)
    non_tensors = _copy_group_non_tensors(source_batch, group_size)
    non_tensors.update(
        {
            "uid": np.array([f"uid_{sample_id}"] * group_size, dtype=object),
            "min_global_steps": min_versions,
            "max_global_steps": max_versions,
            "__num_turns__": np.ones(group_size, dtype=np.int32),
            # Keep the unshaped verifier result separate from DAPO's shaped
            # reward. ``acc`` is the synchronous reward-manager contract;
            # ``raw_correctness`` makes the async transport identity explicit.
            "acc": raw_correctness.copy(),
            "raw_correctness": raw_correctness,
            "overlong_reward": overlong_rewards,
            "overlong": overlong_rewards < 0,
        }
    )

    metrics = [
        {"generate_sequences": float(generation_seconds), "tool_calls": 0.0}
        for _ in range(group_size)
    ]
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "rollout_log_probs": rollout_log_probs,
            "rollout_policy_versions": rollout_policy_versions,
            "verifier_correctness": verifier_correctness,
            "response_ended_with_eos": response_ended_with_eos,
            "rm_scores": rm_scores,
        },
        non_tensors=non_tensors,
        meta_info={
            "metrics": metrics,
            "reward_extra_keys": ["acc", "raw_correctness", "overlong_reward", "overlong"],
        },
    )
