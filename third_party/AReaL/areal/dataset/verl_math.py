# SPDX-License-Identifier: Apache-2.0
"""Loader for VERL-style single-turn math parquet files.

The expected schema is the one used by DAPO and VERL math datasets:
``prompt`` contains chat messages and ``reward_model.ground_truth`` contains the
unboxed reference answer.  The selected parquet file represents the requested
logical split, so ``split`` is retained for the standard AReaL dataset API but
does not need to match a Hugging Face split name.
"""

from typing import Any

from datasets import load_dataset


def _encode_messages(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    text = "\n".join(str(message.get("content", "")) for message in messages)
    return tokenizer.encode(text)


def get_verl_math_rl_dataset(
    path: str,
    split: str | None,
    tokenizer: Any,
    max_length: int | None = None,
):
    """Load a local VERL-format math parquet for RLVR training.

    The ground truth remains unboxed because
    :func:`areal.reward.verl_math.verl_math_reward_fn` applies the canonical
    ``\\boxed{}`` wrapper before parsing it.
    """
    dataset = load_dataset("parquet", data_files=path, split="train")
    required_columns = {"prompt", "reward_model"}
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"VERL math parquet {path!r} is missing required columns: {missing}"
        )

    def process(sample: dict[str, Any]) -> dict[str, Any]:
        reward_model = sample["reward_model"]
        if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
            raise ValueError(
                "VERL math parquet reward_model must contain a ground_truth field"
            )
        return {
            "messages": sample["prompt"],
            "answer": reward_model["ground_truth"],
        }

    dataset = dataset.map(process).remove_columns(["prompt", "reward_model"])

    if max_length is not None:
        if tokenizer is None:
            raise ValueError("tokenizer is required when max_length is set")
        dataset = dataset.filter(
            lambda sample: len(_encode_messages(tokenizer, sample["messages"]))
            <= max_length
        )

    return dataset
