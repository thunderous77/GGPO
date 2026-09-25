from types import SimpleNamespace

from datasets import Dataset

import areal.dataset as dataset_api
from areal.dataset.verl_math import get_verl_math_rl_dataset
from areal.utils import environ


class _Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        return [token for message in messages for token in message["content"].split()]


def test_verl_math_loader_preserves_unboxed_ground_truth(monkeypatch):
    """The reward function, rather than the loader, must box the gold answer."""
    source = Dataset.from_list(
        [
            {
                "data_source": "math",
                "prompt": [
                    {"role": "system", "content": "Reason carefully."},
                    {"role": "user", "content": "What is 1 + 1?"},
                ],
                "reward_model": {"ground_truth": "2", "style": "rule"},
            }
        ]
    )
    monkeypatch.setattr(
        "areal.dataset.verl_math.load_dataset",
        lambda *args, **kwargs: source,
    )

    dataset = get_verl_math_rl_dataset(
        path="arbitrary-name.parquet",
        split="train",
        tokenizer=_Tokenizer(),
        max_length=16,
    )

    assert dataset[0]["answer"] == "2"
    assert dataset[0]["messages"] == source[0]["prompt"]
    assert "prompt" not in dataset.column_names
    assert "reward_model" not in dataset.column_names


def test_dataset_kwargs_select_verl_math_without_path_heuristic(monkeypatch):
    """dataset_name from config must reach the loader on the local path."""
    source = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "What is 1 + 1?"}],
                "reward_model": {"ground_truth": "2", "style": "rule"},
            }
        ]
    )
    config = SimpleNamespace(
        path="plain.parquet",
        type="rl",
        max_length=128,
        dataset_kwargs={"dataset_name": "verl_math"},
        scheduling_spec=None,
    )
    monkeypatch.setattr(environ, "is_single_controller", lambda: False)
    monkeypatch.setattr(
        "areal.dataset.verl_math.load_dataset",
        lambda *args, **kwargs: source,
    )

    result = dataset_api.get_custom_dataset(
        split="train",
        dataset_config=config,
        tokenizer=_Tokenizer(),
    )

    assert result[0]["answer"] == "2"
    assert result[0]["messages"][0]["content"] == "What is 1 + 1?"


def test_verl_math_loader_rejects_missing_schema(monkeypatch):
    """A non-VERL parquet should fail with its missing schema fields named."""
    source = Dataset.from_dict({"question": ["What is 1 + 1?"]})
    monkeypatch.setattr(
        "areal.dataset.verl_math.load_dataset",
        lambda *args, **kwargs: source,
    )

    try:
        get_verl_math_rl_dataset(
            path="plain.parquet",
            split="train",
            tokenizer=None,
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("missing VERL schema should raise ValueError")

    assert "prompt" in message
    assert "reward_model" in message
