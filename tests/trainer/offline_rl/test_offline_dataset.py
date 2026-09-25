# Copyright 2026 OpenAI

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.offline_rl.dataset import OfflineBatchDataset, OfflineDatasetWriter


def _config():
    return OmegaConf.create(
        {
            "offline_data": {
                "dataset_id": "unit-test-dataset",
                "implementation_commit": "0" * 40,
                "behavior_model_id": "test/model",
                "behavior_model_revision": "revision",
                "source_path": "/test/train.parquet",
                "expected_batches": 3,
                "prompt_groups_per_batch": 2,
                "responses_per_group": 2,
                "reward_kind": "binary",
                "binary_rewards": True,
            },
            "data": {
                "seed": 42,
                "shuffle": True,
                "train_batch_size": 2,
                "filter_overlong_prompts": True,
                "truncation": "error",
                "max_prompt_length": 3,
                "max_response_length": 4,
            },
            "actor_rollout_ref": {
                "rollout": {
                    "seed": 42,
                    "logprobs_mode": "processed_logprobs",
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                }
            },
        }
    )


def _batch(offset: int) -> tuple[DataProto, torch.Tensor]:
    response_count = 4
    input_ids = torch.arange(offset, offset + response_count * 7).reshape(response_count, 7)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, -1] = 0
    behavior = torch.linspace(-4, -1, response_count * 4).reshape(response_count, 4)
    rollout = DataProto(
        batch=TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "rollout_log_probs": behavior,
            },
            batch_size=response_count,
        )
    )
    rewards = torch.zeros(response_count, 4)
    rewards[1, -1] = 1
    rewards[3, -1] = 1
    return rollout, rewards


def _seal(root, config) -> None:
    with OfflineDatasetWriter(root, config) as writer:
        for index in range(3):
            batch, rewards = _batch(index * 100)
            writer.append(index, batch, rewards, [index * 2, index * 2 + 1])
        writer.finalize()


def test_schema_v1_round_trip_preserves_frozen_behavior_log_probs(tmp_path):
    config = _config()
    _seal(tmp_path, config)
    assert os.stat(tmp_path / "manifest.json").st_mode & 0o222 == 0

    dataset = OfflineBatchDataset(
        tmp_path,
        expected_batches=3,
        prompt_groups_per_batch=2,
        responses_per_group=2,
        expected_config=config,
    )
    loaded = dataset.load_batch(0)
    expected, _ = _batch(0)
    assert torch.equal(loaded.batch["old_log_probs"], expected.batch["rollout_log_probs"])
    assert torch.equal(loaded.batch["rollout_log_probs"], expected.batch["rollout_log_probs"])
    assert loaded.batch["token_level_scores"].sum().item() == 2
    assert np.array_equal(loaded.non_tensor_batch["source_index"], np.array([0, 0, 1, 1]))


def test_prefix_selection_maps_step_k_to_shard_k_minus_one(tmp_path):
    config = _config()
    _seal(tmp_path, config)

    full = OfflineBatchDataset(
        tmp_path,
        expected_batches=3,
        prompt_groups_per_batch=2,
        responses_per_group=2,
        selection_start=0,
        selection_count=3,
        expected_config=config,
    )
    prefix = OfflineBatchDataset(
        tmp_path,
        expected_batches=3,
        prompt_groups_per_batch=2,
        responses_per_group=2,
        selection_start=0,
        selection_count=2,
        expected_config=config,
    )
    assert full.source_indices == [0, 1, 2]
    assert prefix.source_indices == [0, 1]
    assert prefix.load_batch(0).non_tensor_batch["offline_batch_index"].tolist() == [0] * 4
    assert prefix.load_batch(1).non_tensor_batch["offline_batch_index"].tolist() == [1] * 4
    with pytest.raises(IndexError):
        prefix.load_batch(2)


def test_reader_rejects_a_different_scientific_identity(tmp_path):
    config = _config()
    _seal(tmp_path, config)
    changed = _config()
    changed.actor_rollout_ref.rollout.seed = 7
    with pytest.raises(ValueError, match="scientific identity"):
        OfflineBatchDataset(
            tmp_path,
            expected_batches=3,
            prompt_groups_per_batch=2,
            responses_per_group=2,
            expected_config=changed,
        )


def test_reader_accepts_a_relocated_source(tmp_path):
    config = _config()
    _seal(tmp_path, config)

    relocated = _config()
    relocated.offline_data.source_path = "/another-server/train.parquet"
    dataset = OfflineBatchDataset(
        tmp_path,
        expected_batches=3,
        prompt_groups_per_batch=2,
        responses_per_group=2,
        expected_config=relocated,
    )
    assert dataset.manifest["identity"]["source"]["path"] == "/test/train.parquet"


def test_writer_resume_rejects_a_different_source(tmp_path):
    config = _config()
    with OfflineDatasetWriter(tmp_path, config) as writer:
        batch, rewards = _batch(0)
        writer.append(0, batch, rewards, [0, 1])

    changed = _config()
    changed.offline_data.source_path = "/test/different.parquet"
    with pytest.raises(ValueError, match="different scientific identity"):
        OfflineDatasetWriter(tmp_path, changed)


def test_writer_rejects_nonbinary_reward_before_uint8_conversion(tmp_path):
    config = _config()
    batch, rewards = _batch(0)
    rewards[1, -1] = 0.5
    with OfflineDatasetWriter(tmp_path, config) as writer:
        with pytest.raises(ValueError, match="exactly in"):
            writer.append(0, batch, rewards, [0, 1])
