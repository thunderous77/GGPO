# Copyright 2026 OpenAI

from __future__ import annotations

import json

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.offline_rl.dataset import OfflineBatchDataset, OfflineDatasetWriter
from verl.trainer.offline_rl.preprocess import preprocess_mixed_offline_dataset


def _config():
    return OmegaConf.create(
        {
            "offline_data": {
                "dataset_id": "synthetic-sealed",
                "implementation_commit": "implementation",
                "behavior_model_id": "test/model",
                "behavior_model_revision": "revision",
                "source_path": "/test/train.parquet",
                "expected_batches": 2,
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
                "max_prompt_length": 2,
                "max_response_length": 3,
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


def _batch(offset: int, rewards: tuple[int, int, int, int]) -> tuple[DataProto, torch.Tensor]:
    response_count = 4
    input_ids = torch.arange(offset, offset + response_count * 5).reshape(response_count, 5)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, -1] = 0
    behavior = torch.arange(offset, offset + response_count * 3, dtype=torch.float32).reshape(
        response_count, 3
    )
    batch = DataProto(
        batch=TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "rollout_log_probs": behavior,
            },
            batch_size=response_count,
        )
    )
    reward_tensor = torch.zeros(response_count, 3)
    reward_tensor[:, -1] = torch.tensor(rewards)
    return batch, reward_tensor


def _seal_source(root) -> None:
    config = _config()
    with OfflineDatasetWriter(root, config) as writer:
        batch, rewards = _batch(0, (0, 1, 1, 0))
        writer.append(0, batch, rewards, [10, 11])
        batch, rewards = _batch(100, (0, 0, 1, 0))
        writer.append(1, batch, rewards, [20, 21])
        writer.finalize()


def _dataset(root, *, batches: int) -> OfflineBatchDataset:
    return OfflineBatchDataset(
        root,
        expected_batches=batches,
        prompt_groups_per_batch=2,
        responses_per_group=2,
        selection_count=batches,
    )


def test_mixed_preprocessing_defaults_to_all_source_shards_and_preserves_alignment(tmp_path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "filtered"
    _seal_source(source_root)
    source_manifest_before = (source_root / "manifest.json").read_bytes()

    manifest = preprocess_mixed_offline_dataset(
        source_root,
        output_root,
        seed=7,
    )

    provenance = manifest["identity"]["filter_provenance"]
    assert provenance == {
        "source_dataset_id": "synthetic-sealed",
        "source_batch_count": 2,
        "source_selection": {"method": "prefix", "start": 0, "count": 2},
        "rule": "mixed_raw_correctness",
        "seed": 7,
        "per_source_shard_kept_counts": [
            {"source_shard_index": 0, "kept_groups": 2},
            {"source_shard_index": 1, "kept_groups": 1},
        ],
        "total_kept_groups": 3,
        "dropped_leftover_groups": 1,
        "resulting_round_count": 1,
    }
    assert manifest["identity"]["geometry"]["batch_count"] == 1
    assert len(manifest["shards"]) == 1
    assert (source_root / "manifest.json").read_bytes() == source_manifest_before

    source = _dataset(source_root, batches=2)
    filtered = _dataset(output_root, batches=1)
    output = filtered.load_batch(0)
    source_by_uid = {}
    for source_batch_index in range(2):
        source_batch = source.load_batch(source_batch_index)
        for uid in np.unique(source_batch.non_tensor_batch["uid"]):
            source_by_uid[str(uid)] = (
                source_batch,
                np.flatnonzero(source_batch.non_tensor_batch["uid"] == uid),
            )

    for response_index, uid in enumerate(output.non_tensor_batch["uid"]):
        source_batch, source_response_indices = source_by_uid[str(uid)]
        source_response_index = source_response_indices[response_index % 2]
        for key in output.batch.keys():
            torch.testing.assert_close(
                output.batch[key][response_index],
                source_batch.batch[key][source_response_index],
            )
        for key in output.non_tensor_batch:
            assert output.non_tensor_batch[key][response_index] == source_batch.non_tensor_batch[key][
                source_response_index
            ]

    group_rewards = output.batch["token_level_scores"].sum(-1).reshape(2, 2)
    assert torch.logical_and(group_rewards.sum(-1) > 0, group_rewards.sum(-1) < 2).all()


def test_mixed_preprocessing_seed_is_deterministic(tmp_path):
    source_root = tmp_path / "source"
    first_root = tmp_path / "filtered-a"
    second_root = tmp_path / "filtered-b"
    _seal_source(source_root)

    preprocess_mixed_offline_dataset(source_root, first_root, seed=19, selection_count=2)
    preprocess_mixed_offline_dataset(source_root, second_root, seed=19, selection_count=2)

    first_manifest = json.loads((first_root / "manifest.json").read_text())
    second_manifest = json.loads((second_root / "manifest.json").read_text())
    assert first_manifest == second_manifest
    first = _dataset(first_root, batches=1).load_batch(0)
    second = _dataset(second_root, batches=1).load_batch(0)
    for key in first.batch.keys():
        assert torch.equal(first.batch[key], second.batch[key])
    for key in first.non_tensor_batch:
        np.testing.assert_array_equal(first.non_tensor_batch[key], second.non_tensor_batch[key])
