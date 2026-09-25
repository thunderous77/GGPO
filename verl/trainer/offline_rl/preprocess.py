# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Deterministically build full-round mixed-correctness offline datasets."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verl.trainer.offline_rl.dataset import (
    FILTER_PROVENANCE_KEY,
    MANIFEST_NAME,
    MIXED_FILTER_RULE,
    SOURCE_GROUP_LOCATIONS_KEY,
    OfflineBatchDataset,
    _atomic_json,
    _atomic_torch_save,
    mixed_filtered_dataset_id,
    read_complete_manifest,
    validate_mixed_filtered_manifest,
    validate_offline_payload,
)
from verl.trainer.ppo.mixed_reward_filter import select_mixed_reward_group_indices


def _load_payload(dataset: OfflineBatchDataset, source_batch_index: int) -> dict[str, torch.Tensor]:
    record = dataset.manifest["shards"][source_batch_index]
    path = dataset.root / record["path"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    geometry = dataset.manifest["identity"]["geometry"]
    validate_offline_payload(
        payload,
        prompt_groups=int(geometry["prompt_groups_per_batch"]),
        responses_per_group=int(geometry["responses_per_group"]),
        max_prompt_length=int(geometry["max_prompt_length"]),
        max_response_length=int(geometry["max_response_length"]),
        binary_rewards=bool(dataset.manifest["identity"]["reward"]["binary"]),
        require_source_group_locations=False,
    )
    return payload


def preprocess_mixed_offline_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    seed: int,
    selection_start: int = 0,
    selection_count: int | None = None,
) -> dict[str, Any]:
    """Filter a sealed source interval to mixed groups and write full rounds.

    The global group permutation is generated once from ``seed``. Disk-backed
    arrays allow each source shard to be read only twice even for production
    geometry, without holding the filtered dataset in memory. When count is
    omitted, every source shard at or after ``selection_start`` is consumed.
    """
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("filtered output must be outside the immutable source dataset")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite filtered dataset output: {output_root}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("mixed-filter seed must be a nonnegative integer")
    if isinstance(selection_start, bool) or not isinstance(selection_start, int) or selection_start < 0:
        raise ValueError("mixed-filter source start must be a nonnegative integer")

    source_manifest = read_complete_manifest(source_root)
    source_identity = source_manifest["identity"]
    if FILTER_PROVENANCE_KEY in source_identity:
        raise ValueError("mixed filtering requires an unfiltered sealed source dataset")
    geometry = source_identity["geometry"]
    source_batch_count = int(geometry["batch_count"])
    if selection_count is None:
        selection_count = source_batch_count - selection_start
    if (
        isinstance(selection_count, bool)
        or not isinstance(selection_count, int)
        or selection_count <= 0
        or selection_start + selection_count > source_batch_count
    ):
        raise ValueError("mixed-filter source selection must be a nonempty in-range prefix interval")
    prompt_groups = int(geometry["prompt_groups_per_batch"])
    responses_per_group = int(geometry["responses_per_group"])
    max_response_length = int(geometry["max_response_length"])

    source_dataset = OfflineBatchDataset(
        source_root,
        expected_batches=source_batch_count,
        prompt_groups_per_batch=prompt_groups,
        responses_per_group=responses_per_group,
        selection_method="prefix",
        selection_start=selection_start,
        selection_count=selection_count,
    )

    group_references: list[tuple[int, int, int]] = []
    per_source_counts: list[dict[str, int]] = []
    template_payload: dict[str, torch.Tensor] | None = None
    for selection_index, source_batch_index in enumerate(source_dataset.source_indices):
        payload = _load_payload(source_dataset, source_batch_index)
        if template_payload is None:
            template_payload = payload
        response_indices, stats = select_mixed_reward_group_indices(
            payload["rewards"].float(),
            group_size=responses_per_group,
            max_groups=prompt_groups,
        )
        mixed_group_indices = (
            response_indices.reshape(-1, responses_per_group)[:, 0] // responses_per_group
        ).tolist()
        per_source_counts.append(
            {
                "source_shard_index": source_batch_index,
                "kept_groups": int(stats["selected_groups"]),
            }
        )
        group_references.extend(
            (source_batch_index, selection_index, int(group_index))
            for group_index in mixed_group_indices
        )

    total_kept_groups = len(group_references)
    resulting_round_count, dropped_leftover_groups = divmod(total_kept_groups, prompt_groups)
    if resulting_round_count == 0:
        raise ValueError(
            "mixed filtering produced no full training round: "
            f"kept_groups={total_kept_groups}, groups_per_round={prompt_groups}"
        )
    output_group_count = resulting_round_count * prompt_groups
    permutation = np.random.default_rng(seed).permutation(total_kept_groups)
    ordered_references = [group_references[int(index)] for index in permutation[:output_group_count]]

    assignments: dict[int, list[tuple[int, int, int]]] = {
        source_batch_index: [] for source_batch_index in source_dataset.source_indices
    }
    for destination_group, (source_batch_index, selection_index, source_group) in enumerate(
        ordered_references
    ):
        assignments[source_batch_index].append((source_group, destination_group, selection_index))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.mixedfilt.", dir=output_root.parent)
    )
    try:
        batch_root = staging_root / "batches"
        batch_root.mkdir()
        array_root = staging_root / ".repack-arrays"
        array_root.mkdir()

        assert template_payload is not None
        source_response_count = prompt_groups * responses_per_group
        output_response_count = output_group_count * responses_per_group
        arrays: dict[str, np.memmap] = {}
        alignments: dict[str, str] = {}
        for key, tensor in template_payload.items():
            if not torch.is_tensor(tensor):
                raise TypeError(f"offline shard field {key!r} is not a tensor")
            value = tensor.detach().cpu().numpy()
            if value.shape[0] == source_response_count:
                target_shape = (output_response_count, *value.shape[1:])
                alignments[key] = "response"
            elif value.shape[0] == prompt_groups:
                target_shape = (output_group_count, *value.shape[1:])
                alignments[key] = "group"
            else:
                raise ValueError(f"offline shard field {key!r} is not aligned to responses or groups")
            arrays[key] = np.lib.format.open_memmap(
                array_root / f"{key}.npy",
                mode="w+",
                dtype=value.dtype,
                shape=target_shape,
            )
        source_locations = np.lib.format.open_memmap(
            array_root / f"{SOURCE_GROUP_LOCATIONS_KEY}.npy",
            mode="w+",
            dtype=np.int64,
            shape=(output_group_count, 3),
        )

        offsets = np.arange(responses_per_group, dtype=np.int64)
        for source_batch_index in source_dataset.source_indices:
            shard_assignments = assignments[source_batch_index]
            if not shard_assignments:
                continue
            payload = _load_payload(source_dataset, source_batch_index)
            source_groups = np.asarray([item[0] for item in shard_assignments], dtype=np.int64)
            destination_groups = np.asarray([item[1] for item in shard_assignments], dtype=np.int64)
            selection_indices = np.asarray([item[2] for item in shard_assignments], dtype=np.int64)
            source_responses = (source_groups[:, None] * responses_per_group + offsets).reshape(-1)
            destination_responses = (
                destination_groups[:, None] * responses_per_group + offsets
            ).reshape(-1)
            for key, target in arrays.items():
                value = payload[key].detach().cpu().numpy()
                if alignments[key] == "response":
                    target[destination_responses] = value[source_responses]
                else:
                    target[destination_groups] = value[source_groups]
            source_locations[destination_groups, 0] = source_batch_index
            source_locations[destination_groups, 1] = selection_indices
            source_locations[destination_groups, 2] = source_groups

        for value in (*arrays.values(), source_locations):
            value.flush()

        shards: list[dict[str, Any]] = []
        for round_index in range(resulting_round_count):
            group_start = round_index * prompt_groups
            group_end = group_start + prompt_groups
            response_start = group_start * responses_per_group
            response_end = group_end * responses_per_group
            payload: dict[str, torch.Tensor] = {}
            for key, value in arrays.items():
                selected = (
                    value[response_start:response_end]
                    if alignments[key] == "response"
                    else value[group_start:group_end]
                )
                payload[key] = torch.from_numpy(np.array(selected, copy=True))
            payload[SOURCE_GROUP_LOCATIONS_KEY] = torch.from_numpy(
                np.array(source_locations[group_start:group_end], copy=True)
            )
            validate_offline_payload(
                payload,
                prompt_groups=prompt_groups,
                responses_per_group=responses_per_group,
                max_prompt_length=int(geometry["max_prompt_length"]),
                max_response_length=max_response_length,
                binary_rewards=bool(source_identity["reward"]["binary"]),
                require_source_group_locations=True,
            )
            relative = Path("batches") / f"batch-{round_index:06d}.pt"
            path = staging_root / relative
            _atomic_torch_save(path, payload)
            shards.append(
                {
                    "index": round_index,
                    "path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "reward_sum": int(payload["rewards"].sum().item()),
                    "valid_response_tokens": int(
                        payload["attention_mask"][:, -max_response_length:].sum().item()
                    ),
                }
            )

        for value in (*arrays.values(), source_locations):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
        shutil.rmtree(array_root)

        provenance = {
            "source_dataset_id": str(source_identity["dataset_id"]),
            "source_batch_count": source_batch_count,
            "source_selection": {
                "method": "prefix",
                "start": int(selection_start),
                "count": int(selection_count),
            },
            "rule": MIXED_FILTER_RULE,
            "seed": seed,
            "per_source_shard_kept_counts": per_source_counts,
            "total_kept_groups": total_kept_groups,
            "dropped_leftover_groups": dropped_leftover_groups,
            "resulting_round_count": resulting_round_count,
        }
        identity = json.loads(json.dumps(source_identity))
        identity["dataset_id"] = mixed_filtered_dataset_id(
            str(source_identity["dataset_id"]),
            seed=seed,
            selection_start=selection_start,
            selection_count=selection_count,
        )
        identity["geometry"]["batch_count"] = resulting_round_count
        identity[FILTER_PROVENANCE_KEY] = provenance
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "identity": identity,
            "shards": shards,
        }
        validate_mixed_filtered_manifest(manifest)
        manifest_path = staging_root / MANIFEST_NAME
        _atomic_json(manifest_path, manifest)
        os.chmod(manifest_path, 0o444)
        for record in shards:
            os.chmod(staging_root / record["path"], 0o444)
        os.replace(staging_root, output_root)
        return manifest
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise


def describe_mixed_filtered_dataset(root: str | Path) -> dict[str, Any]:
    """Return launcher-facing identity and geometry from a validated manifest."""
    manifest = read_complete_manifest(root)
    round_count = validate_mixed_filtered_manifest(manifest)
    return {
        "dataset_id": str(manifest["identity"]["dataset_id"]),
        "round_count": round_count,
    }
