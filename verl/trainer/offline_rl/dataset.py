# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Immutable batches for fully offline policy optimization."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
PROGRESS_NAME = "manifest.inprogress.json"
MIXED_FILTER_RULE = "mixed_raw_correctness"
FILTER_PROVENANCE_KEY = "filter_provenance"
SOURCE_GROUP_LOCATIONS_KEY = "source_group_locations"
PREFIX_SELECTION_METHOD = "prefix"
SEEDED_POOL_CYCLE_SELECTION_METHOD = "seeded_pool_cycle"


def seeded_pool_round_schedule(
    *,
    pool_round_count: int,
    total_round_count: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Map logical training rounds to deterministic (pass, pool-round) pairs."""
    pool_round_count = int(pool_round_count)
    total_round_count = int(total_round_count)
    seed = int(seed)
    if pool_round_count <= 0 or total_round_count <= 0:
        raise ValueError("pool and total round counts must both be positive")
    if seed < 0:
        raise ValueError("filtered pool cycle seed must be nonnegative")

    schedule: list[tuple[int, int]] = []
    pass_index = 0
    while len(schedule) < total_round_count:
        if pass_index == 0:
            permutation = np.arange(pool_round_count, dtype=np.int64)
        else:
            permutation = np.random.default_rng(seed + pass_index).permutation(pool_round_count)
        remaining = total_round_count - len(schedule)
        schedule.extend(
            (pass_index, int(pool_round_index))
            for pool_round_index in permutation[:remaining]
        )
        pass_index += 1
    return schedule


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _manifest_identity(config) -> dict[str, Any]:
    offline = config.offline_data
    return {
        "dataset_id": str(offline.dataset_id),
        "implementation_commit": str(offline.implementation_commit),
        "behavior_model": {
            "id": str(offline.behavior_model_id),
            "revision": str(offline.behavior_model_revision),
        },
        "source": {
            "path": str(offline.source_path),
        },
        "source_order": {
            "seed": int(config.data.seed),
            "shuffle": bool(config.data.get("shuffle", True)),
            "train_batch_size": int(config.data.train_batch_size),
            "filter_overlong_prompts": bool(config.data.filter_overlong_prompts),
            "truncation": str(config.data.truncation),
        },
        "sampling": {
            "seed": int(config.actor_rollout_ref.rollout.seed),
            "temperature": float(config.actor_rollout_ref.rollout.temperature),
            "top_p": float(config.actor_rollout_ref.rollout.top_p),
            "top_k": int(config.actor_rollout_ref.rollout.top_k),
        },
        "geometry": {
            "batch_count": int(offline.expected_batches),
            "prompt_groups_per_batch": int(offline.prompt_groups_per_batch),
            "responses_per_group": int(offline.responses_per_group),
            "max_prompt_length": int(config.data.max_prompt_length),
            "max_response_length": int(config.data.max_response_length),
        },
        "reward": {
            "kind": str(offline.reward_kind),
            "binary": bool(offline.binary_rewards),
        },
        "behavior_log_probs": {
            "source": "rollout_behavior_policy",
            "mode": str(config.actor_rollout_ref.rollout.logprobs_mode),
            "frozen": True,
            "training_key": "old_log_probs",
        },
    }


def _identity_for_config_match(identity: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Ignore host-local source directories and optional identity labels."""
    comparable = json.loads(json.dumps(identity))
    comparable.pop("dataset_id", None)
    # A mixed-filtered dataset copies the source scientific identity and adds
    # immutable preprocessing provenance. The runtime config pins the copied
    # science keys; the filtered contract validates the provenance separately.
    comparable.pop(FILTER_PROVENANCE_KEY, None)
    # Manifests written before the sha256 field was removed from
    # _manifest_identity still carry source.sha256; it has no current-config
    # counterpart, so it must not participate in the match.
    comparable.get("source", {}).pop("sha256", None)
    source_path = comparable.get("source", {}).get("path")
    if source_path is not None:
        comparable["source"]["path"] = Path(str(source_path)).name
    if not expected.get("implementation_commit"):
        comparable.pop("implementation_commit", None)
    if not expected.get("behavior_model", {}).get("revision"):
        comparable.get("behavior_model", {}).pop("revision", None)
    return comparable


def validate_offline_payload(
    payload: dict[str, torch.Tensor],
    *,
    prompt_groups: int,
    responses_per_group: int,
    max_prompt_length: int,
    max_response_length: int,
    binary_rewards: bool,
    require_source_group_locations: bool = False,
) -> None:
    response_count = prompt_groups * responses_per_group
    sequence_length = max_prompt_length + max_response_length
    expected_shapes = {
        "input_ids": (response_count, sequence_length),
        "attention_mask": (response_count, sequence_length),
        "behavior_old_log_probs": (response_count, max_response_length),
        "rewards": (response_count,),
        "source_indices": (prompt_groups,),
    }
    if require_source_group_locations:
        expected_shapes[SOURCE_GROUP_LOCATIONS_KEY] = (prompt_groups, 3)
    if set(payload) != set(expected_shapes):
        raise ValueError(
            f"offline shard keys must be exactly {sorted(expected_shapes)}, got {sorted(payload)}"
        )
    for key, shape in expected_shapes.items():
        if tuple(payload[key].shape) != shape:
            raise ValueError(f"offline shard {key} shape must be {shape}, got {tuple(payload[key].shape)}")
    expected_dtypes = {
        "input_ids": torch.int32,
        "attention_mask": torch.bool,
        "behavior_old_log_probs": torch.float32,
        "rewards": torch.uint8,
        "source_indices": torch.int64,
    }
    if require_source_group_locations:
        expected_dtypes[SOURCE_GROUP_LOCATIONS_KEY] = torch.int64
    for key, dtype in expected_dtypes.items():
        if payload[key].dtype != dtype:
            raise ValueError(f"offline shard {key} dtype must be {dtype}, got {payload[key].dtype}")

    attention_mask = payload["attention_mask"]
    response_mask = attention_mask[:, -max_response_length:].bool()
    if not response_mask.any(dim=-1).all():
        raise ValueError("every offline response must contain at least one valid token")
    if not torch.isfinite(payload["behavior_old_log_probs"]).all():
        raise ValueError("behavior old_log_probs contain a non-finite value")
    if binary_rewards and not torch.all((payload["rewards"] == 0) | (payload["rewards"] == 1)):
        raise ValueError("offline binary verifier rewards must be in {0, 1}")


def read_complete_manifest(root: str | Path) -> dict[str, Any]:
    """Read a sealed schema-v1 manifest without touching shard payloads."""
    manifest_path = Path(root).resolve() / MANIFEST_NAME
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise ValueError("offline dataset manifest must be a complete schema-v1 manifest")
    return manifest


def mixed_filtered_dataset_id(
    source_dataset_id: str,
    *,
    seed: int,
    selection_start: int,
    selection_count: int,
) -> str:
    """Return the deterministic label for one source-bound filtered variant."""
    if not isinstance(source_dataset_id, str) or not source_dataset_id:
        raise ValueError("mixed filtering requires a nonempty source dataset_id")
    selection_end = selection_start + selection_count
    return (
        f"{source_dataset_id}.mixedfilt-s{seed}-"
        f"prefix{selection_start}-{selection_end}"
    )


def validate_mixed_filtered_manifest(manifest: dict[str, Any]) -> int:
    """Validate filtered provenance and return its manifest-derived round count."""
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise ValueError("mixed-filtered dataset manifest must be complete schema-v1")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("mixed-filtered dataset manifest is missing its identity")
    provenance = identity.get(FILTER_PROVENANCE_KEY)
    if not isinstance(provenance, dict) or provenance.get("rule") != MIXED_FILTER_RULE:
        raise ValueError(f"mixed-filtered dataset requires filter rule {MIXED_FILTER_RULE!r}")

    integer_fields = (
        "seed",
        "source_batch_count",
        "total_kept_groups",
        "dropped_leftover_groups",
        "resulting_round_count",
    )
    parsed: dict[str, int] = {}
    for key in integer_fields:
        value = provenance.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"mixed-filtered provenance {key} must be a nonnegative integer")
        parsed[key] = value

    source_selection = provenance.get("source_selection")
    if not isinstance(source_selection, dict) or source_selection.get("method") != "prefix":
        raise ValueError("mixed-filtered provenance requires a prefix source selection")
    selection_start = source_selection.get("start")
    selection_count = source_selection.get("count")
    if (
        isinstance(selection_start, bool)
        or not isinstance(selection_start, int)
        or selection_start < 0
        or isinstance(selection_count, bool)
        or not isinstance(selection_count, int)
        or selection_count <= 0
        or selection_start + selection_count > parsed["source_batch_count"]
    ):
        raise ValueError("mixed-filtered provenance has an invalid source selection")

    kept_counts = provenance.get("per_source_shard_kept_counts")
    selected_shards = list(range(selection_start, selection_start + selection_count))
    if not isinstance(kept_counts, list) or len(kept_counts) != selection_count:
        raise ValueError("mixed-filtered provenance must record every selected source shard")
    recorded_counts: list[int] = []
    for expected_index, record in zip(selected_shards, kept_counts, strict=True):
        if not isinstance(record, dict) or record.get("source_shard_index") != expected_index:
            raise ValueError("mixed-filtered per-source-shard provenance is not contiguous")
        kept_groups = record.get("kept_groups")
        if isinstance(kept_groups, bool) or not isinstance(kept_groups, int) or kept_groups < 0:
            raise ValueError("mixed-filtered per-source-shard kept count must be nonnegative")
        recorded_counts.append(kept_groups)
    if sum(recorded_counts) != parsed["total_kept_groups"]:
        raise ValueError("mixed-filtered per-source-shard kept counts do not sum to the total")

    geometry = identity.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("mixed-filtered identity is missing dataset geometry")
    groups_per_round = geometry.get("prompt_groups_per_batch")
    if isinstance(groups_per_round, bool) or not isinstance(groups_per_round, int) or groups_per_round <= 0:
        raise ValueError("mixed-filtered prompt_groups_per_batch must be positive")
    if any(count > groups_per_round for count in recorded_counts):
        raise ValueError("mixed-filtered source shard kept count exceeds its prompt-group geometry")
    round_count, leftover_groups = divmod(parsed["total_kept_groups"], groups_per_round)
    if round_count <= 0:
        raise ValueError("mixed-filtered dataset must contain at least one full round")
    if parsed["resulting_round_count"] != round_count:
        raise ValueError("mixed-filtered resulting round count does not match kept groups")
    if parsed["dropped_leftover_groups"] != leftover_groups:
        raise ValueError("mixed-filtered leftover count does not match kept groups")
    if int(geometry.get("batch_count", -1)) != round_count or len(manifest.get("shards", [])) != round_count:
        raise ValueError("mixed-filtered shard count does not match its derived full-round count")
    for expected_index, record in enumerate(manifest["shards"]):
        if not isinstance(record, dict) or record.get("index") != expected_index:
            raise ValueError("mixed-filtered shard indices are not contiguous")
        relative = Path(str(record.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"mixed-filtered shard {expected_index} has an unsafe path")

    source_dataset_id = provenance.get("source_dataset_id")
    expected_dataset_id = mixed_filtered_dataset_id(
        source_dataset_id,
        seed=parsed["seed"],
        selection_start=selection_start,
        selection_count=selection_count,
    )
    if identity.get("dataset_id") != expected_dataset_id:
        raise ValueError("mixed-filtered dataset_id is not derived from its source provenance")
    return round_count


class OfflineDatasetWriter:
    """Append atomically written shards, then seal them behind one immutable manifest."""

    def __init__(self, root: str | Path, config):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_root = self.root / "batches"
        self.shard_root.mkdir(exist_ok=True)
        self.lock_path = self.root / ".writer.lock"
        self.identity = _manifest_identity(config)
        self._acquire_lock()

        final_path = self.root / MANIFEST_NAME
        progress_path = self.root / PROGRESS_NAME
        if final_path.exists():
            self.close()
            raise FileExistsError(f"offline dataset is already sealed: {final_path}")
        if progress_path.exists():
            with open(progress_path, encoding="utf-8") as stream:
                self.manifest = json.load(stream)
            if _identity_for_config_match(self.manifest["identity"], self.identity) != _identity_for_config_match(
                self.identity, self.identity
            ):
                self.close()
                raise ValueError("existing offline dataset progress belongs to a different scientific identity")
        else:
            self.manifest = {
                "schema_version": SCHEMA_VERSION,
                "status": "materializing",
                "identity": self.identity,
                "created_unix_seconds": time.time(),
                "shards": [],
            }
            _atomic_json(progress_path, self.manifest)
        try:
            self._validate_prefix()
        except Exception:
            self.close()
            raise

    def _acquire_lock(self) -> None:
        self._lock_stream = open(self.lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_stream.close()
            raise RuntimeError(f"another writer owns offline dataset lock {self.lock_path}") from exc
        self._lock_stream.seek(0)
        self._lock_stream.truncate()
        self._lock_stream.write(f"pid={os.getpid()}\n")
        self._lock_stream.flush()

    def _validate_prefix(self) -> None:
        for expected_index, record in enumerate(self.manifest["shards"]):
            if record["index"] != expected_index:
                raise ValueError("offline dataset progress is not a contiguous shard prefix")
            path = self.root / record["path"]
            if not path.is_file():
                raise ValueError(f"offline shard {expected_index} is missing")

    @property
    def next_index(self) -> int:
        return len(self.manifest["shards"])

    @property
    def expected_batches(self) -> int:
        return int(self.identity["geometry"]["batch_count"])

    def append(
        self,
        batch_index: int,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        source_indices: list[int],
    ) -> dict[str, Any]:
        if batch_index != self.next_index:
            raise ValueError(f"next offline shard must be {self.next_index}, got {batch_index}")
        if self.next_index >= self.expected_batches:
            raise ValueError("offline dataset already contains its exact approved batch count")
        geometry = self.identity["geometry"]
        prompt_groups = int(geometry["prompt_groups_per_batch"])
        responses_per_group = int(geometry["responses_per_group"])
        response_count = prompt_groups * responses_per_group
        if len(batch) != response_count:
            raise ValueError(f"materialized batch must contain {response_count} responses, got {len(batch)}")
        if len(source_indices) != prompt_groups:
            raise ValueError(f"source_indices must contain {prompt_groups} prompt rows")
        if "rollout_log_probs" not in batch.batch:
            raise ValueError("offline materialization requires rollout behavior log probabilities")

        max_response_length = int(geometry["max_response_length"])
        sequence_rewards = reward_tensor.float().sum(dim=-1)
        if not torch.isfinite(sequence_rewards).all():
            raise ValueError("offline verifier returned a non-finite sequence reward")
        if bool(self.identity["reward"]["binary"]) and not torch.all(
            (sequence_rewards == 0) | (sequence_rewards == 1)
        ):
            raise ValueError("offline binary verifier sequence rewards must be exactly in {0, 1}")
        payload = {
            "input_ids": batch.batch["input_ids"].to(dtype=torch.int32, device="cpu").contiguous(),
            "attention_mask": batch.batch["attention_mask"].to(dtype=torch.bool, device="cpu").contiguous(),
            "behavior_old_log_probs": batch.batch["rollout_log_probs"]
            .to(dtype=torch.float32, device="cpu")
            .contiguous(),
            "rewards": sequence_rewards.to(dtype=torch.uint8, device="cpu").contiguous(),
            "source_indices": torch.as_tensor(source_indices, dtype=torch.int64),
        }
        if payload["behavior_old_log_probs"].shape[-1] != max_response_length:
            raise ValueError("rollout behavior log probabilities do not use the configured response length")
        validate_offline_payload(
            payload,
            prompt_groups=prompt_groups,
            responses_per_group=responses_per_group,
            max_prompt_length=int(geometry["max_prompt_length"]),
            max_response_length=max_response_length,
            binary_rewards=bool(self.identity["reward"]["binary"]),
        )

        relative = Path("batches") / f"batch-{batch_index:06d}.pt"
        path = self.root / relative
        if path.exists():
            raise FileExistsError(f"refusing to overwrite an unindexed offline shard: {path}")
        _atomic_torch_save(path, payload)
        record = {
            "index": batch_index,
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "reward_sum": int(payload["rewards"].sum().item()),
            "valid_response_tokens": int(payload["attention_mask"][:, -max_response_length:].sum().item()),
        }
        self.manifest["shards"].append(record)
        _atomic_json(self.root / PROGRESS_NAME, self.manifest)
        return record

    def finalize(self) -> Path:
        if self.next_index != self.expected_batches:
            raise ValueError(
                f"cannot seal offline dataset with {self.next_index}/{self.expected_batches} accepted shards"
            )
        self.manifest["status"] = "complete"
        self.manifest["completed_unix_seconds"] = time.time()
        final_path = self.root / MANIFEST_NAME
        _atomic_json(final_path, self.manifest)
        os.chmod(final_path, 0o444)
        for record in self.manifest["shards"]:
            os.chmod(self.root / record["path"], 0o444)
        progress_path = self.root / PROGRESS_NAME
        if progress_path.exists():
            progress_path.unlink()
        return final_path

    def close(self) -> None:
        lock_stream = getattr(self, "_lock_stream", None)
        if lock_stream is not None and not lock_stream.closed:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            lock_stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class OfflineBatchDataset:
    """Read a deterministic logical-round schedule from a sealed dataset."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_batches: int,
        prompt_groups_per_batch: int,
        responses_per_group: int,
        selection_method: str = "prefix",
        selection_start: int = 0,
        selection_count: int | None = None,
        expected_config=None,
    ):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / MANIFEST_NAME
        self.manifest = read_complete_manifest(self.root)
        self.is_mixed_filtered = FILTER_PROVENANCE_KEY in self.manifest["identity"]
        if self.is_mixed_filtered:
            validate_mixed_filtered_manifest(self.manifest)
        if expected_config is not None:
            expected_identity = _manifest_identity(expected_config)
            if _identity_for_config_match(self.manifest["identity"], expected_identity) != _identity_for_config_match(
                expected_identity, expected_identity
            ):
                raise ValueError("offline dataset scientific identity does not match the training configuration")
        geometry = self.manifest["identity"]["geometry"]
        expected = {
            "batch_count": expected_batches,
            "prompt_groups_per_batch": prompt_groups_per_batch,
            "responses_per_group": responses_per_group,
        }
        for key, value in expected.items():
            if int(geometry[key]) != int(value):
                raise ValueError(f"offline dataset {key} is {geometry[key]}, expected {value}")
        if len(self.manifest["shards"]) != expected_batches:
            raise ValueError("offline dataset shard count does not match its fixed batch count")
        for expected_index, record in enumerate(self.manifest["shards"]):
            if int(record["index"]) != expected_index:
                raise ValueError("offline dataset shard index is not contiguous")
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"offline shard {expected_index} has an unsafe path")
        selection_start = int(selection_start)
        selection_count = expected_batches - selection_start if selection_count is None else int(selection_count)
        if selection_method == PREFIX_SELECTION_METHOD:
            if selection_start < 0 or selection_count <= 0 or selection_start + selection_count > expected_batches:
                raise ValueError(
                    "offline prefix selection must be a nonempty in-range interval, got "
                    f"start={selection_start}, count={selection_count}, expected_batches={expected_batches}"
                )
            schedule = [(0, index) for index in range(selection_start, selection_start + selection_count)]
        elif selection_method == SEEDED_POOL_CYCLE_SELECTION_METHOD:
            if not self.is_mixed_filtered:
                raise ValueError("seeded pool cycling requires a mixed-filtered manifest")
            if selection_start != 0 or selection_count <= 0:
                raise ValueError("seeded pool cycling requires start=0 and a positive logical round count")
            filter_seed = int(self.manifest["identity"][FILTER_PROVENANCE_KEY]["seed"])
            schedule = seeded_pool_round_schedule(
                pool_round_count=expected_batches,
                total_round_count=selection_count,
                seed=filter_seed,
            )
        else:
            raise ValueError(
                "offline selection method must be "
                f"{PREFIX_SELECTION_METHOD!r} or {SEEDED_POOL_CYCLE_SELECTION_METHOD!r}, "
                f"got {selection_method!r}"
            )
        self._schedule = schedule
        self.source_indices = [pool_round_index for _, pool_round_index in schedule]
        self.selection = {
            "schema_version": 1,
            "dataset_id": str(self.manifest["identity"]["dataset_id"]),
            "expected_batches": int(expected_batches),
            "method": selection_method,
            "start": selection_start,
            "count": selection_count,
            "selected_shards": self.source_indices,
        }

    def __len__(self) -> int:
        return len(self.source_indices)

    @property
    def pool_round_count(self) -> int:
        return int(self.manifest["identity"]["geometry"]["batch_count"])

    def schedule_entry(self, batch_index: int) -> tuple[int, int]:
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError(batch_index)
        return self._schedule[batch_index]

    def load_batch(self, batch_index: int) -> DataProto:
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError(batch_index)
        source_batch_index = self.source_indices[batch_index]
        record = self.manifest["shards"][source_batch_index]
        if record["index"] != source_batch_index:
            raise ValueError("offline dataset shard index is not contiguous")
        path = self.root / record["path"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        geometry = self.manifest["identity"]["geometry"]
        prompt_groups = int(geometry["prompt_groups_per_batch"])
        responses_per_group = int(geometry["responses_per_group"])
        max_prompt_length = int(geometry["max_prompt_length"])
        max_response_length = int(geometry["max_response_length"])
        validate_offline_payload(
            payload,
            prompt_groups=prompt_groups,
            responses_per_group=responses_per_group,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            binary_rewards=bool(self.manifest["identity"]["reward"]["binary"]),
            require_source_group_locations=self.is_mixed_filtered,
        )

        input_ids = payload["input_ids"].long()
        attention_mask = payload["attention_mask"].long()
        response_mask = attention_mask[:, -max_response_length:].bool()
        responses = input_ids[:, -max_response_length:]
        prompts = input_ids[:, :max_prompt_length]
        token_level_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        response_positions = torch.arange(max_response_length).expand_as(response_mask)
        final_response_positions = response_positions.masked_fill(~response_mask, -1).max(dim=-1).values
        token_level_scores[
            torch.arange(response_mask.shape[0]),
            final_response_positions,
        ] = payload["rewards"].float()

        response_count = prompt_groups * responses_per_group
        if self.is_mixed_filtered:
            source_group_locations = payload[SOURCE_GROUP_LOCATIONS_KEY].numpy()
            source_batch_indices = source_group_locations[:, 0]
            source_selection_indices = source_group_locations[:, 1]
            source_group_indices = source_group_locations[:, 2]
            source_selection = self.manifest["identity"][FILTER_PROVENANCE_KEY]["source_selection"]
            source_start = int(source_selection["start"])
            source_end = source_start + int(source_selection["count"])
            if not np.logical_and(source_batch_indices >= source_start, source_batch_indices < source_end).all():
                raise ValueError("mixed-filtered payload references a source shard outside its sealed prefix")
            if not np.array_equal(source_selection_indices, source_batch_indices - source_start):
                raise ValueError("mixed-filtered payload source selection indices are inconsistent")
            if not np.logical_and(source_group_indices >= 0, source_group_indices < prompt_groups).all():
                raise ValueError("mixed-filtered payload references an out-of-range source group")
        else:
            source_batch_indices = np.full(prompt_groups, source_batch_index, dtype=np.int64)
            source_selection_indices = np.full(prompt_groups, batch_index, dtype=np.int64)
            source_group_indices = np.arange(prompt_groups, dtype=np.int64)
        uids = np.repeat(
            np.array(
                [
                    f"offline-{int(shard):06d}-{int(group):04d}"
                    for shard, group in zip(source_batch_indices, source_group_indices, strict=True)
                ],
                dtype=object,
            ),
            responses_per_group,
        )
        source_indices = payload["source_indices"].numpy()
        repeated_source_indices = np.repeat(source_indices, responses_per_group)
        batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": compute_position_id_with_mask(attention_mask),
                "prompts": prompts,
                "responses": responses,
                "response_mask": response_mask,
                "old_log_probs": payload["behavior_old_log_probs"].float(),
                "rollout_log_probs": payload["behavior_old_log_probs"].float(),
                "token_level_scores": token_level_scores,
            },
            batch_size=response_count,
        )
        return DataProto(
            batch=batch,
            non_tensor_batch={
                "uid": uids,
                "source_index": repeated_source_indices,
                "offline_batch_index": np.repeat(source_batch_indices, responses_per_group).astype(np.int32),
                "offline_selection_index": np.repeat(source_selection_indices, responses_per_group).astype(np.int32),
            },
            meta_info={"offline_dataset_id": str(self.manifest["identity"]["dataset_id"])},
        )
