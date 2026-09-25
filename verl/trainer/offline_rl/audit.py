# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Audit a manifest-bound selection without reading shard payloads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from verl.trainer.offline_rl.dataset import OfflineBatchDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    with open(root / "manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    geometry = manifest["identity"]["geometry"]
    dataset = OfflineBatchDataset(
        root,
        expected_batches=int(geometry["batch_count"]),
        prompt_groups_per_batch=int(geometry["prompt_groups_per_batch"]),
        responses_per_group=int(geometry["responses_per_group"]),
        selection_method="prefix",
        selection_start=0,
        selection_count=100,
    )
    reward_sum = 0
    valid_response_tokens = 0
    bytes_total = 0
    for source_batch_index in dataset.source_indices:
        record = manifest["shards"][source_batch_index]
        reward_sum += int(record["reward_sum"])
        valid_response_tokens += int(record["valid_response_tokens"])
        bytes_total += int(record["bytes"])

    total_steps = 100
    result = {
        "schema_version": 2,
        "status": "accepted",
        "dataset_root": str(root),
        "source_batch_count": int(geometry["batch_count"]),
        "selected_batch_count": len(dataset),
        "selection": dataset.selection,
        "prompt_groups_per_batch": int(geometry["prompt_groups_per_batch"]),
        "responses_per_group": int(geometry["responses_per_group"]),
        "trajectory_count": (
            len(dataset)
            * int(geometry["prompt_groups_per_batch"])
            * int(geometry["responses_per_group"])
        ),
        "reward_sum": reward_sum,
        "valid_response_tokens": valid_response_tokens,
        "bytes": bytes_total,
        "training_optimizer_steps": total_steps,
        "complete_selection_passes": total_steps // len(dataset),
        "tail_batches": total_steps % len(dataset),
        "next_batch_after_step_100": total_steps,
        "shard_payloads_read": 0,
    }
    receipt = Path(args.receipt).resolve()
    if receipt.exists():
        raise FileExistsError(f"audit receipt already exists: {receipt}")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, receipt)
    os.chmod(receipt, 0o444)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
