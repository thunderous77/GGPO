# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Checkpoint-bound Math500 inference that cannot feed back into offline training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from pprint import pprint

from verl.trainer.offline_rl.contract import (
    PARAMETERIZED_OFFLINE_CONTRACT,
    SEALED_MIXED_FILTERED_OFFLINE_CONTRACT,
    SEALED_OFFLINE_CONTRACT,
    validate_checkpoint_dataset_identity,
    validate_parameterized_contract,
    validate_reference_contract,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def validate_evaluation_config(config) -> tuple[Path, Path, int]:
    offline = config.offline_data
    contract = str(offline.get("contract", SEALED_OFFLINE_CONTRACT))
    if contract in {SEALED_OFFLINE_CONTRACT, SEALED_MIXED_FILTERED_OFFLINE_CONTRACT}:
        validate_reference_contract(config, training=False)
    elif contract == PARAMETERIZED_OFFLINE_CONTRACT:
        validate_parameterized_contract(config, training=False)
    else:
        raise ValueError(
            f"offline_data.contract must be {SEALED_OFFLINE_CONTRACT!r}, "
            f"{SEALED_MIXED_FILTERED_OFFLINE_CONTRACT!r}, or {PARAMETERIZED_OFFLINE_CONTRACT!r}, "
            f"got {contract!r}"
        )
    if str(offline.mode) != "evaluate":
        raise ValueError("offline evaluator requires offline_data.mode=evaluate")
    checkpoint = Path(str(offline.eval_checkpoint)).resolve()
    receipt = Path(str(offline.eval_receipt)).resolve()
    expected_step = int(offline.eval_step)
    if contract == SEALED_OFFLINE_CONTRACT:
        if expected_step < 0 or expected_step > 100 or expected_step % 20 != 0:
            raise ValueError("Math500 evaluation step must be a boundary in {0, 20, ..., 100}")
    elif contract == SEALED_MIXED_FILTERED_OFFLINE_CONTRACT:
        total_steps = int(config.trainer.total_training_steps)
        save_freq = int(config.trainer.save_freq)
        if save_freq <= 0:
            raise ValueError("mixed-filtered offline evaluation requires a positive checkpoint frequency")
        final_boundary = expected_step == total_steps and bool(config.trainer.save_final_checkpoint)
        if (
            expected_step < 0
            or expected_step > total_steps
            or (expected_step % save_freq != 0 and not final_boundary)
        ):
            raise ValueError(
                "mixed-filtered offline evaluation step must be a persisted checkpoint boundary"
            )
    else:
        total_steps = int(config.trainer.total_training_steps)
        segment_steps = int(offline.segment_optimizer_steps)
        if expected_step < 0 or expected_step > total_steps or expected_step % segment_steps != 0:
            raise ValueError(
                "parameterized offline evaluation step must be a configured segment boundary"
            )
    if checkpoint.name != f"global_step_{expected_step}" or not checkpoint.is_dir():
        raise ValueError("eval_checkpoint must be the exact expected global-step directory")
    if receipt.is_relative_to(checkpoint):
        raise ValueError("evaluation receipt must be outside the immutable checkpoint directory")
    if receipt.exists():
        raise FileExistsError(f"evaluation receipt is immutable and already exists: {receipt}")
    if list(config.actor_rollout_ref.actor.checkpoint.load_contents) != ["model"]:
        raise ValueError("evaluation must load actor model weights only")
    return checkpoint, receipt, expected_step


class OfflineMathEvaluator(RayPPOTrainer):
    """Run evaluation from one exact offline checkpoint and emit an immutable receipt."""

    def fit(self):
        checkpoint, receipt, expected_step = validate_evaluation_config(self.config)
        contract = str(self.config.offline_data.get("contract", SEALED_OFFLINE_CONTRACT))
        publication_segment_steps = (
            20
            if contract in {SEALED_OFFLINE_CONTRACT, SEALED_MIXED_FILTERED_OFFLINE_CONTRACT}
            else int(self.config.offline_data.segment_optimizer_steps)
        )
        state_path = checkpoint / "offline_state.json"
        with open(state_path, encoding="utf-8") as stream:
            offline_state = json.load(stream)
        if int(offline_state.get("schema_version", -1)) != 2:
            raise ValueError("offline checkpoint state has an unsupported schema")
        if int(offline_state["completed_optimizer_steps"]) != expected_step:
            raise ValueError("offline checkpoint state does not match the requested evaluation step")
        validate_checkpoint_dataset_identity(
            offline_state,
            dataset_id=str(self.config.offline_data.dataset_id),
        )
        implementation_commit = str(self.config.offline_data.implementation_commit)
        if implementation_commit and offline_state.get("implementation_commit") != implementation_commit:
            raise ValueError("evaluation checkpoint implementation commit does not match the configuration")
        if offline_state.get("loss_mode") != str(self.config.actor_rollout_ref.actor.policy_loss.loss_mode):
            raise ValueError("evaluation checkpoint objective does not match the configuration")
        if offline_state.get("experiment_name") != str(self.config.trainer.experiment_name):
            raise ValueError("evaluation checkpoint display identity does not match the configuration")
        expected_wandb_run_id = str(self.config.offline_data.eval_wandb_run_id)
        if offline_state.get("wandb_run_id") != expected_wandb_run_id:
            raise ValueError("evaluation checkpoint does not match the expected training W&B run ID")

        self.global_steps = expected_step
        try:
            self.actor_rollout_wg.load_checkpoint(
                str(checkpoint / "actor"),
                del_local_after_load=False,
            )
            self.checkpoint_manager.update_weights(expected_step)
            metrics = self._validate()
            if not metrics:
                raise ValueError("Math500 evaluation returned no metrics")
            result = {
                "schema_version": 2,
                "status": "accepted",
                "optimizer_step": expected_step,
                "user_step": expected_step,
                "checkpoint": str(checkpoint),
                "wandb_run_id": expected_wandb_run_id,
                "experiment_name": offline_state["experiment_name"],
                "wandb_step_offset": expected_step // publication_segment_steps,
                "validation_path": str(self.config.offline_data.validation_path),
                "metrics": metrics,
            }
            receipt.parent.mkdir(parents=True, exist_ok=True)
            temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, receipt)
            os.chmod(receipt, 0o444)
            pprint(result)
        finally:
            self._shutdown_dump_executor()
