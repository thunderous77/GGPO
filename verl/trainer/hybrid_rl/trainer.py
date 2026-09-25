# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Thin 75/25 historical/on-policy adapter for the synchronous PPO trainer."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.offline_rl.dataset import OfflineBatchDataset
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.tokenizer import normalize_token_ids


HISTORICAL = "H"
ON_POLICY = "O"
PROMPT_IDENTITY_SCHEMA = "source-dataset-index+stored-prompt-tokens-v1"
BOUNDARY_METRIC_KEYS = {
    "training/global_step",
    "hybrid/historical_groups",
    "hybrid/on_policy_groups",
    "critic/score/mean",
    "actor/pg_loss",
    "actor/grad_norm",
    "actor/entropy",
    "val-core/alp_math500/acc/mean@32",
    "val-core/alp_math500/acc/best@32/mean",
    "val-core/alp_math500/acc/best@32/std",
}
SEALED_HYBRID_CONTRACT = "sealed_75h25o_ggpo_100_step"
PARAMETERIZED_HYBRID_CONTRACT = "parameterized"


def _derived_seed(selection_seed: int, step: int, stream: int) -> int:
    words = np.random.SeedSequence([selection_seed, step, stream]).generate_state(2, dtype=np.uint32)
    return int(words[0]) << 32 | int(words[1])


@dataclass(frozen=True)
class HybridPlan:
    step: int
    shard_index: int
    fresh_group_mask: tuple[bool, ...]
    group_order: tuple[int, ...]
    minibatch_seeds: tuple[int, ...]

    @property
    def response_order(self) -> np.ndarray:
        groups = np.asarray(self.group_order, dtype=np.int64)
        offsets = np.arange(8, dtype=np.int64)
        return (groups[:, None] * 8 + offsets[None, :]).reshape(-1)

    @property
    def ordered_group_sources(self) -> np.ndarray:
        mask = np.asarray(self.fresh_group_mask, dtype=bool)
        return np.where(mask[np.asarray(self.group_order)], ON_POLICY, HISTORICAL)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "shard_index": self.shard_index,
            "fresh_group_mask": list(self.fresh_group_mask),
            "group_order": list(self.group_order),
            "minibatch_seeds": list(self.minibatch_seeds),
        }


def build_hybrid_plan(step: int, selection_seed: int = 42) -> HybridPlan:
    """Build eight deterministic 24H+8O group-contiguous minibatches."""
    if step < 1:
        raise ValueError("hybrid learner steps are one-indexed")
    selection_rng = np.random.default_rng(_derived_seed(selection_seed, step, 0))
    fresh = np.sort(selection_rng.choice(256, size=64, replace=False))
    fresh_mask = np.zeros(256, dtype=bool)
    fresh_mask[fresh] = True
    historical = np.flatnonzero(~fresh_mask)

    # Deterministically randomize each pool before assigning its disjoint slices.
    pool_rng = np.random.default_rng(_derived_seed(selection_seed, step, 1))
    historical = historical[pool_rng.permutation(len(historical))]
    fresh = fresh[pool_rng.permutation(len(fresh))]

    group_order: list[int] = []
    minibatch_seeds: list[int] = []
    for minibatch_index in range(8):
        seed = _derived_seed(selection_seed, step, 2 + minibatch_index)
        minibatch_seeds.append(seed)
        groups = np.concatenate(
            [historical[minibatch_index * 24 : (minibatch_index + 1) * 24],
             fresh[minibatch_index * 8 : (minibatch_index + 1) * 8]]
        )
        groups = groups[np.random.default_rng(seed).permutation(32)]
        group_order.extend(int(group) for group in groups)

    plan = HybridPlan(
        step=step,
        shard_index=step - 1,
        fresh_group_mask=tuple(bool(value) for value in fresh_mask),
        group_order=tuple(group_order),
        minibatch_seeds=tuple(minibatch_seeds),
    )
    assert_hybrid_plan(plan)
    return plan


def assert_hybrid_plan(plan: HybridPlan) -> None:
    if len(plan.group_order) != 256 or sorted(plan.group_order) != list(range(256)):
        raise ValueError("hybrid plan must use every prompt group exactly once")
    if sum(plan.fresh_group_mask) != 64:
        raise ValueError("hybrid plan must select exactly 64 fresh prompt groups")
    sources = plan.ordered_group_sources
    for index in range(8):
        minibatch = sources[index * 32 : (index + 1) * 32]
        if int(np.count_nonzero(minibatch == HISTORICAL)) != 24 or int(
            np.count_nonzero(minibatch == ON_POLICY)
        ) != 8:
            raise ValueError(f"hybrid minibatch {index} is not exactly 24H+8O")


class HybridCursorLoader:
    """Checkpointable one-pass loader over the first 100 immutable shards."""

    def __init__(
        self,
        train_dataset,
        collate_fn,
        historical_dataset: OfflineBatchDataset,
        selection_seed: int,
        target_steps: int,
    ):
        self.train_dataset = train_dataset
        self.collate_fn = collate_fn
        self.historical_dataset = historical_dataset
        self.selection_seed = int(selection_seed)
        self.target_steps = int(target_steps)
        dataframe = getattr(train_dataset, "dataframe", None)
        if dataframe is not None and "extra_info" in dataframe.column_names:
            metadata = list(dataframe["extra_info"])
        else:
            metadata = [train_dataset[position].get("extra_info") for position in range(len(train_dataset))]

        self.source_index_to_positions: dict[int, list[int]] = {}
        self.position_to_identity: list[tuple[str, int]] = []
        self.position_to_revision: list[str] = []
        seen_identities: set[tuple[str, int]] = set()
        for position, extra_info in enumerate(metadata):
            if not isinstance(extra_info, dict):
                raise ValueError(f"ALP row {position} is missing extra_info identity metadata")
            try:
                source_index = int(extra_info["index"])
                source_dataset = extra_info["source_dataset"]
                source_revision = extra_info["source_revision"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"ALP row {position} has an incomplete composite source identity") from error
            if (
                not isinstance(source_dataset, str)
                or not source_dataset
                or not isinstance(source_revision, str)
                or not source_revision
            ):
                raise ValueError(f"ALP row {position} has an empty source dataset or revision")
            identity = (source_dataset, source_index)
            if identity in seen_identities:
                raise ValueError(f"ALP source identity {identity!r} occurs more than once after filtering")
            seen_identities.add(identity)
            self.source_index_to_positions.setdefault(source_index, []).append(position)
            self.position_to_identity.append(identity)
            self.position_to_revision.append(source_revision)
        manifest_identity = historical_dataset.manifest["identity"]
        self.prompt_identity_schema = PROMPT_IDENTITY_SCHEMA
        self.behavior_model_revision = str(manifest_identity["behavior_model"]["revision"])
        self._tokenized_prompt_cache: dict[int, tuple[int, ...]] = {}
        self.cursor = 0
        self.active_plan: HybridPlan | None = None
        self.active_historical_batch: DataProto | None = None
        self.last_completed_plan: HybridPlan | None = None

    def __len__(self) -> int:
        return self.target_steps

    def _tokenize_position(self, position: int) -> tuple[int, ...]:
        cached = self._tokenized_prompt_cache.get(position)
        if cached is not None:
            return cached
        if getattr(self.train_dataset, "processor", None) is not None:
            raise ValueError("hybrid prompt identity resolution supports the configured text-only ALP dataset")
        tokenizer = getattr(self.train_dataset, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("hybrid prompt identity resolution requires the training tokenizer")
        row = self.train_dataset[position]
        apply_kwargs = dict(**getattr(self.train_dataset, "apply_chat_template_kwargs", {}))
        tool_schemas = getattr(self.train_dataset, "tool_schemas", None)
        if tool_schemas is not None:
            apply_kwargs["tools"] = tool_schemas
        apply_kwargs.pop("tokenize", None)
        apply_kwargs.pop("return_dict", None)
        apply_kwargs.pop("return_tensors", None)
        tokenized = tokenizer.apply_chat_template(
            row["raw_prompt"], add_generation_prompt=True, tokenize=True, **apply_kwargs
        )
        prompt = tuple(normalize_token_ids(tokenized))
        max_prompt_length = int(self.historical_dataset.manifest["identity"]["geometry"]["max_prompt_length"])
        if not prompt or len(prompt) > max_prompt_length:
            raise ValueError(
                f"ALP source identity {self.position_to_identity[position]!r} tokenizes to "
                f"{len(prompt)} tokens outside the immutable historical prompt geometry"
            )
        self._tokenized_prompt_cache[position] = prompt
        return prompt

    @staticmethod
    def _stored_group_prompt(historical: DataProto, group_index: int) -> tuple[int, tuple[int, ...]]:
        start = group_index * 8
        stop = start + 8
        source_indices = np.asarray(historical.non_tensor_batch["source_index"])[start:stop]
        if source_indices.shape != (8,) or not np.all(source_indices == source_indices[0]):
            raise ValueError(f"historical group {group_index} does not contain one source index across 8 responses")

        prompts = historical.batch["prompts"][start:stop]
        prompt_masks = historical.batch["attention_mask"][start:stop, : prompts.shape[-1]].bool()
        if not torch.equal(prompts, prompts[0].expand_as(prompts)) or not torch.equal(
            prompt_masks, prompt_masks[0].expand_as(prompt_masks)
        ):
            raise ValueError(f"historical group {group_index} does not contain one prompt across 8 responses")
        mask = prompt_masks[0]
        valid_positions = torch.nonzero(mask, as_tuple=False).flatten()
        if len(valid_positions) == 0 or not bool(mask[int(valid_positions[0]) :].all()):
            raise ValueError(f"historical group {group_index} has an invalid left-padded prompt mask")
        prompt = tuple(int(value) for value in prompts[0][mask].tolist())
        return int(source_indices[0]), prompt

    def _resolve_historical_rows(self, historical: DataProto, shard_index: int) -> list[dict[str, Any]]:
        if len(historical) != 256 * 8:
            raise ValueError(f"historical shard {shard_index} is not 256 groups x 8 responses")
        rows: list[dict[str, Any]] = []
        seen_identities: set[tuple[str, int]] = set()
        seen_prompts: set[tuple[int, ...]] = set()
        for group_index in range(256):
            source_index, stored_prompt = self._stored_group_prompt(historical, group_index)
            if stored_prompt in seen_prompts:
                raise ValueError(f"historical shard {shard_index} contains duplicate prompt content")
            seen_prompts.add(stored_prompt)

            candidates = self.source_index_to_positions.get(source_index, [])
            matches = [position for position in candidates if self._tokenize_position(position) == stored_prompt]
            if len(matches) != 1:
                raise ValueError(
                    f"historical shard {shard_index} group {group_index} source index {source_index} "
                    f"resolved to {len(matches)} ALP rows by stored prompt content"
                )
            position = matches[0]
            identity = self.position_to_identity[position]
            row = self.train_dataset[position]
            extra_info = row.get("extra_info")
            if not isinstance(extra_info, dict):
                raise ValueError(f"ALP row {position} lost its composite source identity")
            if (
                identity != (str(extra_info.get("source_dataset")), int(row.get("index")))
                or str(extra_info.get("source_revision")) != self.position_to_revision[position]
            ):
                raise ValueError(f"ALP row {position} changed identity while resolving historical prompts")
            if identity in seen_identities:
                raise ValueError(f"historical shard {shard_index} repeats source identity {identity!r}")
            seen_identities.add(identity)
            rows.append(row)
        return rows

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while self.cursor < self.target_steps:
            step = self.cursor + 1
            plan = build_hybrid_plan(step, self.selection_seed)
            historical = self.historical_dataset.load_batch(plan.shard_index).select_idxs(plan.response_order)
            rows = self._resolve_historical_rows(historical, plan.shard_index)
            batch = self.collate_fn(rows)
            batch["_hybrid_source"] = plan.ordered_group_sources.astype(object)
            batch["_hybrid_shard_index"] = np.full(256, plan.shard_index, dtype=np.int32)
            batch["_hybrid_group_index"] = np.asarray(plan.group_order, dtype=np.int32)

            self.active_plan = plan
            self.active_historical_batch = historical
            self.cursor = step
            self.last_completed_plan = plan
            yield batch

    def state_dict(self) -> dict[str, Any]:
        plan = None if self.last_completed_plan is None else self.last_completed_plan.as_dict()
        return {
            "schema_version": 2,
            "schedule": "one_pass_prefix_75h25o",
            "completed_optimizer_steps": self.cursor,
            "next_shard_index": self.cursor,
            "target_steps": self.target_steps,
            "selection_seed": self.selection_seed,
            "rng_scheme": "numpy.SeedSequence([selection_seed, step, stream])",
            "last_completed_plan": plan,
            "behavior_model_revision": self.behavior_model_revision,
            "prompt_identity_schema": self.prompt_identity_schema,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "schema_version": 2,
            "schedule": "one_pass_prefix_75h25o",
            "target_steps": self.target_steps,
            "selection_seed": self.selection_seed,
            "behavior_model_revision": self.behavior_model_revision,
            "prompt_identity_schema": self.prompt_identity_schema,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"hybrid dataloader checkpoint mismatch for {key}")
        cursor = int(state["completed_optimizer_steps"])
        if cursor < 0 or cursor > self.target_steps or int(state["next_shard_index"]) != cursor:
            raise ValueError("hybrid dataloader checkpoint has an inconsistent cursor")
        plan = state.get("last_completed_plan")
        if cursor == 0:
            if plan is not None:
                raise ValueError("step-zero hybrid state must not contain a completed plan")
        else:
            expected_plan = build_hybrid_plan(cursor, self.selection_seed).as_dict()
            if plan != expected_plan:
                raise ValueError("hybrid dataloader checkpoint plan does not reproduce from its RNG identity")
        self.cursor = cursor
        self.active_plan = None
        self.active_historical_batch = None
        self.last_completed_plan = None if cursor == 0 else build_hybrid_plan(cursor, self.selection_seed)


class HybridRolloutAdapter:
    """Generate only O groups and stitch them into stored H trajectories."""

    def __init__(self, manager, loader: HybridCursorLoader):
        self.manager = manager
        self.loader = loader

    @staticmethod
    def _historical_non_tensor(key: str, fresh_value: np.ndarray, stored: DataProto) -> np.ndarray:
        count = len(stored)
        if key == "__num_turns__":
            return np.ones(count, dtype=fresh_value.dtype)
        if key == "acc":
            return stored.batch["token_level_scores"].sum(dim=-1).cpu().numpy().astype(fresh_value.dtype)
        if key == "valid_response_length":
            return stored.batch["response_mask"].sum(dim=-1).cpu().numpy().astype(fresh_value.dtype)
        if key == "multi_modal_inputs":
            value = np.empty(count, dtype=object)
            value[:] = [{} for _ in range(count)]
            return value
        if fresh_value.dtype == object:
            value = np.empty(count, dtype=object)
            value[:] = None
            return value
        return np.zeros(count, dtype=fresh_value.dtype)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        if prompts.meta_info.get("validate", False):
            return self.manager.generate_sequences(prompts)
        plan = self.loader.active_plan
        stored = self.loader.active_historical_batch
        if plan is None or stored is None:
            raise RuntimeError("hybrid rollout adapter has no active historical shard")
        sources = np.asarray(prompts.non_tensor_batch.get("_hybrid_source"))
        if len(prompts) != 2048 or sources.shape != (2048,):
            raise ValueError("hybrid rollout input must contain 256 contiguous groups x 8 responses")
        expected_sources = np.repeat(plan.ordered_group_sources, 8)
        if not np.array_equal(sources, expected_sources):
            raise ValueError("hybrid rollout input does not match the deterministic H/O plan")
        fresh_indices = np.flatnonzero(sources == ON_POLICY)
        historical_indices = np.flatnonzero(sources == HISTORICAL)
        if len(fresh_indices) != 512 or len(historical_indices) != 1536:
            raise ValueError("hybrid rollout input is not exactly 192H+64O groups")

        fresh = self.manager.generate_sequences(prompts.select_idxs(fresh_indices))
        stored_keys = {
            "prompts", "responses", "response_mask", "input_ids", "attention_mask",
            "position_ids", "rollout_log_probs",
        }
        missing = stored_keys.difference(stored.batch.keys())
        if missing:
            raise ValueError(f"historical shard is missing rollout fields: {sorted(missing)}")
        if "rm_scores" not in fresh.batch:
            raise ValueError("fresh rollout must return rm_scores for all on-policy responses")
        required_reward_extras = {"acc", "valid_response_length"}
        missing_reward_extras = required_reward_extras.difference(fresh.non_tensor_batch)
        if missing_reward_extras:
            raise ValueError(f"fresh rollout is missing reward metadata: {sorted(missing_reward_extras)}")
        if not required_reward_extras.issubset(fresh.meta_info.get("reward_extra_keys", ())):
            raise ValueError("fresh rollout did not declare its required reward metadata")
        allowed_fresh = stored_keys | {"rm_scores"}
        unexpected = set(fresh.batch.keys()).difference(allowed_fresh)
        if unexpected:
            raise ValueError(f"fresh rollout returned unsupported tensor fields: {sorted(unexpected)}")

        tensors = stored.batch.select(*sorted(stored_keys)).clone()
        tensors["rm_scores"] = stored.batch["token_level_scores"].clone()
        for key in fresh.batch.keys():
            if tuple(tensors[key].shape[1:]) != tuple(fresh.batch[key].shape[1:]):
                raise ValueError(f"fresh and historical {key} shapes do not match")
            tensors[key][fresh_indices] = fresh.batch[key].to(
                device=tensors[key].device, dtype=tensors[key].dtype
            )

        non_tensors: dict[str, np.ndarray] = {}
        for key, values in fresh.non_tensor_batch.items():
            values = np.asarray(values)
            if key in prompts.non_tensor_batch:
                combined = np.asarray(prompts.non_tensor_batch[key]).copy()
                if combined.shape[:1] != (len(prompts),) or not np.array_equal(
                    combined[fresh_indices], values
                ):
                    raise ValueError(f"fresh rollout changed prompt metadata field {key}")
            else:
                combined = self._historical_non_tensor(key, values, stored)
                combined[fresh_indices] = values
            non_tensors[key] = combined
        # _get_gen_batch removes these fields from the prompt half before union.
        # Keep response-aligned provenance on the combined actor batch.
        non_tensors["_hybrid_source"] = sources.copy()
        for key in ("_hybrid_shard_index", "_hybrid_group_index"):
            if key in prompts.non_tensor_batch:
                non_tensors[key] = np.asarray(prompts.non_tensor_batch[key]).copy()
        meta_info = dict(fresh.meta_info)
        meta_info["hybrid_composition"] = "192H+64O"
        return DataProto(
            batch=tensors,
            non_tensor_batch=non_tensors,
            meta_info=meta_info,
        )


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"hybrid contract requires the {name} environment variable")
    return value


HYBRID_EXACT_CONFIG = {
        "trainer.trainer_class": "verl.trainer.hybrid_rl.trainer.HybridGGPOTrainer",
        "hybrid_data.historical_groups": 192,
        "hybrid_data.on_policy_groups": 64,
        "hybrid_data.responses_per_group": 8,
        "hybrid_data.minibatches": 8,
        "hybrid_data.selection_seed": 42,
        "hybrid_data.train_source_path": "${HYBRID_TRAIN_SOURCE_PATH}",
        "hybrid_data.dataset_root": "${HYBRID_DATASET_ROOT}",
        "hybrid_data.expected_batches": 286,
        "data.train_batch_size": 256,
        "data.max_prompt_length": 2048,
        "data.max_response_length": 2048,
        "data.filter_overlong_prompts": True,
        "data.truncation": "error",
        "data.seed": 42,
        "data.shuffle": False,
        "data.validation_shuffle": False,
        "algorithm.filter_groups.enable": False,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.rollout.n": 8,
        "actor_rollout_ref.rollout.seed": 42,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.do_sample": True,
        "actor_rollout_ref.rollout.ignore_eos": False,
        "actor_rollout_ref.rollout.repetition_penalty": 1.0,
        "actor_rollout_ref.rollout.full_determinism": True,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.6,
        "actor_rollout_ref.rollout.max_model_len": 4096,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 8192,
        "actor_rollout_ref.rollout.max_num_seqs": 256,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": 4096,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.logprobs_mode": "processed_logprobs",
        "actor_rollout_ref.rollout.val_kwargs.n": 32,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_k": -1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": 4096,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 32,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 8,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 4096,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.shuffle": False,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.entropy_coeff": 0,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.data_loader_seed": 42,
        "actor_rollout_ref.actor.optim.optimizer": "AdamW",
        "actor_rollout_ref.actor.optim.betas": [0.9, 0.999],
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.optim.weight_decay": 0.01,
        "actor_rollout_ref.actor.optim.clip_grad": 1.0,
        "actor_rollout_ref.actor.policy_loss.loss_mode": "ggpo",
        "actor_rollout_ref.actor.policy_loss.beta": 0.001,
        "actor_rollout_ref.actor.policy_loss.sequence_log_prob_reduction": "mean",
        "actor_rollout_ref.actor.policy_loss.memory_safe_surrogate": False,
        "reward.reward_manager.name": "batch",
        "reward.custom_reward_function.path": None,
        "reward.custom_reward_function.name": "compute_score",
        "trainer.balance_batch": False,
        "trainer.total_epochs": 1,
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": 4,
}


def resolved_hybrid_exact_config() -> dict[str, Any]:
    exact = dict(HYBRID_EXACT_CONFIG)
    exact["hybrid_data.train_source_path"] = _required_env("HYBRID_TRAIN_SOURCE_PATH")
    exact["hybrid_data.dataset_root"] = _required_env("HYBRID_DATASET_ROOT")
    model_revision = os.environ.get("GGPO_MODEL_REVISION", "")
    if model_revision:
        exact["trainer.base_model_revision"] = model_revision
    return exact


def _validate_sealed_hybrid_config(config) -> None:
    hybrid = config.hybrid_data
    exact = resolved_hybrid_exact_config()
    for path, expected in exact.items():
        actual = OmegaConf.select(config, path)
        if actual != expected:
            raise ValueError(f"hybrid contract requires {path}={expected!r}, got {actual!r}")
    expected_model = os.environ.get("MODEL_PATH")
    if not expected_model:
        raise ValueError("hybrid contract requires the launcher-verified MODEL_PATH")
    configured_model = Path(str(config.actor_rollout_ref.model.path)).resolve()
    if configured_model != Path(expected_model).resolve():
        raise ValueError("hybrid contract model path was overridden after launcher verification")
    expected_checkpoint_dir = os.environ.get("CKPTS_DIR")
    if not expected_checkpoint_dir:
        raise ValueError("hybrid contract requires CKPTS_DIR")
    configured_checkpoint_dir = Path(str(config.trainer.default_local_dir)).resolve()
    if configured_checkpoint_dir != Path(expected_checkpoint_dir).resolve():
        raise ValueError("hybrid checkpoint directory was overridden after lineage validation")
    if config.trainer.project_name != "Gibbs-Gap Policy Optimization":
        raise ValueError("hybrid W&B project was overridden after lineage validation")
    expected_run_name = os.environ.get("RUN_NAME")
    if not expected_run_name or config.trainer.experiment_name != expected_run_name:
        raise ValueError("hybrid W&B display identity was overridden after lineage validation")
    if config.trainer.get("wandb_display_name") != expected_run_name:
        raise ValueError("hybrid W&B canonical display name was overridden after lineage validation")
    contract = config.trainer.ggpo_theory_contract
    expected_contract = {
        "run_id": os.environ.get("WANDB_RUN_ID"),
    }
    model_revision = os.environ.get("GGPO_MODEL_REVISION", "")
    if model_revision:
        expected_contract["reference_policy_identity"] = f"hf:Qwen/Qwen3-1.7B-Base@{model_revision}"
    if os.environ.get("GGPO_DATASET_ID", ""):
        expected_contract["behavior_policy_identity"] = (
            f"hybrid75h25o:{hybrid.dataset_id}+fresh:{os.environ.get('WANDB_RUN_ID')}"
        )
    for key, expected in expected_contract.items():
        if not expected or contract.get(key) != expected:
            raise ValueError(f"hybrid theory contract identity drifted: {key}")
    target = int(hybrid.target_optimizer_steps)
    segment = int(hybrid.segment_optimizer_steps)
    if bool(hybrid.smoke):
        if target != 2 or segment != 2:
            raise ValueError("hybrid smoke must run exactly two full-shape steps")
        if int(config.trainer.save_freq) != -1:
            raise ValueError("hybrid smoke must disable checkpoint saves")
        if bool(config.trainer.save_final_checkpoint):
            raise ValueError("hybrid smoke must disable the final checkpoint")
        if bool(config.trainer.val_before_train):
            raise ValueError("hybrid smoke must disable step-zero validation")
    elif target != 100 or segment != 20 or int(config.trainer.save_freq) != 20:
        raise ValueError("formal hybrid training requires a 100-step target in exact 20-step segments")
    elif int(config.trainer.test_freq) != 20 or not bool(config.trainer.save_final_checkpoint):
        raise ValueError("formal hybrid training requires 20-step validation and a final checkpoint")
    elif str(config.trainer.resume_mode) == "resume_path":
        if bool(config.trainer.val_before_train):
            raise ValueError("resumed hybrid segments must not repeat step-zero validation")
    elif not bool(config.trainer.val_before_train):
        raise ValueError("formal fresh/auto hybrid training requires step-zero validation")
    if int(config.trainer.total_training_steps) != target:
        raise ValueError("trainer.total_training_steps must equal the immutable hybrid target")


def _validate_parameterized_hybrid_config(config) -> None:
    """Validate new hybrid studies without changing the sealed 75H/25O result."""
    hybrid = config.hybrid_data
    if str(config.trainer.trainer_class) != "verl.trainer.hybrid_rl.trainer.ParameterizedHybridRLTrainer":
        raise ValueError("parameterized hybrid runs require ParameterizedHybridRLTrainer")
    structural = {
        "hybrid_data.historical_groups": 192,
        "hybrid_data.on_policy_groups": 64,
        "hybrid_data.responses_per_group": 8,
        "hybrid_data.minibatches": 8,
        "data.train_batch_size": 256,
        "actor_rollout_ref.rollout.n": 8,
    }
    for path, expected in structural.items():
        actual = OmegaConf.select(config, path)
        if actual != expected:
            raise ValueError(f"parameterized hybrid implementation requires {path}={expected!r}, got {actual!r}")

    integer_fields = {
        "hybrid_data.expected_batches": hybrid.expected_batches,
        "hybrid_data.target_optimizer_steps": hybrid.target_optimizer_steps,
        "hybrid_data.segment_optimizer_steps": hybrid.segment_optimizer_steps,
    }
    parsed: dict[str, int] = {}
    for path, value in integer_fields.items():
        if isinstance(value, bool):
            raise ValueError(f"parameterized hybrid contract requires positive integer {path}")
        try:
            parsed[path] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"parameterized hybrid contract requires integer {path}") from exc
        if parsed[path] != value or parsed[path] <= 0:
            raise ValueError(f"parameterized hybrid contract requires positive integer {path}")
    expected_batches = parsed["hybrid_data.expected_batches"]
    target = parsed["hybrid_data.target_optimizer_steps"]
    segment = parsed["hybrid_data.segment_optimizer_steps"]
    if target > expected_batches:
        raise ValueError("parameterized hybrid target exceeds the immutable historical batch count")
    if target % segment != 0:
        raise ValueError("parameterized hybrid target must be divisible by segment_optimizer_steps")
    if int(config.trainer.total_training_steps) != target:
        raise ValueError("trainer.total_training_steps must equal the parameterized hybrid target")
    loss_mode = str(config.actor_rollout_ref.actor.policy_loss.loss_mode).strip()
    if not loss_mode:
        raise ValueError("parameterized hybrid policy loss mode must be nonempty")
    if loss_mode == "ggpo" and (
        int(config.actor_rollout_ref.actor.ppo_epochs) != 1
        or bool(config.actor_rollout_ref.actor.shuffle)
    ):
        raise ValueError("parameterized hybrid GGPO requires actor ppo_epochs=1 and shuffle=false")

    for environment_name, config_path in (
        ("MODEL_PATH", "actor_rollout_ref.model.path"),
        ("CKPTS_DIR", "trainer.default_local_dir"),
    ):
        expected = os.environ.get(environment_name)
        if not expected:
            raise ValueError(f"parameterized hybrid contract requires {environment_name}")
        if Path(str(OmegaConf.select(config, config_path))).resolve() != Path(expected).resolve():
            raise ValueError(f"parameterized hybrid {config_path} drifted from {environment_name}")
    if not str(config.trainer.experiment_name).strip():
        raise ValueError("parameterized hybrid training requires a nonempty experiment_name")
    if bool(hybrid.smoke):
        if int(config.trainer.save_freq) != -1 or bool(config.trainer.save_final_checkpoint):
            raise ValueError("parameterized hybrid smoke must disable checkpoints")
    elif int(config.trainer.save_freq) <= 0 or target % int(config.trainer.save_freq) != 0:
        raise ValueError("parameterized hybrid save_freq must be a positive divisor of the target")


def validate_hybrid_config(config) -> None:
    """Dispatch to either the immutable published contract or the new surface."""
    contract = str(config.hybrid_data.get("contract", SEALED_HYBRID_CONTRACT))
    if contract == SEALED_HYBRID_CONTRACT:
        _validate_sealed_hybrid_config(config)
    elif contract == PARAMETERIZED_HYBRID_CONTRACT:
        _validate_parameterized_hybrid_config(config)
    else:
        raise ValueError(
            f"hybrid_data.contract must be {SEALED_HYBRID_CONTRACT!r} or "
            f"{PARAMETERIZED_HYBRID_CONTRACT!r}, got {contract!r}"
        )


def _hybrid_checkpoint_objective(config) -> str:
    contract = str(config.hybrid_data.get("contract", SEALED_HYBRID_CONTRACT))
    if contract == SEALED_HYBRID_CONTRACT:
        return "ggpo-exp-seqmean-beta0.001"
    policy_loss = config.actor_rollout_ref.actor.policy_loss
    loss_mode = str(policy_loss.loss_mode)
    reduction = str(policy_loss.get("sequence_log_prob_reduction", "not_applicable"))
    beta = policy_loss.get("beta", None)
    return f"{loss_mode}:sequence_log_prob_reduction={reduction}:beta={beta}"


class HybridGGPOTrainer(RayPPOTrainer):
    """Existing synchronous GGPO trainer plus deterministic hybrid data provenance."""

    _checkpoint_requires_validation_metrics = True

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        super()._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        hybrid = self.config.hybrid_data
        self.historical_dataset = OfflineBatchDataset(
            hybrid.dataset_root,
            expected_batches=int(hybrid.expected_batches),
            prompt_groups_per_batch=256,
            responses_per_group=8,
            selection_method="prefix",
            selection_start=0,
            selection_count=int(hybrid.target_optimizer_steps),
        )
        train_files = self.config.data.train_files
        if OmegaConf.is_config(train_files):
            train_files = OmegaConf.to_container(train_files, resolve=True)
        if isinstance(train_files, str):
            train_files = [train_files]
        if not isinstance(train_files, list) or len(train_files) != 1:
            raise ValueError("hybrid training requires exactly one ALP train parquet")
        train_source_path = Path(str(train_files[0])).resolve()
        expected_train_source_path = Path(str(hybrid.train_source_path)).resolve()
        if train_source_path != expected_train_source_path:
            raise ValueError("hybrid training parquet path does not match its immutable identity")
        self.train_dataloader = HybridCursorLoader(
            self.train_dataset,
            collate_fn,
            self.historical_dataset,
            selection_seed=int(hybrid.selection_seed),
            target_steps=int(hybrid.target_optimizer_steps),
        )
        self.total_training_steps = int(hybrid.target_optimizer_steps)

    def init_workers(self):
        validate_hybrid_config(self.config)
        super().init_workers()
        self.async_rollout_manager = HybridRolloutAdapter(self.async_rollout_manager, self.train_dataloader)

    def _hybrid_state_path(self, folder: str | Path) -> Path:
        return Path(folder) / "hybrid_state.json"

    def _save_checkpoint(self):
        folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        folder.mkdir(parents=True, exist_ok=True)
        state = self.train_dataloader.state_dict()
        state.update(
            {
                "dataset_id": str(self.config.hybrid_data.dataset_id),
                "experiment_name": str(self.config.trainer.experiment_name),
                "wandb_run_id": os.environ.get("WANDB_RUN_ID"),
                "objective": _hybrid_checkpoint_objective(self.config),
            }
        )
        if not state["wandb_run_id"]:
            raise ValueError("formal hybrid checkpoints require immutable WANDB_RUN_ID")
        path = self._hybrid_state_path(folder)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        boundary_metrics = getattr(self, "_checkpoint_boundary_metrics", None)
        if boundary_metrics is not None:
            missing = BOUNDARY_METRIC_KEYS.difference(boundary_metrics)
            if missing:
                raise ValueError(f"hybrid checkpoint boundary metrics are missing: {sorted(missing)}")
            persisted_metrics = {}
            for key in sorted(BOUNDARY_METRIC_KEYS):
                value = boundary_metrics[key]
                if isinstance(value, (torch.Tensor, np.generic)):
                    value = value.item()
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f"hybrid checkpoint boundary metric is not finite: {key}")
                persisted_metrics[key] = value
            metrics_path = folder / "boundary_metrics.json"
            metrics_temporary = metrics_path.with_name(f".{metrics_path.name}.{os.getpid()}.tmp")
            with open(metrics_temporary, "w", encoding="utf-8") as stream:
                json.dump(
                    {"schema_version": 1, "step": self.global_steps, "metrics": persisted_metrics},
                    stream,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(metrics_temporary, metrics_path)
        directory = os.open(folder, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        # The parent publishes latest_checkpointed_iteration.txt last. Keep the
        # hybrid identity durable before that commit point so a published
        # boundary is always loadable by this trainer.
        super()._save_checkpoint()

    def _load_checkpoint(self):
        loaded_folder = super()._load_checkpoint()
        if self.global_steps:
            if loaded_folder is None:
                raise ValueError("hybrid checkpoint path was not resolved")
            folder = Path(loaded_folder).resolve()
            path = self._hybrid_state_path(folder)
            with open(path, encoding="utf-8") as stream:
                state = json.load(stream)
            expected = {
                "completed_optimizer_steps": self.global_steps,
                "next_shard_index": self.global_steps,
                "experiment_name": str(self.config.trainer.experiment_name),
                "wandb_run_id": os.environ.get("WANDB_RUN_ID"),
                "objective": _hybrid_checkpoint_objective(self.config),
            }
            for key, value in expected.items():
                if state.get(key) != value:
                    raise ValueError(f"hybrid checkpoint identity mismatch for {key}")
            if self.train_dataloader.cursor != self.global_steps:
                raise ValueError("hybrid checkpoint loader cursor does not match optimizer step")
        segment = int(self.config.hybrid_data.segment_optimizer_steps)
        target = int(self.config.hybrid_data.target_optimizer_steps)
        if self.global_steps >= target and not bool(self.config.trainer.val_only):
            raise ValueError(f"hybrid checkpoint already reached target step {target}")
        self.total_training_steps = min(self.global_steps + segment, target)

    def _ggpo_theory_policy_identity(self, batch: DataProto):
        sources = np.asarray(batch.non_tensor_batch["_hybrid_source"])
        if sources.shape != (len(batch),):
            raise ValueError("hybrid source identity is not response-aligned")
        training_version = max(int(self.global_steps) - 1, 0)
        versions = torch.where(
            torch.from_numpy(sources == ON_POLICY),
            torch.full((len(batch),), training_version, dtype=torch.long),
            torch.zeros(len(batch), dtype=torch.long),
        )
        return {
            "actor_update_step": int(self.global_steps),
            "training_policy_version": training_version,
            "behavior_policy_versions": versions,
            "behavior_version_stratification": True,
        }

    def _validate(self, merged: bool = False):
        metrics = super()._validate(merged=merged)
        if metrics:
            # Keep the validation family exclusively under val-core while
            # supplying its true scientific x-coordinate on the same row.
            metrics["training/global_step"] = int(self.global_steps)
        return metrics

    def _update_actor(self, batch: DataProto) -> DataProto:
        sources = np.asarray(batch.non_tensor_batch["_hybrid_source"])
        rewards = batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy()
        output = super()._update_actor(batch)
        metrics = output.meta_info["metrics"]
        for source, label, expected_groups in ((HISTORICAL, "historical", 192), (ON_POLICY, "on_policy", 64)):
            mask = sources == source
            count = int(np.count_nonzero(mask))
            if count != expected_groups * 8:
                raise ValueError(f"hybrid actor batch has {count // 8} {source} groups, expected {expected_groups}")
            metrics[f"hybrid/{label}_groups"] = [float(expected_groups)]
            metrics[f"hybrid/{label}_reward_mean"] = [float(rewards[mask].mean())]
        return output


class ParameterizedHybridRLTrainer(HybridGGPOTrainer):
    """Hybrid trainer entry point gated by the separate parameterized contract."""
