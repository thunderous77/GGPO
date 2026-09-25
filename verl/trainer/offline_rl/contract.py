# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Fail-closed scientific contract for the approved offline objectives."""

from __future__ import annotations

import os

from omegaconf import OmegaConf

from verl.trainer.offline_rl.dataset import (
    FILTER_PROVENANCE_KEY,
    _identity_for_config_match,
    _manifest_identity,
    read_complete_manifest,
    validate_mixed_filtered_manifest,
)

BEHAVIOR_MODEL_ID = "Qwen/Qwen3-1.7B-Base"
# Keep the serialized identifier stable for existing dataset/checkpoint manifests.
SEALED_OFFLINE_CONTRACT = "sealed_agro_asympo_100_step"
SEALED_MIXED_FILTERED_OFFLINE_CONTRACT = "sealed_mixedfilt"
PARAMETERIZED_OFFLINE_CONTRACT = "parameterized"


def validate_checkpoint_dataset_identity(
    state: dict,
    *,
    dataset_id: str,
) -> None:
    """Bind a checkpoint to the configured dataset label when one is declared."""
    if not os.environ.get("GGPO_DATASET_ID", ""):
        return
    if state.get("dataset_id") != dataset_id:
        raise ValueError("offline checkpoint dataset ID does not match the configuration")


def _require(config, path: str, expected) -> None:
    actual = OmegaConf.select(config, path)
    if actual != expected:
        raise ValueError(f"offline reference contract requires {path}={expected!r}, got {actual!r}")


def filtered_manifest_round_count(config) -> int:
    """Validate the filtered dataset binding and return its pool round count."""
    manifest = read_complete_manifest(config.offline_data.dataset_root)
    round_count = validate_mixed_filtered_manifest(manifest)
    provenance = manifest["identity"][FILTER_PROVENANCE_KEY]
    if provenance["source_batch_count"] != 286:
        raise ValueError("sealed mixed-filtered contract requires the approved 286-shard source dataset")
    expected_identity = _manifest_identity(config)
    if _identity_for_config_match(manifest["identity"], expected_identity) != _identity_for_config_match(
        expected_identity, expected_identity
    ):
        raise ValueError("sealed mixed-filtered manifest science identity does not match the configuration")
    _require(config, "offline_data.dataset_id", manifest["identity"]["dataset_id"])
    _require(config, "offline_data.expected_batches", round_count)
    return round_count


def filtered_training_round_count(config) -> int:
    """Return the configured logical training length for a filtered pool."""
    value = OmegaConf.select(config, "offline_data.total_training_rounds")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("sealed mixed-filtered offline_data.total_training_rounds must be a positive integer")
    return value


def validate_reference_contract(config, *, training: bool) -> str:
    """Validate the pinned Qwen3-1.7B science and offline dataset structure."""
    loss_mode = str(config.actor_rollout_ref.actor.policy_loss.loss_mode)
    contract = str(config.offline_data.get("contract", SEALED_OFFLINE_CONTRACT))
    mixed_filtered = contract == SEALED_MIXED_FILTERED_OFFLINE_CONTRACT
    if contract not in {SEALED_OFFLINE_CONTRACT, SEALED_MIXED_FILTERED_OFFLINE_CONTRACT}:
        raise ValueError(f"validate_reference_contract received unsupported contract {contract!r}")
    if mixed_filtered:
        filtered_manifest_round_count(config)
        training_steps = filtered_training_round_count(config)
    else:
        training_steps = 100
    behavior_model_revision = os.environ.get("GGPO_MODEL_REVISION", "")
    dataset_implementation_commit = os.environ.get("GGPO_DATASET_IMPLEMENTATION_COMMIT", "")
    if dataset_implementation_commit:
        _require(config, "offline_data.implementation_commit", dataset_implementation_commit)
    _require(
        config,
        "offline_data.selection.method",
        "seeded_pool_cycle" if mixed_filtered else "prefix",
    )
    _require(config, "offline_data.selection.start", 0)
    _require(config, "offline_data.selection.count", training_steps)

    if loss_mode != "ggpo":
        _require(config, "algorithm.adv_estimator", "grpo")
        _require(config, "algorithm.gamma", 1.0)
        _require(config, "algorithm.lam", 1.0)
        _require(config, "algorithm.use_kl_in_reward", False)

    _require(config, "data.train_batch_size", 256)
    _require(config, "data.max_prompt_length", 2048)
    _require(config, "data.max_response_length", 2048)
    _require(config, "data.filter_overlong_prompts", True)
    _require(config, "data.truncation", "error")
    _require(config, "data.seed", 42)
    _require(config, "data.shuffle", True)
    _require(config, "data.dataloader_num_workers", 8)
    _require(config, "data.filter_overlong_prompts_workers", 8)

    _require(config, "offline_data.behavior_model_id", BEHAVIOR_MODEL_ID)
    if behavior_model_revision:
        _require(config, "offline_data.behavior_model_revision", behavior_model_revision)

    # DeltaAI's approved container does not ship flash-attn.  Keep samples
    # isolated by pairing the SDPA backend with the padded model-input path;
    # SDPA plus the concatenated remove-padding path would allow attention
    # across response boundaries.
    _require(config, "actor_rollout_ref.model.override_config.attn_implementation", "sdpa")
    _require(config, "actor_rollout_ref.model.use_remove_padding", False)
    _require(config, "actor_rollout_ref.model.enable_gradient_checkpointing", True)
    _require(config, "actor_rollout_ref.actor.optim.optimizer", "AdamW")
    _require(config, "actor_rollout_ref.actor.optim.lr", 1e-6)
    _require(config, "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio", 0.0)
    _require(config, "actor_rollout_ref.actor.optim.weight_decay", 0.01)
    _require(config, "actor_rollout_ref.actor.optim.betas", [0.9, 0.999])
    _require(config, "actor_rollout_ref.actor.optim.clip_grad", 1.0)
    _require(config, "actor_rollout_ref.actor.optim.lr_scheduler_type", "constant")
    _require(config, "actor_rollout_ref.actor.optim.zero_indexed_step", True)
    _require(config, "actor_rollout_ref.actor.ppo_mini_batch_size", 32)
    _require(config, "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", 8)
    _require(config, "actor_rollout_ref.actor.ppo_max_token_len_per_gpu", 4096)
    _require(config, "actor_rollout_ref.actor.ppo_epochs", 1)
    _require(config, "actor_rollout_ref.actor.shuffle", False)
    _require(config, "actor_rollout_ref.actor.use_dynamic_bsz", False)
    _require(config, "actor_rollout_ref.actor.entropy_coeff", 0)
    _require(config, "actor_rollout_ref.actor.data_loader_seed", 42)
    if loss_mode != "ggpo":
        _require(config, "actor_rollout_ref.actor.loss_agg_mode", "token-mean")
    _require(config, "actor_rollout_ref.actor.fsdp_config.param_offload", False)
    _require(config, "actor_rollout_ref.actor.fsdp_config.optimizer_offload", False)
    _require(config, "actor_rollout_ref.actor.fsdp_config.model_dtype", "fp32")
    _require(config, "actor_rollout_ref.actor.fsdp_config.dtype", "bfloat16")
    _require(config, "actor_rollout_ref.actor.fsdp_config.fsdp_size", -1)

    # These rollout fields are retained as frozen source metadata and actor
    # log-prob inference geometry; offline training never creates a rollout.
    _require(config, "actor_rollout_ref.rollout.calculate_log_probs", True)
    _require(config, "actor_rollout_ref.rollout.logprobs_mode", "processed_logprobs")
    _require(config, "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", 1)
    _require(config, "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz", False)
    _require(config, "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu", 4096)
    _require(config, "actor_rollout_ref.rollout.tensor_model_parallel_size", 1)
    _require(config, "actor_rollout_ref.rollout.gpu_memory_utilization", 0.6)
    _require(config, "actor_rollout_ref.rollout.max_model_len", 4096)
    _require(config, "actor_rollout_ref.rollout.max_num_batched_tokens", 8192)
    _require(config, "actor_rollout_ref.rollout.max_num_seqs", 256)
    _require(config, "actor_rollout_ref.rollout.seed", 42)
    _require(config, "actor_rollout_ref.rollout.n", 8)
    _require(config, "actor_rollout_ref.rollout.temperature", 1.0)
    _require(config, "actor_rollout_ref.rollout.top_p", 1.0)
    _require(config, "actor_rollout_ref.rollout.top_k", -1)

    _require(config, "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", 1)
    _require(config, "actor_rollout_ref.ref.log_prob_use_dynamic_bsz", False)
    _require(config, "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu", 4096)
    _require(config, "actor_rollout_ref.ref.fsdp_config.param_offload", False)
    _require(config, "actor_rollout_ref.ref.fsdp_config.model_dtype", "fp32")
    _require(config, "actor_rollout_ref.ref.fsdp_config.dtype", "bfloat16")
    _require(config, "critic.enable", False)
    _require(config, "trainer.balance_batch", False)
    _require(config, "trainer.critic_warmup", 0)
    _require(config, "trainer.n_gpus_per_node", 4)
    _require(config, "trainer.nnodes", 1)
    _require(config, "trainer.total_training_steps", training_steps)
    _require(config, "reward.reward_model.enable", False)
    _require(config, "reward.custom_reward_function.path", None)

    if loss_mode == "agro":
        _require(config, "algorithm.norm_adv_by_std_in_grpo", False)
        _require(config, "actor_rollout_ref.actor.policy_loss.beta", 0.01)
        _require(config, "actor_rollout_ref.actor.policy_loss.sequence_log_prob_reduction", "sum")
        expected_name = (
            f"agro-seqsum-beta0.01-qwen3-1.7b-base-offline-mixedfilt-maxstep{training_steps}"
            if mixed_filtered
            else "agro-seqsum-beta0.01-qwen3-1.7b-base-offline-maxstep100"
        )
    elif loss_mode == "asympo":
        _require(config, "algorithm.norm_adv_by_std_in_grpo", False)
        expected_name = (
            f"asympo-core-qwen3-1.7b-base-offline-mixedfilt-maxstep{training_steps}"
            if mixed_filtered
            else "asympo-core-qwen3-1.7b-base-offline-maxstep100"
        )
    elif loss_mode == "ggpo":
        _require(config, "actor_rollout_ref.actor.policy_loss.beta", 0.01)
        _require(config, "actor_rollout_ref.actor.policy_loss.sequence_log_prob_reduction", "sum")
        _require(config, "algorithm.filter_groups.enable", True)
        _require(config, "algorithm.filter_groups.metric", "raw_correctness")
        _require(config, "algorithm.filter_groups.max_num_gen_batches", 32)
        # The offline arms are compared head to head, so they must be anchored the same way:
        # AGRO uses its prescribed fixed reference and GRPO a fixed behavior KL, so GGPO holds
        # its reference fixed here too. Refreshing it re-anchors to the current policy and lets
        # the run walk away from the sealed behavior policy, which is the very drift this
        # experiment measures. Reference refresh is studied by the online refresh ablation.
        _require(config, "trainer.ggpo_moving_reference.enabled", False)
        expected_name = (
            "ggpo-exp-seqsum-beta0.01-qwen3-1.7b-base-"
            + (
                f"dapo-prefilter-offline-fixed-ref-mixedfilt-maxstep{training_steps}"
                if mixed_filtered
                else "dapo-filter-offline-fixed-ref-maxstep100"
            )
        )
    elif loss_mode == "vanilla":
        _require(config, "algorithm.norm_adv_by_std_in_grpo", True)
        _require(config, "actor_rollout_ref.actor.clip_ratio", 0.2)
        _require(config, "actor_rollout_ref.actor.clip_ratio_low", 0.2)
        _require(config, "actor_rollout_ref.actor.clip_ratio_high", 0.2)
        _require(config, "actor_rollout_ref.actor.use_kl_loss", True)
        kl_coef = config.actor_rollout_ref.actor.kl_loss_coef
        expected_name = str(config.trainer.experiment_name)
        for marker in ("fixed-behavior", f"kl{kl_coef}", "maxstep100"):
            if marker not in expected_name:
                raise ValueError(f"offline vanilla run name must contain {marker!r}")
    else:
        raise ValueError(f"offline experiment supports only AGRO, ASymPO, GGPO, or vanilla, got {loss_mode!r}")

    if loss_mode != "vanilla":
        _require(config, "actor_rollout_ref.actor.use_kl_loss", False)
    _require(config, "actor_rollout_ref.actor.calculate_sum_pi_squared", False)
    _require(config, "actor_rollout_ref.actor.use_prefix_grouper", False)
    if training:
        _require(config, "trainer.experiment_name", expected_name)
        _require(config, "trainer.test_freq", -1)
        _require(config, "trainer.val_before_train", False)
        _require(config, "actor_rollout_ref.actor.checkpoint.async_save", False)
        if bool(config.offline_data.smoke):
            _require(config, "trainer.logger", ["console"])
            _require(config, "trainer.save_freq", -1)
            _require(config, "trainer.save_final_checkpoint", False)
            smoke_steps = int(config.offline_data.segment_optimizer_steps)
            if smoke_steps not in {1, 2}:
                raise ValueError(
                    "offline smoke segment_optimizer_steps must be 1, or 2 for the "
                    "full-shape consecutive-batch memory gate"
                )
        else:
            save_freq = OmegaConf.select(config, "trainer.save_freq")
            if mixed_filtered:
                if isinstance(save_freq, bool) or not isinstance(save_freq, int) or save_freq <= 0:
                    raise ValueError("sealed mixed-filtered training requires a positive trainer.save_freq")
                if 20 % save_freq != 0:
                    raise ValueError(
                        "sealed mixed-filtered trainer.save_freq must divide the 20-step interactive segment"
                    )
            else:
                _require(config, "trainer.save_freq", 20)
            _require(config, "trainer.save_final_checkpoint", True)
            _require(config, "offline_data.segment_optimizer_steps", 20)
            _require(
                config,
                "actor_rollout_ref.actor.checkpoint.save_contents",
                ["model", "optimizer", "extra"],
            )
    return loss_mode


def validate_parameterized_contract(config, *, training: bool) -> str:
    """Validate the configurable offline surface without weakening the sealed pair."""
    offline = config.offline_data
    integer_fields = {
        "offline_data.expected_batches": offline.expected_batches,
        "offline_data.selection.start": offline.selection.start,
        "offline_data.selection.count": offline.selection.count,
        "offline_data.prompt_groups_per_batch": offline.prompt_groups_per_batch,
        "offline_data.responses_per_group": offline.responses_per_group,
        "offline_data.segment_optimizer_steps": offline.segment_optimizer_steps,
        "trainer.total_training_steps": config.trainer.total_training_steps,
    }
    parsed: dict[str, int] = {}
    for path, value in integer_fields.items():
        if isinstance(value, bool):
            raise ValueError(f"parameterized offline contract requires positive integer {path}")
        try:
            parsed[path] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"parameterized offline contract requires integer {path}") from exc
        if parsed[path] != value or parsed[path] < (0 if path.endswith("selection.start") else 1):
            raise ValueError(f"parameterized offline contract requires valid integer {path}")

    expected_batches = parsed["offline_data.expected_batches"]
    selection_start = parsed["offline_data.selection.start"]
    selection_count = parsed["offline_data.selection.count"]
    total_steps = parsed["trainer.total_training_steps"]
    segment_steps = parsed["offline_data.segment_optimizer_steps"]
    if str(offline.selection.method) != "prefix":
        raise ValueError("parameterized offline training currently requires deterministic prefix selection")
    if selection_start + selection_count > expected_batches:
        raise ValueError("parameterized offline selection exceeds the immutable dataset batch count")
    if selection_count != total_steps:
        raise ValueError("parameterized offline selection.count must equal trainer.total_training_steps")
    if total_steps % segment_steps != 0:
        raise ValueError("parameterized offline total steps must be divisible by segment_optimizer_steps")
    if int(config.data.train_batch_size) != parsed["offline_data.prompt_groups_per_batch"]:
        raise ValueError("parameterized offline data.train_batch_size must equal prompt_groups_per_batch")
    if int(config.actor_rollout_ref.rollout.n) != parsed["offline_data.responses_per_group"]:
        raise ValueError("parameterized offline rollout.n must equal responses_per_group")

    loss_mode = str(config.actor_rollout_ref.actor.policy_loss.loss_mode).strip()
    if not loss_mode:
        raise ValueError("parameterized offline policy loss mode must be nonempty")
    if loss_mode == "ggpo" and (
        int(config.actor_rollout_ref.actor.ppo_epochs) != 1
        or bool(config.actor_rollout_ref.actor.shuffle)
    ):
        raise ValueError("parameterized offline GGPO requires actor ppo_epochs=1 and shuffle=false")
    if training and not str(config.trainer.experiment_name).strip():
        raise ValueError("parameterized offline training requires a nonempty experiment_name")
    return loss_mode
