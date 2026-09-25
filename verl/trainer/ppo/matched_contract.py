# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Immutable scientific identity for the matched GGPO/DAPO launchers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


RUN_NAMES = {
    "ggpo": "ggpo-exp-seqsum-beta0.001-qwen3-1.7b-base-dapo-filter",
    "dapo": "grpo-kl0-qwen3-1.7b-base-dapo-filter-dapo-cliphigher-token-mean",
}
def optional_model_revision() -> str:
    return os.environ.get("GGPO_MODEL_REVISION", "")


def run_display_name(method: str, entropy_coeff: float = 0.0) -> str:
    """Canonical W&B display name for a matched run.

    A nonzero entropy coefficient is a different experiment, so it gets its own
    name segment. Baseline names stay byte-identical to the established runs.
    This is the single source of truth: the launcher asks for the name rather
    than formatting the coefficient itself, so the shell and the contract can
    never disagree about how a value like 0.001 is spelled.
    """
    if method not in RUN_NAMES:
        raise ValueError(f"unsupported matched method: {method}")
    coeff = float(entropy_coeff)
    if coeff == 0.0:
        return RUN_NAMES[method]
    rendered = f"{coeff:.10f}".rstrip("0").rstrip(".")
    return f"{RUN_NAMES[method]}-entropy{rendered}"


def expected_scientific_config(
    method: str,
    *,
    model_path: str | Path,
    train_source: str | Path,
    validation_source: str | Path,
    entropy_coeff: float = 0.0,
) -> dict[str, Any]:
    if method not in RUN_NAMES:
        raise ValueError(f"unsupported matched method: {method}")
    model_path = str(Path(model_path).resolve())
    train_source = str(Path(train_source).resolve())
    validation_source = str(Path(validation_source).resolve())
    values: dict[str, Any] = {
        "trainer.trainer_class": None,
        "trainer.project_name": "Gibbs-Gap Policy Optimization",
        "trainer.experiment_name": run_display_name(method, entropy_coeff),
        "trainer.wandb_display_name": run_display_name(method, entropy_coeff),
        "trainer.total_training_steps": 800,
        "trainer.total_epochs": 100,
        "trainer.save_freq": 20,
        "trainer.test_freq": 20,
        "trainer.val_before_train": True,
        "trainer.save_final_checkpoint": True,
        "trainer.balance_batch": False,
        "data.train_files": train_source,
        "data.val_files": validation_source,
        "data.train_batch_size": 256,
        "data.max_prompt_length": 2048,
        "data.max_response_length": 2048,
        "data.filter_overlong_prompts": True,
        "data.truncation": "error",
        "data.seed": 42,
        "data.shuffle": False,
        "data.validation_shuffle": False,
        "algorithm.filter_groups.enable": True,
        "algorithm.filter_groups.metric": "raw_correctness",
        "algorithm.filter_groups.max_num_gen_batches": 32,
        "actor_rollout_ref.model.path": model_path,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 32,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 8,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 4096,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.shuffle": False,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.entropy_coeff": float(entropy_coeff),
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.kl_loss_coef": 0.0,
        "actor_rollout_ref.actor.data_loader_seed": 42,
        "actor_rollout_ref.actor.optim.optimizer": "AdamW",
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.optim.weight_decay": 0.01,
        "actor_rollout_ref.actor.optim.betas": [0.9, 0.999],
        "actor_rollout_ref.actor.optim.clip_grad": 1.0,
        "actor_rollout_ref.rollout.n": 8,
        "actor_rollout_ref.rollout.seed": 42,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.repetition_penalty": 1.0,
        "actor_rollout_ref.rollout.do_sample": True,
        "actor_rollout_ref.rollout.ignore_eos": False,
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
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_k": -1,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": 4096,
        "reward.reward_manager.name": "dapo",
        "reward.custom_reward_function.name": "compute_score",
        "reward.reward_kwargs.max_resp_len": 2048,
        "reward.reward_kwargs.overlong_buffer_cfg.enable": True,
        "reward.reward_kwargs.overlong_buffer_cfg.len": 410,
        "reward.reward_kwargs.overlong_buffer_cfg.penalty_factor": 1.0,
        "reward.reward_kwargs.overlong_buffer_cfg.log": True,
    }
    model_revision = optional_model_revision()
    if model_revision:
        values["trainer.base_model_revision"] = model_revision
    if method == "ggpo":
        # Provenance note. This arm ran with the overlong-buffer penalty active, so its
        # response rewards are continuous rather than 0/1, and the completed run recorded
        # that as policy_loss.require_binary_rewards=false. That key no longer exists:
        # constant-group classification now reads the unshaped verifier acc directly, which
        # is exactly binary by construction. The loss and gradient are unchanged by that
        # switch (classification feeds diagnostics only), but the group-composition and
        # tied-max-reward metrics of a re-run will not match the completed run's.
        values.update(
            {
                "actor_rollout_ref.actor.policy_loss.loss_mode": "ggpo",
                "actor_rollout_ref.actor.policy_loss.beta": 0.001,
                "actor_rollout_ref.actor.policy_loss.sequence_log_prob_reduction": "sum",
                "actor_rollout_ref.actor.policy_loss.memory_safe_surrogate": False,
            }
        )
    else:
        values.update(
            {
                "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
                "actor_rollout_ref.actor.clip_ratio": 0.2,
                "actor_rollout_ref.actor.clip_ratio_low": 0.2,
                "actor_rollout_ref.actor.clip_ratio_high": 0.28,
                "actor_rollout_ref.actor.clip_ratio_c": 10.0,
                "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
                "algorithm.adv_estimator": "grpo",
                "algorithm.gamma": 1.0,
                "algorithm.lam": 1.0,
                "algorithm.use_kl_in_reward": False,
                "algorithm.norm_adv_by_std_in_grpo": True,
                "algorithm.rollout_correction.bypass_mode": True,
            }
        )
    return values


def _select(config: Any, path: str) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            return OmegaConf.select(config, path)
    except ImportError:
        pass
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _normalize(path: str, value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    if path in {"data.train_files", "data.val_files"}:
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        return str(Path(str(value)).resolve())
    if path == "actor_rollout_ref.model.path":
        return str(Path(str(value)).resolve())
    if path == "actor_rollout_ref.actor.optim.betas" and isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return value


def validate_resolved_config(config: Any, contract: dict[str, Any]) -> None:
    expected = expected_scientific_config(
        str(contract.get("method")),
        model_path=str(contract.get("model_path")),
        train_source=str(contract.get("train_source")),
        validation_source=str(contract.get("validation_source")),
        entropy_coeff=float(contract.get("entropy_coeff", 0.0)),
    )
    for path, expected_value in expected.items():
        value = _normalize(path, _select(config, path))
        if value != expected_value:
            raise ValueError(f"matched contract requires {path}={expected_value!r}, got {value!r}")


def validate_matched_config(config: Any) -> None:
    contract_value = _select(config, "trainer.matched_contract")
    try:
        from omegaconf import OmegaConf

        contract = OmegaConf.to_container(contract_value, resolve=True)
    except ImportError:
        contract = contract_value
    if not isinstance(contract, dict):
        return
    required = {
        "run_id",
        "method",
        "model_revision",
        "model_path",
        "train_source",
        "validation_source",
        "target_optimizer_steps",
        "entropy_coeff",
    }
    if set(contract) != required:
        raise ValueError("matched immutable contract fields are incomplete")
    if contract["run_id"] != os.environ.get("WANDB_RUN_ID"):
        raise ValueError("matched immutable W&B run ID drifted")
    model_revision = optional_model_revision()
    if (model_revision and contract["model_revision"] != model_revision) or contract["target_optimizer_steps"] != 800:
        raise ValueError("matched model or target identity drifted")
    validate_resolved_config(config, contract)


def main() -> None:
    """Print the canonical display name so launchers do not format it twice."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(RUN_NAMES))
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    args = parser.parse_args()
    print(run_display_name(args.method, args.entropy_coeff))


if __name__ == "__main__":
    main()
