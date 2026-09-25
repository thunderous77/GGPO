# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import operator
import os
import tempfile
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ggpo import (
    GGPO_RAW_CORRECTNESS,
    GGPO_SURROGATE_CURRENT_LOG_PROBS,
    GGPO_SURROGATE_METRICS_META,
    GGPO_SURROGATE_WEIGHTS,
    POLICY_GROUP_IDS,
    build_policy_group_ids,
    compute_ggpo_loss,
    compute_ggpo_surrogate_weights,
    ggpo_conditional_metric_fraction,
)
from verl.trainer.ppo.ggpo_theory import (
    GGPO_THEORY_CURRENT_LOG_PROBS,
    GGPO_THEORY_METRICS_SCHEMA_VERSION,
    GGPO_THEORY_METRICS_SENTINEL,
    compute_ggpo_theory_metrics,
    seal_ggpo_theory_run_receipt,
)
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.interactive_segment import resolve_segment_end
from verl.trainer.ppo.mixed_reward_filter import (
    raw_correctness_tensor,
    require_complete_mixed_batch,
    select_mixed_reward_group_indices,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.rollout_training_mismatch import (
    INTERNAL_MISMATCH_STATS_KEY,
    MISMATCH_AGGREGATION_META,
    finalize_mismatch_metrics,
    prune_sync_alp_math_training_metrics,
    prune_sync_alp_math_validation_metrics,
)
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    create_rl_dataset,
    create_rl_sampler,
    need_critic,
    need_reference_policy,
    need_reward_model,
    need_teacher_policy,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass, validate_policy_loss_trainer_compatibility
from verl.utils.debug import marked_timer
from verl.utils.import_utils import deprecated, load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.skip.skip_manager import SkipManager
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import DistillationConfig, EngineConfig
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _merge_validation_reward_extra_info(
    aggregate: dict[str, list],
    reward_extra_info: dict[str, Any],
    canonical_diagnostics: dict[str, list],
) -> None:
    """Merge one validation batch without duplicating canonical diagnostics.

    Reward managers may expose diagnostics that the trainer also computes from
    the generated batch.  Accept those duplicate keys only when their values
    exactly match the trainer's canonical values for the current batch.
    """

    normalized_reward_extra_info = {}
    for key, values in reward_extra_info.items():
        normalized_values = values.tolist() if isinstance(values, np.ndarray) else values
        if not isinstance(normalized_values, list):
            normalized_values = [normalized_values]
        normalized_reward_extra_info[key] = normalized_values

        if key in canonical_diagnostics:
            canonical_values = canonical_diagnostics[key]
            if len(normalized_values) != len(canonical_values) or any(
                observed != expected for observed, expected in zip(normalized_values, canonical_values, strict=True)
            ):
                raise ValueError(f"reward extra info conflicts with canonical validation diagnostic: {key}")

    for key, values in canonical_diagnostics.items():
        aggregate.setdefault(key, []).extend(values)
    for key, normalized_values in normalized_reward_extra_info.items():
        if key not in canonical_diagnostics:
            aggregate.setdefault(key, []).extend(normalized_values)


def compute_spec_decode_metrics(
    spec_drafts,
    spec_accepts,
    spec_verifies,
    non_padding_mask=None,
) -> dict:
    """Aggregate per-request speculative decoding stats.

    Ratios are computed per request and then averaged, so long and short
    responses have equal metric weight.

    The three inputs come from the rollout engine (vLLM request spec-decode
    stats or sglang ``meta_info["spec_*"]`` keys). Either all three are ``None``
    (caller didn't fetch them, e.g. spec rollout disabled) and the function
    is a no-op, or all three are populated; mixed state is a programmer error.

    ``non_padding_mask`` is a numpy bool array used by sync PPO to drop padded
    placeholder samples; pass ``None`` for async PPO.
    """
    if spec_drafts is None and spec_accepts is None and spec_verifies is None:
        return {}
    assert spec_drafts is not None and spec_accepts is not None and spec_verifies is not None, (
        "spec_decode metrics require all three of spec_num_draft_tokens / "
        "spec_num_accepted_tokens / spec_num_verify_steps; got partial inputs"
    )

    drafts = spec_drafts.tolist() if hasattr(spec_drafts, "tolist") else list(spec_drafts)
    accepts = spec_accepts.tolist() if hasattr(spec_accepts, "tolist") else list(spec_accepts)
    verifies = spec_verifies.tolist() if hasattr(spec_verifies, "tolist") else list(spec_verifies)

    if non_padding_mask is not None:
        drafts = [d for d, keep in zip(drafts, non_padding_mask, strict=True) if keep]
        accepts = [a for a, keep in zip(accepts, non_padding_mask, strict=True) if keep]
        verifies = [v for v, keep in zip(verifies, non_padding_mask, strict=True) if keep]

    if len(drafts) == 0:
        return {}

    # Treat zero-denominator samples as 0.0 and keep them in the mean.
    per_sample_accept_rate = [(a / d) if d > 0 else 0.0 for a, d in zip(accepts, drafts, strict=True)]
    per_sample_accept_length = [(1.0 + a / v) if v > 0 else 0.0 for a, v in zip(accepts, verifies, strict=True)]

    n = len(drafts)
    return {
        "rollout/spec_accept_rate": float(sum(per_sample_accept_rate) / n),
        "rollout/spec_accept_length": float(sum(per_sample_accept_length) / n),
    }


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # old_log_probs needed for path-variance proxy: w_t = 1 - 2*exp(old_log_probs) + sum_pi_squared
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


def is_ggpo_policy_loss(config) -> bool:
    """Return whether the actor policy loss is configured for GGPO."""
    return OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode") == "ggpo"


def is_group_reference_policy_loss(config) -> bool:
    """Return whether the actor loss needs complete groups and a reference policy."""
    return OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode") in {"ggpo", "agro"}


def get_policy_group_size(config) -> int:
    """Derive the GGPO/AGRO group size from the rollout response count."""
    loss_mode = OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode")
    group_size = OmegaConf.select(config, "actor_rollout_ref.rollout.n")
    if group_size is None:
        raise ValueError(f"{str(loss_mode).upper()} requires actor_rollout_ref.rollout.n to be set.")
    try:
        group_size = operator.index(group_size)
    except TypeError as exc:
        raise ValueError(
            f"{str(loss_mode).upper()} actor_rollout_ref.rollout.n must be an integer greater than 1, "
            f"got {group_size!r}."
        ) from exc
    if group_size <= 1:
        raise ValueError(
            f"{str(loss_mode).upper()} actor_rollout_ref.rollout.n must be greater than 1, got {group_size}."
        )
    return group_size


def _ggpo_memory_safe_surrogate_enabled(config) -> bool:
    """Return the opt-in fully-async GGPO dynamic-packing mode."""
    return bool(
        OmegaConf.select(
            config,
            "actor_rollout_ref.actor.policy_loss.memory_safe_surrogate",
            default=False,
        )
    )


def _ggpo_actor_data_parallel_size(config) -> int:
    """Resolve the same effective FSDP DP size used by config validation."""
    from verl.utils.config import get_trainer_process_on_nodes

    total_trainer_gpus = sum(get_trainer_process_on_nodes(config))
    try:
        legacy_sp = operator.index(
            OmegaConf.select(config, "actor_rollout_ref.actor.ulysses_sequence_parallel_size", default=1)
        )
        engine_sp = operator.index(
            OmegaConf.select(
                config,
                "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size",
                default=1,
            )
        )
    except TypeError as exc:
        raise ValueError("GGPO memory-safe surrogate requires integral actor Ulysses SP settings.") from exc
    sequence_parallel_size = legacy_sp if legacy_sp > 1 else engine_sp
    if sequence_parallel_size <= 0 or total_trainer_gpus % sequence_parallel_size != 0:
        raise ValueError(
            "GGPO memory-safe surrogate requires trainer GPU count to be divisible by actor Ulysses SP: "
            f"gpus={total_trainer_gpus}, sp={sequence_parallel_size}."
        )
    return total_trainer_gpus // sequence_parallel_size


def prepare_group_reference_actor_update_batch(batch: DataProto, config) -> dict[str, int]:
    """Attach group IDs and translate the GGPO/AGRO response micro-batch size."""
    if not is_group_reference_policy_loss(config):
        return {}

    loss_mode = OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode")
    label = str(loss_mode).upper()

    if "uid" not in batch.non_tensor_batch:
        raise ValueError(f"{label} actor update requires batch.non_tensor_batch['uid'] to build group IDs.")

    group_size = get_policy_group_size(config)
    response_micro_batch_size = OmegaConf.select(
        config, "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
    )
    try:
        response_micro_batch_size = operator.index(response_micro_batch_size)
    except TypeError as exc:
        raise ValueError(
            f"{label} actor.ppo_micro_batch_size_per_gpu must be an integer response count, "
            f"got {response_micro_batch_size!r}."
        ) from exc
    if response_micro_batch_size <= 0 or response_micro_batch_size % group_size != 0:
        raise ValueError(
            f"{label} actor.ppo_micro_batch_size_per_gpu counts responses and must be a positive multiple "
            f"of actor_rollout_ref.rollout.n ({group_size}), got {response_micro_batch_size}."
        )

    batch.batch[POLICY_GROUP_IDS] = build_policy_group_ids(batch.non_tensor_batch["uid"], group_size=group_size)
    if loss_mode == "ggpo":
        batch.batch[GGPO_RAW_CORRECTNESS] = raw_correctness_tensor(
            batch.non_tensor_batch, expected_responses=len(batch)
        ).to(batch.batch.device)
    if _ggpo_memory_safe_surrogate_enabled(config):
        # This mode is intentionally fail-closed. It needs the padded async
        # layout and the current learner-policy diagnostic captured immediately
        # before actor update; silently falling back to whole-group packing
        # would reintroduce the OOM failure this mode is designed to prevent.
        from verl.experimental.fully_async_policy.actor_update_padding import (
            LOGICAL_MINI_BATCH_SIZE_META,
            PHYSICAL_MINI_BATCH_SIZE_META,
            REAL_SAMPLE_MASK,
        )

        required_tensor_keys = {
            GGPO_SURROGATE_CURRENT_LOG_PROBS,
            "ref_log_prob",
            "response_mask",
            "token_level_scores",
            REAL_SAMPLE_MASK,
        }
        missing = required_tensor_keys.difference(batch.batch.keys())
        if missing:
            raise ValueError(
                "GGPO memory-safe surrogate requires fully-async pre-update tensors; "
                f"missing={sorted(missing)}."
            )
        logical_mini_batch_size = batch.meta_info.get(LOGICAL_MINI_BATCH_SIZE_META)
        physical_mini_batch_size = batch.meta_info.get(PHYSICAL_MINI_BATCH_SIZE_META)
        if (
            not isinstance(logical_mini_batch_size, int)
            or not isinstance(physical_mini_batch_size, int)
            or logical_mini_batch_size <= 0
            or physical_mini_batch_size <= logical_mini_batch_size
            or len(batch) % physical_mini_batch_size != 0
        ):
            raise ValueError(
                "GGPO memory-safe surrogate requires valid fully-async padded mini-batch metadata: "
                f"logical={logical_mini_batch_size}, physical={physical_mini_batch_size}, responses={len(batch)}."
            )

        current_log_probs = batch.batch.pop(GGPO_SURROGATE_CURRENT_LOG_PROBS)
        real_sample_mask = batch.batch[REAL_SAMPLE_MASK].bool()
        if current_log_probs.shape != batch.batch["ref_log_prob"].shape:
            raise ValueError(
                "GGPO memory-safe surrogate current/ref log-probability shapes differ: "
                f"current={tuple(current_log_probs.shape)}, ref={tuple(batch.batch['ref_log_prob'].shape)}."
            )
        weights = torch.zeros(
            len(batch),
            dtype=current_log_probs.dtype,
            device=current_log_probs.device,
        )
        dp_size = _ggpo_actor_data_parallel_size(config)
        sequence_reduction = config.actor_rollout_ref.actor.policy_loss.get(
            "sequence_log_prob_reduction", "sum"
        )
        for start in range(0, len(batch), physical_mini_batch_size):
            stop = start + physical_mini_batch_size
            local_real_mask = real_sample_mask[start:stop]
            real_count = int(local_real_mask.sum().item())
            if real_count != logical_mini_batch_size:
                raise ValueError(
                    "GGPO memory-safe surrogate physical mini-batch has the wrong real-response count: "
                    f"expected={logical_mini_batch_size}, observed={real_count}."
                )
            selected = slice(start, stop)
            response_mask = batch.batch["response_mask"][selected][local_real_mask]
            batch_num_tokens = int(response_mask.sum().item())
            sequence_scores = (
                batch.batch["token_level_scores"][selected][local_real_mask].float() * response_mask
            ).sum(dim=-1)
            local_weights = compute_ggpo_surrogate_weights(
                log_prob=current_log_probs[selected][local_real_mask],
                ref_log_prob=batch.batch["ref_log_prob"][selected][local_real_mask],
                rewards=sequence_scores,
                response_mask=response_mask,
                group_ids=batch.batch[POLICY_GROUP_IDS][selected][local_real_mask],
                beta=config.actor_rollout_ref.actor.policy_loss.beta,
                group_size=group_size,
                global_batch_size=logical_mini_batch_size,
                batch_num_tokens=batch_num_tokens,
                dp_size=dp_size,
                sequence_log_prob_reduction=sequence_reduction,
                reward_kl=config.actor_rollout_ref.actor.policy_loss.get("reward_kl", True),
            )
            local_weight_view = weights[start:stop]
            local_weight_view[local_real_mask] = local_weights

        all_real_response_mask = batch.batch["response_mask"][real_sample_mask]
        all_real_scores = (
            batch.batch["token_level_scores"][real_sample_mask].float() * all_real_response_mask
        ).sum(dim=-1)
        _, exact_metrics = compute_ggpo_loss(
            log_prob=current_log_probs[real_sample_mask],
            ref_log_prob=batch.batch["ref_log_prob"][real_sample_mask],
            rewards=all_real_scores,
            raw_correctness=batch.batch[GGPO_RAW_CORRECTNESS][real_sample_mask],
            response_mask=all_real_response_mask,
            group_ids=batch.batch[POLICY_GROUP_IDS][real_sample_mask],
            beta=config.actor_rollout_ref.actor.policy_loss.beta,
            group_size=group_size,
            global_batch_size=int(real_sample_mask.sum().item()),
            batch_num_tokens=int(all_real_response_mask.sum().item()),
            dp_size=1,
            sequence_log_prob_reduction=sequence_reduction,
            reward_kl=config.actor_rollout_ref.actor.policy_loss.get("reward_kl", True),
        )
        batch.batch[GGPO_SURROGATE_WEIGHTS] = weights
        batch.meta_info[GGPO_SURROGATE_METRICS_META] = {
            name: float(value.detach().float().cpu().item())
            for name, value in exact_metrics.items()
            if ggpo_conditional_metric_fraction(name, exact_metrics) != 0
        }
        # Do not force a group to remain in one dynamic micro-batch. The
        # precomputed detached weights retain the exact GGPO gradient.
        return {"force_group_size": 1}

    # The shared engine API predates GGPO and interprets micro_batch_size_per_gpu
    # as a group count whenever force_group_size is present. Keep that internal
    # contract stable while exposing the same response-count meaning as GRPO.
    return {
        "force_group_size": group_size,
        "micro_batch_size_per_gpu": response_micro_batch_size // group_size,
    }


def maybe_apply_group_reference_advantage_bypass(batch: DataProto, config) -> bool:
    """Add compatibility advantages/returns for GGPO/AGRO and skip estimation."""
    if not is_group_reference_policy_loss(config):
        return False

    if "token_level_rewards" not in batch.batch.keys():
        raise ValueError("Group-reference advantage bypass requires token_level_rewards to be set.")

    batch.batch["advantages"] = torch.zeros_like(batch.batch["token_level_rewards"])
    batch.batch["returns"] = torch.zeros_like(batch.batch["token_level_rewards"])
    return True


def shutdown_stateful_dataloader_workers(*dataloaders) -> None:
    """Stop live StatefulDataLoader workers and release iterator references.

    PyTorch keeps worker processes behind the loader's private iterator.  A
    bounded Slurm segment must not leave those processes alive while its Ray
    actors and task-owned temporary directory are being cleaned up.
    """
    for dataloader in dataloaders:
        if dataloader is None:
            continue
        iterator = getattr(dataloader, "_iterator", None)
        if iterator is not None:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
            dataloader._iterator = None


@deprecated("Legacy trainer is deprecated, and wil be removed in v0.9.0. Please use `trainer.use_v1=True` instead.")
class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self._collection_remainder = None
        validate_policy_loss_trainer_compatibility(self.config)
        self._ggpo_theory_run_receipt = (
            seal_ggpo_theory_run_receipt(self.config)
            if is_ggpo_policy_loss(self.config) and self.config.trainer.get("ggpo_theory_contract")
            else None
        )

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            offline_actor_only = (
                OmegaConf.select(config, "offline_data.mode") in {"train", "initialize"}
                and Role.Actor in role_worker_mapping
            )
            assert offline_actor_only or Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self._ggpo_moving_reference_enabled = bool(
            OmegaConf.select(self.config, "trainer.ggpo_moving_reference.enabled", default=False)
        )
        if self._ggpo_moving_reference_enabled:
            self._ggpo_reference_version = 0
            self._ggpo_optimizer_updates = 0
        self.use_teacher_policy = need_teacher_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if OmegaConf.select(self.config, "algorithm.use_kl_in_reward", default=False):
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self._init_fixed_lag_state()

        self.checkpoint_manager = None
        self._init_dump_executor()

    def _init_fixed_lag_state(self) -> None:
        """Initialize the manifest-only fixed-lag FIFO without touching durable state."""
        lag = OmegaConf.select(self.config, "trainer.fixed_policy_lag", default=None)
        self._fixed_policy_lag = None if lag is None else operator.index(lag)
        self._fixed_lag_queue = deque()
        self._fixed_lag_entries = {}
        self._fixed_lag_checkpoint_events = {}
        self._fixed_lag_known_entry_ids = set()
        self._fixed_lag_consumed_entry_ids = set()
        self._fixed_lag_protected_entry_ids = set()
        self._fixed_lag_next_event_seq = 1
        if self._fixed_policy_lag is None:
            self._fixed_lag_ledger_path = None
            self._fixed_lag_batch_dir = None
            return

        ledger_path = OmegaConf.select(self.config, "trainer.fixed_lag_ledger_path")
        self._fixed_lag_ledger_path = os.path.realpath(os.path.abspath(ledger_path))
        self._fixed_lag_batch_dir = f"{self._fixed_lag_ledger_path}.batches"

    def _fixed_lag_header(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "lag": self._fixed_policy_lag,
            "group_size": int(self.config.actor_rollout_ref.rollout.n),
            "train_batch_size": int(self.config.data.train_batch_size),
            "total_training_steps": int(self.total_training_steps),
        }

    @staticmethod
    def _fsync_directory(path: str) -> None:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _atomic_write_json(cls, path: str, payload: dict[str, Any]) -> None:
        directory = os.path.dirname(path)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            cls._fsync_directory(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def _atomic_save_dataproto(cls, batch: DataProto, path: str) -> None:
        """Publish a complete DataProto with one atomic rename."""
        directory = os.path.dirname(path)
        descriptor, temporary = tempfile.mkstemp(prefix=".fixed_lag_batch.", suffix=".dp", dir=directory)
        os.close(descriptor)
        try:
            batch.save_to_disk(temporary)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            cls._fsync_directory(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _append_fixed_lag_record(self, record_type: str, **payload) -> dict[str, Any]:
        event = {
            "record_type": record_type,
            "event_seq": self._fixed_lag_next_event_seq,
            **payload,
        }
        encoded = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        descriptor = os.open(self._fixed_lag_ledger_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise OSError(f"short fixed-lag ledger write: {written} != {len(encoded)}")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._fsync_directory(os.path.dirname(self._fixed_lag_ledger_path))
        self._fixed_lag_next_event_seq += 1
        return event

    def _initialize_fixed_lag_ledger(self) -> None:
        if self._fixed_policy_lag is None:
            return
        if os.path.exists(self._fixed_lag_ledger_path) or os.path.exists(self._fixed_lag_batch_dir):
            raise RuntimeError(
                "fresh fixed-lag training requires a new ledger path; existing durable state must be resumed"
            )
        os.makedirs(os.path.dirname(self._fixed_lag_ledger_path), exist_ok=True)
        os.makedirs(self._fixed_lag_batch_dir)
        self._fsync_directory(os.path.dirname(self._fixed_lag_batch_dir))
        header = self._append_fixed_lag_record("header", **self._fixed_lag_header())
        self._fixed_lag_header_event = header

    def _fixed_lag_batch_path(self, relative_path: str) -> str:
        candidate = os.path.realpath(os.path.join(os.path.dirname(self._fixed_lag_ledger_path), relative_path))
        if os.path.commonpath((candidate, self._fixed_lag_batch_dir)) != self._fixed_lag_batch_dir:
            raise RuntimeError(f"fixed-lag batch path escapes the run-owned batch directory: {relative_path!r}")
        return candidate

    def _validate_fixed_lag_batch(self, batch: DataProto, entry: Optional[dict[str, Any]] = None) -> None:
        if "rollout_log_probs" not in batch.batch or "response_mask" not in batch.batch:
            raise RuntimeError("fixed-lag batch requires rollout_log_probs and response_mask")
        rollout_log_probs = batch.batch["rollout_log_probs"]
        response_mask = batch.batch["response_mask"]
        if rollout_log_probs.shape != response_mask.shape:
            raise RuntimeError(
                "fixed-lag rollout_log_probs shape does not match response_mask: "
                f"{tuple(rollout_log_probs.shape)} != {tuple(response_mask.shape)}"
            )
        if not torch.isfinite(rollout_log_probs.masked_select(response_mask.bool())).all():
            raise RuntimeError("fixed-lag rollout_log_probs contains a non-finite value on the response mask")

        group_size = int(self.config.actor_rollout_ref.rollout.n)
        if len(batch) % group_size != 0:
            raise RuntimeError(
                f"fixed-lag response count {len(batch)} is not divisible by rollout.n={group_size}"
            )
        if "uid" not in batch.non_tensor_batch:
            raise RuntimeError("fixed-lag batch is missing uid group metadata")
        build_policy_group_ids(batch.non_tensor_batch["uid"], group_size)

        required_meta = {
            "fixed_lag_entry_id": str,
            "fixed_lag_generation_step": int,
            "fixed_lag_behavior_policy_version": int,
            "fixed_lag_kind": str,
        }
        for key, expected_type in required_meta.items():
            value = batch.meta_info.get(key)
            if isinstance(value, bool) or not isinstance(value, expected_type):
                raise RuntimeError(f"fixed-lag batch has invalid {key}: {value!r}")
        if batch.meta_info["fixed_lag_kind"] not in {"warmup", "steady"}:
            raise RuntimeError(f"fixed-lag batch has invalid kind: {batch.meta_info['fixed_lag_kind']!r}")
        if entry is not None:
            expected = {
                "fixed_lag_entry_id": entry["entry_id"],
                "fixed_lag_generation_step": entry["logical_generation_step"],
                "fixed_lag_behavior_policy_version": entry["behavior_policy_version"],
                "fixed_lag_kind": entry["kind"],
            }
            for key, value in expected.items():
                if batch.meta_info.get(key) != value:
                    raise RuntimeError(
                        f"fixed-lag batch metadata disagrees with ledger for {key}: "
                        f"{batch.meta_info.get(key)!r} != {value!r}"
                    )
            if len(batch) != entry["response_count"] or len(batch) // group_size != entry["group_count"]:
                raise RuntimeError("fixed-lag batch geometry disagrees with its ledger manifest")

    def _tag_fixed_lag_batch(
        self,
        batch: DataProto,
        logical_generation_step: int,
        behavior_policy_version: int,
        kind: str,
        sampling_metrics: Optional[dict[str, Any]] = None,
    ) -> DataProto:
        entry_id = uuid.uuid4().hex
        if entry_id in self._fixed_lag_known_entry_ids:
            raise RuntimeError(f"fixed-lag entry ID collision: {entry_id}")
        batch.meta_info.update(
            {
                "fixed_lag_entry_id": entry_id,
                "fixed_lag_generation_step": int(logical_generation_step),
                "fixed_lag_behavior_policy_version": int(behavior_policy_version),
                "fixed_lag_kind": kind,
                "fixed_lag_sampling_metrics": {
                    key: value for key, value in (sampling_metrics or {}).items() if key.startswith("sampling/")
                },
            }
        )
        self._validate_fixed_lag_batch(batch)
        return batch

    def _enqueue_fixed_lag_batch(self, batch: DataProto) -> dict[str, Any]:
        self._validate_fixed_lag_batch(batch)
        entry_id = batch.meta_info["fixed_lag_entry_id"]
        if entry_id in self._fixed_lag_known_entry_ids or entry_id in self._fixed_lag_consumed_entry_ids:
            raise RuntimeError(f"fixed-lag entry {entry_id} was already enqueued or consumed")

        event_seq = self._fixed_lag_next_event_seq
        generation_step = batch.meta_info["fixed_lag_generation_step"]
        filename = f"generation_{generation_step:+08d}_event_{event_seq:08d}.dp"
        batch_path = os.path.join(self._fixed_lag_batch_dir, filename)
        self._atomic_save_dataproto(batch, batch_path)
        relative_path = os.path.relpath(batch_path, os.path.dirname(self._fixed_lag_ledger_path))
        entry = {
            "entry_id": entry_id,
            "logical_generation_step": generation_step,
            "behavior_policy_version": batch.meta_info["fixed_lag_behavior_policy_version"],
            "kind": batch.meta_info["fixed_lag_kind"],
            "batch_path": relative_path,
            "response_count": len(batch),
            "group_count": len(batch) // int(self.config.actor_rollout_ref.rollout.n),
        }
        event = self._append_fixed_lag_record("produce", **entry)
        entry["produce_event_seq"] = event["event_seq"]
        self._fixed_lag_entries[entry_id] = entry
        self._fixed_lag_known_entry_ids.add(entry_id)
        self._fixed_lag_queue.append(entry)
        return entry

    def _load_fixed_lag_batch(self, entry: dict[str, Any]) -> DataProto:
        path = self._fixed_lag_batch_path(entry["batch_path"])
        if not os.path.isfile(path):
            raise RuntimeError(f"fixed-lag batch file is missing: {path}")
        try:
            batch = DataProto.load_from_disk(path)
        except Exception as exc:
            raise RuntimeError(f"fixed-lag batch file is corrupt or unreadable: {path}") from exc
        self._validate_fixed_lag_batch(batch, entry)
        return batch

    def _consume_fixed_lag_batch(self, learner_step: int) -> DataProto:
        if not self._fixed_lag_queue:
            raise RuntimeError("fixed-lag FIFO is empty")
        entry = self._fixed_lag_queue[0]
        expected_generation_step = int(learner_step) - self._fixed_policy_lag
        if entry["logical_generation_step"] != expected_generation_step:
            raise RuntimeError(
                "fixed-lag FIFO generation slot mismatch: "
                f"{entry['logical_generation_step']} != {expected_generation_step}"
            )
        if entry["entry_id"] in self._fixed_lag_consumed_entry_ids:
            raise RuntimeError(f"fixed-lag entry {entry['entry_id']} was already consumed")
        batch = self._load_fixed_lag_batch(entry)
        self._append_fixed_lag_record(
            "consume",
            entry_id=entry["entry_id"],
            learner_step=int(learner_step),
            effective_lag=int(learner_step) - entry["logical_generation_step"],
        )
        self._fixed_lag_queue.popleft()
        self._fixed_lag_consumed_entry_ids.add(entry["entry_id"])
        return batch

    def _fixed_lag_consumption_metrics(self, batch: DataProto) -> dict[str, Any]:
        generation_step = batch.meta_info["fixed_lag_generation_step"]
        behavior_version = batch.meta_info["fixed_lag_behavior_policy_version"]
        metrics = dict(batch.meta_info.get("fixed_lag_sampling_metrics", {}))
        metrics.update(
            {
                "actor/fixed_lag_generation_step": generation_step,
                "actor/fixed_lag_behavior_policy_version": behavior_version,
                "actor/fixed_lag_effective": int(self.global_steps) - generation_step,
                "actor/fixed_lag_policy_version_lag": max(int(self.global_steps) - 1, 0) - behavior_version,
            }
        )
        return metrics

    def _read_fixed_lag_ledger(self) -> list[dict[str, Any]]:
        if not os.path.isfile(self._fixed_lag_ledger_path):
            raise RuntimeError(f"fixed-lag ledger is missing: {self._fixed_lag_ledger_path}")
        records = []
        try:
            with open(self._fixed_lag_ledger_path, encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        raise RuntimeError(f"fixed-lag ledger has a blank record at line {line_number}")
                    record = json.loads(line)
                    if record.get("event_seq") != line_number:
                        raise RuntimeError(
                            f"fixed-lag ledger event sequence is invalid at line {line_number}: "
                            f"{record.get('event_seq')!r}"
                        )
                    records.append(record)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"fixed-lag ledger is corrupt: {self._fixed_lag_ledger_path}") from exc
        if not records or records[0].get("record_type") != "header":
            raise RuntimeError("fixed-lag ledger is missing its header record")
        header = records[0]
        for key, value in self._fixed_lag_header().items():
            if header.get(key) != value:
                raise RuntimeError(
                    f"fixed-lag ledger header mismatch for {key}: {header.get(key)!r} != {value!r}"
                )

        self._fixed_lag_entries = {}
        self._fixed_lag_checkpoint_events = {}
        self._fixed_lag_known_entry_ids = set()
        self._fixed_lag_consumed_entry_ids = set()
        for record in records[1:]:
            record_type = record.get("record_type")
            if record_type == "produce":
                entry_id = record.get("entry_id")
                if not isinstance(entry_id, str) or not entry_id or entry_id in self._fixed_lag_entries:
                    raise RuntimeError(f"fixed-lag ledger has an invalid or duplicate entry ID: {entry_id!r}")
                entry = {
                    key: record.get(key)
                    for key in (
                        "entry_id",
                        "logical_generation_step",
                        "behavior_policy_version",
                        "kind",
                        "batch_path",
                        "response_count",
                        "group_count",
                    )
                }
                entry["produce_event_seq"] = record["event_seq"]
                integer_fields = (
                    "logical_generation_step",
                    "behavior_policy_version",
                    "response_count",
                    "group_count",
                )
                if any(
                    not isinstance(entry[field], int) or isinstance(entry[field], bool)
                    for field in integer_fields
                ):
                    raise RuntimeError(f"fixed-lag ledger produce record has invalid geometry: {record!r}")
                if (
                    entry["kind"] not in {"warmup", "steady"}
                    or not isinstance(entry["batch_path"], str)
                    or not entry["batch_path"]
                    or entry["response_count"] <= 0
                    or entry["group_count"] <= 0
                ):
                    raise RuntimeError(f"fixed-lag ledger produce record is invalid: {record!r}")
                self._fixed_lag_batch_path(entry["batch_path"])
                self._fixed_lag_entries[entry_id] = entry
                self._fixed_lag_known_entry_ids.add(entry_id)
            elif record_type == "consume":
                entry_id = record.get("entry_id")
                if entry_id not in self._fixed_lag_entries:
                    raise RuntimeError(f"fixed-lag ledger consumes unknown entry {entry_id!r}")
                self._fixed_lag_consumed_entry_ids.add(entry_id)
            elif record_type == "checkpoint":
                self._fixed_lag_checkpoint_events[record["event_seq"]] = record
            elif record_type == "resume":
                continue
            else:
                raise RuntimeError(f"fixed-lag ledger has unknown record type: {record_type!r}")
        self._fixed_lag_next_event_seq = len(records) + 1
        return records

    def _restore_fixed_lag_checkpoint(
        self,
        state: dict[str, Any],
        checkpoint_step: int,
    ) -> None:
        if state.get("lag") != self._fixed_policy_lag:
            raise RuntimeError(
                f"fixed-lag checkpoint lag does not match config: {state.get('lag')!r} != {self._fixed_policy_lag}"
            )
        state_path = state.get("ledger_path")
        canonical_state_path = (
            os.path.realpath(os.path.abspath(state_path)) if isinstance(state_path, str) else None
        )
        if canonical_state_path != self._fixed_lag_ledger_path:
            raise RuntimeError("fixed-lag checkpoint ledger path does not match the configured canonical path")
        checkpoint_event_seq = state.get("checkpoint_event_seq")
        if isinstance(checkpoint_event_seq, bool) or not isinstance(checkpoint_event_seq, int):
            raise RuntimeError("fixed-lag checkpoint pointer is invalid")

        records = self._read_fixed_lag_ledger()
        checkpoint_event = self._fixed_lag_checkpoint_events.get(checkpoint_event_seq)
        if checkpoint_event is None or checkpoint_event.get("global_step") != checkpoint_step:
            raise RuntimeError("fixed-lag checkpoint pointer does not name the matching ledger checkpoint event")
        pending_entry_ids = checkpoint_event.get("pending_entry_ids")
        if not isinstance(pending_entry_ids, list) or len(pending_entry_ids) != self._fixed_policy_lag:
            raise RuntimeError("fixed-lag checkpoint does not contain exactly k pending entries")
        if len(set(pending_entry_ids)) != len(pending_entry_ids):
            raise RuntimeError("fixed-lag checkpoint contains duplicate pending entries")

        consumed_before_checkpoint = {
            record["entry_id"]
            for record in records
            if record.get("record_type") == "consume" and record["event_seq"] <= checkpoint_event_seq
        }
        # Produce/consume records after the checkpoint pointer are an abandoned
        # crash tail. They remain in the audit log but are not active state.
        self._fixed_lag_consumed_entry_ids = consumed_before_checkpoint
        expected_slots = list(range(checkpoint_step - self._fixed_policy_lag + 1, checkpoint_step + 1))
        restored_queue = deque()
        for entry_id, expected_slot in zip(pending_entry_ids, expected_slots, strict=True):
            entry = self._fixed_lag_entries.get(entry_id)
            if entry is None or entry["produce_event_seq"] >= checkpoint_event_seq:
                raise RuntimeError(f"fixed-lag checkpoint references an invalid pending entry: {entry_id!r}")
            if entry_id in consumed_before_checkpoint:
                raise RuntimeError(f"fixed-lag checkpoint entry was already consumed: {entry_id}")
            expected_behavior_version = max(expected_slot - 1, 0)
            expected_kind = "warmup" if expected_slot <= 0 else "steady"
            if (
                entry["logical_generation_step"] != expected_slot
                or entry["behavior_policy_version"] != expected_behavior_version
                or entry["kind"] != expected_kind
            ):
                raise RuntimeError(
                    "fixed-lag checkpoint pending-entry indexing is invalid: "
                    f"slot={entry['logical_generation_step']!r}, behavior={entry['behavior_policy_version']!r}, "
                    f"kind={entry['kind']!r}"
                )
            self._load_fixed_lag_batch(entry)
            restored_queue.append(entry)
        self._fixed_lag_queue = restored_queue
        self._append_fixed_lag_record(
            "resume",
            checkpoint_event_seq=checkpoint_event_seq,
            global_step=checkpoint_step,
        )
        self._refresh_fixed_lag_protected_entries()
        self._prune_fixed_lag_batches()

    def _checkpoint_fixed_lag_state(self) -> int:
        if len(self._fixed_lag_queue) != self._fixed_policy_lag:
            raise RuntimeError(
                f"fixed-lag checkpoint requires exactly {self._fixed_policy_lag} pending entries, "
                f"got {len(self._fixed_lag_queue)}"
            )
        expected_slots = list(
            range(self.global_steps - self._fixed_policy_lag + 1, self.global_steps + 1)
        )
        actual_slots = [entry["logical_generation_step"] for entry in self._fixed_lag_queue]
        if actual_slots != expected_slots:
            raise RuntimeError(f"fixed-lag checkpoint queue slots are invalid: {actual_slots} != {expected_slots}")
        for entry, expected_slot in zip(self._fixed_lag_queue, expected_slots, strict=True):
            if (
                entry["behavior_policy_version"] != max(expected_slot - 1, 0)
                or entry["kind"] != ("warmup" if expected_slot <= 0 else "steady")
            ):
                raise RuntimeError("fixed-lag checkpoint queue behavior-policy indexing is invalid")
            path = self._fixed_lag_batch_path(entry["batch_path"])
            if not os.path.isfile(path):
                raise RuntimeError(f"fixed-lag pending batch is missing before checkpoint publication: {path}")
            with open(path, "rb") as stream:
                os.fsync(stream.fileno())
        checkpoint_event = self._append_fixed_lag_record(
            "checkpoint",
            global_step=int(self.global_steps),
            pending_entry_ids=[entry["entry_id"] for entry in self._fixed_lag_queue],
        )
        self._fixed_lag_checkpoint_events[checkpoint_event["event_seq"]] = checkpoint_event
        return checkpoint_event["event_seq"]

    def _refresh_fixed_lag_protected_entries(self) -> None:
        if self._fixed_policy_lag is None:
            return
        protected = {entry["entry_id"] for entry in self._fixed_lag_queue}
        checkpoint_root = os.path.realpath(os.path.abspath(self.config.trainer.default_local_dir))
        if not os.path.isdir(checkpoint_root):
            self._fixed_lag_protected_entry_ids = protected
            return
        try:
            for item in os.scandir(checkpoint_root):
                if not item.is_dir() or not item.name.startswith("global_step_"):
                    continue
                if not os.path.isdir(os.path.join(item.path, "actor")):
                    continue
                trainer_state_path = os.path.join(item.path, "trainer_state.json")
                if not os.path.isfile(trainer_state_path):
                    continue
                with open(trainer_state_path, encoding="utf-8") as stream:
                    trainer_state = json.load(stream)
                fixed_lag_state = trainer_state.get("fixed_lag")
                if not isinstance(fixed_lag_state, dict):
                    continue
                ledger_path = fixed_lag_state.get("ledger_path")
                if not isinstance(ledger_path, str):
                    continue
                if os.path.realpath(os.path.abspath(ledger_path)) != self._fixed_lag_ledger_path:
                    continue
                checkpoint_event = self._fixed_lag_checkpoint_events.get(
                    fixed_lag_state.get("checkpoint_event_seq")
                )
                if checkpoint_event is None:
                    self._fixed_lag_protected_entry_ids = set(self._fixed_lag_entries)
                    return
                protected.update(checkpoint_event.get("pending_entry_ids", ()))
        except Exception:
            # A retained checkpoint with unreadable state must never cause its
            # batches to be collected. Resume will report the corruption.
            self._fixed_lag_protected_entry_ids = set(self._fixed_lag_entries)
            return
        self._fixed_lag_protected_entry_ids = protected

    def _prune_fixed_lag_batches(self) -> None:
        if getattr(self, "_fixed_policy_lag", None) is None:
            return
        self._refresh_fixed_lag_protected_entries()
        for entry_id, entry in self._fixed_lag_entries.items():
            if entry_id in self._fixed_lag_protected_entry_ids:
                continue
            path = self._fixed_lag_batch_path(entry["batch_path"])
            if os.path.isfile(path):
                os.unlink(path)
        self._fsync_directory(self._fixed_lag_batch_dir)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        validation_enabled = bool(self.config.trainer.get("val_before_train", True)) or bool(
            self.config.trainer.get("val_only", False)
        ) or int(self.config.trainer.get("test_freq", -1)) > 0
        if validation_enabled and val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        self.val_dataloader = None
        if validation_enabled:
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)

            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        if validation_enabled:
            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        val_size = len(self.val_dataloader) if self.val_dataloader is not None else "disabled"
        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {val_size}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        """Write generation samples as JSONL (runs in background thread)."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL asynchronously."""
        global_steps = self.global_steps
        future = self._dump_executor.submit(
            self._write_generations,
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path,
            global_steps,
        )
        self._dump_futures.append(future)
        # Clean up completed futures and surface any exceptions early
        still_pending = []
        for f in self._dump_futures:
            if f.done():
                f.result()  # re-raises if the write failed
            else:
                still_pending.append(f)
        self._dump_futures = still_pending

    def _init_dump_executor(self):
        """Create or recreate the dump executor and futures list."""
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _shutdown_dump_executor(self):
        """Drain pending dump futures and shut down the executor."""
        for f in self._dump_futures:
            f.result()
        self._dump_futures.clear()
        self._dump_executor.shutdown(wait=True)

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = {
                k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in reward_extra_infos_dict.items()
            }
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _select_reward_loop_worker_handles(self, enable_agent_reward_loop: bool):
        if not enable_agent_reward_loop:
            return None
        return self.reward_loop_manager.reward_loop_worker_handles

    @staticmethod
    def _needs_colocated_reward(batch: DataProto) -> bool:
        return "rm_scores" not in batch.batch.keys()

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self._needs_colocated_reward(test_output_gen_batch_padded):
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            prompt_width = test_batch.batch["prompts"].shape[-1]
            valid_response_lengths = test_batch.batch["attention_mask"][:, prompt_width:].sum(dim=-1)
            eos_token_id = self.tokenizer.eos_token_id
            eos_flags = []
            for response_ids, valid_length in zip(output_ids, valid_response_lengths, strict=True):
                length = int(valid_length.item())
                eos_flags.append(float(length > 0 and int(response_ids[length - 1].item()) == eos_token_id))
            truncation_flags = [
                float(int(length.item()) == output_ids.shape[-1] and not bool(eos))
                for length, eos in zip(valid_response_lengths, eos_flags, strict=True)
            ]

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            _merge_validation_reward_extra_info(
                aggregate=reward_extra_infos_dict,
                reward_extra_info=reward_extra_info,
                canonical_diagnostics={
                    "valid_response_length": valid_response_lengths.cpu().tolist(),
                    "eos": eos_flags,
                    "truncation": truncation_flags,
                },
            )

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict={
                    **reward_extra_infos_dict,
                    "data_source": np.concatenate(data_source_lst).tolist(),
                    "uid": sample_uids,
                },
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return prune_sync_alp_math_validation_metrics(metric_dict)

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            # convert critic_cfg into TrainingWorkerConfig for the unified model engine worker
            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            engine_config: EngineConfig = orig_critic_cfg.engine
            engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

            critic_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
                extra_context=getattr(self, "_critic_extra_context", {}),
            )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/verl-project/verl/blob/master/examples/tutorial/ray/tutorial.ipynb
        # for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            # assign critic loss
            from functools import partial

            from verl.workers.utils.losses import value_loss

            value_loss_ = partial(value_loss, config=orig_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # initialize teacher loop manager
        if self.use_teacher_policy:
            from verl.experimental.teacher_loop import MultiTeacherModelManager

            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        self.llm_server_manager = LLMServerManager.create(
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        # To stream teacher computation with actor rollout, we instead pass the full manager so that the
        # teacher loop workers can sleep/wake together with rollout workers
        reward_loop_worker_handles = self._select_reward_loop_worker_handles(enable_agent_reward_loop)
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            teacher_client=self.teacher_model_manager.get_client() if self.use_teacher_policy else None,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        # Support custom CheckpointEngineManager via config
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _persist_matched_boundary_metrics(self, checkpoint_folder: str) -> None:
        contract = self.config.trainer.get("matched_contract")
        if contract is None:
            return
        boundary_metrics = getattr(self, "_checkpoint_boundary_metrics", None)
        if boundary_metrics is None:
            raise ValueError("matched checkpoints require durable boundary metrics before publication")
        retained_prefixes = (
            "training/",
            "actor/",
            "critic/",
            "ggpo/",
            "sampling/",
            "rollout_training_mismatch/",
            "val-core/",
        )
        persisted_metrics = {}
        for key, value in boundary_metrics.items():
            if not key.startswith(retained_prefixes):
                continue
            if isinstance(value, (torch.Tensor, np.generic)):
                if getattr(value, "numel", lambda: 1)() != 1:
                    continue
                value = value.item()
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not np.isfinite(value):
                raise ValueError(f"matched checkpoint boundary metric is not finite: {key}")
            persisted_metrics[key] = value
        required = {
            "training/global_step",
            "critic/score/mean",
            "actor/pg_loss",
            "actor/grad_norm",
            "actor/entropy",
        }
        missing = required.difference(persisted_metrics)
        if missing or persisted_metrics.get("training/global_step") != self.global_steps:
            raise ValueError(f"matched checkpoint boundary metrics are incomplete: {sorted(missing)}")
        validation_keys = {
            "val-core/alp_math500/acc/mean@32",
            "val-core/alp_math500/acc/best@32/mean",
            "val-core/alp_math500/acc/best@32/std",
        }
        requires_validation = self.global_steps >= self.total_training_steps or (
            self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0
        )
        if requires_validation and not validation_keys.issubset(persisted_metrics):
            raise ValueError("matched checkpoint boundary is missing canonical Math500 validation metrics")
        contract = OmegaConf.to_container(contract, resolve=True) if OmegaConf.is_config(contract) else dict(contract)
        payload = {
            "schema_version": 1,
            "step": self.global_steps,
            "run_id": os.environ.get("WANDB_RUN_ID"),
            "name": str(self.config.trainer.experiment_name),
            "contract": contract,
            "metrics": persisted_metrics,
        }
        if not payload["run_id"] or payload["run_id"] != contract.get("run_id"):
            raise ValueError("matched checkpoint W&B run identity drifted")
        path = os.path.join(checkpoint_folder, "matched_boundary_metrics.json")
        descriptor, temporary = tempfile.mkstemp(prefix=".matched_boundary_metrics.", dir=checkpoint_folder)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(checkpoint_folder, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _ggpo_reference_version_after_step(self) -> Optional[int]:
        if not getattr(self, "_ggpo_moving_reference_enabled", False):
            return None
        if OmegaConf.select(
            self.config, "trainer.ggpo_moving_reference.refresh_every_updates", default=None
        ) is not None:
            return int(self._ggpo_reference_version)
        update_freq = int(self.config.trainer.ggpo_moving_reference.update_freq)
        if self.global_steps % update_freq == 0:
            return int(self.global_steps)
        return int(self._ggpo_reference_version)

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        fixed_policy_lag = getattr(self, "_fixed_policy_lag", None)
        if fixed_policy_lag is not None and isinstance(max_actor_ckpt_to_keep, int) and max_actor_ckpt_to_keep > 0:
            # Keep the previously published actor until the new root marker is
            # durable. Worker-level retention runs before trainer publication.
            max_actor_ckpt_to_keep += 1
        if (
            getattr(self, "_ggpo_moving_reference_enabled", False)
            and OmegaConf.select(
                self.config, "trainer.ggpo_moving_reference.refresh_every_updates", default=None
            )
            is None
            and self._ggpo_reference_version > 0
            and self._ggpo_reference_version_after_step() == self._ggpo_reference_version
        ):
            # An arbitrary segment checkpoint must not evict the actor checkpoint
            # that still backs the active moving reference.
            max_actor_ckpt_to_keep = None
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        if fixed_policy_lag is not None and isinstance(max_critic_ckpt_to_keep, int) and max_critic_ckpt_to_keep > 0:
            max_critic_ckpt_to_keep += 1

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        with open(dataloader_local_path, "rb") as stream:
            os.fsync(stream.fileno())

        filter_config = self.config.algorithm.get("filter_groups", None)
        mixed_filter_enabled = bool(filter_config and filter_config.get("enable", False))
        moving_reference_enabled = getattr(self, "_ggpo_moving_reference_enabled", False)
        fixed_lag_checkpoint_event_seq = None
        if fixed_policy_lag is not None:
            fixed_lag_checkpoint_event_seq = self._checkpoint_fixed_lag_state()

        if mixed_filter_enabled or fixed_policy_lag is not None or moving_reference_enabled:
            trainer_state = {
                "schema_version": 2,
                "global_step": int(self.global_steps),
            }
            if mixed_filter_enabled:
                trainer_state["dynamic_filter_epoch"] = self._current_dynamic_filter_epoch
            if fixed_policy_lag is not None:
                trainer_state["fixed_lag"] = {
                    "lag": fixed_policy_lag,
                    "ledger_path": self._fixed_lag_ledger_path,
                    "checkpoint_event_seq": fixed_lag_checkpoint_event_seq,
                }
            reference_version_after_step = self._ggpo_reference_version_after_step()
            if reference_version_after_step is not None:
                trainer_state["ggpo_reference_version_after_step"] = reference_version_after_step
            if OmegaConf.select(
                self.config, "trainer.ggpo_moving_reference.refresh_every_updates", default=None
            ) is not None:
                trainer_state["ggpo_optimizer_updates"] = int(self._ggpo_optimizer_updates)
            trainer_state_path = os.path.join(local_global_step_folder, "trainer_state.json")
            self._atomic_write_json(trainer_state_path, trainer_state)

        if self.config.trainer.get("consume_collection_minibatches", False):
            self._save_collection_state(local_global_step_folder)

        # Publish the matched replay payload before the root marker. A
        # successor can then repair a preemption between checkpoint save and
        # the W&B log without retraining or losing the scientific boundary.
        self._persist_matched_boundary_metrics(local_global_step_folder)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        descriptor, temporary = tempfile.mkstemp(
            prefix=".latest_checkpointed_iteration.", dir=self.config.trainer.default_local_dir
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(str(self.global_steps))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, local_latest_checkpointed_iteration)
            directory = os.open(
                self.config.trainer.default_local_dir,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

        reference_version_after_step = self._ggpo_reference_version_after_step()
        if reference_version_after_step is not None and reference_version_after_step != self._ggpo_reference_version:
            self._update_ggpo_reference(local_global_step_folder, reference_version_after_step)
        self._prune_fixed_lag_batches()

    def _update_ggpo_reference(self, global_step_folder, reference_version: int):
        actor_path = os.path.join(global_step_folder, "actor")
        if not os.path.isdir(actor_path):
            raise RuntimeError(f"GGPO moving-reference actor checkpoint is missing: {actor_path}")
        self.ref_policy_wg.load_reference_checkpoint(actor_path)
        self._ggpo_reference_version = int(reference_version)

    def _refresh_ggpo_reference_at_update(self, optimizer_update: int) -> None:
        snapshot_folder = os.path.join(
            self.config.trainer.default_local_dir, f"ggpo_reference_update_{optimizer_update}"
        )
        actor_path = os.path.join(snapshot_folder, "actor")
        self.actor_rollout_wg.save_reference_checkpoint(actor_path)
        self._update_ggpo_reference(snapshot_folder, optimizer_update)

    def _restore_ggpo_reference(self, loaded_checkpoint: str) -> None:
        reference_version = getattr(self, "_restored_ggpo_reference_version", None)
        if reference_version is None:
            raise RuntimeError("GGPO moving-reference checkpoint did not restore an active reference version")
        if reference_version == 0:
            self._ggpo_reference_version = 0
            return
        refresh_every_updates = OmegaConf.select(
            self.config, "trainer.ggpo_moving_reference.refresh_every_updates", default=None
        )
        if refresh_every_updates is None:
            reference_folder = os.path.join(os.path.dirname(loaded_checkpoint), f"global_step_{reference_version}")
        else:
            reference_folder = os.path.join(
                os.path.dirname(loaded_checkpoint), f"ggpo_reference_update_{reference_version}"
            )
        actor_path = os.path.join(reference_folder, "actor")
        if not os.path.isdir(actor_path):
            raise RuntimeError(
                "GGPO moving-reference resume requires the persisted backing actor checkpoint: "
                f"{actor_path}"
            )
        self._update_ggpo_reference(reference_folder, reference_version)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return None

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return None
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        filter_config = self.config.algorithm.get("filter_groups", None)
        mixed_filter_enabled = bool(filter_config and filter_config.get("enable", False))
        fixed_lag_enabled = getattr(self, "_fixed_policy_lag", None) is not None
        moving_reference_enabled = getattr(self, "_ggpo_moving_reference_enabled", False)
        trainer_state_path = os.path.join(global_step_folder, "trainer_state.json")
        trainer_state = None
        fixed_lag_state = None
        if os.path.isfile(trainer_state_path):
            try:
                with open(trainer_state_path, encoding="utf-8") as stream:
                    trainer_state = json.load(stream)
            except Exception as exc:
                raise RuntimeError("trainer_state.json is corrupt or unreadable") from exc
        elif mixed_filter_enabled or fixed_lag_enabled or moving_reference_enabled:
            raise RuntimeError("checkpoint is missing trainer_state.json required by the active trainer features")

        if trainer_state is not None:
            schema_version = trainer_state.get("schema_version")
            if trainer_state.get("global_step") != self.global_steps:
                raise RuntimeError("trainer_state.json global_step does not match its checkpoint directory")
            if schema_version == 1:
                if fixed_lag_enabled or moving_reference_enabled or not mixed_filter_enabled:
                    raise RuntimeError(
                        "trainer_state.json schema version 1 is accepted only for old dynamic-filter checkpoints "
                        "when fixed lag and moving reference are disabled"
                    )
            elif schema_version != 2:
                raise RuntimeError(f"unsupported trainer_state.json schema version: {schema_version!r}")

            if mixed_filter_enabled:
                dynamic_filter_epoch = trainer_state.get("dynamic_filter_epoch")
                if (
                    not isinstance(dynamic_filter_epoch, int)
                    or isinstance(dynamic_filter_epoch, bool)
                    or dynamic_filter_epoch < 0
                ):
                    raise RuntimeError("dynamic-filter trainer_state.json is invalid")
                self._restored_dynamic_filter_epoch = dynamic_filter_epoch

            checkpoint_fixed_lag_state = trainer_state.get("fixed_lag")
            if fixed_lag_enabled:
                if schema_version != 2 or not isinstance(checkpoint_fixed_lag_state, dict):
                    raise RuntimeError("fixed-lag checkpoint is missing its schema-v2 trainer state")
                fixed_lag_state = checkpoint_fixed_lag_state
            elif checkpoint_fixed_lag_state is not None:
                raise RuntimeError("checkpoint contains fixed-lag state but trainer.fixed_policy_lag is disabled")

            if moving_reference_enabled:
                reference_version = trainer_state.get("ggpo_reference_version_after_step")
                refresh_every_updates = OmegaConf.select(
                    self.config, "trainer.ggpo_moving_reference.refresh_every_updates", default=None
                )
                if (
                    schema_version != 2
                    or not isinstance(reference_version, int)
                    or isinstance(reference_version, bool)
                    or reference_version < 0
                    or (refresh_every_updates is None and reference_version > self.global_steps)
                ):
                    raise RuntimeError("GGPO moving-reference trainer_state.json is invalid")
                self._restored_ggpo_reference_version = reference_version
                if refresh_every_updates is not None:
                    self._ggpo_optimizer_updates = trainer_state["ggpo_optimizer_updates"]

        # load dataloader; fixed lag and dynamic filtering must restore their
        # advanced candidate cursor exactly, including at epoch boundaries.
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            steps_per_epoch = len(self.train_dataloader)
            at_epoch_boundary = steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0
            if mixed_filter_enabled or fixed_lag_enabled or self.config.trainer.get("consume_collection_minibatches", False):
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
            elif at_epoch_boundary:
                print(
                    f"Skipping dataloader state restore: global_steps={self.global_steps} "
                    f"is at an epoch boundary (steps_per_epoch={steps_per_epoch}). "
                    f"The saved state marks the dataloader as exhausted. "
                    f"Next epoch will iterate from scratch."
                )
            else:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            if mixed_filter_enabled or fixed_lag_enabled:
                raise RuntimeError(
                    f"checkpoint is missing required dataloader state: {dataloader_local_path}"
                )
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
        if fixed_lag_enabled:
            self._restore_fixed_lag_checkpoint(fixed_lag_state, self.global_steps)
        if self.config.trainer.get("consume_collection_minibatches", False):
            self._restore_collection_state(global_step_folder)
        return global_step_folder

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td)
        output = output.get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        values = DataProto.from_tensordict(values)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        metadata = {"calculate_entropy": False, "compute_loss": False}
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        # gather output
        log_probs = tu.get(output, "log_probs")
        # step 4. No padding to padding
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        ref_log_prob = DataProto.from_tensordict(ref_log_prob)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto) -> DataProto:
        """Score the batch under the actor before its first learner update."""
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        log_probs = no_padding_2_padding(tu.get(output, "log_probs"), batch_td)
        return DataProto.from_tensordict(tu.get_tensordict({"old_log_probs": log_probs.float()}))

    def _prepare_behavior_log_probs(
        self,
        batch: DataProto,
        timing_raw: dict[str, Any],
    ) -> tuple[DataProto, dict[str, Any]]:
        """Bind PPO and optional Token-TIS probabilities without relabeling behavior."""
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if loss_mode == "ggpo":
            return batch, {}
        rollout_correction = self.config.algorithm.get("rollout_correction", None)
        decoupled_rollout_is = bool(
            rollout_correction
            and not rollout_correction.get("bypass_mode", False)
            and rollout_correction.get("rollout_is", None) in {"token", "sequence"}
        )
        if not decoupled_rollout_is:
            batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
            return batch, {}

        with marked_timer("old_log_prob", timing_raw, color="blue"):
            old_log_prob = self._compute_old_log_prob(batch)
            batch = batch.union(old_log_prob)
        return compute_rollout_correction_and_add_to_batch(batch, rollout_correction)

    def _actor_update_calculate_entropy(self) -> bool:
        """Return whether the shared actor dispatcher requests entropy."""
        return bool(self.config.actor_rollout_ref.actor.calculate_entropy)

    def _update_actor(self, batch: DataProto, update_lr_scheduler: bool = True) -> DataProto:
        from verl.experimental.fully_async_policy.actor_update_padding import (
            LOGICAL_MINI_BATCH_SIZE_META,
            PHYSICAL_MINI_BATCH_SIZE_META,
        )

        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        actor_update_metadata = prepare_group_reference_actor_update_batch(batch, self.config)
        surrogate_metrics = batch.meta_info.pop(GGPO_SURROGATE_METRICS_META, None)
        # update actor
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        # Synchronous GGPO/GRPO obtains entropy and mismatch from this same loss
        # forward; the removed trainer-side old-log-prob forward must not be
        # reintroduced for observability.
        # Synchronous trainers retain entropy observability by default.  A
        # caller may explicitly skip the full-vocabulary diagnostic when the
        # configured entropy coefficient is zero and entropy is not part of
        # its acceptance contract (for example, bounded offline segments).
        calculate_entropy = self._actor_update_calculate_entropy()
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        distillation_only = False  # distillation_only flag means we can skip policy loss and reduce mem footprint
        if is_distillation_enabled(self.config.get("distillation")):
            distillation_loss_cfg = self.distillation_config.distillation_loss
            distillation_only = (
                distillation_use_topk
                and not distillation_loss_cfg.use_task_rewards
                and not distillation_loss_cfg.use_policy_gradient
            )
        logical_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        logical_mini_batch_size = logical_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        declared_logical_mini_batch_size = batch.meta_info.get(
            LOGICAL_MINI_BATCH_SIZE_META,
            logical_mini_batch_size,
        )
        if declared_logical_mini_batch_size != logical_mini_batch_size:
            raise ValueError(
                "Actor-update padding changed the scientific mini-batch denominator: "
                f"configured={logical_mini_batch_size}, declared={declared_logical_mini_batch_size}."
            )
        physical_mini_batch_size = batch.meta_info.get(
            PHYSICAL_MINI_BATCH_SIZE_META,
            logical_mini_batch_size,
        )
        if (
            PHYSICAL_MINI_BATCH_SIZE_META in batch.meta_info
            and LOGICAL_MINI_BATCH_SIZE_META not in batch.meta_info
        ):
            raise ValueError("Actor-update padding omitted its logical mini-batch metadata.")
        if PHYSICAL_MINI_BATCH_SIZE_META in batch.meta_info and (
            not isinstance(physical_mini_batch_size, int)
            or physical_mini_batch_size <= logical_mini_batch_size
            or len(batch) % physical_mini_batch_size != 0
        ):
            raise ValueError(
                "Actor-update padding declared an invalid physical mini-batch: "
                f"logical={logical_mini_batch_size}, physical={physical_mini_batch_size}, "
                f"update_responses={len(batch)}."
            )
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        collect_ggpo_theory_log_probs = self._ggpo_theory_run_receipt is not None
        collect_rollout_training_mismatch = "rollout_log_probs" in batch.batch
        if collect_ggpo_theory_log_probs and (int(ppo_epochs) != 1 or bool(shuffle)):
            raise ValueError(
                "Synchronous GGPO Set-A loss-forward reuse requires actor ppo_epochs=1 and shuffle=false "
                "so the collected current log-probabilities preserve logical-batch order."
            )
        actor_update_metadata = {
            "calculate_entropy": calculate_entropy,
            "distillation_use_topk": distillation_use_topk,
            "distillation_only": distillation_only,
            # Losses normalize by the unpadded scientific geometry. The
            # iterator alone uses the physical, DP-divisible mini-batch.
            "global_batch_size": logical_mini_batch_size,
            "mini_batch_size": physical_mini_batch_size,
            "epochs": ppo_epochs,
            "seed": seed,
            "dataloader_kwargs": {"shuffle": shuffle},
            "compute_loss": True,
            "collect_rollout_training_mismatch": collect_rollout_training_mismatch,
            "collect_ggpo_theory_current_log_probs": collect_ggpo_theory_log_probs,
            **actor_update_metadata,
        }
        if not update_lr_scheduler:
            actor_update_metadata["update_lr_scheduler"] = False
        tu.assign_non_tensor(batch_td, **actor_update_metadata)
        actor_result = self.actor_rollout_wg.update_actor(batch_td)
        if collect_ggpo_theory_log_probs:
            if GGPO_THEORY_CURRENT_LOG_PROBS not in actor_result:
                raise RuntimeError("Synchronous GGPO actor update omitted Set-A current log-probabilities.")
            current_log_probs = tu.get(actor_result, GGPO_THEORY_CURRENT_LOG_PROBS)
            if current_log_probs.shape != batch.batch["response_mask"].shape:
                raise RuntimeError(
                    "Synchronous GGPO Set-A current log-probability shape does not match the logical batch: "
                    f"{tuple(current_log_probs.shape)} != {tuple(batch.batch['response_mask'].shape)}."
                )
            batch.batch[GGPO_THEORY_CURRENT_LOG_PROBS] = current_log_probs
        actor_output = tu.get(actor_result, "metrics")
        mismatch_metrics = {}
        mismatch_aggregation = {}
        if collect_rollout_training_mismatch:
            mismatch_stats = actor_output.pop(INTERNAL_MISMATCH_STATS_KEY)
            mismatch_metrics = finalize_mismatch_metrics(mismatch_stats)
            mismatch_aggregation = mismatch_metrics.aggregation
        actor_output = rename_dict(actor_output, "actor/")
        if surrogate_metrics is not None:
            actor_output.update({f"actor/{name}": value for name, value in surrogate_metrics.items()})
        # modify key name
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output.update(mismatch_metrics)
        self._record_ggpo_theory_metrics(batch, actor_output)
        actor_output = DataProto.from_single_dict(
            data={},
            meta_info={
                "metrics": actor_output,
                MISMATCH_AGGREGATION_META: mismatch_aggregation,
            },
        )

        return actor_output

    def _update_actor_with_ggpo_refresh(self, batch: DataProto) -> DataProto:
        refresh_every_updates = self.config.trainer.ggpo_moving_reference.refresh_every_updates
        mini_batch_size = (
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        )
        total_updates = len(batch) // mini_batch_size
        completed = 0
        outputs = []

        while completed < total_updates:
            next_refresh = (self._ggpo_optimizer_updates // refresh_every_updates + 1) * refresh_every_updates
            segment_updates = min(total_updates - completed, next_refresh - self._ggpo_optimizer_updates)
            segment_end = completed + segment_updates
            part = batch[completed * mini_batch_size : segment_end * mini_batch_size]
            outputs.append(self._update_actor(part, update_lr_scheduler=segment_end == total_updates))
            self._ggpo_optimizer_updates += segment_updates
            completed = segment_end

            if self._ggpo_optimizer_updates == next_refresh:
                self._refresh_ggpo_reference_at_update(self._ggpo_optimizer_updates)
                if completed < total_updates:
                    remainder = batch[completed * mini_batch_size :]
                    refreshed = self._compute_ref_log_prob(remainder)
                    batch.batch["ref_log_prob"][completed * mini_batch_size :] = refreshed.batch["ref_log_prob"]

        merged_metrics = {}
        for output in outputs:
            for name, values in output.meta_info["metrics"].items():
                merged_metrics.setdefault(name, []).extend(values if isinstance(values, list) else [values])
        outputs[0].meta_info["metrics"] = merged_metrics
        return outputs[0]

    def _update_critic(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        # modify key name
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        return critic_output

    def _ggpo_theory_policy_identity(self, batch: DataProto):
        training_version = max(int(self.global_steps) - 1, 0)
        behavior_version = batch.meta_info.get("fixed_lag_behavior_policy_version", training_version)
        if isinstance(behavior_version, bool) or not isinstance(behavior_version, int):
            raise ValueError(f"GGPO batch has an invalid behavior-policy version: {behavior_version!r}")
        return {
            "actor_update_step": int(self.global_steps),
            "training_policy_version": training_version,
            "behavior_policy_versions": torch.full((len(batch),), behavior_version, dtype=torch.long),
            "behavior_version_stratification": False,
        }

    def _record_ggpo_theory_metrics(self, batch: DataProto, metrics: dict) -> None:
        if self._ggpo_theory_run_receipt is None:
            return
        if GGPO_THEORY_METRICS_SENTINEL in batch.batch:
            raise ValueError("GGPO Set-A metrics were already recorded for this batch")
        required = {GGPO_THEORY_CURRENT_LOG_PROBS, "rollout_log_probs", "ref_log_prob", "token_level_scores"}
        missing = required.difference(batch.batch.keys())
        if missing:
            raise ValueError(f"GGPO Set-A input is incomplete: {sorted(missing)}")
        if "rollout_is_weights" in batch.batch:
            raise ValueError("GGPO forbids rollout importance-sampling weights")
        receipt = self._ggpo_theory_run_receipt
        group_size = int(self.config.actor_rollout_ref.rollout.n)
        group_ids = build_policy_group_ids(batch.non_tensor_batch["uid"], group_size)
        theory = compute_ggpo_theory_metrics(
            training_log_prob=batch.batch.pop(GGPO_THEORY_CURRENT_LOG_PROBS),
            behavior_log_prob=batch.batch["rollout_log_probs"],
            ref_log_prob=batch.batch["ref_log_prob"],
            rewards=batch.batch["token_level_scores"],
            response_mask=batch.batch["response_mask"],
            group_ids=group_ids,
            beta=float(self.config.actor_rollout_ref.actor.policy_loss.beta),
            reward_kl=self.config.actor_rollout_ref.actor.policy_loss.get("reward_kl", True),
            group_size=group_size,
            run_receipt_schema_version=receipt["schema_version"],
            run_receipt_digest_u52=receipt["digest_u52"],
            real_sample_mask=batch.batch.get("_fully_async_actor_update_real_mask"),
            **self._ggpo_theory_policy_identity(batch),
        )
        metrics.update(theory)
        runtime_root = OmegaConf.select(self.config, "async_training.runtime_root", default=None)
        if runtime_root:
            set_a_path = os.path.join(os.path.abspath(runtime_root), "ggpo_set_a.jsonl")
            os.makedirs(os.path.dirname(set_a_path), exist_ok=True)
            with open(set_a_path, "a", encoding="utf-8") as stream:
                json.dump(theory, stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        batch.batch[GGPO_THEORY_METRICS_SENTINEL] = torch.full(
            (len(batch),), GGPO_THEORY_METRICS_SCHEMA_VERSION, dtype=torch.long
        )

    def _generate_standard_training_batch(self, batch_dict, timing_raw, curr_step_profile):
        """Generate the ordinary single-candidate training batch."""
        batch = DataProto.from_single_dict(batch_dict)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
        )
        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        rollout_n = self.config.actor_rollout_ref.rollout.n
        gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

        adv_estimator = OmegaConf.select(self.config, "algorithm.adv_estimator", default=None)
        if adv_estimator == AdvantageEstimator.REMAX:
            gen_batch_output.non_tensor_batch["__do_sample__"] = np.ones(len(gen_batch_output), dtype=bool)
            gen_baseline_batch = gen_batch.slice(0, None)
            gen_baseline_batch.non_tensor_batch["__do_sample__"] = np.zeros(len(gen_baseline_batch), dtype=bool)
            combined_gen_batch = DataProto.concat([gen_batch_output, gen_baseline_batch])
            num_sampled_prompts = len(gen_batch_output)
        else:
            combined_gen_batch = gen_batch_output
            num_sampled_prompts = len(gen_batch_output)

        with marked_timer("gen", timing_raw, color="red"):
            if curr_step_profile:
                self.llm_server_manager.start_profile()
            combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
            self.checkpoint_manager.sleep_replicas()
            if curr_step_profile:
                self.llm_server_manager.stop_profile()
            timing_raw.update(combined_gen_output.meta_info["timing"])
            combined_gen_output.meta_info.pop("timing", None)

        gen_batch_output = combined_gen_output.slice(0, num_sampled_prompts)
        if "__do_sample__" in gen_batch_output.non_tensor_batch:
            gen_batch_output.pop(non_tensor_batch_keys=["__do_sample__"])

        if adv_estimator == AdvantageEstimator.REMAX:
            gen_baseline_output = combined_gen_output.slice(num_sampled_prompts, None)
            if "__do_sample__" in gen_baseline_output.non_tensor_batch:
                gen_baseline_output.pop(non_tensor_batch_keys=["__do_sample__"])
            if self._needs_colocated_reward(gen_baseline_output):
                gen_baseline_output = gen_baseline_output.union(self._compute_reward_colocate(gen_baseline_output))
            batch.batch["reward_baselines"] = gen_baseline_output.batch["rm_scores"].sum(dim=-1)
            del gen_baseline_output

        del combined_gen_batch, combined_gen_output
        batch = batch.repeat(repeat_times=rollout_n, interleave=True).union(gen_batch_output)
        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)
        return batch

    def _collect_mixed_reward_batch(
        self, first_batch_dict, train_iterator, epoch, metrics, timing_raw, curr_step_profile
    ):
        """Refill one learner batch from fresh prompts under one frozen pre-update policy."""
        filter_config = self.config.algorithm.get("filter_groups", None)
        if not filter_config or not filter_config.get("enable", False):
            raise ValueError("mixed-reward collection requires algorithm.filter_groups.enable=true")
        if filter_config.get("metric") != "raw_correctness":
            raise ValueError("mixed-reward filtering must use raw_correctness")
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if loss_mode == "vanilla" and self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise ValueError("mixed-reward refill requires GRPO advantages")

        # AGRO is a group-reference loss like GGPO and consumes exactly what this refill
        # produces (ref_log_prob, token_level_scores, POLICY_GROUP_IDS); only GGPO also needs
        # GGPO_RAW_CORRECTNESS, which is written under its own loss_mode check. The refill
        # itself is loss-agnostic, so keeping AGRO out of this set only meant every online
        # AGRO arm died on its first optimizer step.
        if loss_mode not in {"ggpo", "agro", "vanilla"}:
            raise ValueError("mixed-reward refill supports only GGPO, AGRO and matched DAPO policy losses")
        if loss_mode == "vanilla" and self.config.actor_rollout_ref.actor.loss_agg_mode != "token-mean":
            raise ValueError("matched DAPO requires actor.loss_agg_mode=token-mean")

        max_rounds = int(filter_config.get("max_num_gen_batches", 0))
        if max_rounds <= 0:
            raise ValueError("algorithm.filter_groups.max_num_gen_batches must be a finite positive integer")
        group_size = int(self.config.actor_rollout_ref.rollout.n)
        target_groups = int(self.config.data.train_batch_size)
        accepted_parts = []
        accepted_groups = 0
        aggregate = {
            key: 0
            for key in (
                "generated_groups",
                "mixed_groups",
                "all_zero_groups",
                "all_one_groups",
                "selected_groups",
                "discarded_mixed_groups",
            )
        }
        generation_rounds = 0
        pending_batch_dict = first_batch_dict
        reward_extra_keys = set()

        while accepted_groups < target_groups and generation_rounds < max_rounds:
            if pending_batch_dict is not None:
                batch_dict = pending_batch_dict
                pending_batch_dict = None
            else:
                try:
                    batch_dict = next(train_iterator)
                except StopIteration:
                    epoch += 1
                    if epoch >= self.config.trainer.total_epochs:
                        raise RuntimeError("dynamic sampling exhausted trainer.total_epochs")
                    train_iterator = iter(self.train_dataloader)
                    try:
                        batch_dict = next(train_iterator)
                    except StopIteration:
                        raise RuntimeError("train dataloader yielded no batches")

            prompt_batch = DataProto.from_single_dict(batch_dict)
            if len(prompt_batch) != target_groups:
                raise ValueError(
                    f"candidate prompt batch must contain exactly {target_groups} fresh groups, got {len(prompt_batch)}"
                )
            prompt_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            prompt_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(prompt_batch.batch))], dtype=object
            )
            gen_batch = self._get_gen_batch(prompt_batch)
            gen_batch.meta_info["global_steps"] = self.global_steps
            repeated_gen_batch = gen_batch.repeat(repeat_times=group_size, interleave=True)

            round_timing = {}
            with marked_timer("gen", round_timing, color="red"):
                if curr_step_profile:
                    self.llm_server_manager.start_profile()
                gen_output = self.async_rollout_manager.generate_sequences(repeated_gen_batch)
                if curr_step_profile:
                    self.llm_server_manager.stop_profile()
                for key, value in gen_output.meta_info.get("timing", {}).items():
                    timing_raw[key] = timing_raw.get(key, 0) + value
                gen_output.meta_info.pop("timing", None)
            timing_raw["gen"] = timing_raw.get("gen", 0) + round_timing["gen"]

            candidate = prompt_batch.repeat(repeat_times=group_size, interleave=True).union(gen_output)
            if "response_mask" not in candidate.batch:
                candidate.batch["response_mask"] = compute_response_mask(candidate)
            round_timing = {}
            with marked_timer("reward", round_timing, color="yellow"):
                if self._needs_colocated_reward(candidate):
                    candidate = candidate.union(self._compute_reward_colocate(candidate))
                shaped_reward, reward_extra = extract_reward(candidate)
            reward_extra_keys.update(reward_extra)
            timing_raw["reward"] = timing_raw.get("reward", 0) + round_timing["reward"]

            raw_correctness = raw_correctness_tensor(reward_extra, expected_responses=len(candidate))
            response_indices, round_stats = select_mixed_reward_group_indices(
                raw_correctness,
                group_size=group_size,
                max_groups=target_groups - accepted_groups,
            )
            candidate.batch["token_level_scores"] = shaped_reward
            candidate.non_tensor_batch.update({key: np.asarray(value) for key, value in reward_extra.items()})
            if response_indices.numel() > 0:
                accepted_parts.append(candidate.select_idxs(response_indices.cpu()))
                accepted_groups += round_stats["selected_groups"]
            for key in aggregate:
                aggregate[key] += round_stats[key]
            generation_rounds += 1

        require_complete_mixed_batch(
            accepted_groups=accepted_groups,
            target_groups=target_groups,
            generation_rounds=generation_rounds,
        )
        batch = DataProto.concat(accepted_parts)
        batch.meta_info["reward_extra_keys"] = sorted(reward_extra_keys)
        if len(batch) != target_groups * group_size:
            raise RuntimeError("dynamic sampling did not produce the exact learner response batch")
        accepted_raw = raw_correctness_tensor(batch.non_tensor_batch, expected_responses=len(batch))
        grouped_raw = accepted_raw.reshape(target_groups, group_size)
        if not ((grouped_raw.sum(dim=-1) > 0) & (grouped_raw.sum(dim=-1) < group_size)).all():
            raise RuntimeError("accepted learner batch contains an all-0 or all-1 raw-correctness group")

        generated_groups = aggregate["generated_groups"]
        response_lengths = batch.batch["response_mask"].sum(dim=-1).float()
        eos_token_id = self.tokenizer.eos_token_id
        eos = ((batch.batch["responses"] == eos_token_id) & batch.batch["response_mask"].bool()).any(dim=-1)
        max_response_length = int(self.config.data.max_response_length)
        truncation = (response_lengths >= max_response_length) & ~eos
        overlong_penalty = np.asarray(batch.non_tensor_batch.get("overlong_reward", np.zeros(len(batch))), dtype=float)
        metrics.update(
            {
                "sampling/generated_groups": generated_groups,
                "sampling/accepted_groups": target_groups,
                "sampling/rejected_groups": generated_groups - target_groups,
                "sampling/rejected_constant_groups": aggregate["all_zero_groups"] + aggregate["all_one_groups"],
                "sampling/discarded_mixed_groups": aggregate["discarded_mixed_groups"],
                "sampling/acceptance_rate": target_groups / generated_groups,
                "sampling/raw_all_zero_groups": aggregate["all_zero_groups"],
                "sampling/raw_all_one_groups": aggregate["all_one_groups"],
                "sampling/generation_rounds": generation_rounds,
                "sampling/response_length_mean": response_lengths.mean().item(),
                "sampling/response_length_max": response_lengths.max().item(),
                "sampling/overlong_penalty_mean": float(overlong_penalty.mean()),
                "sampling/overlong_penalty_min": float(overlong_penalty.min()),
                "sampling/eos_rate": eos.float().mean().item(),
                "sampling/truncation_rate": truncation.float().mean().item(),
            }
        )
        return batch, train_iterator, epoch

    def _next_training_candidate(self, train_iterator, epoch: int, require_batch: bool = True):
        try:
            return next(train_iterator), train_iterator, epoch
        except StopIteration:
            epoch += 1
            if epoch >= self.config.trainer.total_epochs:
                if require_batch:
                    raise RuntimeError("training exhausted trainer.total_epochs while producing a learner batch")
                return None, train_iterator, epoch
            train_iterator = iter(self.train_dataloader)
            try:
                return next(train_iterator), train_iterator, epoch
            except StopIteration as exc:
                raise RuntimeError("train dataloader yielded no batches") from exc

    def _produce_training_batch(
        self,
        batch_dict,
        train_iterator,
        epoch: int,
        mixed_filter_enabled: bool,
        generation_metrics: dict[str, Any],
        timing_raw: dict[str, Any],
        curr_step_profile: bool,
    ):
        """Produce one complete storable learner batch under the current actor."""
        if mixed_filter_enabled:
            batch, train_iterator, epoch = self._collect_mixed_reward_batch(
                batch_dict,
                train_iterator,
                epoch,
                generation_metrics,
                timing_raw,
                curr_step_profile,
            )
            self.checkpoint_manager.sleep_replicas()
        else:
            batch = self._generate_standard_training_batch(batch_dict, timing_raw, curr_step_profile)
        return batch, train_iterator, epoch

    def _take_collection_minibatch(self, collection: DataProto) -> DataProto:
        """Consume one complete prompt minibatch; preserve behavior probabilities."""
        size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        batch = collection[:size]
        batch.meta_info = dict(collection.meta_info)
        self._collection_remainder = collection[size:] if len(collection) > size else None
        if self._collection_remainder is not None:
            self._collection_remainder.meta_info = dict(collection.meta_info)
        return batch

    def _save_collection_state(self, checkpoint_folder: str) -> None:
        torch.save(
            {
                "remainder": self._collection_remainder,
                "epoch": self._current_dynamic_filter_epoch,
                "metrics": getattr(self, "_collection_metrics", None),
            },
            os.path.join(checkpoint_folder, "collection.pt"),
            pickle_protocol=4,
        )

    def _restore_collection_state(self, checkpoint_folder: str) -> None:
        state = torch.load(os.path.join(checkpoint_folder, "collection.pt"), weights_only=False)
        self._collection_remainder = state["remainder"]
        self._restored_dynamic_filter_epoch = state["epoch"]
        self._collection_metrics = state.get("metrics")

    def _warm_up_fixed_lag(self, train_iterator, epoch: int, mixed_filter_enabled: bool):
        if self._fixed_lag_queue:
            raise RuntimeError("fresh fixed-lag warmup requires an empty FIFO")
        for logical_generation_step in range(1 - self._fixed_policy_lag, 1):
            batch_dict, train_iterator, epoch = self._next_training_candidate(train_iterator, epoch)
            generation_metrics = {}
            batch, train_iterator, epoch = self._produce_training_batch(
                batch_dict,
                train_iterator,
                epoch,
                mixed_filter_enabled,
                generation_metrics,
                {},
                False,
            )
            batch = self._tag_fixed_lag_batch(
                batch,
                logical_generation_step=logical_generation_step,
                behavior_policy_version=0,
                kind="warmup",
                sampling_metrics=generation_metrics,
            )
            self._enqueue_fixed_lag_batch(batch)
            # Each production sleeps the rollout replicas (level-2 sleep frees
            # weights and KV cache) and, unlike the main loop, warmup performs
            # no learner update to wake them. Restore weights before the next
            # generation — including after the final warmup batch, because the
            # main loop produces its first steady batch before any update.
            self.checkpoint_manager.update_weights(self.global_steps)
        if len(self._fixed_lag_queue) != self._fixed_policy_lag:
            raise RuntimeError("fixed-lag warmup did not produce exactly k pending entries")
        return train_iterator, epoch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        if self._dump_executor._shutdown:
            self._init_dump_executor()

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        loaded_checkpoint = self._load_checkpoint()
        if getattr(self, "_ggpo_moving_reference_enabled", False) and loaded_checkpoint is not None:
            self._restore_ggpo_reference(loaded_checkpoint)
        self.checkpoint_manager.update_weights(self.global_steps)

        segment_end = resolve_segment_end(
            checkpoint_step=self.global_steps,
            total_training_steps=self.total_training_steps,
            start_raw=os.environ.get("INTERACTIVE_SEGMENT_START_STEP"),
            end_raw=os.environ.get("INTERACTIVE_SEGMENT_END_STEP"),
        )

        current_epoch = getattr(
            self,
            "_restored_dynamic_filter_epoch",
            self.global_steps // len(self.train_dataloader),
        )

        SkipManager.init(self.config)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        val_only = self.config.trainer.get("val_only", False)
        if val_only or (self.global_steps == 0 and self.config.trainer.get("val_before_train", True)):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            val_metrics["training/global_step"] = self.global_steps
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if val_only:
                self._shutdown_dump_executor()
                return

        filter_config = self.config.algorithm.get("filter_groups", None)
        mixed_filter_enabled = bool(filter_config and filter_config.get("enable", False))
        epoch = current_epoch
        consume_minibatches = self.config.trainer.get("consume_collection_minibatches", False)
        rollout_awake = True
        train_iterator = iter(self.train_dataloader)
        if self._fixed_policy_lag is not None:
            if loaded_checkpoint is None:
                self._initialize_fixed_lag_ledger()
                train_iterator, epoch = self._warm_up_fixed_lag(
                    train_iterator,
                    epoch,
                    mixed_filter_enabled,
                )
            elif len(self._fixed_lag_queue) != self._fixed_policy_lag:
                raise RuntimeError("fixed-lag resume did not restore exactly k pending entries")

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        SkipManager.set_step(self.global_steps)

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        while self.global_steps <= segment_end and epoch < self.config.trainer.total_epochs:
            if not consume_minibatches or self._collection_remainder is None:
                batch_dict, train_iterator, epoch = self._next_training_candidate(
                    train_iterator,
                    epoch,
                    require_batch=self._fixed_policy_lag is not None,
                )
                if batch_dict is None:
                    break
            if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
            metrics = {}
            timing_raw = {}
            ggpo_reference_version_used = None

            with marked_timer("start_profile", timing_raw):
                self._start_profiling(
                    not prev_step_profile and curr_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
            is_last_step = self.global_steps >= segment_end
            is_final_training_step = self.global_steps >= self.total_training_steps
            should_save_checkpoint = False
            with marked_timer("step", timing_raw):
                generation_metrics = {}
                if consume_minibatches and self._collection_remainder is not None:
                    batch = self._collection_remainder
                    # Resume or validation can leave rollout replicas awake.
                    if rollout_awake:
                        self.checkpoint_manager.sleep_replicas()
                        rollout_awake = False
                else:
                    batch, train_iterator, epoch = self._produce_training_batch(
                        batch_dict,
                        train_iterator,
                        epoch,
                        mixed_filter_enabled,
                        generation_metrics,
                        timing_raw,
                        curr_step_profile,
                    )
                    rollout_awake = False
                    if consume_minibatches and self._fixed_policy_lag is None:
                        # Freeze old probabilities and TIS weights before splitting the collection.
                        # The checkpointed remainder carries these same tensors across updates.
                        batch, rollout_correction_metrics = self._prepare_behavior_log_probs(batch, timing_raw)
                        metrics.update(rollout_correction_metrics)
                if consume_minibatches:
                    batch = self._take_collection_minibatch(batch)
                    collection_updates = self.config.data.train_batch_size // self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                    metrics["training/optimizer_updates"] = self.global_steps
                    metrics["training/sample_age_updates"] = (self.global_steps - 1) % collection_updates
                if self._fixed_policy_lag is not None:
                    batch = self._tag_fixed_lag_batch(
                        batch,
                        logical_generation_step=self.global_steps,
                        behavior_policy_version=max(self.global_steps - 1, 0),
                        kind="steady",
                        sampling_metrics=generation_metrics,
                    )
                    self._enqueue_fixed_lag_batch(batch)
                    batch = self._consume_fixed_lag_batch(self.global_steps)
                    if len(self._fixed_lag_queue) != self._fixed_policy_lag:
                        raise RuntimeError("fixed-lag learner iteration did not retain exactly k pending entries")
                    metrics.update(self._fixed_lag_consumption_metrics(batch))
                else:
                    metrics.update(generation_metrics)

                if mixed_filter_enabled:
                    reward_tensor = batch.batch["token_level_scores"]
                    reward_extra_infos_dict = {
                        key: batch.non_tensor_batch[key]
                        for key in batch.meta_info.get("reward_extra_keys", [])
                        if key in batch.non_tensor_batch
                    }
                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                # get images_seqlens
                images_seqlens_all = []
                for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                    if "image_grid_thw" not in multi_modal_input.keys():
                        continue
                    images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                batch.meta_info["images_seqlens"] = images_seqlens_all
                if not mixed_filter_enabled:
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self._needs_colocated_reward(batch):
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                if "rollout_log_probs" not in batch.batch:
                    raise ValueError("Synchronous GGPO/GRPO requires rollout_log_probs from sampled responses.")
                if not consume_minibatches or self._fixed_policy_lag is not None:
                    batch, rollout_correction_metrics = self._prepare_behavior_log_probs(batch, timing_raw)
                    metrics.update(rollout_correction_metrics)
                if self.use_reference_policy:
                    if getattr(self, "_ggpo_moving_reference_enabled", False):
                        ggpo_reference_version_used = int(self._ggpo_reference_version)
                    # compute reference log_prob
                    with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                        ref_log_prob = self._compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        values = self._compute_values(batch)
                        batch = batch.union(values)

                with marked_timer("adv", timing_raw, color="brown"):
                    # we combine with rule-based rm
                    reward_extra_infos_dict: dict[str, list]
                    batch.batch["token_level_scores"] = reward_tensor

                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # compute rewards. apply_kl_penalty if available
                    if OmegaConf.select(self.config, "algorithm.use_kl_in_reward", default=False):
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    if not maybe_apply_group_reference_advantage_bypass(batch, self.config):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                # update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self._update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup > self.global_steps:
                    # Still in critic warmup, only update weights to wake up rollout replicas.
                    self.checkpoint_manager.update_weights(self.global_steps)
                    rollout_awake = True
                else:
                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        if (
                            getattr(self, "_ggpo_moving_reference_enabled", False)
                            and OmegaConf.select(
                                self.config,
                                "trainer.ggpo_moving_reference.refresh_every_updates",
                                default=None,
                            )
                            is not None
                        ):
                            actor_output = self._update_actor_with_ggpo_refresh(batch)
                        else:
                            actor_output = self._update_actor(batch)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        should_save_checkpoint = True

                    # update weights from trainer to rollout
                    needs_validation = self.config.trainer.test_freq > 0 and (
                        is_final_training_step
                        or self.global_steps % self.config.trainer.test_freq == 0
                        or (should_save_checkpoint and getattr(self, "_checkpoint_requires_validation_metrics", False))
                    )
                    if not consume_minibatches or self._collection_remainder is None or needs_validation:
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)
                            rollout_awake = True

                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

            # validate
            requires_checkpoint_validation = should_save_checkpoint and getattr(
                self, "_checkpoint_requires_validation_metrics", False
            )
            if self.config.trainer.test_freq > 0 and (
                is_final_training_step
                or self.global_steps % self.config.trainer.test_freq == 0
                or requires_checkpoint_validation
            ):
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            with marked_timer("stop_profile", timing_raw):
                next_step_profile = (
                    self.global_steps + 1 in self.config.global_profiler.steps
                    if self.config.global_profiler.steps is not None
                    else False
                )
                self._stop_profiling(
                    curr_step_profile and not next_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
                prev_step_profile = curr_step_profile
                curr_step_profile = next_step_profile

            steps_duration = timing_raw["step"]
            self.max_steps_duration = max(self.max_steps_duration, steps_duration)

            # training metrics
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                }
            )
            # collect metrics
            metrics.update(
                prune_sync_alp_math_training_metrics(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
            )
            # GDPO per-component reward metrics
            gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
            if gdpo_reward_keys and OmegaConf.select(
                self.config, "algorithm.adv_estimator", default=None
            ) in ("gdpo", AdvantageEstimator.GDPO):
                for key in gdpo_reward_keys:
                    if key in batch.non_tensor_batch:
                        vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                        metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                        metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                        metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                        metrics[f"gdpo/{key}/min"] = float(np.min(vals))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            # compute variance proxy metrics
            gradient_norm = metrics.get("actor/grad_norm", None)
            metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
            # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

            # Per-request spec decode metrics.
            metrics.update(
                compute_spec_decode_metrics(
                    batch.non_tensor_batch.get("spec_num_draft_tokens", None),
                    batch.non_tensor_batch.get("spec_num_accepted_tokens", None),
                    batch.non_tensor_batch.get("spec_num_verify_steps", None),
                )
            )

            # Timing metrics are added after compute_data_metrics(), and the
            # agent-loop timing schema can itself contain single-turn-only
            # num_turns/tool_calls dimensions.  Prune the complete payload
            # at the synchronous logger boundary so later metric producers
            # cannot reintroduce them.
            metrics = prune_sync_alp_math_training_metrics(metrics)

            if getattr(self, "_ggpo_moving_reference_enabled", False):
                if ggpo_reference_version_used is None:
                    raise RuntimeError("GGPO moving reference did not record the version used by the learner loss")
                metrics["actor/ggpo_reference_version"] = ggpo_reference_version_used

            if should_save_checkpoint:
                # Hybrid checkpoints persist this exact scientific boundary so
                # a marker published just before preemption can be reconciled
                # without retraining or inventing metrics.
                self._checkpoint_boundary_metrics = dict(metrics)
                self._current_dynamic_filter_epoch = epoch
                try:
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()
                finally:
                    del self._checkpoint_boundary_metrics
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1
            SkipManager.set_step(self.global_steps)

            if is_last_step:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                self._shutdown_dump_executor()
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            # this is experimental and may be changed/removed in the future
            # in favor of a general-purpose data buffer pool
            if hasattr(self.train_dataset, "on_batch_end") and (
                not consume_minibatches or self._collection_remainder is None
            ):
                # The dataset may be changed after each training batch
                self.train_dataset.on_batch_end(batch=batch)

        # Ensure dump executor is shut down when training loop ends without reaching is_last_step
        self._shutdown_dump_executor()
