# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
from tensordict import TensorDict

from verl.experimental.fully_async_policy.actor_update_padding import (
    DUMMY_SAFE_RESPONSE_MASK,
    REAL_SAMPLE_MASK,
)
from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.agro import compute_agro_loss
from verl.trainer.ppo.ggpo import (
    GGPO_METRIC_NAMES,
    GGPO_RAW_CORRECTNESS,
    GGPO_SURROGATE_WEIGHTS,
    POLICY_GROUP_IDS,
    compute_ggpo_loss,
    ggpo_conditional_metric_fraction,
)
from verl.trainer.ppo.rollout_training_mismatch import (
    INTERNAL_MISMATCH_STATS_KEY,
    compute_mismatch_stats,
    empty_mismatch_stats,
)
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def _zero_group_reference_metrics(log_prob: torch.Tensor, *, loss_mode: str):
    """Return a group-reference metric schema with zero scientific weight."""
    zero = log_prob.sum() * 0.0
    if loss_mode == "agro":
        return zero, {
            "agro_surrogate_loss": zero.detach(),
            "agro_sample_variance": zero.detach(),
            "agro_regularized_reward_mean": zero.detach(),
            "agro_regularized_reward_min": zero.detach(),
            "agro_regularized_reward_max": zero.detach(),
            "agro_loo_delta_mean": zero.detach(),
            "agro_loo_delta_abs_mean": zero.detach(),
            "agro_loo_delta_abs_max": zero.detach(),
            "agro_sequence_log_ratio_mean": zero.detach(),
            "agro_sequence_log_ratio_abs_max": zero.detach(),
            "agro_reference_sequence_log_prob_mean": zero.detach(),
        }
    return zero, {name: zero.detach() for name in GGPO_METRIC_NAMES}


def _zero_asympo_metrics(log_prob: torch.Tensor):
    zero = log_prob.sum() * 0.0
    names = (
        "asympo_loss",
        "asympo_response_nll_mean",
        "asympo_response_nll_min",
        "asympo_response_nll_max",
        "asympo_inverse_scale_mean",
        "asympo_inverse_scale_max",
        "asympo_positive_contribution",
        "asympo_negative_contribution",
        "asympo_gradient_coefficient_abs_mean",
    )
    return zero, {name: zero.detach() for name in names}


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute the configured policy loss from model output and its required batch fields."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    collect_rollout_training_mismatch = tu.get_non_tensor_data(
        data=data, key="collect_rollout_training_mismatch", default=False
    )
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    group_reference_loss = loss_mode in {"ggpo", "agro"}
    if group_reference_loss and config.use_kl_loss:
        raise ValueError(
            f"{loss_mode.upper()} does not support actor use_kl_loss because it embeds reference regularization."
        )

    # select fields and convert to padded tensor
    padding_mask_key = REAL_SAMPLE_MASK
    has_padding_mask = padding_mask_key in data
    memory_safe_ggpo_surrogate = loss_mode == "ggpo" and GGPO_SURROGATE_WEIGHTS in data
    if group_reference_loss:
        if memory_safe_ggpo_surrogate:
            fields = ["response_mask", GGPO_SURROGATE_WEIGHTS]
        else:
            fields = ["response_mask", "ref_log_prob", "token_level_scores", POLICY_GROUP_IDS]
            if loss_mode == "ggpo":
                fields.append(GGPO_RAW_CORRECTNESS)
    elif loss_mode == "asympo":
        fields = ["response_mask", "advantages"]
    else:
        fields = ["response_mask", "old_log_probs", "advantages"]
        if "rollout_is_weights" in data:
            fields.append("rollout_is_weights")
        if "ref_log_prob" in data:
            fields.append("ref_log_prob")
    if collect_rollout_training_mismatch:
        if "rollout_log_probs" not in data:
            raise ValueError("Synchronous mismatch measurement requires saved rollout_log_probs.")
        fields.append("rollout_log_probs")
    if has_padding_mask:
        fields.append(padding_mask_key)
        fields.append(DUMMY_SAFE_RESPONSE_MASK)
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    regularization_response_mask = response_mask
    if has_padding_mask:
        real_sample_mask = data.pop(padding_mask_key).to(bool)
        dummy_safe_response_mask = data.pop(DUMMY_SAFE_RESPONSE_MASK).to(bool)
        if real_sample_mask.ndim != 1:
            raise ValueError(f"{padding_mask_key} must contain one boolean value per response.")
        if dummy_safe_response_mask.shape != response_mask.shape:
            raise ValueError(
                f"{DUMMY_SAFE_RESPONSE_MASK} must match response_mask shape "
                f"{tuple(response_mask.shape)}, got {tuple(dummy_safe_response_mask.shape)}."
            )
    else:
        real_sample_mask = torch.ones(response_mask.shape[0], dtype=torch.bool, device=response_mask.device)
    loss_agg_mode = config.loss_agg_mode
    if group_reference_loss:
        group_size = config.rollout_n
        real_response_count = int(real_sample_mask.sum().item())
        if not memory_safe_ggpo_surrogate and real_response_count % group_size != 0:
            raise ValueError(
                f"Fully async actor-update padding split a {loss_mode.upper()} group: "
                f"{real_response_count} real responses is not divisible by group_size={group_size}."
            )
        scientific_metric_weight = (
            real_response_count if memory_safe_ggpo_surrogate else real_response_count // group_size
        )
        sequence_reduction = config.policy_loss.get("sequence_log_prob_reduction", "sum")
        log_prob = log_prob[real_sample_mask]
        rollout_log_prob = data["rollout_log_probs"][real_sample_mask] if collect_rollout_training_mismatch else None
        if entropy is not None:
            entropy = entropy[real_sample_mask]
        response_mask = response_mask[real_sample_mask]
        regularization_response_mask = response_mask
        if real_response_count:
            if memory_safe_ggpo_surrogate:
                surrogate_weights = data[GGPO_SURROGATE_WEIGHTS][real_sample_mask]
                if surrogate_weights.shape != (real_response_count,):
                    raise ValueError(
                        "GGPO memory-safe surrogate weights must contain one scalar per real response, "
                        f"got {tuple(surrogate_weights.shape)} for {real_response_count} responses."
                    )
                if not torch.isfinite(surrogate_weights).all():
                    raise FloatingPointError("GGPO memory-safe surrogate weights are not finite.")
                response_lengths = response_mask.sum(dim=-1).float()
                sequence_log_prob = (log_prob.float() * response_mask).sum(dim=-1)
                if sequence_reduction == "mean":
                    sequence_log_prob = sequence_log_prob / response_lengths
                pg_loss = (surrogate_weights.detach().float() * sequence_log_prob).sum()
                pg_metrics = {}
            else:
                ref_log_prob = data["ref_log_prob"][real_sample_mask]
                token_level_scores = data["token_level_scores"][real_sample_mask]
                group_ids = data[POLICY_GROUP_IDS][real_sample_mask]
                sequence_scores = (token_level_scores.float() * response_mask).sum(dim=-1)
                common_kwargs = {
                    "log_prob": log_prob,
                    "ref_log_prob": ref_log_prob,
                    "rewards": sequence_scores,
                    "response_mask": response_mask,
                    "group_ids": group_ids,
                    "beta": config.policy_loss.beta,
                    "group_size": group_size,
                    "global_batch_size": config.global_batch_info["global_batch_size"],
                    "batch_num_tokens": config.global_batch_info["batch_num_tokens"],
                    "dp_size": config.global_batch_info["dp_size"],
                }
                if loss_mode == "ggpo":
                    pg_loss, pg_metrics = compute_ggpo_loss(
                        **common_kwargs,
                        raw_correctness=data[GGPO_RAW_CORRECTNESS][real_sample_mask],
                        sequence_log_prob_reduction=sequence_reduction,
                        reward_kl=config.policy_loss.get("reward_kl", True),
                    )
                else:
                    pg_loss, pg_metrics = compute_agro_loss(**common_kwargs)
        else:
            pg_loss, pg_metrics = _zero_group_reference_metrics(log_prob, loss_mode=loss_mode)
    else:
        scientific_metric_weight = int(real_sample_mask.sum().item())
        scientific_response_mask = response_mask & real_sample_mask.unsqueeze(-1)
        regularization_response_mask = scientific_response_mask
        if scientific_metric_weight:
            response_mask = scientific_response_mask
        else:
            # Rollout-correction helpers intentionally reject an empty token
            # mask. DP3 can produce one padding-only microbatch, so run the
            # normal schema-producing path on its cloned valid token mask while
            # zero advantages make its loss and gradient exactly zero.
            if not dummy_safe_response_mask.any():
                raise ValueError("Padding-only microbatch has no valid dummy response tokens.")
            response_mask = dummy_safe_response_mask
        # compute policy loss
        old_log_prob = None if loss_mode == "asympo" else data["old_log_probs"]
        advantages = data["advantages"]
        if loss_mode == "asympo" and scientific_metric_weight:
            log_prob = log_prob[real_sample_mask]
            advantages = advantages[real_sample_mask]
            response_mask = response_mask[real_sample_mask]
            regularization_response_mask = response_mask
            if entropy is not None:
                entropy = entropy[real_sample_mask]
        if not scientific_metric_weight:
            if loss_mode == "asympo":
                pg_loss, pg_metrics = _zero_asympo_metrics(log_prob)
            else:
                advantages = torch.zeros_like(advantages)
        rollout_is_weights = None if loss_mode == "asympo" else data.get("rollout_is_weights", None)

        if scientific_metric_weight or loss_mode != "asympo":
            policy_loss_fn = get_policy_loss_fn(loss_mode)
            pg_loss, pg_metrics = policy_loss_fn(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                response_mask=response_mask,
                loss_agg_mode=loss_agg_mode,
                config=config,
                rollout_is_weights=rollout_is_weights,
            )

    if collect_rollout_training_mismatch:
        mismatch_rollout_log_prob = (
            rollout_log_prob
            if group_reference_loss
            else data["rollout_log_probs"][real_sample_mask]
            if loss_mode == "asympo" and scientific_metric_weight
            else data["rollout_log_probs"]
        )
        metrics[INTERNAL_MISMATCH_STATS_KEY] = (
            compute_mismatch_stats(
                current_log_probs=log_prob,
                rollout_log_probs=mismatch_rollout_log_prob,
                response_mask=response_mask,
            )
            if scientific_metric_weight
            else empty_mismatch_stats()
        )

    if group_reference_loss:
        min_metrics = {"ggpo_adjusted_reward_min", "agro_regularized_reward_min"}
        max_metrics = {
            "ggpo_adjusted_reward_max",
            "ggpo_sequence_log_ratio_abs_max",
            "ggpo_raw_sequence_log_ratio_abs_max",
            "agro_regularized_reward_max",
            "agro_loo_delta_abs_max",
            "agro_sequence_log_ratio_abs_max",
        }
        weighted_pg_metrics = {}
        for name, value in pg_metrics.items():
            metric_weight = scientific_metric_weight
            if loss_mode == "ggpo":
                selected_fraction = ggpo_conditional_metric_fraction(name, pg_metrics)
            else:
                selected_fraction = None
            if selected_fraction is not None:
                metric_weight *= selected_fraction
                if metric_weight == 0:
                    continue
            weighted_pg_metrics[name] = Metric(
                value=value,
                weight=metric_weight,
                aggregation=(
                    AggregationType.MIN
                    if name in min_metrics
                    else AggregationType.MAX
                    if name in max_metrics
                    else AggregationType.MEAN
                ),
            )
        pg_metrics = weighted_pg_metrics
    elif loss_mode == "asympo":
        min_metrics = {"asympo_response_nll_min"}
        max_metrics = {"asympo_response_nll_max", "asympo_inverse_scale_max"}
        pg_metrics = {
            name: Metric(
                value=value,
                weight=scientific_metric_weight,
                aggregation=(
                    AggregationType.MIN
                    if name in min_metrics
                    else AggregationType.MAX
                    if name in max_metrics
                    else AggregationType.MEAN
                ),
            )
            for name, value in pg_metrics.items()
        }
    else:
        # AggregationType.MEAN assumes the policy loss function normalizes by
        # local batch size or local token count.
        pg_metrics = Metric.from_dict(
            pg_metrics,
            aggregation=AggregationType.MEAN,
            weight=scientific_metric_weight,
        )

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(
        value=pg_loss,
        aggregation=metric_aggregation,
        weight=scientific_metric_weight,
    )
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=regularization_response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy"] = Metric(
            value=entropy_loss,
            aggregation=metric_aggregation,
            weight=scientific_metric_weight,
        )

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld,
            loss_mask=regularization_response_mask,
            loss_agg_mode=config.loss_agg_mode,
            **config.global_batch_info,
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(
            value=kl_loss,
            aggregation=metric_aggregation,
            weight=scientific_metric_weight,
        )
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
