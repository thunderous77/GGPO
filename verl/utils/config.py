# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import math
import os
from dataclasses import is_dataclass
from numbers import Real
from operator import index
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = [
    "get_train_batch_size_for_validation",
    "get_fully_async_prompts_per_update",
    "get_trainer_process_on_nodes",
    "omega_conf_to_dataclass",
    "validate_config",
]


_CONFIG_VALUE_MISSING = object()
_GGPO_INERT_CONFIG_PATH_REASONS = {
    "actor_rollout_ref.actor.clip_ratio": "GGPO is ratio-free and has no importance ratio to clip",
    "actor_rollout_ref.actor.clip_ratio_low": "GGPO is ratio-free and has no importance ratio to clip",
    "actor_rollout_ref.actor.clip_ratio_high": "GGPO is ratio-free and has no importance ratio to clip",
    "actor_rollout_ref.actor.clip_ratio_c": "GGPO is ratio-free and has no importance ratio to clip",
    "actor_rollout_ref.actor.policy_loss.ppo_kl_coef": (
        "GGPO gets its reference-policy strength from actor.policy_loss.beta"
    ),
    "actor_rollout_ref.actor.policy_loss.clip_cov_lb": "GGPO does not use the clip-cov ratio objective",
    "actor_rollout_ref.actor.policy_loss.clip_cov_ub": "GGPO does not use the clip-cov ratio objective",
    "actor_rollout_ref.actor.policy_loss.clip_cov_ratio": "GGPO does not use the clip-cov ratio objective",
    "actor_rollout_ref.actor.policy_loss.kl_cov_ratio": "GGPO does not use the kl-cov ratio objective",
    "actor_rollout_ref.actor.loss_agg_mode": "GGPO implements its own group and global-token normalization",
    "actor_rollout_ref.actor.loss_scale_factor": "GGPO implements its own group and global-token normalization",
    "algorithm.adv_estimator": "GGPO consumes verifier rewards directly and does not consume advantages",
    "algorithm.gamma": "GGPO consumes verifier rewards directly and does not compute discounted returns",
    "algorithm.lam": "GGPO consumes verifier rewards directly and does not compute GAE",
    "algorithm.norm_adv_by_std_in_grpo": "GGPO does not consume GRPO advantages",
    "algorithm.kl_ctrl.target_kl": "GGPO uses the fixed actor.policy_loss.beta coefficient, not adaptive KL control",
    "algorithm.use_kl_in_reward": "GGPO embeds reference regularization directly in its objective",
    "algorithm.pf_ppo.reweight_method": "GGPO does not consume PF-PPO advantage weights",
    "algorithm.pf_ppo.weight_pow": "GGPO does not consume PF-PPO advantage weights",
    "algorithm.rollout_correction.rollout_is": "GGPO is ratio-free and does not consume rollout IS correction",
    "algorithm.rollout_correction.rollout_rs": "GGPO is ratio-free and does not consume rollout rejection sampling",
    "algorithm.rollout_correction.bypass_mode": "GGPO has one dedicated ratio-free loss path",
    "algorithm.rollout_correction.loss_type": "GGPO does not use a rollout-correction loss replacement",
    "algorithm.rollout_correction.rollout_is_batch_normalize": (
        "GGPO does not consume rollout importance weights"
    ),
}


def _validate_ggpo_inert_config_is_absent(config: DictConfig) -> None:
    if OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode", default="vanilla") != "ggpo":
        return
    for path, reason in _GGPO_INERT_CONFIG_PATH_REASONS.items():
        value = OmegaConf.select(config, path, default=_CONFIG_VALUE_MISSING)
        # A null-valued key carries no configuration and cannot be removed by a Hydra `~` override:
        # Hydra reports a None-valued node as non-existent and refuses to delete it.
        if value is not _CONFIG_VALUE_MISSING and value is not None:
            raise ValueError(f"GGPO forbids {path}: {reason}; remove this key from the resolved GGPO config.")


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got an object
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_. "
            "See trainer/config/ppo_trainer.yaml algorithm section for an example. "
            f"Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    # pop _target_ to avoid hydra instantiate error, as most dataclass do not have _target_
    # Updated (vermouth1992) We add _target_ to BaseConfig so that it is compatible.
    # Otherwise, this code path can't support recursive instantiation.
    # if "_target_" in cfg:
    #     cfg.pop("_target_")
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def validate_ggpo_trainer_compatibility(config: DictConfig) -> None:
    """Reject trainer modes or probability conventions that invalidate GGPO.

    ``rollout_log_probs`` is interpreted as the behavior-policy probability in
    the GGPO theory contract.  The learner scorer is a teacher-forced raw model
    softmax, so every rollout sampling transform must be the identity for those
    two tensors to describe the same distribution.
    """
    if OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode") != "ggpo":
        return
    offline_mode = OmegaConf.select(config, "offline_data.mode", default=None)
    if offline_mode in {
        "materialize",
        "seal_theory_plan",
        "evaluate",
        "analyze",
        "theory_validate",
        "theory_finalize",
    }:
        # These are read-only/materialization workflows, not learner updates.
        # Their separate immutable offline contracts validate sampling and
        # policy identities without creating a training Set-A run receipt.
        return
    if OmegaConf.select(config, "trainer.use_v1", default=False):
        raise ValueError(
            "GGPO requires trainer.use_v1=False because the v1 trainer does not yet preserve GGPO group metadata "
            "or implement the mandatory pre-update Set-A theory metrics during actor updates."
        )

    # Paper reproduction does not require the historical theory-run identity receipt.
    if OmegaConf.select(config, "trainer.ggpo_theory_contract") is None:
        return

    identity_sampling_fields = {
        "actor_rollout_ref.rollout.temperature": (1.0, 1.0),
        "actor_rollout_ref.rollout.top_p": (1.0, 1.0),
        "actor_rollout_ref.rollout.top_k": (-1, -1),
        "actor_rollout_ref.rollout.repetition_penalty": (1.0, 1.0),
    }
    for path, (default, expected) in identity_sampling_fields.items():
        try:
            value = OmegaConf.select(config, path, default=default)
        except Exception as exc:
            raise ValueError(f"GGPO requires a resolved identity sampling transform at {path}.") from exc
        if isinstance(value, bool):
            matches = False
        elif path.endswith(".top_k"):
            try:
                matches = index(value) == expected
            except TypeError:
                matches = False
        else:
            if not isinstance(value, Real):
                matches = False
            else:
                numeric = float(value)
                matches = math.isfinite(numeric) and numeric == expected
        if not matches:
            raise ValueError(
                "GGPO Set-A density ratios require rollout probabilities to use the same "
                "raw-softmax convention as the teacher-forced learner scorer; "
                f"{path} must be {expected!r}, got {value!r}."
            )

    try:
        do_sample = OmegaConf.select(
            config,
            "actor_rollout_ref.rollout.do_sample",
            default=True,
        )
    except Exception as exc:
        raise ValueError("GGPO requires a resolved actor_rollout_ref.rollout.do_sample value.") from exc
    if do_sample is not True:
        raise ValueError(
            "GGPO Set-A density ratios require actor_rollout_ref.rollout.do_sample=true; "
            "a greedy delta policy is not the teacher-forced raw-softmax behavior distribution."
        )

    try:
        contract = OmegaConf.select(config, "trainer.ggpo_theory_contract")
    except Exception as exc:
        raise ValueError("GGPO requires a resolved trainer.ggpo_theory_contract block.") from exc
    if contract is None or not isinstance(contract, DictConfig | dict):
        raise ValueError(
            "GGPO requires trainer.ggpo_theory_contract with run, source, initial-reference-policy, "
            "and behavior-policy identities."
        )

    required_identities = (
        "run_id",
        "source_identity",
        "reference_policy_identity",
        "behavior_policy_identity",
    )
    resolved_identities = {}
    for field in required_identities:
        try:
            value = OmegaConf.select(config, f"trainer.ggpo_theory_contract.{field}")
        except Exception as exc:
            raise ValueError(
                f"GGPO requires resolved trainer.ggpo_theory_contract.{field}."
            ) from exc
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"GGPO requires nonempty trainer.ggpo_theory_contract.{field}."
            )
        resolved_identities[field] = value.strip()

    logger_backends = OmegaConf.select(config, "trainer.logger", default=())
    if isinstance(logger_backends, str):
        logger_backends = (logger_backends,)
    elif logger_backends is None:
        logger_backends = ()
    if "wandb" in logger_backends or "tracking" in logger_backends:
        wandb_run_id = os.environ.get("WANDB_RUN_ID")
        if not wandb_run_id:
            raise ValueError(
                "GGPO with W&B logging requires an explicit WANDB_RUN_ID so the sealed "
                "theory receipt cannot bind to a tracker-generated, unknown run identity."
            )
        if wandb_run_id != resolved_identities["run_id"]:
            raise ValueError(
                "GGPO theory run identity does not match WANDB_RUN_ID: "
                f"{resolved_identities['run_id']!r} != {wandb_run_id!r}."
            )


def _validate_fixed_policy_lag(config: DictConfig) -> None:
    """Validate the V0 online execution contract for durable fixed policy lag."""
    fixed_policy_lag = OmegaConf.select(config, "trainer.fixed_policy_lag", default=None)
    fixed_lag_ledger_path = OmegaConf.select(config, "trainer.fixed_lag_ledger_path", default=None)
    if fixed_policy_lag is None:
        if fixed_lag_ledger_path is not None:
            raise ValueError("trainer.fixed_lag_ledger_path must be null when fixed_policy_lag is disabled.")
    else:
        if isinstance(fixed_policy_lag, bool):
            raise ValueError("trainer.fixed_policy_lag must be an integer greater than zero.")
        try:
            fixed_policy_lag = index(fixed_policy_lag)
        except TypeError as exc:
            raise ValueError("trainer.fixed_policy_lag must be an integer greater than zero.") from exc
        if fixed_policy_lag <= 0:
            raise ValueError("trainer.fixed_policy_lag must be an integer greater than zero.")
        if not isinstance(fixed_lag_ledger_path, str) or not fixed_lag_ledger_path.strip():
            raise ValueError("trainer.fixed_lag_ledger_path must be a nonempty string when fixed lag is enabled.")
        if OmegaConf.select(config, "trainer.use_v1", default=False):
            raise ValueError("Fixed policy lag supports only the V0 trainer (trainer.use_v1=false).")
        nnodes = OmegaConf.select(config, "trainer.nnodes", default=None)
        if isinstance(nnodes, bool):
            raise ValueError("Fixed policy lag requires trainer.nnodes=1.")
        try:
            nnodes = index(nnodes)
        except TypeError as exc:
            raise ValueError("Fixed policy lag requires trainer.nnodes=1.") from exc
        if nnodes != 1:
            raise ValueError("Fixed policy lag requires trainer.nnodes=1.")
        if OmegaConf.select(config, "actor_rollout_ref.hybrid_engine", default=False) is not True:
            raise ValueError("Fixed policy lag requires actor_rollout_ref.hybrid_engine=true.")
        if OmegaConf.select(config, "async_training", default=None) is not None:
            raise ValueError("Fixed policy lag does not support fully-async training.")
        if OmegaConf.select(config, "offline_data", default=None) is not None:
            raise ValueError("Fixed policy lag supports only online V0 training.")
        if OmegaConf.select(config, "actor_rollout_ref.rollout.calculate_log_probs", default=False) is not True:
            raise ValueError("Fixed policy lag requires actor_rollout_ref.rollout.calculate_log_probs=true.")
        group_size = OmegaConf.select(config, "actor_rollout_ref.rollout.n", default=None)
        if isinstance(group_size, bool):
            raise ValueError("Fixed policy lag requires actor_rollout_ref.rollout.n greater than one.")
        try:
            group_size = index(group_size)
        except TypeError as exc:
            raise ValueError("Fixed policy lag requires actor_rollout_ref.rollout.n greater than one.") from exc
        if group_size <= 1:
            raise ValueError("Fixed policy lag requires actor_rollout_ref.rollout.n greater than one.")


def validate_policy_loss_trainer_compatibility(config: DictConfig) -> None:
    """Validate execution-mode and correction contracts for dedicated losses."""
    _validate_fixed_policy_lag(config)

    loss_mode = OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode", default="vanilla")
    moving_ref = OmegaConf.select(config, "trainer.ggpo_moving_reference.enabled", default=False)
    if not isinstance(moving_ref, bool):
        raise ValueError("trainer.ggpo_moving_reference.enabled must be a boolean.")
    if moving_ref:
        if loss_mode not in {"ggpo", "agro", "vanilla"}:
            raise ValueError("trainer.ggpo_moving_reference requires loss_mode ggpo, agro, or vanilla.")
        if loss_mode == "vanilla" and not OmegaConf.select(
            config, "actor_rollout_ref.actor.use_kl_loss", default=False
        ):
            raise ValueError(
                "Vanilla moving reference requires actor_rollout_ref.actor.use_kl_loss=true "
                "so the reference worker exists."
            )
        if OmegaConf.select(
            config, "trainer.ggpo_moving_reference.refresh_every_updates", default=None
        ) is None:
            update_freq = OmegaConf.select(config, "trainer.ggpo_moving_reference.update_freq", default=None)
            save_freq = OmegaConf.select(config, "trainer.save_freq", default=None)
            if isinstance(update_freq, bool) or isinstance(save_freq, bool):
                raise ValueError("GGPO moving-reference update_freq and trainer.save_freq must be integers.")
            try:
                update_freq, save_freq = index(update_freq), index(save_freq)
            except TypeError as exc:
                raise ValueError("GGPO moving-reference update_freq and trainer.save_freq must be integers.") from exc
            offline_smoke = bool(OmegaConf.select(config, "offline_data.smoke", default=False))
            offline_configured = OmegaConf.select(config, "offline_data", default=None) is not None
            if offline_configured:
                invalid_cadence = update_freq <= 0 or (offline_smoke and save_freq != -1) or (
                    not offline_smoke and (save_freq <= 0 or update_freq % save_freq != 0)
                )
            else:
                invalid_cadence = update_freq <= 0 or update_freq != save_freq
            if invalid_cadence:
                if offline_configured:
                    raise ValueError(
                        "offline GGPO moving-reference update_freq must be a positive multiple of "
                        "trainer.save_freq (offline smoke instead requires trainer.save_freq=-1): "
                        f"update_freq={update_freq}, save_freq={save_freq}."
                    )
                raise ValueError(
                    "trainer.ggpo_moving_reference.update_freq must be positive and equal trainer.save_freq "
                    "(offline smoke instead requires trainer.save_freq=-1): "
                    f"update_freq={update_freq}, save_freq={save_freq}."
                )
        if OmegaConf.select(config, "trainer.use_v1", default=False):
            raise ValueError("GGPO moving reference supports only the V0 trainer (trainer.use_v1=false).")
        if OmegaConf.select(config, "async_training", default=None) is not None:
            raise ValueError("GGPO moving reference does not support fully-async training.")
        offline_mode = OmegaConf.select(config, "offline_data.mode", default=None)
        if OmegaConf.select(config, "offline_data", default=None) is not None and offline_mode not in {
            "train",
            "initialize",
            "evaluate",
        }:
            raise ValueError("GGPO moving reference received an unsupported offline_data.mode.")
        if OmegaConf.select(config, "actor_rollout_ref.actor.strategy") != "fsdp" or OmegaConf.select(
            config, "actor_rollout_ref.ref.strategy"
        ) != "fsdp":
            raise ValueError("GGPO moving reference supports only synchronous FSDP actor and ref workers.")
        if OmegaConf.select(config, "actor_rollout_ref.ref.fsdp_config.forward_only", default=True) is not True:
            raise ValueError("GGPO moving reference requires a forward-only reference worker.")
        lora_rank = OmegaConf.select(config, "actor_rollout_ref.model.lora.rank", default=0) or OmegaConf.select(
            config, "actor_rollout_ref.model.lora_rank", default=0
        )
        if lora_rank != 0 or OmegaConf.select(config, "actor_rollout_ref.model.lora_adapter_path", default=None):
            raise ValueError("GGPO moving reference does not support LoRA.")
        checkpoint = OmegaConf.select(config, "actor_rollout_ref.actor.checkpoint")
        if "model" not in checkpoint.get("save_contents", ()) or "model" not in checkpoint.get("load_contents", ()):
            raise ValueError("GGPO moving reference requires actor checkpoint save/load contents to include model.")
        if checkpoint.get("async_save", False):
            raise ValueError("GGPO moving reference requires synchronous actor checkpoint saves.")
        if OmegaConf.select(config, "trainer.resume_mode", default="auto") != "disable":
            if OmegaConf.select(config, "trainer.del_local_ckpt_after_load", default=False):
                raise ValueError("GGPO moving-reference resume requires trainer.del_local_ckpt_after_load=false.")
            if OmegaConf.select(config, "trainer.default_hdfs_dir", default=None) is not None:
                raise ValueError("GGPO moving-reference resume requires a local actor checkpoint.")
    if loss_mode in {"ggpo", "agro"} and OmegaConf.select(config, "trainer.balance_batch", default=False):
        raise ValueError(
            f"{str(loss_mode).upper()} requires trainer.balance_batch=False because trainer-level length balancing "
            "can split or reorder complete response groups."
        )
    _validate_ggpo_inert_config_is_absent(config)
    validate_ggpo_trainer_compatibility(config)
    if loss_mode not in {"agro", "asympo"}:
        return

    offline_mode = OmegaConf.select(config, "offline_data.mode", default=None)
    fully_async = OmegaConf.select(config, "async_training", default=None) is not None
    if loss_mode == "asympo" and not fully_async and offline_mode not in {"train", "initialize", "evaluate"}:
        raise ValueError("ASymPO supports only fully-async or offline training; synchronous training is rejected.")
    if OmegaConf.select(config, "algorithm.use_kl_in_reward", default=False):
        raise ValueError(f"{loss_mode.upper()} requires algorithm.use_kl_in_reward=false.")
    if OmegaConf.select(config, "actor_rollout_ref.actor.use_kl_loss", default=False):
        raise ValueError(f"{loss_mode.upper()} requires actor.use_kl_loss=false.")

    for path in (
        "algorithm.rollout_correction.rollout_is",
        "algorithm.rollout_correction.rollout_rs",
    ):
        if OmegaConf.select(config, path, default=None) is not None:
            raise ValueError(f"{loss_mode.upper()} forbids rollout correction at {path}.")
    if bool(OmegaConf.select(config, "algorithm.rollout_correction.bypass_mode", default=False)):
        raise ValueError(f"{loss_mode.upper()} must retain its dedicated loss_mode and forbids bypass_mode.")

    if loss_mode == "agro":
        if OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.sequence_log_prob_reduction") != "sum":
            raise ValueError("AGRO requires response sequence log probabilities to use sum reduction.")
    else:
        adv_estimator = str(OmegaConf.select(config, "algorithm.adv_estimator"))
        if adv_estimator != "grpo":
            raise ValueError("ASymPO requires algorithm.adv_estimator=grpo for raw-reward group mean centering.")
        if bool(OmegaConf.select(config, "algorithm.norm_adv_by_std_in_grpo", default=True)):
            raise ValueError("ASymPO requires algorithm.norm_adv_by_std_in_grpo=false.")


def get_fully_async_prompts_per_update(config: DictConfig) -> int:
    """Resolve one fail-closed prompt-group budget for all async components."""
    ppo_mini_batch_size = OmegaConf.select(config, "actor_rollout_ref.actor.ppo_mini_batch_size")
    require_batches = OmegaConf.select(config, "async_training.require_batches")
    try:
        ppo_mini_batch_size = index(ppo_mini_batch_size)
        require_batches = index(require_batches)
    except TypeError as exc:
        raise ValueError(
            "Fully async batch validation requires integer actor.ppo_mini_batch_size and "
            "async_training.require_batches."
        ) from exc
    if ppo_mini_batch_size <= 0 or require_batches <= 0:
        raise ValueError(
            "Fully async batch validation requires actor.ppo_mini_batch_size and "
            "async_training.require_batches to be positive."
        )

    canonical = ppo_mini_batch_size * require_batches
    explicit = OmegaConf.select(config, "async_training.prompts_per_update", default=None)
    if explicit is None:
        return canonical
    try:
        explicit = index(explicit)
    except TypeError as exc:
        raise ValueError("async_training.prompts_per_update must be an integer when set.") from exc
    if explicit != canonical:
        raise ValueError(
            "async_training.prompts_per_update must equal "
            "actor.ppo_mini_batch_size * async_training.require_batches: "
            f"{explicit} != {ppo_mini_batch_size} * {require_batches}."
        )
    return canonical


def get_train_batch_size_for_validation(config: DictConfig) -> int:
    """Return the prompt batch size used for trainer-side config validation.

    Fully asynchronous rollout keeps ``data.train_batch_size`` at zero because
    its data loader streams one prompt at a time. The trainer still consumes a
    fixed prompt batch on every update, so fixed-batch validation must use that
    effective size without mutating the loader configuration.
    """
    train_batch_size = config.data.train_batch_size
    if train_batch_size != 0 or OmegaConf.select(config, "async_training") is None:
        return train_batch_size

    return get_fully_async_prompts_per_update(config)


def get_trainer_process_on_nodes(config: DictConfig) -> list[int]:
    """Return the resolved trainer GPU-process count for each node.

    ``trainer.process_on_nodes`` is an optional fully-async topology override
    for uneven pools such as five trainer workers on two four-GPU nodes.  The
    legacy rectangular ``nnodes``/``n_gpus_per_node`` settings remain the
    default.  When the override is present, the legacy fields still describe
    the node count and maximum per-node occupancy so downstream configuration
    remains explicit and auditable.
    """

    try:
        nnodes = index(config.trainer.nnodes)
        n_gpus_per_node = index(config.trainer.n_gpus_per_node)
    except TypeError as exc:
        raise ValueError("trainer.nnodes and trainer.n_gpus_per_node must be integers.") from exc
    if nnodes <= 0 or n_gpus_per_node <= 0:
        raise ValueError("trainer.nnodes and trainer.n_gpus_per_node must be positive.")

    raw_process_on_nodes = OmegaConf.select(config, "trainer.process_on_nodes")
    if raw_process_on_nodes is None:
        return [n_gpus_per_node] * nnodes
    if not isinstance(raw_process_on_nodes, ListConfig | list | tuple):
        raise ValueError("trainer.process_on_nodes must be a list of positive integers.")

    process_on_nodes = []
    for value in raw_process_on_nodes:
        try:
            value = index(value)
        except TypeError as exc:
            raise ValueError("trainer.process_on_nodes must contain only positive integers.") from exc
        if value <= 0:
            raise ValueError("trainer.process_on_nodes must contain only positive integers.")
        process_on_nodes.append(value)

    if len(process_on_nodes) != nnodes:
        raise ValueError(
            "trainer.process_on_nodes length must equal trainer.nnodes: "
            f"{len(process_on_nodes)} != {nnodes}."
        )
    if max(process_on_nodes) != n_gpus_per_node:
        raise ValueError(
            "trainer.n_gpus_per_node must equal the maximum trainer.process_on_nodes entry: "
            f"{n_gpus_per_node} != {max(process_on_nodes)}."
        )
    return process_on_nodes


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    validate_policy_loss_trainer_compatibility(config)

    loss_mode = OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode", default="vanilla")
    if loss_mode == "agro" and not use_reference_policy:
        raise ValueError("AGRO requires a reference policy worker.")

    # number of trainer GPUs total
    n_gpus = sum(get_trainer_process_on_nodes(config))
    validation_train_batch_size = get_train_batch_size_for_validation(config)
    from verl.experimental.fully_async_policy.actor_update_padding import get_actor_update_padding_spec

    actor_update_padding = get_actor_update_padding_spec(config)
    ggpo_memory_safe_surrogate = bool(
        OmegaConf.select(
            config,
            "actor_rollout_ref.actor.policy_loss.memory_safe_surrogate",
            default=False,
        )
    )
    if ggpo_memory_safe_surrogate:
        if not actor_update_padding.enabled:
            raise ValueError(
                "GGPO memory_safe_surrogate requires fully-async actor_update_padding so "
                "the driver can bind exact logical and physical mini-batch geometry."
            )
        if OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode") != "ggpo":
            raise ValueError("GGPO memory_safe_surrogate requires actor.policy_loss.loss_mode=ggpo.")
    if actor_update_padding.enabled:
        if use_critic:
            raise ValueError("Fully async actor-update padding does not support a critic worker.")
        if not bool(OmegaConf.select(config, "actor_rollout_ref.model.use_remove_padding", default=False)):
            raise ValueError("Fully async actor-update padding requires actor_rollout_ref.model.use_remove_padding=true.")
        if bool(
            OmegaConf.select(
                config,
                "algorithm.rollout_correction.rollout_is_batch_normalize",
                default=False,
            )
        ):
            raise ValueError(
                "Fully async actor-update padding does not support "
                "algorithm.rollout_correction.rollout_is_batch_normalize=true."
            )
    if config.actor_rollout_ref.actor.strategy == "megatron":
        model_parallel_size = (
            config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
            * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
        )
        assert (
            n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
        ), (
            f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
            f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
        )
        effective_actor_dp_size = n_gpus // (
            model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
        )
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            minimal_bsz = effective_actor_dp_size * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
    else:
        try:
            legacy_ulysses_sp_value = OmegaConf.select(
                config,
                "actor_rollout_ref.actor.ulysses_sequence_parallel_size",
                default=1,
            )
            engine_ulysses_sp_value = OmegaConf.select(
                config,
                "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size",
                default=1,
            )
            if isinstance(legacy_ulysses_sp_value, bool) or isinstance(engine_ulysses_sp_value, bool):
                raise TypeError
            legacy_ulysses_sp_size = index(legacy_ulysses_sp_value)
            engine_ulysses_sp_size = index(engine_ulysses_sp_value)
        except TypeError as exc:
            raise ValueError(
                "actor FSDP Ulysses sequence-parallel size must be an integer."
            ) from exc
        # Match FSDPActorConfig's backward-compatibility behavior: a
        # legacy actor-level value above one overrides the engine value.
        ulysses_sp_size = (
            legacy_ulysses_sp_size
            if legacy_ulysses_sp_size > 1
            else engine_ulysses_sp_size
        )
        if ulysses_sp_size <= 0 or n_gpus % ulysses_sp_size != 0:
            raise ValueError(
                f"n_gpus ({n_gpus}) must be divisible by "
                "the actor FSDP Ulysses sequence-parallel size "
                f"({ulysses_sp_size})"
            )
        # Ulysses ranks jointly process one sequence shard. Batch
        # divisibility therefore follows the effective data-parallel
        # world size, not the total FSDP world size.
        effective_actor_dp_size = n_gpus // ulysses_sp_size
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            minimal_bsz = effective_actor_dp_size

    if loss_mode in {"ggpo", "agro"} and not ggpo_memory_safe_surrogate:
        group_size = OmegaConf.select(config, "actor_rollout_ref.rollout.n")
        physical_responses_per_mini = (
            actor_update_padding.physical_responses_per_mini
            if actor_update_padding.enabled
            else (
                config.actor_rollout_ref.actor.ppo_mini_batch_size
                * config.actor_rollout_ref.rollout.n
            )
        )
        complete_group_dispatch_size = effective_actor_dp_size * group_size
        if physical_responses_per_mini % complete_group_dispatch_size != 0:
            raise ValueError(
                f"Each effective data-parallel rank must receive complete {loss_mode.upper()} groups in every "
                "actor mini-batch: "
                f"{physical_responses_per_mini} responses must be divisible by effective_dp "
                f"({effective_actor_dp_size}) * group_size ({group_size})."
            )

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        # 1. Check total batch size for data correctness
        logical_train_batch_size = validation_train_batch_size * config.actor_rollout_ref.rollout.n
        physical_train_batch_size = (
            actor_update_padding.physical_responses_per_update
            if actor_update_padding.enabled
            else logical_train_batch_size
        )
        if actor_update_padding.enabled:
            if logical_train_batch_size != actor_update_padding.logical_responses_per_update:
                raise ValueError(
                    "Fully async actor-update padding logical geometry disagrees with trainer validation: "
                    f"{logical_train_batch_size} != {actor_update_padding.logical_responses_per_update}."
                )
            if actor_update_padding.physical_responses_per_mini % minimal_bsz != 0:
                raise ValueError(
                    "Physical actor mini-batch must be divisible by the effective FSDP data-parallel size: "
                    f"{actor_update_padding.physical_responses_per_mini} % {minimal_bsz} != 0."
                )
        if physical_train_batch_size % minimal_bsz != 0:
            raise ValueError(
                f"physical_train_batch_size ({physical_train_batch_size}) must be divisible by minimal possible "
                f"batch size ({minimal_bsz})"
            )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    # Actor validation done in ActorConfig.__post_init__ and validate()
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor_config.validate(n_gpus, validation_train_batch_size, config.actor_rollout_ref.model)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )

    if config.algorithm.get("use_kl_in_reward", False) and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(n_gpus, validation_train_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # check LoRA rank in vLLM
    lora_config = config.actor_rollout_ref.model.get("lora", {})
    lora_rank = lora_config.get("rank", 0)
    if lora_rank <= 0:
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
    if lora_config.get("merge", False):
        lora_rank = 0
    if lora_rank > 0 and config.actor_rollout_ref.rollout.name == "vllm":
        from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

        get_vllm_max_lora_rank(lora_rank)

    print("[validate_config] All configuration checks passed successfully!")
