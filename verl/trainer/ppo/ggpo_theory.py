# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Pure group-level GGPO objective."""

from __future__ import annotations

import hashlib
import json
import math
import operator
import os
import stat
from pathlib import Path
from typing import Any

import torch


GGPO_THEORY_METRICS_SCHEMA_VERSION = 5
GGPO_THEORY_METRICS_SENTINEL = "ggpo_theory_metrics_schema_version"
GGPO_THEORY_CURRENT_LOG_PROBS = "_ggpo_theory_current_log_probs"
GGPO_THEORY_Q_RHO_FORMULA_VERSION = 1
GGPO_THEORY_J_H_FORMULA_VERSION = 1
GGPO_THEORY_RUN_RECEIPT_SCHEMA_VERSION = 3
GGPO_THEORY_PROBABILITY_CONVENTION_SCHEMA_VERSION = 1
GGPO_THEORY_BEHAVIOR_VERSION_STRATIFICATION_SCHEMA_VERSION = 3
_THEORY_QUANTILES = {
    "p05": 0.05,
    "p95": 0.95,
    "p99": 0.99,
    "p999": 0.999,
}
_CORE_DISTRIBUTION_FIELDS = {
    "ggpo_theory/behavior/sequence_log_ratio": (
        "mean",
        "std",
        "p95",
        "p99",
        "p999",
    ),
    "ggpo_theory/behavior/prompt_sample_log_ratio_max": (
        "mean",
        "p95",
        "p99",
        "max",
    ),
    "ggpo_theory/response/adjusted_reward": ("std", "p95", "p99", "max"),
    "ggpo_theory/group/eta_beh": ("mean", "p95", "max"),
    "ggpo_theory/group/j_beh": ("p95", "max"),
    "ggpo_theory/group/z_abs_max": ("p95", "max"),
    "ggpo_theory/group/q_entropy": ("mean",),
    "ggpo_theory/group/q_max": ("mean", "p95"),
    "ggpo_theory/group/q_ess_normalized": ("mean",),
    "ggpo_theory/group/q_l1_rho": ("mean", "p95"),
    "ggpo_theory/group/r_variance": ("mean", "p95", "max"),
}
_CONFIG_VALUE_MISSING = object()
_BEHAVIOR_VERSION_STRATUM_FIELDS = (
    "majority_token_owner_behavior_policy_version",
    "training_behavior_version_lag",
    "response_count",
    "prompt_count",
    "ell_mean",
    "ell_max",
    "ell_p99",
    "j_beh_mean",
    "j_beh_max",
    "q_entropy_mean",
    "q_max_mean",
    "q_ess_normalized_mean",
    "q_l1_rho_mean",
)


def _resolved_config_value(config, path: str, default=_CONFIG_VALUE_MISSING):
    from omegaconf import OmegaConf

    try:
        value = OmegaConf.select(config, path, default=_CONFIG_VALUE_MISSING)
    except Exception as exc:
        raise ValueError(f"GGPO theory receipt requires resolved configuration value {path}.") from exc
    if value is _CONFIG_VALUE_MISSING:
        if default is _CONFIG_VALUE_MISSING:
            raise ValueError(f"GGPO theory receipt requires configuration value {path}.")
        return default
    return value


def build_ggpo_theory_run_receipt(config) -> dict[str, Any]:
    """Build the canonical immutable run-level GGPO theory contract.

    Human-readable identities stay in this receipt rather than being coerced
    into tracker scalars.  Each Set-A record logs an exact 52-bit prefix of the
    receipt SHA-256, which is also carried to the final actor-update boundary.
    """
    contract_path = "trainer.ggpo_theory_contract"
    identity_fields = (
        "run_id",
        "source_identity",
        "reference_policy_identity",
        "behavior_policy_identity",
    )
    identities = {}
    for field in identity_fields:
        value = _resolved_config_value(config, f"{contract_path}.{field}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"GGPO theory receipt identity {contract_path}.{field} must be a nonempty string."
            )
        identities[field] = value.strip()

    moving_reference = bool(
        _resolved_config_value(config, "trainer.ggpo_moving_reference.enabled", default=False)
    )
    reference_policy = {
        "initial_identity" if moving_reference else "immutable_identity": identities["reference_policy_identity"],
        "model_path": str(_resolved_config_value(config, "actor_rollout_ref.model.path", default="")),
        "hf_config_path": _resolved_config_value(
            config, "actor_rollout_ref.model.hf_config_path", default=None
        ),
        "tokenizer_path": _resolved_config_value(
            config, "actor_rollout_ref.model.tokenizer_path", default=None
        ),
        "probability": "teacher_forced_raw_model_softmax",
    }
    if moving_reference:
        refresh_every_updates = _resolved_config_value(
            config, "trainer.ggpo_moving_reference.refresh_every_updates", default=None
        )
        reference_policy["update"] = (
            {
                "mode": "optimizer_updates",
                "refresh_every_updates": int(refresh_every_updates),
                "version_metric": "actor/ggpo_reference_version",
                "resume_source": "reference_model_snapshot",
            }
            if refresh_every_updates is not None
            else {
                "mode": "latest_saved_actor_checkpoint",
                "update_freq": int(_resolved_config_value(config, "trainer.ggpo_moving_reference.update_freq")),
                "version_metric": "actor/ggpo_reference_version",
                "resume_source": "resumed_actor_checkpoint",
            }
        )

    receipt = {
        "schema_version": GGPO_THEORY_RUN_RECEIPT_SCHEMA_VERSION,
        "run": {
            "run_id": identities["run_id"],
            "project_name": str(
                _resolved_config_value(config, "trainer.project_name", default="")
            ),
            "experiment_name": str(
                _resolved_config_value(config, "trainer.experiment_name", default="")
            ),
            "wandb_entity": os.environ.get("WANDB_ENTITY"),
            "wandb_run_id": os.environ.get("WANDB_RUN_ID"),
        },
        "source": {
            "immutable_identity": identities["source_identity"],
        },
        "objective": {
            "loss_mode": str(
                _resolved_config_value(
                    config,
                    "actor_rollout_ref.actor.policy_loss.loss_mode",
                )
            ),
            "sequence_log_prob_reduction": str(
                _resolved_config_value(
                    config,
                    "actor_rollout_ref.actor.policy_loss.sequence_log_prob_reduction",
                    default="sum",
                )
            ),
            "beta": float(
                _resolved_config_value(
                    config,
                    "actor_rollout_ref.actor.policy_loss.beta",
                )
            ),
            "group_size": int(
                _resolved_config_value(
                    config,
                    "actor_rollout_ref.rollout.n",
                )
            ),
        },
        "reference_policy": reference_policy,
        "behavior_policy": {
            "immutable_identity": identities["behavior_policy_identity"],
            "model_path": str(
                _resolved_config_value(config, "actor_rollout_ref.model.path", default="")
            ),
            "rollout_name": str(
                _resolved_config_value(config, "actor_rollout_ref.rollout.name", default="")
            ),
            "rollout_backend": str(
                _resolved_config_value(
                    config,
                    "async_training.rollout_backend",
                    default="synchronous",
                )
            ),
            "per_response_policy_version_bound_in_set_a": True,
        },
        "probability_convention": {
            "schema_version": GGPO_THEORY_PROBABILITY_CONVENTION_SCHEMA_VERSION,
            "training_log_prob": "teacher_forced_raw_model_softmax",
            "behavior_log_prob": "rollout_log_probs_identity_sampling_distribution",
            "reference_log_prob": "teacher_forced_raw_model_softmax",
            "response_log_ratio_reduction": "sum_valid_response_tokens",
            "response_mask_applied": True,
            "identity_sampling_transforms": {
                "do_sample": _resolved_config_value(
                    config,
                    "actor_rollout_ref.rollout.do_sample",
                    default=True,
                ),
                "temperature": _resolved_config_value(
                    config,
                    "actor_rollout_ref.rollout.temperature",
                    default=1.0,
                ),
                "top_p": _resolved_config_value(
                    config,
                    "actor_rollout_ref.rollout.top_p",
                    default=1.0,
                ),
                "top_k": _resolved_config_value(
                    config,
                    "actor_rollout_ref.rollout.top_k",
                    default=-1,
                ),
                "repetition_penalty": _resolved_config_value(
                    config,
                    "actor_rollout_ref.rollout.repetition_penalty",
                    default=1.0,
                ),
            },
        },
        "behavior_version_stratification": {
            "schema_version": GGPO_THEORY_BEHAVIOR_VERSION_STRATIFICATION_SCHEMA_VERSION,
            "dynamic_key": "group_majority_valid_token_owner_behavior_policy_version",
            "group_attribution": "majority_valid_tokens_with_newest_version_tie_break",
            "non_overlapping": True,
            "fields": list(_BEHAVIOR_VERSION_STRATUM_FIELDS),
            "stratum_statistics": "complete_groups_attributed_once_to_their_majority_token_owner",
            "async_nonlinear_authority": "dynamic_group_owner_strata",
            "global_nonlinear_summaries_when_stratified": "descriptive_only",
        },
    }
    # Preserve existing default receipts when resuming runs with KL enabled.
    if not _resolved_config_value(config, "actor_rollout_ref.actor.policy_loss.reward_kl", default=True):
        receipt["objective"]["reward_kl"] = False
    return receipt


def _canonical_receipt_bytes(receipt: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def seal_ggpo_theory_run_receipt(config) -> dict[str, Any]:
    """Create or exactly verify the immutable GGPO theory run receipt."""
    receipt = build_ggpo_theory_run_receipt(config)
    payload = _canonical_receipt_bytes(receipt)
    sha256 = hashlib.sha256(payload).hexdigest()
    digest_u52 = int(sha256[:13], 16)
    configured_run_root = _resolved_config_value(config, "trainer.default_local_dir")
    if not isinstance(configured_run_root, str) or not configured_run_root.strip():
        raise ValueError(
            "GGPO theory receipt requires trainer.default_local_dir to be a nonempty path."
        )
    run_root = Path(configured_run_root).expanduser()
    receipt_dir = run_root / "ggpo_theory"
    receipt_path = receipt_dir / "run_receipt.json"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    if receipt_path.is_symlink():
        raise ValueError(f"GGPO theory run receipt must not be a symlink: {receipt_path}")

    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(receipt_path, flags, 0o600)
    except FileExistsError:
        if receipt_path.is_symlink():
            raise ValueError(
                f"GGPO theory run receipt must not be a symlink: {receipt_path}"
            )
        read_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            read_flags |= os.O_NOFOLLOW
        read_fd = os.open(receipt_path, read_flags)
        with os.fdopen(read_fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError(
                    f"GGPO theory run receipt must be a regular file: {receipt_path}"
                )
            existing = handle.read()
            if existing != payload:
                raise ValueError(
                    "Existing GGPO theory run receipt does not exactly match the resolved "
                    f"run contract: {receipt_path}"
                )
            os.fchmod(handle.fileno(), 0o400)
    else:
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fchmod(handle.fileno(), 0o400)
                os.fsync(handle.fileno())
        except BaseException:
            # Leave a failed partial file visible so a retry fails closed rather
            # than silently replacing evidence of an interrupted seal.
            raise

    read_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        read_flags |= os.O_NOFOLLOW
    read_fd = os.open(receipt_path, read_flags)
    with os.fdopen(read_fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError(
                f"GGPO theory run receipt must be a regular file: {receipt_path}"
            )
        readback = handle.read()
    if readback != payload:
        raise ValueError(f"GGPO theory run receipt readback mismatch: {receipt_path}")
    return {
        "path": str(receipt_path.resolve()),
        "sha256": sha256,
        "digest_u52": digest_u52,
        "schema_version": GGPO_THEORY_RUN_RECEIPT_SCHEMA_VERSION,
        "probability_convention_schema_version": (
            GGPO_THEORY_PROBABILITY_CONVENTION_SCHEMA_VERSION
        ),
    }


def _theory_distribution_metrics(
    values: torch.Tensor,
    prefix: str,
    fields: tuple[str, ...],
) -> dict[str, float]:
    """Return only the schema-selected exact finite-batch statistics."""
    values = values.detach().double()
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("GGPO theory distribution metrics require a nonempty vector")
    if not torch.isfinite(values).all():
        raise ValueError("GGPO theory distribution metrics require finite values")
    unknown = set(fields).difference(
        {"min", "mean", "std", "max", *_THEORY_QUANTILES}
    )
    if unknown:
        raise ValueError(f"GGPO theory distribution schema has unknown fields: {sorted(unknown)}")

    metrics: dict[str, float] = {}
    direct = {
        "min": values.min,
        "mean": values.mean,
        "std": lambda: values.std(unbiased=False),
        "max": values.max,
    }
    for field in fields:
        if field in direct:
            metrics[f"{prefix}_{field}"] = float(direct[field]().item())

    quantile_fields = [field for field in fields if field in _THEORY_QUANTILES]
    if quantile_fields:
        points = torch.tensor(
            [_THEORY_QUANTILES[field] for field in quantile_fields],
            dtype=values.dtype,
            device=values.device,
        )
        estimates = torch.quantile(values, points)
        metrics.update(
            {
                f"{prefix}_{field}": float(estimate.item())
                for field, estimate in zip(quantile_fields, estimates, strict=True)
            }
        )
    return metrics


def _behavior_version_stratum_summary(
    *,
    sequence_log_ratio: torch.Tensor,
    j_behavior: torch.Tensor,
    q_entropy: torch.Tensor,
    q_max: torch.Tensor,
    q_ess_normalized: torch.Tensor,
    q_l1_rho: torch.Tensor,
) -> dict[str, float]:
    response_values = sequence_log_ratio.double()
    group_values = (
        j_behavior.double(),
        q_entropy.double(),
        q_max.double(),
        q_ess_normalized.double(),
        q_l1_rho.double(),
    )
    if response_values.numel() == 0 or any(values.numel() == 0 for values in group_values):
        raise ValueError("GGPO behavior-version strata must contain responses and complete groups")

    def p99(values):
        return float(torch.quantile(values, 0.99).item())

    return {
        "ell_mean": float(response_values.mean().item()),
        "ell_max": float(response_values.max().item()),
        "ell_p99": p99(response_values),
        "j_beh_mean": float(j_behavior.double().mean().item()),
        "j_beh_max": float(j_behavior.double().max().item()),
        "q_entropy_mean": float(q_entropy.double().mean().item()),
        "q_max_mean": float(q_max.double().mean().item()),
        "q_ess_normalized_mean": float(q_ess_normalized.double().mean().item()),
        "q_l1_rho_mean": float(q_l1_rho.double().mean().item()),
    }


def _group_majority_version_owners(
    *,
    response_versions: torch.Tensor,
    token_versions: torch.Tensor | None,
    response_mask: torch.Tensor,
    grouped_order: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Assign every prompt group once by valid-token majority, newest on ties."""
    owners: list[torch.Tensor] = []
    multi_version_group_count = 0
    for response_indices in grouped_order:
        if token_versions is None:
            response_token_counts = response_mask[response_indices].sum(dim=-1)
            values = torch.repeat_interleave(
                response_versions[response_indices],
                response_token_counts,
            )
        else:
            group_versions = token_versions[response_indices]
            group_mask = response_mask[response_indices]
            values = group_versions[group_mask]
        unique_versions, counts = torch.unique(values, return_counts=True)
        if unique_versions.numel() > 1:
            multi_version_group_count += 1
        majority_count = counts.max()
        # A partial rollout can split exactly at half the valid tokens.  The
        # newest tied version is deterministic and reflects the most recent
        # policy contributing to the stitched group.
        owners.append(unique_versions[counts == majority_count].max())
    return torch.stack(owners), multi_version_group_count


def _ggpo_theory_required_metric_names() -> frozenset[str]:
    distribution_names = {
        f"{prefix}_{field}"
        for prefix, fields in _CORE_DISTRIBUTION_FIELDS.items()
        for field in fields
    }
    return frozenset(
        {
            "ggpo_theory/schema_version",
            "ggpo_theory/run_receipt/schema_version",
            "ggpo_theory/run_receipt/digest_u52",
            "ggpo_theory/probability_convention/schema_version",
            "ggpo_theory/behavior_version_stratification/schema_version",
            "ggpo_theory/behavior_version_stratification/enabled",
            "ggpo_theory/behavior_version_stratification/group_owner_version_count",
            "ggpo_theory/behavior_version_stratification/multi_version_prompt_count",
            "ggpo_theory/behavior_version_stratification/multi_version_prompt_fraction",
            "ggpo_theory/behavior_version_stratification/majority_valid_tokens_newest_tie_break",
            "ggpo_theory/measurement/pre_update",
            "ggpo_theory/formula/q_rho_version",
            "ggpo_theory/formula/j_h_identity_version",
            "ggpo_theory/config/beta",
            "ggpo_theory/config/group_size",
            "ggpo_theory/identity/actor_update_step",
            "ggpo_theory/identity/pre_update_training_policy_version",
            "ggpo_theory/identity/behavior_policy_version_available",
            "ggpo_theory/behavior/log_prob_source_is_rollout",
            "ggpo_theory/behavior/response_count",
            "ggpo_theory/behavior/prompt_count",
            "ggpo_theory/behavior/valid_response_token_count",
            "ggpo_theory/behavior/sequence_log_ratio_positive_fraction",
            "ggpo_theory/behavior/log_mean_importance_weight",
            "ggpo_theory/behavior/d2_log_second_moment_proxy",
            "ggpo_theory/behavior/is_ess_normalized",
            "ggpo_theory/behavior/is_max_normalized_weight",
            "ggpo_theory/reference/sequence_log_ratio_mean",
            "ggpo_theory/reference/sequence_log_ratio_abs_max",
            *distribution_names,
        }
    )


GGPO_THEORY_REQUIRED_METRICS = _ggpo_theory_required_metric_names()


def _exact_metric_integer(metrics: dict[str, float], name: str) -> int:
    value = metrics[name]
    if isinstance(value, bool):
        raise ValueError(f"GGPO theory metric {name} must be an exact integer")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric != float(int(numeric)):
        raise ValueError(f"GGPO theory metric {name} must be an exact integer")
    return int(numeric)


def validate_ggpo_theory_metrics(metrics: dict[str, float]) -> None:
    """Fail closed when a mandatory GGPO theory metric is absent or non-finite."""
    missing = GGPO_THEORY_REQUIRED_METRICS.difference(metrics)
    if missing:
        raise ValueError(f"GGPO theory metrics are missing required keys: {sorted(missing)}")
    nonfinite = [
        name
        for name in GGPO_THEORY_REQUIRED_METRICS
        if not math.isfinite(float(metrics[name]))
    ]
    if nonfinite:
        raise FloatingPointError(f"GGPO theory metrics contain non-finite values: {sorted(nonfinite)}")
    if (
        _exact_metric_integer(metrics, "ggpo_theory/schema_version")
        != GGPO_THEORY_METRICS_SCHEMA_VERSION
    ):
        raise ValueError("GGPO theory metric schema version does not match the implementation")
    if (
        _exact_metric_integer(metrics, "ggpo_theory/run_receipt/schema_version")
        != GGPO_THEORY_RUN_RECEIPT_SCHEMA_VERSION
    ):
        raise ValueError("GGPO theory run-receipt schema version does not match the implementation")
    receipt_digest = _exact_metric_integer(
        metrics,
        "ggpo_theory/run_receipt/digest_u52",
    )
    if receipt_digest < 0 or receipt_digest > (1 << 52) - 1:
        raise ValueError("GGPO theory run receipt digest must be an exact unsigned 52-bit integer")
    if (
        _exact_metric_integer(
            metrics,
            "ggpo_theory/probability_convention/schema_version",
        )
        != GGPO_THEORY_PROBABILITY_CONVENTION_SCHEMA_VERSION
    ):
        raise ValueError("GGPO theory probability-convention schema does not match the implementation")
    if (
        _exact_metric_integer(
            metrics,
            "ggpo_theory/behavior_version_stratification/schema_version",
        )
        != GGPO_THEORY_BEHAVIOR_VERSION_STRATIFICATION_SCHEMA_VERSION
    ):
        raise ValueError("GGPO behavior-version stratification schema does not match")
    stratification_enabled = metrics[
        "ggpo_theory/behavior_version_stratification/enabled"
    ]
    if stratification_enabled not in (0.0, 1.0):
        raise ValueError("GGPO behavior-version stratification enabled flag must be 0 or 1")
    if stratification_enabled:
        owner_version_count = _exact_metric_integer(
            metrics,
            "ggpo_theory/behavior_version_stratification/group_owner_version_count",
        )
        if owner_version_count < 1:
            raise ValueError("GGPO behavior-version stratification requires a group owner")
        attribution_flag = metrics[
            "ggpo_theory/behavior_version_stratification/majority_valid_tokens_newest_tie_break"
        ]
        if attribution_flag != 1.0:
            raise ValueError("GGPO behavior-version attribution rule must be majority-token ownership")
        multi_version_prompt_count = _exact_metric_integer(
            metrics,
            "ggpo_theory/behavior_version_stratification/multi_version_prompt_count",
        )
        prompt_count = _exact_metric_integer(metrics, "ggpo_theory/behavior/prompt_count")
        if multi_version_prompt_count < 0 or multi_version_prompt_count > prompt_count:
            raise ValueError("GGPO multi-version prompt count is outside the logical batch")
        expected_multi_fraction = multi_version_prompt_count / prompt_count
        if (
            metrics["ggpo_theory/behavior_version_stratification/multi_version_prompt_fraction"]
            != expected_multi_fraction
        ):
            raise ValueError("GGPO multi-version prompt fraction does not match its count")

        dynamic_prefix = "ggpo_theory/group_majority_token_owner_stratum/version_"
        emitted_strata = {name for name in metrics if name.startswith(dynamic_prefix)}
        owner_versions: set[int] = set()
        for name in emitted_strata:
            version_text, separator, _ = name.removeprefix(dynamic_prefix).partition("/")
            if not separator or not version_text.isdigit():
                raise ValueError(f"GGPO behavior-version stratum has an invalid dynamic key: {name}")
            owner_versions.add(int(version_text))
        if len(owner_versions) != owner_version_count:
            raise ValueError("GGPO dynamic behavior-version stratum count does not match its metadata")
        stratum_names = {
            f"{dynamic_prefix}{version}/{field}"
            for version in owner_versions
            for field in _BEHAVIOR_VERSION_STRATUM_FIELDS
        }
        missing_strata = stratum_names.difference(metrics)
        if missing_strata:
            raise ValueError(
                f"GGPO behavior-version stratification is missing active keys: {sorted(missing_strata)}"
            )
        unexpected_strata = emitted_strata.difference(stratum_names)
        if unexpected_strata:
            raise ValueError(
                "GGPO behavior-version stratification emitted inactive or unknown keys: "
                f"{sorted(unexpected_strata)}"
            )
        nonfinite_strata = [
            name for name in stratum_names if not math.isfinite(float(metrics[name]))
        ]
        if nonfinite_strata:
            raise FloatingPointError(
                f"GGPO behavior-version strata contain non-finite values: {sorted(nonfinite_strata)}"
            )
        stratum_response_count = 0.0
        stratum_prompt_count = 0.0
        configured_group_size = _exact_metric_integer(
            metrics,
            "ggpo_theory/config/group_size",
        )
        training_policy_version = _exact_metric_integer(
            metrics,
            "ggpo_theory/identity/pre_update_training_policy_version",
        )
        for owner_version in sorted(owner_versions, reverse=True):
            prefix = f"{dynamic_prefix}{owner_version}/"
            version = _exact_metric_integer(
                metrics,
                f"{prefix}majority_token_owner_behavior_policy_version",
            )
            lag = _exact_metric_integer(
                metrics,
                f"{prefix}training_behavior_version_lag",
            )
            response_count = _exact_metric_integer(
                metrics,
                f"{prefix}response_count",
            )
            prompt_count = _exact_metric_integer(
                metrics,
                f"{prefix}prompt_count",
            )
            if version < 0 or lag < 0 or response_count <= 0 or prompt_count <= 0:
                raise ValueError(
                    "GGPO active behavior-version strata require non-negative "
                    "identities and positive counts"
                )
            if lag != training_policy_version - version:
                raise ValueError(
                    "GGPO behavior-version stratum lag does not match its bound "
                    "training and behavior policy versions"
                )
            if version != owner_version:
                raise ValueError("GGPO behavior-version dynamic key does not match its owner metric")
            if response_count != prompt_count * configured_group_size:
                raise ValueError(
                    "GGPO behavior-version stratum response count must contain "
                    "complete prompt groups"
                )
            stratum_response_count += response_count
            stratum_prompt_count += prompt_count
        if stratum_response_count != metrics["ggpo_theory/behavior/response_count"]:
            raise ValueError(
                "GGPO behavior-version strata do not cover every real response"
            )
        if stratum_prompt_count != metrics["ggpo_theory/behavior/prompt_count"]:
            raise ValueError(
                "GGPO behavior-version strata do not cover every prompt group"
            )
    elif any(
        name.startswith("ggpo_theory/group_majority_token_owner_stratum/version_")
        for name in metrics
    ):
        raise ValueError(
            "Synchronous GGPO theory records must not emit behavior-version strata"
        )
    if metrics["ggpo_theory/behavior/response_count"] <= 0:
        raise ValueError("GGPO theory metrics contain no real behavior responses")
    actor_update_step = _exact_metric_integer(
        metrics,
        "ggpo_theory/identity/actor_update_step",
    )
    training_policy_version = _exact_metric_integer(
        metrics,
        "ggpo_theory/identity/pre_update_training_policy_version",
    )
    if actor_update_step < 1:
        raise ValueError("GGPO theory actor_update_step must be a positive integer")
    if training_policy_version < 0:
        raise ValueError("GGPO theory pre-update training-policy version must be non-negative")

    behavior_version_available = metrics[
        "ggpo_theory/identity/behavior_policy_version_available"
    ]
    if behavior_version_available not in (0.0, 1.0):
        raise ValueError("GGPO theory behavior-policy version availability must be 0 or 1")
    if behavior_version_available:
        version_metric_names = {
            "ggpo_theory/identity/behavior_rollout_version_min",
            "ggpo_theory/identity/behavior_rollout_version_max",
            "ggpo_theory/identity/behavior_rollout_version_diversity",
            "ggpo_theory/identity/training_behavior_version_lag_min",
            "ggpo_theory/identity/training_behavior_version_lag_max",
            "ggpo_theory/identity/training_behavior_version_lag_mean",
            "ggpo_theory/identity/training_behavior_version_lag_p95",
        }
        missing_versions = version_metric_names.difference(metrics)
        if missing_versions:
            raise ValueError(
                "GGPO theory metrics are missing available behavior-policy versions: "
                f"{sorted(missing_versions)}"
            )
        nonfinite_versions = [
            name for name in version_metric_names if not math.isfinite(float(metrics[name]))
        ]
        if nonfinite_versions:
            raise FloatingPointError(
                "GGPO theory behavior-policy version metrics are non-finite: "
                f"{sorted(nonfinite_versions)}"
            )


def compute_ggpo_theory_metrics(
    training_log_prob: torch.Tensor,
    behavior_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    beta: float,
    group_size: int,
    actor_update_step: int,
    training_policy_version: int,
    run_receipt_schema_version: int,
    run_receipt_digest_u52: int,
    behavior_policy_versions: torch.Tensor | None = None,
    behavior_version_stratification: bool = False,
    real_sample_mask: torch.Tensor | None = None,
    reward_kl: bool = True,
) -> dict[str, float]:
    """Compute the mandatory finite-batch GGPO theory diagnostics.

    All density ratios and adjusted rewards use the raw complete-response
    token-log-probability sum. This definition is intentionally independent of
    the reduction used by the training loss. ``behavior_log_prob`` must be the
    log probability recorded by the policy that generated each response (the
    canonical ``rollout_log_probs`` field), never PPO's proximal
    ``old_log_probs``.

    The caller invokes this once on the complete logical batch immediately
    before one externally logged actor-update step. It is intentionally not a
    record for each engine-internal mini/microbatch optimizer substep.

    ``behavior_policy_versions``, when available, may contain one rollout-policy
    version per input response or one per already-selected real response. The
    same ``real_sample_mask`` is applied when full-batch versions are supplied.
    Immutable string identities remain bound through the sealed run receipt
    rather than coerced into scalars. ``run_receipt_digest_u52`` is an exact
    tracker-safe prefix of that receipt's SHA-256.

    Fully asynchronous callers enable ``behavior_version_stratification``.
    Each complete prompt group is attributed exactly once to the behavior
    version owning the majority of its valid response tokens, with the newest
    version winning a tie. Dynamic version-keyed records have no fixed-width
    cap. Sequence-ratio and nonlinear J/q statistics use the same non-
    overlapping complete groups; token-specific mismatch remains in separate lag
    buckets.
    """
    if isinstance(beta, bool) or not math.isfinite(beta) or beta <= 0:
        raise ValueError(f"GGPO theory metrics require positive finite beta, got {beta}")
    if isinstance(group_size, bool):
        raise ValueError(f"GGPO theory metrics require integer group_size > 1, got {group_size}")
    try:
        group_size = operator.index(group_size)
    except TypeError as exc:
        raise ValueError(
            f"GGPO theory metrics require integer group_size > 1, got {group_size}"
        ) from exc
    if group_size <= 1:
        raise ValueError(f"GGPO theory metrics require group_size > 1, got {group_size}")
    integer_inputs = (
        actor_update_step,
        training_policy_version,
        run_receipt_schema_version,
        run_receipt_digest_u52,
    )
    if any(isinstance(value, bool) for value in integer_inputs):
        raise ValueError(
            "GGPO theory update step, policy version, and run receipt binding must be integers"
        )
    try:
        actor_update_step = operator.index(actor_update_step)
        training_policy_version = operator.index(training_policy_version)
        run_receipt_schema_version = operator.index(run_receipt_schema_version)
        run_receipt_digest_u52 = operator.index(run_receipt_digest_u52)
    except TypeError as exc:
        raise ValueError(
            "GGPO theory update step, policy version, and run receipt binding must be integers"
        ) from exc
    if actor_update_step < 1:
        raise ValueError("GGPO theory actor_update_step must be positive")
    if training_policy_version < 0:
        raise ValueError("GGPO theory training_policy_version must be non-negative")
    if run_receipt_schema_version != GGPO_THEORY_RUN_RECEIPT_SCHEMA_VERSION:
        raise ValueError(
            "GGPO theory run receipt schema does not match the metrics implementation"
        )
    if run_receipt_digest_u52 < 0 or run_receipt_digest_u52 > (1 << 52) - 1:
        raise ValueError("GGPO theory run receipt digest must be an unsigned 52-bit integer")
    if not isinstance(behavior_version_stratification, bool):
        raise ValueError("GGPO behavior_version_stratification must be boolean")
    if behavior_version_stratification and behavior_policy_versions is None:
        raise ValueError(
            "GGPO behavior-version stratification requires behavior_policy_versions"
        )
    if training_log_prob.ndim != 2:
        raise ValueError("GGPO theory log probabilities must have shape [responses, response_tokens]")
    if (
        behavior_log_prob.shape != training_log_prob.shape
        or ref_log_prob.shape != training_log_prob.shape
        or response_mask.shape != training_log_prob.shape
    ):
        raise ValueError("GGPO theory log probabilities and response_mask must have identical shapes")
    if (
        behavior_log_prob.device != training_log_prob.device
        or ref_log_prob.device != training_log_prob.device
        or response_mask.device != training_log_prob.device
        or rewards.device != training_log_prob.device
    ):
        raise ValueError("GGPO theory tensors must be colocated on one device")
    response_count = training_log_prob.shape[0]
    if group_ids.shape != (response_count,):
        raise ValueError("GGPO theory group_ids must contain one ID per response")
    if rewards.ndim == 2:
        if rewards.shape != response_mask.shape:
            raise ValueError("token-level GGPO theory rewards must match response_mask")
    elif rewards.shape != (response_count,):
        raise ValueError("GGPO theory rewards must be sequence-level or token-level")

    if real_sample_mask is None:
        real_sample_mask = torch.ones(
            response_count,
            dtype=torch.bool,
            device=training_log_prob.device,
        )
    if real_sample_mask.shape != (response_count,) or real_sample_mask.dtype != torch.bool:
        raise ValueError("GGPO theory real_sample_mask must be a boolean value per response")
    real_sample_mask = real_sample_mask.detach().to(device=training_log_prob.device)
    if not real_sample_mask.any():
        raise ValueError("GGPO theory metrics require at least one real response")

    with torch.no_grad():
        training = training_log_prob.detach()[real_sample_mask].double()
        behavior = behavior_log_prob.detach()[real_sample_mask].double()
        reference = ref_log_prob.detach()[real_sample_mask].double()
        mask = response_mask.detach()[real_sample_mask].bool()
        ids = group_ids.detach().to(device=training_log_prob.device)[real_sample_mask]
        selected_rewards = rewards.detach()[real_sample_mask].double()

        if not mask.any(dim=-1).all():
            raise ValueError("every real GGPO theory response must contain at least one valid token")
        for name, value in (
            ("training", training),
            ("behavior", behavior),
            ("reference", reference),
        ):
            if not torch.isfinite(value[mask]).all():
                raise ValueError(f"GGPO theory {name} log probabilities must be finite on valid tokens")
        if not torch.isfinite(selected_rewards).all():
            raise ValueError("GGPO theory rewards must be finite")
        if ids.dtype == torch.bool or ids.is_complex():
            raise ValueError("GGPO theory group IDs must be real integer values")
        if ids.is_floating_point() and (
            not torch.isfinite(ids).all() or not torch.all(ids == ids.trunc())
        ):
            raise ValueError("GGPO theory group IDs must be finite integers")

        ids = ids.long()
        _, counts = torch.unique(ids, return_counts=True)
        if counts.numel() == 0 or not torch.all(counts == group_size):
            raise ValueError(f"every GGPO theory prompt must contain exactly group_size={group_size} responses")
        order = torch.argsort(ids, stable=True)
        prompt_count = counts.numel()

        identity_metrics: dict[str, float] = {
            "ggpo_theory/identity/actor_update_step": float(actor_update_step),
            "ggpo_theory/identity/pre_update_training_policy_version": float(
                training_policy_version
            ),
            "ggpo_theory/identity/behavior_policy_version_available": float(
                behavior_policy_versions is not None
            ),
        }
        versions = None
        token_versions = None
        identity_versions = None
        if behavior_policy_versions is not None:
            versions = behavior_policy_versions.detach()
            if versions.ndim == 2:
                if versions.shape != response_mask.shape:
                    raise ValueError("token behavior-policy versions must match response_mask")
                token_versions = versions[real_sample_mask.to(device=versions.device)]
                selected_version_mask = response_mask.detach()[real_sample_mask.to(device=response_mask.device)].bool()
                if versions.dtype == torch.bool or versions.is_complex():
                    raise ValueError("GGPO theory behavior-policy versions must be real integers")
                if versions.is_floating_point() and (
                    not torch.isfinite(versions).all() or not torch.all(versions == versions.trunc())
                ):
                    raise ValueError("GGPO theory behavior-policy versions must be finite integers")
                token_versions = token_versions.to(device=training.device, dtype=torch.int64)
                selected_version_mask = selected_version_mask.to(device=training.device)
                if torch.any(token_versions[selected_version_mask] < 0):
                    raise ValueError("valid token behavior-policy versions must be non-negative")
                if torch.any(token_versions[~selected_version_mask] != -1):
                    raise ValueError("token behavior-policy version padding must be -1")
                identity_versions = token_versions[selected_version_mask]
                versions = token_versions.max(dim=-1).values
            elif versions.ndim != 1:
                raise ValueError("GGPO theory behavior_policy_versions must be one- or two-dimensional")
            elif versions.numel() == response_count:
                versions = versions[
                    real_sample_mask.to(device=versions.device)
                ]
            elif versions.numel() != training.shape[0]:
                raise ValueError(
                    "GGPO theory behavior_policy_versions must contain one value per "
                    "input response or per selected real response"
                )
            if versions.dtype == torch.bool:
                raise ValueError("GGPO theory behavior-policy versions must be integer values")
            if versions.is_complex():
                raise ValueError("GGPO theory behavior-policy versions must be real integers")
            if versions.is_floating_point() and (
                not torch.isfinite(versions).all()
                or not torch.all(versions == versions.trunc())
            ):
                raise ValueError("GGPO theory behavior-policy versions must be finite integers")
            versions = versions.to(device=training.device, dtype=torch.int64)
            if torch.any(versions < 0):
                raise ValueError("GGPO theory behavior-policy versions must be non-negative")
            if identity_versions is None:
                identity_versions = versions
            version_lag = training_policy_version - identity_versions
            if torch.any(version_lag < 0):
                raise ValueError("GGPO theory behavior-policy version cannot be from the future")
            version_lag_p95 = torch.quantile(version_lag.double(), 0.95)
            identity_metrics.update(
                {
                    "ggpo_theory/identity/behavior_rollout_version_min": float(
                        identity_versions.min().item()
                    ),
                    "ggpo_theory/identity/behavior_rollout_version_max": float(
                        identity_versions.max().item()
                    ),
                    "ggpo_theory/identity/behavior_rollout_version_diversity": float(
                        torch.unique(identity_versions).numel()
                    ),
                    "ggpo_theory/identity/training_behavior_version_lag_min": float(
                        version_lag.min().item()
                    ),
                    "ggpo_theory/identity/training_behavior_version_lag_max": float(
                        version_lag.max().item()
                    ),
                    "ggpo_theory/identity/training_behavior_version_lag_mean": float(
                        version_lag.double().mean().item()
                    ),
                    "ggpo_theory/identity/training_behavior_version_lag_p95": float(
                        version_lag_p95.item()
                    ),
                }
            )

        token_training_behavior = training - behavior
        sequence_log_ratio = torch.where(
            mask,
            token_training_behavior,
            torch.zeros_like(token_training_behavior),
        ).sum(dim=-1)
        grouped_sequence_log_ratio = sequence_log_ratio[order].reshape(-1, group_size)
        prompt_sample_max = grouped_sequence_log_ratio.max(dim=-1).values

        log_n = math.log(sequence_log_ratio.numel())
        log_weight_sum = torch.logsumexp(sequence_log_ratio, dim=0)
        log_weight_square_sum = torch.logsumexp(2.0 * sequence_log_ratio, dim=0)
        log_ess_normalized = 2.0 * log_weight_sum - log_weight_square_sum - log_n
        # Exact arithmetic gives <= 0. Clamp only the tiny positive round-off.
        log_ess_normalized = torch.clamp(log_ess_normalized, max=0.0)
        maximum = sequence_log_ratio.max()

        token_training_reference = training - reference
        sequence_reference_log_ratio = torch.where(
            mask,
            token_training_reference,
            torch.zeros_like(token_training_reference),
        ).sum(dim=-1)
        if selected_rewards.ndim == 2:
            sequence_rewards = torch.where(
                mask,
                selected_rewards,
                torch.zeros_like(selected_rewards),
            ).sum(dim=-1)
        else:
            sequence_rewards = selected_rewards

        adjusted_reward = sequence_rewards
        if reward_kl:
            adjusted_reward = sequence_rewards - beta * sequence_reference_log_ratio
        grouped_adjusted_reward = adjusted_reward[order].reshape(-1, group_size)
        scaled = grouped_adjusted_reward / beta
        eta = beta * (torch.logsumexp(scaled, dim=-1) - math.log(group_size))
        j_behavior = eta - grouped_adjusted_reward.mean(dim=-1)
        z = scaled - eta.unsqueeze(-1) / beta
        q = torch.softmax(scaled, dim=-1)
        q_entropy = -(q * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum(dim=-1)
        q_max = q.max(dim=-1).values
        q_ess = q.square().sum(dim=-1).reciprocal()
        q_ess_normalized = q_ess / group_size
        # rho is the empirical behavior measure over the n exchangeable
        # behavior samples in this prompt group: rho_i = 1/n.
        q_l1_rho = (q - 1.0 / group_size).abs().sum(dim=-1)
        reward_variance = grouped_adjusted_reward.var(dim=-1, unbiased=False)

        group_distributions = {
            "ggpo_theory/group/eta_beh": eta,
            "ggpo_theory/group/j_beh": j_behavior,
            "ggpo_theory/group/z_abs_max": z.abs().max(dim=-1).values,
            "ggpo_theory/group/q_entropy": q_entropy,
            "ggpo_theory/group/q_max": q_max,
            "ggpo_theory/group/q_ess_normalized": q_ess_normalized,
            "ggpo_theory/group/q_l1_rho": q_l1_rho,
            "ggpo_theory/group/r_variance": reward_variance,
        }
        group_owner_versions = None
        multi_version_prompt_count = 0
        if behavior_version_stratification:
            assert versions is not None
            grouped_order = order.reshape(-1, group_size)
            group_owner_versions, multi_version_prompt_count = _group_majority_version_owners(
                response_versions=versions,
                token_versions=token_versions,
                response_mask=mask,
                grouped_order=grouped_order,
            )

        metrics: dict[str, float] = {
            "ggpo_theory/schema_version": float(GGPO_THEORY_METRICS_SCHEMA_VERSION),
            "ggpo_theory/run_receipt/schema_version": float(
                run_receipt_schema_version
            ),
            "ggpo_theory/run_receipt/digest_u52": float(run_receipt_digest_u52),
            "ggpo_theory/probability_convention/schema_version": float(
                GGPO_THEORY_PROBABILITY_CONVENTION_SCHEMA_VERSION
            ),
            "ggpo_theory/behavior_version_stratification/schema_version": float(
                GGPO_THEORY_BEHAVIOR_VERSION_STRATIFICATION_SCHEMA_VERSION
            ),
            "ggpo_theory/behavior_version_stratification/enabled": float(
                behavior_version_stratification
            ),
            "ggpo_theory/behavior_version_stratification/group_owner_version_count": float(
                torch.unique(group_owner_versions).numel()
                if group_owner_versions is not None
                else 0
            ),
            "ggpo_theory/behavior_version_stratification/multi_version_prompt_count": float(
                multi_version_prompt_count
            ),
            "ggpo_theory/behavior_version_stratification/multi_version_prompt_fraction": float(
                multi_version_prompt_count / prompt_count if behavior_version_stratification else 0.0
            ),
            "ggpo_theory/behavior_version_stratification/majority_valid_tokens_newest_tie_break": float(
                behavior_version_stratification
            ),
            "ggpo_theory/measurement/pre_update": 1.0,
            "ggpo_theory/formula/q_rho_version": float(GGPO_THEORY_Q_RHO_FORMULA_VERSION),
            "ggpo_theory/formula/j_h_identity_version": float(GGPO_THEORY_J_H_FORMULA_VERSION),
            "ggpo_theory/config/beta": float(beta),
            "ggpo_theory/config/group_size": float(group_size),
            "ggpo_theory/behavior/log_prob_source_is_rollout": 1.0,
            "ggpo_theory/behavior/response_count": float(sequence_log_ratio.numel()),
            "ggpo_theory/behavior/prompt_count": float(prompt_count),
            "ggpo_theory/behavior/valid_response_token_count": float(mask.sum().item()),
            "ggpo_theory/behavior/sequence_log_ratio_positive_fraction": float(
                (sequence_log_ratio > 0).double().mean().item()
            ),
            "ggpo_theory/behavior/log_mean_importance_weight": float(
                (log_weight_sum - log_n).item()
            ),
            "ggpo_theory/behavior/d2_log_second_moment_proxy": float(
                (log_weight_square_sum - log_n).item()
            ),
            "ggpo_theory/behavior/is_ess_normalized": float(torch.exp(log_ess_normalized).item()),
            "ggpo_theory/behavior/is_max_normalized_weight": float(
                torch.exp(maximum - log_weight_sum).item()
            ),
            "ggpo_theory/reference/sequence_log_ratio_mean": float(
                sequence_reference_log_ratio.mean().item()
            ),
            "ggpo_theory/reference/sequence_log_ratio_abs_max": float(
                sequence_reference_log_ratio.abs().max().item()
            ),
            **identity_metrics,
        }
        distributions = {
            "ggpo_theory/behavior/sequence_log_ratio": sequence_log_ratio,
            "ggpo_theory/behavior/prompt_sample_log_ratio_max": prompt_sample_max,
            "ggpo_theory/response/adjusted_reward": adjusted_reward,
            **group_distributions,
        }
        if distributions.keys() != _CORE_DISTRIBUTION_FIELDS.keys():
            raise AssertionError("GGPO theory core distribution schema and values differ")
        for prefix, fields in _CORE_DISTRIBUTION_FIELDS.items():
            metrics.update(
                _theory_distribution_metrics(
                    distributions[prefix],
                    prefix,
                    fields,
                )
            )
        if behavior_version_stratification:
            assert group_owner_versions is not None
            grouped_order = order.reshape(-1, group_size)
            unique_owner_versions = torch.sort(
                torch.unique(group_owner_versions),
                descending=True,
            ).values
            for behavior_version_tensor in unique_owner_versions:
                behavior_version = int(behavior_version_tensor.item())
                group_selector = group_owner_versions == behavior_version
                response_indices = grouped_order[group_selector].reshape(-1)
                response_count_for_version = int(response_indices.numel())
                prompt_count_for_version = int(group_selector.sum().item())
                if response_count_for_version != prompt_count_for_version * group_size:
                    raise ValueError(
                        "Fully async GGPO group-owner stratum does not contain complete prompt groups."
                    )
                summary = _behavior_version_stratum_summary(
                    sequence_log_ratio=sequence_log_ratio[response_indices],
                    j_behavior=j_behavior[group_selector],
                    q_entropy=q_entropy[group_selector],
                    q_max=q_max[group_selector],
                    q_ess_normalized=q_ess_normalized[group_selector],
                    q_l1_rho=q_l1_rho[group_selector],
                )
                prefix = f"ggpo_theory/group_majority_token_owner_stratum/version_{behavior_version}/"
                metrics.update(
                    {
                        f"{prefix}majority_token_owner_behavior_policy_version": float(
                            behavior_version
                        ),
                        f"{prefix}training_behavior_version_lag": float(
                            training_policy_version - behavior_version
                        ),
                        f"{prefix}response_count": float(response_count_for_version),
                        f"{prefix}prompt_count": float(prompt_count_for_version),
                        **{
                            f"{prefix}{field}": value
                            for field, value in summary.items()
                        },
                    }
                )
        validate_ggpo_theory_metrics(metrics)
        return metrics
