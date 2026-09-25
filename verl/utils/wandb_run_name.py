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

"""Canonical W&B display names for the Gibbs-Gap Policy Optimization project."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

GGPO_PROJECT = "Gibbs-Gap Policy Optimization"
SHORT_MAXSTEP_THRESHOLD = 100

_MISSING = object()
_NAME_TOKEN_RE = re.compile(r"[a-z0-9.]+(?:-[a-z0-9.]+)*")
_HF_CACHE_MODEL_RE = re.compile(r"(?:^|/)models--[^/]+--([^/]+)(?:/|$)")


def _select(config: Mapping[str, Any], path: str, default: Any = _MISSING) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            if default is _MISSING:
                raise ValueError(f"Cannot build canonical W&B run name: missing config field {path!r}.")
            return default
        value = value[key]
    return value


def _format_number(value: Any, *, field: str, allow_zero: bool = False) -> str:
    if isinstance(value, bool):
        raise ValueError(f"Cannot build canonical W&B run name: {field} must be numeric, got {value!r}.")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Cannot build canonical W&B run name: {field} must be numeric, got {value!r}.") from exc
    if not number.is_finite() or number < 0 or (number == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"Cannot build canonical W&B run name: {field} must be finite and {qualifier}, got {value!r}.")
    rendered = format(number.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _model_slug(model_path: Any) -> str:
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("Cannot build canonical W&B run name: actor_rollout_ref.model.path must be non-empty.")
    normalized = model_path.strip().rstrip("/")
    cache_match = _HF_CACHE_MODEL_RE.search(normalized)
    model_name = cache_match.group(1) if cache_match else normalized.rsplit("/", 1)[-1]
    slug = re.sub(r"[^a-z0-9.]+", "-", model_name.lower()).strip("-")
    if not slug or not _NAME_TOKEN_RE.fullmatch(slug):
        raise ValueError(f"Cannot build canonical W&B run name: unsupported model path {model_path!r}.")
    return slug


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Cannot build canonical W&B run name: {field} must be a positive integer, got {value!r}.")
    return value


def _contains_dapo_filter(config: Mapping[str, Any]) -> bool:
    filter_groups = _select(config, "algorithm.filter_groups", default=None)
    if isinstance(filter_groups, Mapping) and filter_groups.get("enable") is True:
        return True
    train_files = _select(config, "data.train_files", default=())
    if isinstance(train_files, str):
        candidates = (train_files,)
    elif isinstance(train_files, Sequence):
        candidates = tuple(str(item) for item in train_files)
    else:
        candidates = (str(train_files),)
    return any("dapo_filter" in path.lower() or "dapo-filter" in path.lower() for path in candidates)


def _extra_settings(config: Mapping[str, Any]) -> list[str]:
    configured = _select(config, "trainer.wandb_name_extra_settings", default=())
    if configured is None:
        return []
    if isinstance(configured, str):
        values = (configured,)
    elif isinstance(configured, Sequence):
        values = configured
    else:
        raise ValueError(
            "Cannot build canonical W&B run name: trainer.wandb_name_extra_settings "
            "must be a string or a list of strings."
        )

    settings = []
    for value in values:
        if not isinstance(value, str) or not _NAME_TOKEN_RE.fullmatch(value):
            raise ValueError(
                "Cannot build canonical W&B run name: every extra setting must use lowercase "
                f"ASCII letters, digits, dots, and hyphens; got {value!r}."
            )
        settings.append(value)
    return settings


def canonical_ggpo_wandb_name(config: Mapping[str, Any]) -> str:
    """Build the canonical semantic W&B display name from a resolved trainer config."""

    if not isinstance(config, Mapping):
        raise ValueError("Cannot build canonical W&B run name: resolved config must be a mapping.")

    policy_loss_path = "actor_rollout_ref.actor.policy_loss"
    actor_path = "actor_rollout_ref.actor"
    loss_mode = _select(config, f"{policy_loss_path}.loss_mode")

    if loss_mode == "ggpo":
        reduction = _select(config, f"{policy_loss_path}.sequence_log_prob_reduction", default="sum")
        if reduction not in {"mean", "sum"}:
            raise ValueError(
                "Cannot build canonical W&B run name: GGPO sequence reduction must be 'mean' or 'sum', "
                f"got {reduction!r}."
            )
        objective_token = f"ggpo-exp-seq{reduction}"
        beta = _format_number(_select(config, f"{policy_loss_path}.beta"), field="GGPO beta")
        core_parameter = f"beta{beta}"
    elif loss_mode == "agro":
        reduction = _select(config, f"{policy_loss_path}.sequence_log_prob_reduction", default="sum")
        if reduction != "sum":
            raise ValueError("Cannot build canonical W&B run name: AGRO requires sequence sum reduction.")
        beta = _format_number(_select(config, f"{policy_loss_path}.beta"), field="AGRO beta")
        objective_token = "agro-seqsum"
        core_parameter = f"beta{beta}"
    elif loss_mode == "asympo":
        objective_token = "asympo"
        core_parameter = "core"
    else:
        advantage_estimator = _select(config, "algorithm.adv_estimator")
        if loss_mode != "vanilla" or advantage_estimator != "grpo":
            raise ValueError(
                "Cannot build canonical W&B run name: Gibbs-Gap Policy Optimization runs must use "
                "GGPO, AGRO, ASymPO, or vanilla GRPO policy loss."
            )
        objective_token = "grpo"
        use_kl_loss = _select(config, f"{actor_path}.use_kl_loss")
        use_kl_in_reward = _select(config, "algorithm.use_kl_in_reward", default=False)
        if use_kl_loss and use_kl_in_reward:
            raise ValueError(
                "Cannot build canonical W&B run name: GRPO cannot enable both actor-loss KL "
                "and reward-side KL because the single kl token would be ambiguous."
            )
        if use_kl_loss:
            kl_coef = _select(config, f"{actor_path}.kl_loss_coef")
        elif use_kl_in_reward:
            kl_coef = _select(config, "algorithm.kl_ctrl.kl_coef")
        else:
            kl_coef = 0
        core_parameter = f"kl{_format_number(kl_coef, field='GRPO KL coefficient', allow_zero=True)}"

    name_parts = [
        objective_token,
        core_parameter,
        _model_slug(_select(config, "actor_rollout_ref.model.path")),
    ]

    if _contains_dapo_filter(config):
        name_parts.append("dapo-filter")

    if _select(config, "offline_data.fully_offline", default=False) is True:
        name_parts.append("offline")
    elif isinstance(_select(config, "async_training", default=None), Mapping):
        backend = _select(config, "async_training.rollout_backend", default="verl")
        if backend not in {"verl", "areal"}:
            raise ValueError(
                "Cannot build canonical W&B run name: async rollout backend must be 'verl' or 'areal', "
                f"got {backend!r}."
            )
        name_parts.append(f"async-{backend}")

    if _select(config, "algorithm.rollout_correction.bypass_mode", default=False) is True:
        name_parts.append("no-is")

    if _select(config, "trainer.ggpo_moving_reference.enabled", default=False) is True:
        name_parts.append("dynamic-ref")

    fixed_policy_lag = _select(config, "trainer.fixed_policy_lag", default=None)
    if fixed_policy_lag is not None:
        lag = _positive_int(fixed_policy_lag, field="trainer.fixed_policy_lag")
        name_parts.append(f"fixed-lag{lag}")

    for setting in _extra_settings(config):
        if setting not in name_parts:
            name_parts.append(setting)

    max_optimizer_steps = _select(config, "trainer.max_optimizer_steps", default=None)
    if max_optimizer_steps is not None:
        maxstep = _positive_int(max_optimizer_steps, field="trainer.max_optimizer_steps")
        name_parts.append(f"maxstep{maxstep}")
    else:
        total_training_steps = _select(config, "trainer.total_training_steps", default=None)
        if (
            isinstance(total_training_steps, int)
            and not isinstance(total_training_steps, bool)
            and 1 <= total_training_steps <= SHORT_MAXSTEP_THRESHOLD
        ):
            name_parts.append(f"maxstep{total_training_steps}")

    name = "-".join(name_parts)
    if not _NAME_TOKEN_RE.fullmatch(name):
        raise ValueError(f"Generated invalid canonical W&B run name {name!r}.")
    return name


def resolve_wandb_run_name(project_name: str, experiment_name: str, config: Mapping[str, Any] | None) -> str:
    """Resolve the target project's canonical name or an explicitly bound non-GGPO name."""

    if project_name != GGPO_PROJECT:
        return experiment_name
    if config is None:
        raise ValueError("Gibbs-Gap Policy Optimization W&B runs require a resolved config for canonical naming.")
    if _select(config, "actor_rollout_ref.actor.policy_loss.loss_mode") == "ggpo":
        return canonical_ggpo_wandb_name(config)
    display_name = _select(config, "trainer.wandb_display_name", default=None)
    if display_name is not None:
        if not isinstance(display_name, str) or not _NAME_TOKEN_RE.fullmatch(display_name):
            raise ValueError(
                "Gibbs-Gap Policy Optimization trainer.wandb_display_name must use lowercase ASCII letters, "
                f"digits, dots, and hyphens; got {display_name!r}."
            )
        if display_name != experiment_name:
            raise ValueError(
                "Gibbs-Gap Policy Optimization trainer.wandb_display_name must exactly match "
                "trainer.experiment_name."
            )
        return display_name
    return canonical_ggpo_wandb_name(config)
