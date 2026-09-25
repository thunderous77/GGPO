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

import pytest

from verl.utils.wandb_run_name import (
    canonical_ggpo_wandb_name,
    resolve_wandb_run_name,
)


def _config(*, loss_mode="ggpo"):
    beta = 0.001 if loss_mode == "agro" else 0.1
    reduction = "sum" if loss_mode == "agro" else "mean"
    return {
        "actor_rollout_ref": {
            "actor": {
                "policy_loss": {
                    "loss_mode": loss_mode,
                    "sequence_log_prob_reduction": reduction,
                    "beta": beta,
                },
                "use_kl_loss": loss_mode == "vanilla",
                "kl_loss_coef": 0.001,
            },
            "model": {"path": "Qwen/Qwen3-1.7B-Base"},
        },
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
            "rollout_correction": {"bypass_mode": False},
        },
        "data": {"train_files": ["/data/alp/train.parquet"]},
        "trainer": {"total_training_steps": None},
    }


def test_ggpo_name_contains_reduction_beta_and_model():
    assert canonical_ggpo_wandb_name(_config()) == "ggpo-exp-seqmean-beta0.1-qwen3-1.7b-base"


def test_ggpo_sum_normalizes_numeric_and_hf_cache_model():
    config = _config()
    config["actor_rollout_ref"]["actor"]["policy_loss"].update(
        sequence_log_prob_reduction="sum",
        beta="0.0100",
    )
    config["actor_rollout_ref"]["model"]["path"] = "/cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/012345"

    assert canonical_ggpo_wandb_name(config) == "ggpo-exp-seqsum-beta0.01-qwen2.5-7b-instruct"


def test_agro_and_asympo_have_exact_canonical_names():
    agro = _config(loss_mode="agro")
    asympo = _config(loss_mode="asympo")
    assert canonical_ggpo_wandb_name(agro) == "agro-seqsum-beta0.001-qwen3-1.7b-base"
    assert canonical_ggpo_wandb_name(asympo) == "asympo-core-qwen3-1.7b-base"


def test_agro_name_tracks_configurable_beta():
    config = _config(loss_mode="agro")
    config["actor_rollout_ref"]["actor"]["policy_loss"]["beta"] = 0.25

    assert canonical_ggpo_wandb_name(config) == "agro-seqsum-beta0.25-qwen3-1.7b-base"


def test_grpo_name_uses_enabled_actor_loss_kl():
    config = _config(loss_mode="vanilla")

    assert canonical_ggpo_wandb_name(config) == "grpo-kl0.001-qwen3-1.7b-base"


def test_grpo_name_uses_enabled_reward_kl():
    config = _config(loss_mode="vanilla")
    config["actor_rollout_ref"]["actor"]["use_kl_loss"] = False
    config["algorithm"].update(use_kl_in_reward=True, kl_ctrl={"kl_coef": 0.02})

    assert canonical_ggpo_wandb_name(config) == "grpo-kl0.02-qwen3-1.7b-base"


def test_grpo_without_active_kl_uses_kl_zero():
    config = _config(loss_mode="vanilla")
    config["actor_rollout_ref"]["actor"]["use_kl_loss"] = False

    assert canonical_ggpo_wandb_name(config) == "grpo-kl0-qwen3-1.7b-base"


def test_optional_settings_have_stable_order():
    config = _config()
    config["data"]["train_files"] = ["/data/dapo_filter/train.parquet"]
    config["async_training"] = {"rollout_backend": "areal"}
    config["algorithm"]["rollout_correction"]["bypass_mode"] = True
    config["trainer"].update(
        fixed_policy_lag=8,
        max_optimizer_steps=20,
        wandb_name_extra_settings=["tp2"],
    )
    config["trainer"]["ggpo_moving_reference"] = {"enabled": True, "update_freq": 20}

    assert canonical_ggpo_wandb_name(config) == (
        "ggpo-exp-seqmean-beta0.1-qwen3-1.7b-base-dapo-filter-async-areal-no-is-"
        "dynamic-ref-fixed-lag8-tp2-maxstep20"
    )


def test_enabled_dynamic_filter_marks_canonical_name_without_path_hint():
    config = _config()
    config["algorithm"]["filter_groups"] = {
        "enable": True,
        "metric": "raw_correctness",
        "max_num_gen_batches": 32,
    }

    assert canonical_ggpo_wandb_name(config) == (
        "ggpo-exp-seqmean-beta0.1-qwen3-1.7b-base-dapo-filter"
    )


def test_moving_reference_marks_the_canonical_name():
    """Without this token a moving-reference run is indistinguishable by name from a
    plain one, which is how the first such run came out misnamed."""
    config = _config()
    config["trainer"]["ggpo_moving_reference"] = {"enabled": True, "update_freq": 20}

    assert canonical_ggpo_wandb_name(config) == (
        "ggpo-exp-seqmean-beta0.1-qwen3-1.7b-base-dynamic-ref"
    )


def test_disabled_moving_reference_leaves_the_name_unchanged():
    config = _config()
    config["trainer"]["ggpo_moving_reference"] = {"enabled": False, "update_freq": None}

    assert canonical_ggpo_wandb_name(config) == "ggpo-exp-seqmean-beta0.1-qwen3-1.7b-base"


def test_fixed_policy_lag_marks_the_canonical_name():
    config = _config()
    config["trainer"]["fixed_policy_lag"] = 8

    assert canonical_ggpo_wandb_name(config) == (
        "ggpo-exp-seqmean-beta0.1-qwen3-1.7b-base-fixed-lag8"
    )


def test_null_fixed_policy_lag_leaves_the_name_unchanged():
    config = _config()
    config["trainer"]["fixed_policy_lag"] = None

    assert canonical_ggpo_wandb_name(config) == "ggpo-exp-seqmean-beta0.1-qwen3-1.7b-base"


def test_offline_takes_precedence_over_async_backend():
    config = _config()
    config["offline_data"] = {"fully_offline": True}
    config["async_training"] = {"rollout_backend": "verl"}

    assert canonical_ggpo_wandb_name(config) == ("ggpo-exp-seqmean-beta0.1-qwen3-1.7b-base-offline")


def test_short_total_training_steps_uses_maxstep():
    config = _config()
    config["trainer"]["total_training_steps"] = 100

    assert canonical_ggpo_wandb_name(config) == (
        "ggpo-exp-seqmean-beta0.1-qwen3-1.7b-base-maxstep100"
    )


def test_long_total_training_steps_does_not_use_maxstep():
    config = _config()
    config["trainer"]["total_training_steps"] = 101

    assert canonical_ggpo_wandb_name(config) == ("ggpo-exp-seqmean-beta0.1-qwen3-1.7b-base")


def test_target_project_fails_closed_on_ambiguous_grpo_kl():
    config = _config(loss_mode="vanilla")
    config["algorithm"].update(use_kl_in_reward=True, kl_ctrl={"kl_coef": 0.02})

    with pytest.raises(ValueError, match="both actor-loss KL and reward-side KL"):
        canonical_ggpo_wandb_name(config)


def test_resolver_preserves_other_projects():
    assert resolve_wandb_run_name("another-project", "manual-name", None) == "manual-name"


def test_resolver_enforces_target_project():
    config = _config()

    assert resolve_wandb_run_name("Gibbs-Gap Policy Optimization", "ignored-manual-name", config) == (
        "ggpo-exp-seqmean-beta0.1-qwen3-1.7b-base"
    )


def test_resolver_allows_an_explicitly_bound_target_project_display_name():
    config = _config(loss_mode="vanilla")
    name = "grpo-kl0.001-qwen3-1.7b-base-async"
    config["trainer"]["wandb_display_name"] = name

    assert resolve_wandb_run_name("Gibbs-Gap Policy Optimization", name, config) == name


def test_resolver_ignores_a_hand_written_ggpo_display_name():
    config = _config()
    config["trainer"]["wandb_display_name"] = "ggpo-exp-seqmean-beta0.1-gtm-hand-written"

    assert resolve_wandb_run_name("Gibbs-Gap Policy Optimization", "different-name", config) == (
        "ggpo-exp-seqmean-beta0.1-qwen3-1.7b-base"
    )


def test_resolver_rejects_a_non_ggpo_display_name_that_does_not_match_experiment_name():
    config = _config(loss_mode="vanilla")
    config["trainer"]["wandb_display_name"] = "grpo-kl0.001-qwen3-1.7b-base-async"

    with pytest.raises(ValueError, match="must exactly match"):
        resolve_wandb_run_name("Gibbs-Gap Policy Optimization", "different-name", config)
