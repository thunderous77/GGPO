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

import math
import stat
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ggpo_theory import (
    GGPO_THEORY_CURRENT_LOG_PROBS,
    GGPO_THEORY_J_H_FORMULA_VERSION,
    GGPO_THEORY_METRICS_SCHEMA_VERSION,
    GGPO_THEORY_METRICS_SENTINEL,
    GGPO_THEORY_PROBABILITY_CONVENTION_SCHEMA_VERSION,
    GGPO_THEORY_Q_RHO_FORMULA_VERSION,
    GGPO_THEORY_REQUIRED_METRICS,
    GGPO_THEORY_RUN_RECEIPT_SCHEMA_VERSION,
    build_ggpo_theory_run_receipt,
    compute_ggpo_theory_metrics,
    seal_ggpo_theory_run_receipt,
    validate_ggpo_theory_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.config import validate_ggpo_trainer_compatibility


_TEST_RUN_RECEIPT_DIGEST_U52 = 0x123456789ABCD


def _theory_inputs():
    behavior = torch.full((4, 3), -5.0)
    training = behavior + torch.tensor(
        [
            [1.0, 2.0, 100.0],
            [3.0, 0.0, -100.0],
            [-1.0, -2.0, 50.0],
            [2.0, 2.0, -50.0],
        ]
    )
    response_mask = torch.tensor([[1, 1, 0]] * 4, dtype=torch.bool)
    rewards = torch.tensor(
        [
            [0.0, 0.0, 77.0],
            [1.0, 0.0, 77.0],
            [2.0, 0.0, 77.0],
            [0.0, 0.0, 77.0],
        ]
    )
    return {
        "training_log_prob": training,
        "behavior_log_prob": behavior,
        # This isolates reward-side identities: the raw current/reference
        # complete-response log ratio is exactly zero.
        "ref_log_prob": training.clone(),
        "rewards": rewards,
        "response_mask": response_mask,
        "group_ids": torch.tensor([0, 0, 1, 1]),
        "beta": 0.5,
        "group_size": 2,
        "actor_update_step": 1,
        "training_policy_version": 0,
        "run_receipt_schema_version": GGPO_THEORY_RUN_RECEIPT_SCHEMA_VERSION,
        "run_receipt_digest_u52": _TEST_RUN_RECEIPT_DIGEST_U52,
    }


def test_theory_metrics_use_signed_raw_response_sum_and_locked_q_rho_definition():
    metrics = compute_ggpo_theory_metrics(**_theory_inputs())

    assert metrics.keys() == GGPO_THEORY_REQUIRED_METRICS
    assert len(metrics) == 58
    assert metrics["ggpo_theory/behavior/sequence_log_ratio_mean"] == pytest.approx(1.75)
    assert metrics["ggpo_theory/behavior/sequence_log_ratio_std"] == pytest.approx(
        torch.tensor([3.0, 3.0, -3.0, 4.0], dtype=torch.float64).std(
            unbiased=False
        ).item()
    )
    assert metrics["ggpo_theory/behavior/prompt_sample_log_ratio_max_mean"] == pytest.approx(3.5)
    assert metrics["ggpo_theory/behavior/prompt_sample_log_ratio_max_max"] == pytest.approx(4.0)
    assert metrics["ggpo_theory/response/adjusted_reward_max"] == pytest.approx(2.0)
    assert "ggpo_theory/behavior/reverse_kl_estimate" not in metrics
    assert "ggpo_theory/behavior/log10_cx_sample_lower_proxy" not in metrics
    assert "ggpo_theory/group/q_ess_unnormalized_mean" not in metrics
    assert "ggpo_theory/response/adjusted_reward_mean" not in metrics
    assert "ggpo_theory/group/j_beh_mean" not in metrics
    q_first = torch.softmax(torch.tensor([0.0, 1.0], dtype=torch.float64) / 0.5, dim=0)
    q_second = torch.softmax(torch.tensor([2.0, 0.0], dtype=torch.float64) / 0.5, dim=0)
    assert metrics["ggpo_theory/group/q_ess_normalized_mean"] == pytest.approx(
        torch.stack(
            [q.square().sum().reciprocal() / 2 for q in (q_first, q_second)]
        ).mean().item()
    )
    expected_l1 = torch.stack(
        [
            (q_first - 0.5).abs().sum(),
            (q_second - 0.5).abs().sum(),
        ]
    ).mean()
    assert metrics["ggpo_theory/group/q_l1_rho_mean"] == pytest.approx(expected_l1.item())
    assert metrics["ggpo_theory/formula/q_rho_version"] == GGPO_THEORY_Q_RHO_FORMULA_VERSION
    assert metrics["ggpo_theory/formula/j_h_identity_version"] == GGPO_THEORY_J_H_FORMULA_VERSION
    assert not any("j_identity_" in name for name in metrics)
    assert not any("locality_" in name for name in metrics)
    assert "ggpo_theory/group/q_max_max" not in metrics
    assert "ggpo_theory/group/q_ess_normalized_min" not in metrics
    assert "ggpo_theory/group/q_ess_normalized_p05" not in metrics
    assert "ggpo_theory/group/q_l1_rho_max" not in metrics
    assert metrics["ggpo_theory/measurement/pre_update"] == 1.0
    assert metrics["ggpo_theory/identity/actor_update_step"] == 1.0
    assert metrics["ggpo_theory/identity/pre_update_training_policy_version"] == 0.0
    assert metrics["ggpo_theory/identity/behavior_policy_version_available"] == 0.0


def test_j_h_algebraic_identity_holds_at_float64_precision():
    beta = 0.5
    grouped_adjusted_reward = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [2.0, 0.0]],
        dtype=torch.float64,
    )
    scaled = grouped_adjusted_reward / beta
    eta = beta * (torch.logsumexp(scaled, dim=-1) - math.log(2))
    j_behavior = eta - grouped_adjusted_reward.mean(dim=-1)
    z = scaled - eta.unsqueeze(-1) / beta
    j_from_h = beta * (torch.expm1(z) - z).mean(dim=-1)

    absolute_residual = (j_behavior - j_from_h).abs()
    identity_scale = torch.maximum(j_behavior.abs(), j_from_h.abs())
    relative_residual = torch.where(
        identity_scale > 0,
        absolute_residual / identity_scale,
        torch.zeros_like(absolute_residual),
    )
    zero_denominator_fraction = (identity_scale == 0).double().mean()
    tolerance = 8 * torch.finfo(torch.float64).eps

    assert absolute_residual.mean() <= tolerance
    assert absolute_residual.max() <= tolerance
    assert relative_residual.max() <= tolerance
    assert zero_denominator_fraction == pytest.approx(1 / 3)


def test_theory_metrics_bind_available_behavior_versions_and_lag():
    inputs = _theory_inputs()
    inputs["actor_update_step"] = 8
    inputs["training_policy_version"] = 7
    inputs["behavior_policy_versions"] = torch.tensor([7, 7, 5, 5])

    metrics = compute_ggpo_theory_metrics(**inputs)

    assert metrics["ggpo_theory/identity/behavior_policy_version_available"] == 1.0
    assert metrics["ggpo_theory/identity/behavior_rollout_version_min"] == 5.0
    assert metrics["ggpo_theory/identity/behavior_rollout_version_max"] == 7.0
    assert metrics["ggpo_theory/identity/behavior_rollout_version_diversity"] == 2.0
    assert metrics["ggpo_theory/identity/training_behavior_version_lag_min"] == 0.0
    assert metrics["ggpo_theory/identity/training_behavior_version_lag_max"] == 2.0
    assert metrics["ggpo_theory/identity/training_behavior_version_lag_mean"] == 1.0
    assert len(metrics) == 65


def test_fully_async_theory_metrics_emit_only_active_behavior_version_strata():
    inputs = _theory_inputs()
    inputs.update(
        {
            "actor_update_step": 8,
            "training_policy_version": 7,
            "behavior_policy_versions": torch.tensor([7, 7, 5, 5]),
            "behavior_version_stratification": True,
        }
    )

    metrics = compute_ggpo_theory_metrics(**inputs)

    assert metrics["ggpo_theory/behavior_version_stratification/enabled"] == 1.0
    assert metrics["ggpo_theory/behavior_version_stratification/group_owner_version_count"] == 2.0
    assert metrics["ggpo_theory/behavior_version_stratification/multi_version_prompt_count"] == 0.0
    version_7 = "ggpo_theory/group_majority_token_owner_stratum/version_7/"
    version_5 = "ggpo_theory/group_majority_token_owner_stratum/version_5/"
    assert metrics[f"{version_7}majority_token_owner_behavior_policy_version"] == 7.0
    assert metrics[f"{version_7}training_behavior_version_lag"] == 0.0
    assert metrics[f"{version_7}ell_mean"] == pytest.approx(3.0)
    assert metrics[f"{version_7}ell_max"] == pytest.approx(3.0)
    assert metrics[f"{version_5}majority_token_owner_behavior_policy_version"] == 5.0
    assert metrics[f"{version_5}training_behavior_version_lag"] == 2.0
    assert metrics[f"{version_5}ell_mean"] == pytest.approx(0.5)
    assert metrics[f"{version_5}ell_max"] == pytest.approx(4.0)
    stratum_keys = [
        name
        for name in metrics
        if name.startswith("ggpo_theory/group_majority_token_owner_stratum/version_")
    ]
    assert len(stratum_keys) == 2 * 13
    assert len(metrics) == 91


def test_fully_async_theory_metrics_reject_inactive_stratum_fields():
    inputs = _theory_inputs()
    inputs.update(
        {
            "training_policy_version": 7,
            "behavior_policy_versions": torch.tensor([7, 7, 5, 5]),
            "behavior_version_stratification": True,
        }
    )
    metrics = compute_ggpo_theory_metrics(**inputs)
    metrics["ggpo_theory/group_majority_token_owner_stratum/version_7/unknown"] = 0.0

    with pytest.raises(ValueError, match="inactive or unknown"):
        validate_ggpo_theory_metrics(metrics)


def test_fully_async_theory_strata_attribute_mixed_groups_once_by_majority_with_newest_tie_break():
    inputs = _theory_inputs()
    inputs.update(
        {
            "training_policy_version": 7,
            "behavior_policy_versions": torch.tensor(
                [[7, 5, -1], [5, 7, -1], [5, 5, -1], [5, 7, -1]]
            ),
            "behavior_version_stratification": True,
        }
    )

    metrics = compute_ggpo_theory_metrics(**inputs)

    assert metrics["ggpo_theory/behavior_version_stratification/multi_version_prompt_count"] == 2.0
    assert metrics["ggpo_theory/behavior_version_stratification/multi_version_prompt_fraction"] == 1.0
    assert metrics["ggpo_theory/group_majority_token_owner_stratum/version_7/prompt_count"] == 1.0
    assert metrics["ggpo_theory/group_majority_token_owner_stratum/version_5/prompt_count"] == 1.0
    assert sum(
        value
        for name, value in metrics.items()
        if name.endswith("/prompt_count") and "group_majority_token_owner_stratum" in name
    ) == 2.0


def test_response_level_version_owner_is_weighted_by_valid_tokens():
    metrics = compute_ggpo_theory_metrics(
        training_log_prob=torch.zeros(2, 3),
        behavior_log_prob=torch.zeros(2, 3),
        ref_log_prob=torch.zeros(2, 3),
        rewards=torch.zeros(2, 3),
        response_mask=torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool),
        group_ids=torch.zeros(2, dtype=torch.long),
        beta=0.5,
        group_size=2,
        actor_update_step=8,
        training_policy_version=7,
        run_receipt_schema_version=GGPO_THEORY_RUN_RECEIPT_SCHEMA_VERSION,
        run_receipt_digest_u52=_TEST_RUN_RECEIPT_DIGEST_U52,
        behavior_policy_versions=torch.tensor([5, 7]),
        behavior_version_stratification=True,
    )

    assert "ggpo_theory/group_majority_token_owner_stratum/version_5/prompt_count" in metrics
    assert "ggpo_theory/group_majority_token_owner_stratum/version_7/prompt_count" not in metrics


def test_fully_async_theory_strata_use_dynamic_keys_above_the_old_fixed_version_limit():
    group_count = 33
    response_count = group_count * 2

    metrics = compute_ggpo_theory_metrics(
        training_log_prob=torch.zeros(response_count, 1),
        behavior_log_prob=torch.zeros(response_count, 1),
        ref_log_prob=torch.zeros(response_count, 1),
        rewards=torch.zeros(response_count, 1),
        response_mask=torch.ones(response_count, 1, dtype=torch.bool),
        group_ids=torch.arange(group_count).repeat_interleave(2),
        beta=0.5,
        group_size=2,
        actor_update_step=group_count + 1,
        training_policy_version=group_count,
        run_receipt_schema_version=GGPO_THEORY_RUN_RECEIPT_SCHEMA_VERSION,
        run_receipt_digest_u52=_TEST_RUN_RECEIPT_DIGEST_U52,
        behavior_policy_versions=torch.arange(group_count).repeat_interleave(2),
        behavior_version_stratification=True,
    )

    assert metrics["ggpo_theory/behavior_version_stratification/group_owner_version_count"] == 33.0
    assert "ggpo_theory/group_majority_token_owner_stratum/version_0/prompt_count" in metrics
    assert "ggpo_theory/group_majority_token_owner_stratum/version_32/prompt_count" in metrics


def test_theory_metrics_ignore_complete_dummy_groups_via_real_sample_mask():
    logical = _theory_inputs()
    logical["actor_update_step"] = 8
    logical["training_policy_version"] = 7
    logical["behavior_policy_versions"] = torch.tensor([7, 7, 5, 5])
    logical_metrics = compute_ggpo_theory_metrics(**logical)

    padded = {}
    for name in ("training_log_prob", "behavior_log_prob", "ref_log_prob", "rewards", "response_mask"):
        padded[name] = torch.cat((logical[name], logical[name][:2].clone()), dim=0)
    padded["training_log_prob"][-2:] = 10_000.0
    padded["behavior_log_prob"][-2:] = -10_000.0
    padded["ref_log_prob"][-2:] = -5_000.0
    padded["rewards"][-2:] = 9_000.0
    padded.update(
        {
            "group_ids": torch.tensor([0, 0, 1, 1, 2, 2]),
            "beta": logical["beta"],
            "group_size": logical["group_size"],
            "actor_update_step": logical["actor_update_step"],
            "training_policy_version": logical["training_policy_version"],
            "run_receipt_schema_version": logical[
                "run_receipt_schema_version"
            ],
            "run_receipt_digest_u52": logical["run_receipt_digest_u52"],
            # Exercise the full padded-batch contract: version values for
            # synthetic responses must be removed by the same real-sample mask.
            "behavior_policy_versions": torch.tensor([7, 7, 5, 5, 0, 0]),
            "real_sample_mask": torch.tensor([True, True, True, True, False, False]),
        }
    )
    padded_metrics = compute_ggpo_theory_metrics(**padded)

    assert padded_metrics.keys() == logical_metrics.keys()
    for name in logical_metrics:
        assert padded_metrics[name] == pytest.approx(logical_metrics[name]), name


def test_theory_metrics_keep_extreme_cx_evidence_finite_without_clamping_log_value():
    inputs = _theory_inputs()
    inputs["training_log_prob"][0, 0] += 1_000.0

    metrics = compute_ggpo_theory_metrics(**inputs)

    assert metrics["ggpo_theory/behavior/prompt_sample_log_ratio_max_max"] == pytest.approx(1003.0)
    assert not any("display" in name for name in metrics)
    assert all(math.isfinite(metrics[name]) for name in GGPO_THEORY_REQUIRED_METRICS)


def test_base_online_trainer_records_one_set_a_contract_before_actor_update():
    trainer = object.__new__(RayPPOTrainer)
    trainer.global_steps = 3
    trainer._ggpo_theory_run_receipt = {
        "schema_version": GGPO_THEORY_RUN_RECEIPT_SCHEMA_VERSION,
        "digest_u52": _TEST_RUN_RECEIPT_DIGEST_U52,
    }
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {
                        "loss_mode": "ggpo",
                        "beta": 0.5,
                    }
                },
                "rollout": {"n": 2},
            }
        }
    )
    inputs = _theory_inputs()
    batch = DataProto.from_single_dict(
        {
            GGPO_THEORY_CURRENT_LOG_PROBS: inputs["training_log_prob"],
            "rollout_log_probs": inputs["behavior_log_prob"],
            "ref_log_prob": inputs["ref_log_prob"],
            "token_level_scores": inputs["rewards"],
            "response_mask": inputs["response_mask"],
            "uid": np.array(["prompt-0", "prompt-0", "prompt-1", "prompt-1"], dtype=object),
        }
    )
    metrics = {}

    trainer._record_ggpo_theory_metrics(batch, metrics)

    assert GGPO_THEORY_CURRENT_LOG_PROBS not in batch.batch
    assert torch.all(
        batch.batch[GGPO_THEORY_METRICS_SENTINEL] == GGPO_THEORY_METRICS_SCHEMA_VERSION
    )
    assert metrics["ggpo_theory/run_receipt/digest_u52"] == _TEST_RUN_RECEIPT_DIGEST_U52
    assert metrics["ggpo_theory/identity/actor_update_step"] == 3.0
    assert metrics["ggpo_theory/identity/pre_update_training_policy_version"] == 2.0
    assert metrics["ggpo_theory/identity/behavior_policy_version_available"] == 1.0
    assert metrics["ggpo_theory/identity/behavior_rollout_version_min"] == 2.0
    assert metrics["ggpo_theory/identity/behavior_rollout_version_max"] == 2.0
    assert metrics["ggpo_theory/identity/training_behavior_version_lag_max"] == 0.0
    assert metrics["ggpo_theory/behavior/prompt_sample_log_ratio_max_max"] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="already recorded"):
        trainer._record_ggpo_theory_metrics(batch, metrics)


def _theory_contract_config(tmp_path):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {
                        "loss_mode": "ggpo",
                        "sequence_log_prob_reduction": "mean",
                        "beta": 0.5,
                    }
                },
                "model": {
                    "path": "Qwen/example",
                    "hf_config_path": "Qwen/example",
                    "tokenizer_path": "Qwen/example",
                },
                "rollout": {
                    "n": 2,
                    "name": "vllm",
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "repetition_penalty": 1.0,
                },
            },
            "trainer": {
                "balance_batch": False,
                "use_v1": False,
                "project_name": "ggpo-theory-test",
                "experiment_name": "ggpo-theory-test-run",
                "logger": ["console"],
                "default_local_dir": str(tmp_path / "run"),
                "ggpo_theory_contract": {
                    "run_id": "run-id-001",
                    "source_identity": "git-sha256:source",
                    "reference_policy_identity": "hf:reference@revision",
                    "behavior_policy_identity": "rollout-lineage:behavior",
                },
            },
            "async_training": {"rollout_backend": "synchronous"},
        }
    )


def test_ggpo_run_receipt_seals_exact_objective_probability_and_identities(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    config = _theory_contract_config(tmp_path)

    receipt = build_ggpo_theory_run_receipt(config)
    sealed = seal_ggpo_theory_run_receipt(config)
    resealed = seal_ggpo_theory_run_receipt(config)

    assert receipt["objective"] == {
        "loss_mode": "ggpo",
        "sequence_log_prob_reduction": "mean",
        "beta": 0.5,
        "group_size": 2,
    }
    assert receipt["run"]["run_id"] == "run-id-001"
    assert receipt["source"]["immutable_identity"] == "git-sha256:source"
    assert (
        receipt["reference_policy"]["immutable_identity"]
        == "hf:reference@revision"
    )
    assert (
        receipt["behavior_policy"]["immutable_identity"]
        == "rollout-lineage:behavior"
    )
    assert receipt["probability_convention"]["identity_sampling_transforms"] == {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
    }
    stratification = receipt["behavior_version_stratification"]
    assert stratification["dynamic_key"] == "group_majority_valid_token_owner_behavior_policy_version"
    assert stratification["group_attribution"] == "majority_valid_tokens_with_newest_version_tie_break"
    assert stratification["non_overlapping"] is True
    assert len(stratification["fields"]) == 13
    assert sealed == resealed
    assert len(sealed["sha256"]) == 64
    assert 0 <= sealed["digest_u52"] < 1 << 52
    receipt_path = Path(sealed["path"])
    assert "git-sha256:source" in receipt_path.read_text(encoding="utf-8")
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o400

    config.trainer.ggpo_theory_contract.source_identity = "git-sha256:changed"
    with pytest.raises(ValueError, match="does not exactly match"):
        seal_ggpo_theory_run_receipt(config)


def test_ggpo_run_receipt_describes_moving_reference_state(tmp_path):
    config = _theory_contract_config(tmp_path)
    config.trainer.ggpo_moving_reference = {"enabled": True, "update_freq": 20}

    reference = build_ggpo_theory_run_receipt(config)["reference_policy"]

    assert reference["initial_identity"] == "hf:reference@revision"
    assert "immutable_identity" not in reference
    assert reference["update"] == {
        "mode": "latest_saved_actor_checkpoint",
        "update_freq": 20,
        "version_metric": "actor/ggpo_reference_version",
        "resume_source": "resumed_actor_checkpoint",
    }


def test_ggpo_preflight_accepts_absent_fields_with_identity_defaults(tmp_path):
    config = _theory_contract_config(tmp_path)
    del config.actor_rollout_ref.rollout.temperature
    del config.actor_rollout_ref.rollout.top_p
    del config.actor_rollout_ref.rollout.top_k
    del config.actor_rollout_ref.rollout.repetition_penalty
    del config.actor_rollout_ref.rollout.do_sample

    validate_ggpo_trainer_compatibility(config)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("actor_rollout_ref.rollout.temperature", 0.8),
        ("actor_rollout_ref.rollout.top_p", 0.95),
        ("actor_rollout_ref.rollout.top_k", 20),
        ("actor_rollout_ref.rollout.repetition_penalty", 1.1),
        ("actor_rollout_ref.rollout.temperature", "1.0"),
        ("actor_rollout_ref.rollout.top_k", "-1"),
    ],
)
def test_ggpo_preflight_rejects_nonidentity_sampling_transforms(
    tmp_path,
    path,
    value,
):
    config = _theory_contract_config(tmp_path)
    OmegaConf.update(config, path, value)

    with pytest.raises(ValueError, match=path):
        validate_ggpo_trainer_compatibility(config)


def test_ggpo_preflight_rejects_greedy_behavior_distribution(tmp_path):
    config = _theory_contract_config(tmp_path)
    config.actor_rollout_ref.rollout.do_sample = False

    with pytest.raises(ValueError, match="do_sample=true"):
        validate_ggpo_trainer_compatibility(config)


def test_ggpo_wandb_preflight_binds_explicit_run_id(tmp_path, monkeypatch):
    config = _theory_contract_config(tmp_path)
    config.trainer.logger = ["console", "wandb"]
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)

    with pytest.raises(ValueError, match="explicit WANDB_RUN_ID"):
        validate_ggpo_trainer_compatibility(config)

    monkeypatch.setenv("WANDB_RUN_ID", "wrong-run")
    with pytest.raises(ValueError, match="does not match WANDB_RUN_ID"):
        validate_ggpo_trainer_compatibility(config)

    monkeypatch.setenv("WANDB_RUN_ID", "run-id-001")
    validate_ggpo_trainer_compatibility(config)
