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

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.config import RolloutCorrectionConfig
from verl.trainer.ppo.utils import need_reference_policy
from verl.utils import tensordict_utils as tu
from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from verl.workers.engine.utils import prepare_micro_batches


def test_ggpo_needs_reference_policy_without_kl_loss_or_reward_kl():
    """GGPO always needs the frozen reference policy for its embedded regularizer."""
    config = OmegaConf.create(
        {
            "algorithm": {"use_kl_in_reward": False},
            "actor_rollout_ref": {
                "actor": {
                    "use_kl_loss": False,
                    "policy_loss": {"loss_mode": "ggpo"},
                }
            },
        }
    )

    assert need_reference_policy(config) is True


def test_missing_reference_signals_with_null_policy_loss_does_not_need_reference_policy():
    """A null policy_loss should behave like missing vanilla policy loss and not raise."""
    config = OmegaConf.create(
        {
            "algorithm": {"use_kl_in_reward": False},
            "actor_rollout_ref": {
                "actor": {
                    "use_kl_loss": False,
                    "policy_loss": None,
                }
            },
        }
    )

    assert need_reference_policy(config) is False


@pytest.mark.parametrize(
    "config_dict",
    [
        {
            "algorithm": {"use_kl_in_reward": True},
            "actor_rollout_ref": {
                "actor": {
                    "use_kl_loss": False,
                }
            },
        },
        {
            "algorithm": {"use_kl_in_reward": False},
            "actor_rollout_ref": {
                "actor": {
                    "use_kl_loss": True,
                    "policy_loss": None,
                }
            },
        },
    ],
)
def test_legacy_reference_policy_activation_paths_remain_enabled(config_dict):
    """Legacy KL-in-reward and actor KL-loss paths each independently create a reference policy."""
    config = OmegaConf.create(config_dict)

    assert need_reference_policy(config) is True


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {
                "algorithm": {"use_kl_in_reward": False},
                "actor_rollout_ref": {
                    "actor": {
                        "use_kl_loss": False,
                        "policy_loss": {"loss_mode": "ggpo"},
                    }
                },
            },
            True,
            id="ggpo",
        ),
        pytest.param(
            {
                "algorithm": {"use_kl_in_reward": False},
                "actor_rollout_ref": {
                    "actor": {
                        "use_kl_loss": False,
                    }
                },
            },
            False,
            id="missing-policy-loss",
        ),
        pytest.param(
            {
                "algorithm": {"use_kl_in_reward": False},
                "actor_rollout_ref": {
                    "actor": {
                        "use_kl_loss": False,
                        "policy_loss": None,
                    }
                },
            },
            False,
            id="null-policy-loss",
        ),
        pytest.param(
            {
                "algorithm": {"use_kl_in_reward": True},
                "actor_rollout_ref": {
                    "actor": {
                        "use_kl_loss": False,
                    }
                },
            },
            True,
            id="kl-in-reward",
        ),
        pytest.param(
            {
                "algorithm": {"use_kl_in_reward": False},
                "actor_rollout_ref": {
                    "actor": {
                        "use_kl_loss": True,
                        "policy_loss": None,
                    }
                },
            },
            True,
            id="actor-kl-loss",
        ),
    ],
)
def test_plain_dict_reference_policy_activation(config, expected):
    """Plain dict configs follow the same reference-policy activation rules."""
    assert need_reference_policy(config) is expected


def test_build_group_ids_preserves_first_seen_prompt_order():
    from verl.trainer.ppo.ggpo import build_policy_group_ids

    group_ids = build_policy_group_ids(np.array(["a"] * 8 + ["b"] * 8, dtype=object), group_size=8)

    torch.testing.assert_close(group_ids, torch.tensor([0] * 8 + [1] * 8, dtype=torch.long))


def test_build_group_ids_rejects_interleaved_incomplete_group():
    from verl.trainer.ppo.ggpo import build_policy_group_ids

    with pytest.raises(ValueError, match="contiguous block"):
        build_policy_group_ids(np.array(["a", "b"] * 8, dtype=object), group_size=8)


@pytest.mark.parametrize("group_size", [0, 1])
def test_build_group_ids_rejects_invalid_group_size(group_size):
    from verl.trainer.ppo.ggpo import build_policy_group_ids

    with pytest.raises(ValueError, match="group_size"):
        build_policy_group_ids(np.array(["a"] * 8, dtype=object), group_size=group_size)


def test_build_group_ids_rejects_non_divisible_response_count():
    from verl.trainer.ppo.ggpo import build_policy_group_ids

    with pytest.raises(ValueError, match="divisible"):
        build_policy_group_ids(np.array(["a"] * 8 + ["b"] * 7, dtype=object), group_size=8)


def test_build_group_ids_rejects_non_contiguous_repeated_uid_block():
    from verl.trainer.ppo.ggpo import build_policy_group_ids

    with pytest.raises(ValueError, match="non-contiguous"):
        build_policy_group_ids(np.array(["a"] * 8 + ["b"] * 8 + ["a"] * 8, dtype=object), group_size=8)


def test_build_group_ids_rejects_array_valued_uids_with_ggpo_error():
    from verl.trainer.ppo.ggpo import build_policy_group_ids

    with pytest.raises(ValueError, match="GGPO uid values must be scalar-comparable"):
        build_policy_group_ids(np.array([np.array([1, 2])] * 8, dtype=object), group_size=8)


def test_prepare_micro_batches_keeps_complete_ggpo_groups_with_internal_group_count():
    data = TensorDict(
        {
            "responses": torch.arange(32),
            "policy_group_ids": torch.tensor([0] * 8 + [1] * 8 + [2] * 8 + [3] * 8, dtype=torch.long),
        },
        batch_size=[32],
    )
    tu.assign_non_tensor(data, use_dynamic_bsz=False, micro_batch_size_per_gpu=2, force_group_size=8)

    micro_batches, batch_idx_list = prepare_micro_batches(data)

    assert batch_idx_list is None
    assert len(micro_batches) == 2
    for micro_batch_index, micro_batch in enumerate(micro_batches):
        assert len(micro_batch) == 16
        first_group_id = micro_batch_index * 2
        torch.testing.assert_close(
            micro_batch["policy_group_ids"],
            torch.tensor([first_group_id] * 8 + [first_group_id + 1] * 8, dtype=torch.long),
        )


def test_prepare_micro_batches_grpo_micro_batch_counts_responses():
    data = TensorDict({"responses": torch.arange(32)}, batch_size=[32])
    tu.assign_non_tensor(data, use_dynamic_bsz=False, micro_batch_size_per_gpu=16)

    micro_batches, batch_idx_list = prepare_micro_batches(data)

    assert batch_idx_list is None
    assert [len(micro_batch) for micro_batch in micro_batches] == [16, 16]


def _ggpo_config(*, loss_mode="ggpo", group_size=8, micro_batch_size=16, balance_batch=False, use_v1=False):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "ppo_micro_batch_size_per_gpu": micro_batch_size,
                    "use_dynamic_bsz": False,
                    "policy_loss": {"loss_mode": loss_mode},
                },
                "rollout": {"n": group_size},
            },
            "trainer": {"balance_batch": balance_batch, "use_v1": use_v1},
        }
    )


def _ppo_actor_config(
    *,
    loss_mode="vanilla",
    group_size=2,
    ggpo_sequence_log_prob_reduction="sum",
    reward_kl=True,
    memory_safe_surrogate=False,
    rollout_correction=None,
    use_kl_loss=False,
):
    policy_loss_kwargs = {}
    if rollout_correction is not None:
        policy_loss_kwargs["rollout_correction"] = rollout_correction
    return ActorConfig(
        strategy="fsdp",
        ppo_mini_batch_size=4,
        ppo_micro_batch_size_per_gpu=4,
        rollout_n=group_size if loss_mode in {"ggpo", "agro"} else 1,
        policy_loss=PolicyLossConfig(
            loss_mode=loss_mode,
            sequence_log_prob_reduction=ggpo_sequence_log_prob_reduction,
            reward_kl=reward_kl,
            memory_safe_surrogate=memory_safe_surrogate,
            **policy_loss_kwargs,
        ),
        use_kl_loss=use_kl_loss,
        use_torch_compile=False,
    )


def _ppo_loss_omegaconf_config(*, loss_mode="ggpo", group_size=2, use_kl_loss=False):
    return OmegaConf.create(
        {
            "rollout_n": group_size,
            "policy_loss": {"loss_mode": loss_mode, "beta": 0.01},
            "loss_agg_mode": "token-mean",
            "loss_scale_factor": None,
            "entropy_coeff": 0.0,
            "use_kl_loss": use_kl_loss,
            "kl_loss_type": "low_var_kl",
            "kl_loss_coef": 0.001,
            "global_batch_info": {},
        }
    )


def _make_ppo_loss_batch(*, old_log_prob_shift=0.0, advantage_scale=1.0):
    response_mask = torch.ones(4, 3, dtype=torch.long)
    data = TensorDict(
        {
            "prompts": torch.zeros(4, 1, dtype=torch.long),
            "responses": torch.zeros(4, 3, dtype=torch.long),
            "input_ids": torch.zeros(4, 4, dtype=torch.long),
            "attention_mask": torch.ones(4, 4, dtype=torch.long),
            "response_mask": response_mask,
            "old_log_probs": torch.full((4, 3), -0.25 + old_log_prob_shift, dtype=torch.float32),
            "ref_log_prob": torch.zeros(4, 3, dtype=torch.float32, requires_grad=True),
            "token_level_scores": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            "advantages": advantage_scale
            * torch.tensor(
                [[0.2, 0.1, -0.3], [-0.4, 0.5, 0.6], [0.7, -0.8, 0.9], [-1.0, 1.1, -1.2]],
                dtype=torch.float32,
            ),
            "policy_group_ids": torch.tensor([0, 0, 1, 1], dtype=torch.long),
            "_ggpo_raw_correctness": torch.tensor([1, 0, 0, 1], dtype=torch.float32),
        },
        batch_size=[4],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=int(response_mask.sum().item()), global_batch_size=4)
    return data


def _make_group_padded_ppo_loss_batch():
    from verl.experimental.fully_async_policy.actor_update_padding import (
        DUMMY_SAFE_RESPONSE_MASK,
        REAL_SAMPLE_MASK,
    )

    logical = _make_ppo_loss_batch()
    tensor_keys = (
        "prompts",
        "responses",
        "input_ids",
        "attention_mask",
        "response_mask",
        "old_log_probs",
        "ref_log_prob",
        "token_level_scores",
        "advantages",
        "policy_group_ids",
        "_ggpo_raw_correctness",
    )
    padded = TensorDict(
        {
            name: torch.cat((logical[name], logical[name][:2].detach().clone()), dim=0)
            for name in tensor_keys
        },
        batch_size=[6],
    )
    padded["policy_group_ids"][-2:] = 2
    padded[DUMMY_SAFE_RESPONSE_MASK] = padded["response_mask"].clone()
    padded["response_mask"][-2:].zero_()
    padded["old_log_probs"][-2:].zero_()
    padded["ref_log_prob"][-2:].zero_()
    padded["token_level_scores"][-2:].zero_()
    padded["advantages"][-2:].zero_()
    padded[REAL_SAMPLE_MASK] = torch.tensor([True, True, True, True, False, False])
    tu.assign_non_tensor(
        padded,
        dp_size=1,
        batch_num_tokens=int(logical["response_mask"].sum().item()),
        global_batch_size=4,
    )
    return padded


def _flat_model_log_probs_from_response_values(response_values):
    flat_values = torch.zeros(response_values.shape[0] * (response_values.shape[1] + 1), dtype=torch.float32)
    seq_len = response_values.shape[1] + 1
    for row_idx, row in enumerate(response_values):
        start = row_idx * seq_len
        flat_values[start : start + response_values.shape[1]] = row
    return flat_values.requires_grad_()


def test_ppo_loss_routes_ggpo_without_ratio():
    from verl.workers.utils.losses import ppo_loss

    current_response_log_probs = torch.tensor(
        [[-0.10, -0.20, -0.30], [-0.40, -0.50, -0.60], [-0.70, -0.80, -0.90], [-0.15, -0.25, -0.35]],
        dtype=torch.float32,
    )
    log_probs = _flat_model_log_probs_from_response_values(current_response_log_probs)
    data = _make_ppo_loss_batch()
    changed_log_probs = _flat_model_log_probs_from_response_values(current_response_log_probs)
    changed_data = _make_ppo_loss_batch(old_log_prob_shift=10.0, advantage_scale=-7.0)
    without_behavior_log_probs = _make_ppo_loss_batch(old_log_prob_shift=-123.0, advantage_scale=99.0)
    del without_behavior_log_probs["old_log_probs"]
    del without_behavior_log_probs["advantages"]
    log_probs_without_behavior = _flat_model_log_probs_from_response_values(current_response_log_probs)

    loss, metrics = ppo_loss(
        _ppo_actor_config(loss_mode="ggpo", group_size=2),
        {"log_probs": log_probs},
        data,
    )
    changed_loss, _ = ppo_loss(
        _ppo_actor_config(loss_mode="ggpo", group_size=2),
        {"log_probs": changed_log_probs},
        changed_data,
    )
    loss_without_behavior, _ = ppo_loss(
        _ppo_actor_config(loss_mode="ggpo", group_size=2),
        {"log_probs": log_probs_without_behavior},
        without_behavior_log_probs,
    )

    assert loss.requires_grad
    assert "ggpo_loss" in metrics
    torch.testing.assert_close(loss.detach(), changed_loss.detach(), atol=0.0, rtol=0.0)
    torch.testing.assert_close(loss.detach(), loss_without_behavior.detach(), atol=0.0, rtol=0.0)

    loss.backward()
    changed_loss.backward()
    loss_without_behavior.backward()
    assert log_probs.grad is not None
    assert changed_log_probs.grad is not None
    assert log_probs_without_behavior.grad is not None
    assert log_probs.grad.abs().sum().item() > 0
    assert torch.isfinite(log_probs.grad).all()
    assert torch.isfinite(changed_log_probs.grad).all()
    assert torch.isfinite(log_probs_without_behavior.grad).all()
    torch.testing.assert_close(log_probs.grad, changed_log_probs.grad, atol=0.0, rtol=0.0)
    torch.testing.assert_close(log_probs.grad, log_probs_without_behavior.grad, atol=0.0, rtol=0.0)
    assert data["ref_log_prob"].grad is None


def test_ppo_loss_routes_agro_without_ratio():
    # Both losses require the global token count; exercise the shared dispatch.
    from verl.workers.utils.losses import ppo_loss

    current_response_log_probs = torch.tensor(
        [[-0.10, -0.20, -0.30], [-0.40, -0.50, -0.60], [-0.70, -0.80, -0.90], [-0.15, -0.25, -0.35]],
        dtype=torch.float32,
    )
    log_probs = _flat_model_log_probs_from_response_values(current_response_log_probs)
    data = _make_ppo_loss_batch()

    loss, metrics = ppo_loss(
        _ppo_actor_config(loss_mode="agro", group_size=2),
        {"log_probs": log_probs},
        data,
    )

    assert loss.requires_grad
    assert "agro_surrogate_loss" in metrics
    assert torch.isfinite(loss)

    loss.backward()
    assert log_probs.grad is not None
    assert log_probs.grad.abs().sum().item() > 0
    assert torch.isfinite(log_probs.grad).all()
    assert data["ref_log_prob"].grad is None


def test_mismatch_diagnostics_do_not_change_ggpo_loss_or_gradient():
    from verl.trainer.ppo.rollout_training_mismatch import INTERNAL_MISMATCH_STATS_KEY
    from verl.workers.utils.losses import ppo_loss

    response_values = torch.tensor(
        [[-0.10, -0.20, -0.30], [-0.40, -0.50, -0.60], [-0.70, -0.80, -0.90], [-0.15, -0.25, -0.35]],
        dtype=torch.float32,
    )
    plain_log_probs = _flat_model_log_probs_from_response_values(response_values)
    observed_log_probs = _flat_model_log_probs_from_response_values(response_values)
    plain_data = _make_ppo_loss_batch()
    observed_data = _make_ppo_loss_batch()
    observed_data["rollout_log_probs"] = torch.full((4, 3), -0.35, dtype=torch.float32)
    tu.assign_non_tensor(observed_data, collect_rollout_training_mismatch=True)

    plain_loss, plain_metrics = ppo_loss(
        _ppo_actor_config(loss_mode="ggpo", group_size=2),
        {"log_probs": plain_log_probs},
        plain_data,
    )
    observed_loss, observed_metrics = ppo_loss(
        _ppo_actor_config(loss_mode="ggpo", group_size=2),
        {"log_probs": observed_log_probs},
        observed_data,
    )
    torch.testing.assert_close(observed_loss, plain_loss, atol=0.0, rtol=0.0)
    assert INTERNAL_MISMATCH_STATS_KEY not in plain_metrics
    assert INTERNAL_MISMATCH_STATS_KEY in observed_metrics

    plain_loss.backward()
    observed_loss.backward()
    torch.testing.assert_close(observed_log_probs.grad, plain_log_probs.grad, atol=0.0, rtol=0.0)
    assert plain_data["ref_log_prob"].grad is None
    assert observed_data["ref_log_prob"].grad is None


def test_ppo_loss_ggpo_complete_dummy_group_preserves_real_loss_and_gradient():
    from verl.workers.utils.losses import ppo_loss

    real_response_log_probs = torch.tensor(
        [[-0.10, -0.20, -0.30], [-0.40, -0.50, -0.60], [-0.70, -0.80, -0.90], [-0.15, -0.25, -0.35]],
        dtype=torch.float32,
    )
    logical_log_probs = _flat_model_log_probs_from_response_values(real_response_log_probs)
    physical_log_probs = _flat_model_log_probs_from_response_values(
        torch.cat(
            (
                real_response_log_probs,
                torch.tensor([[10_000.0, -10_000.0, 5_000.0], [-7_000.0, 9_000.0, -8_000.0]]),
            ),
            dim=0,
        )
    )
    config = _ppo_actor_config(loss_mode="ggpo", group_size=2)

    logical_loss, logical_metrics = ppo_loss(
        config,
        {"log_probs": logical_log_probs},
        _make_ppo_loss_batch(),
    )
    physical_loss, physical_metrics = ppo_loss(
        config,
        {"log_probs": physical_log_probs},
        _make_group_padded_ppo_loss_batch(),
    )

    torch.testing.assert_close(physical_loss, logical_loss, atol=0.0, rtol=0.0)
    for name in logical_metrics.keys() & physical_metrics.keys():
        if name.startswith("ggpo_"):
            # nan_ok covers the diagnostics that are NaN when their group
            # subset is empty; padding must still reproduce them exactly.
            assert physical_metrics[name].aggregate() == pytest.approx(
                logical_metrics[name].aggregate(),
                abs=0.0,
                nan_ok=True,
            )

    logical_loss.backward()
    physical_loss.backward()
    logical_gradient = logical_log_probs.grad.reshape(4, 4)
    physical_gradient = physical_log_probs.grad.reshape(6, 4)
    torch.testing.assert_close(physical_gradient[:4], logical_gradient, atol=0.0, rtol=0.0)
    torch.testing.assert_close(physical_gradient[4:], torch.zeros(2, 4), atol=0.0, rtol=0.0)


def test_ppo_loss_ggpo_dummy_only_microbatch_is_finite_graph_zero():
    from verl.experimental.fully_async_policy.actor_update_padding import (
        DUMMY_SAFE_RESPONSE_MASK,
        REAL_SAMPLE_MASK,
    )
    from verl.workers.utils.losses import ppo_loss

    data = _make_ppo_loss_batch()
    data[DUMMY_SAFE_RESPONSE_MASK] = data["response_mask"].clone()
    data[REAL_SAMPLE_MASK] = torch.zeros(4, dtype=torch.bool)
    data["response_mask"].zero_()
    log_probs = _flat_model_log_probs_from_response_values(
        torch.tensor(
            [[10_000.0, -10_000.0, 5_000.0], [-7_000.0, 9_000.0, -8_000.0]] * 2,
            dtype=torch.float32,
        )
    )

    loss, metrics = ppo_loss(
        _ppo_actor_config(loss_mode="ggpo", group_size=2),
        {"log_probs": log_probs},
        data,
    )

    assert loss.requires_grad
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=0.0, rtol=0.0)
    for name, metric in metrics.items():
        assert all(np.isfinite(value) for value in metric.values), name
        if name.startswith("ggpo_"):
            assert metric.weights == [0.0]

    loss.backward()
    assert log_probs.grad is not None
    assert torch.isfinite(log_probs.grad).all()
    torch.testing.assert_close(log_probs.grad, torch.zeros_like(log_probs), atol=0.0, rtol=0.0)


def test_ppo_loss_bypass_dummy_only_microbatch_keeps_schema_and_zero_gradient():
    from verl.experimental.fully_async_policy.actor_update_padding import (
        DUMMY_SAFE_RESPONSE_MASK,
        REAL_SAMPLE_MASK,
    )
    from verl.workers.utils.losses import ppo_loss

    data = _make_ppo_loss_batch()
    data[DUMMY_SAFE_RESPONSE_MASK] = data["response_mask"].clone()
    data[REAL_SAMPLE_MASK] = torch.zeros(4, dtype=torch.bool)
    data["response_mask"].zero_()
    data["advantages"].fill_(123.0)
    log_probs = _flat_model_log_probs_from_response_values(
        torch.tensor(
            [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6], [-0.7, -0.8, -0.9], [-0.2, -0.3, -0.4]],
            dtype=torch.float32,
        )
    )
    config = _ppo_actor_config(
        loss_mode="bypass_mode",
        rollout_correction=RolloutCorrectionConfig.bypass_ppo_clip(),
    )

    loss, metrics = ppo_loss(config, {"log_probs": log_probs}, data)

    assert loss.requires_grad
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=0.0, rtol=0.0)
    assert {"actor/pg_clipfrac", "actor/ppo_kl", "actor/pg_clipfrac_lower"} <= metrics.keys()
    for name, metric in metrics.items():
        assert all(np.isfinite(value) for value in metric.values), name
        assert metric.weights == [0.0], name

    loss.backward()
    assert log_probs.grad is not None
    assert torch.isfinite(log_probs.grad).all()
    torch.testing.assert_close(log_probs.grad, torch.zeros_like(log_probs), atol=0.0, rtol=0.0)


def test_ppo_loss_a1_dummy_only_microbatch_masks_kl_loss_and_gradient():
    from verl.experimental.fully_async_policy.actor_update_padding import (
        DUMMY_SAFE_RESPONSE_MASK,
        REAL_SAMPLE_MASK,
    )
    from verl.workers.utils.losses import ppo_loss

    data = _make_ppo_loss_batch()
    data[DUMMY_SAFE_RESPONSE_MASK] = data["response_mask"].clone()
    data[REAL_SAMPLE_MASK] = torch.zeros(4, dtype=torch.bool)
    data["response_mask"].zero_()
    data["advantages"].fill_(123.0)
    data["ref_log_prob"] = torch.full(
        data["ref_log_prob"].shape,
        -7.0,
        dtype=data["ref_log_prob"].dtype,
        requires_grad=True,
    )
    log_probs = _flat_model_log_probs_from_response_values(
        torch.tensor(
            [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6], [-0.7, -0.8, -0.9], [-0.2, -0.3, -0.4]],
            dtype=torch.float32,
        )
    )
    config = _ppo_actor_config(
        loss_mode="bypass_mode",
        rollout_correction=RolloutCorrectionConfig.bypass_ppo_clip(),
        use_kl_loss=True,
    )

    loss, metrics = ppo_loss(config, {"log_probs": log_probs}, data)

    assert loss.requires_grad
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=0.0, rtol=0.0)
    assert metrics["kl_loss"].weights == [0.0]
    assert all(np.isfinite(value) for value in metrics["kl_loss"].values)

    loss.backward()
    assert log_probs.grad is not None
    assert torch.isfinite(log_probs.grad).all()
    torch.testing.assert_close(log_probs.grad, torch.zeros_like(log_probs), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("reward_kl", [True, False])
def test_ppo_loss_uses_binary_verifier_scores_as_zero_one_ggpo_rewards(reward_kl):
    from verl.trainer.ppo.ggpo import compute_ggpo_loss
    from verl.workers.utils.losses import ppo_loss
    from verl.workers.utils.padding import no_padding_2_padding

    current_response_log_probs = torch.tensor(
        [[-0.10, -0.20, -0.30], [-0.40, -0.50, -0.60], [-0.70, -0.80, -0.90], [-0.15, -0.25, -0.35]],
        dtype=torch.float32,
    )
    log_probs = _flat_model_log_probs_from_response_values(current_response_log_probs)
    data = _make_ppo_loss_batch()
    config = _ppo_actor_config(loss_mode="ggpo", group_size=2, reward_kl=reward_kl)

    loss, _ = ppo_loss(config, {"log_probs": log_probs}, data)

    padded_log_probs = no_padding_2_padding(log_probs, data)
    padded_data = data.select(
        "response_mask", "ref_log_prob", "token_level_scores", "policy_group_ids"
    ).to_padded_tensor()
    sequence_scores = (padded_data["token_level_scores"].float() * padded_data["response_mask"].to(bool)).sum(dim=-1)
    expected, _ = compute_ggpo_loss(
        log_prob=padded_log_probs,
        ref_log_prob=padded_data["ref_log_prob"],
        rewards=sequence_scores,
        raw_correctness=sequence_scores,
        response_mask=padded_data["response_mask"].to(bool),
        group_ids=padded_data["policy_group_ids"],
        beta=config.policy_loss.beta,
        reward_kl=reward_kl,
        group_size=config.rollout_n,
        global_batch_size=4,
        batch_num_tokens=int(padded_data["response_mask"].sum().item()),
        dp_size=1,
    )

    torch.testing.assert_close(loss, expected, atol=1e-7, rtol=0.0)


@pytest.mark.parametrize("sequence_reduction", ["sum", "mean"])
@pytest.mark.parametrize("reward_kl", [True, False])
def test_ggpo_memory_safe_surrogate_matches_exact_gradient_across_split_groups(sequence_reduction, reward_kl):
    """Detached GGPO derivative weights permit arbitrary dynamic group splits."""
    from verl.trainer.ppo.ggpo import compute_ggpo_loss, compute_ggpo_surrogate_weights

    response_mask = torch.tensor(
        [[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]],
        dtype=torch.bool,
    )
    initial_log_probs = torch.tensor(
        [
            [-0.10, -0.20, -0.30],
            [-0.15, -0.25, -0.35],
            [-0.20, -0.30, -0.40],
            [-0.25, -0.35, -0.45],
            [-0.30, -0.40, -0.50],
            [-0.35, -0.45, -0.55],
            [-0.40, -0.50, -0.60],
            [-0.45, -0.55, -0.65],
        ],
        dtype=torch.float32,
    )
    ref_log_probs = torch.zeros_like(initial_log_probs)
    rewards = torch.tensor([1, 0, 0, 1, 0, 1, 1, 0], dtype=torch.float32)
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    beta = 0.1
    dp_size = 3

    direct_log_probs = initial_log_probs.clone().requires_grad_(True)
    direct_loss, _ = compute_ggpo_loss(
        direct_log_probs,
        ref_log_probs,
        rewards,
        response_mask,
        group_ids,
        raw_correctness=rewards,
        beta=beta,
        group_size=4,
        global_batch_size=8,
        batch_num_tokens=int(response_mask.sum().item()),
        dp_size=dp_size,
        sequence_log_prob_reduction=sequence_reduction,
        reward_kl=reward_kl,
    )
    direct_loss.backward()

    surrogate_log_probs = initial_log_probs.clone().requires_grad_(True)
    weights = compute_ggpo_surrogate_weights(
        surrogate_log_probs.detach(),
        ref_log_probs,
        rewards,
        response_mask,
        group_ids,
        beta=beta,
        group_size=4,
        global_batch_size=8,
        batch_num_tokens=int(response_mask.sum().item()),
        dp_size=dp_size,
        sequence_log_prob_reduction=sequence_reduction,
        reward_kl=reward_kl,
    )
    surrogate_loss = surrogate_log_probs.sum() * 0.0
    # These pieces split both complete groups, exactly as dynamic token packing
    # may do once whole-group co-location is removed.
    for start, stop in ((0, 2), (2, 5), (5, 8)):
        local_mask = response_mask[start:stop]
        local_sequence_log_prob = (surrogate_log_probs[start:stop] * local_mask).sum(dim=-1)
        if sequence_reduction == "mean":
            local_sequence_log_prob = local_sequence_log_prob / local_mask.sum(dim=-1)
        surrogate_loss = surrogate_loss + (weights[start:stop] * local_sequence_log_prob).sum()
    surrogate_loss.backward()

    torch.testing.assert_close(surrogate_log_probs.grad, direct_log_probs.grad, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_reward_only_ggpo_has_analytic_policy_gradient_independent_of_reference(reduction):
    from verl.trainer.ppo.ggpo import compute_ggpo_loss

    rewards = torch.tensor([1., 0., 0., 1., 0., 0.])
    groups = torch.tensor([9, 2, 9, 2, 4, 4])
    mask = torch.tensor([[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], [1, 0]], dtype=torch.bool)
    initial = -torch.arange(1, 13, dtype=torch.float32).reshape(6, 2) / 10
    # For binary rewards in groups of two, the mixed-group coefficient is
    # +/-tanh(1/(2*beta)); constant-reward groups have zero coefficients.
    coefficients = torch.tensor([1., -1., -1., 1., 0., 0.]) * math.tanh(1.)
    lengths = mask.sum(-1)
    denominator = mask.sum() if reduction == "sum" else len(rewards)
    expected_gradient = -2 * coefficients[:, None] * mask / denominator
    if reduction == "mean":
        expected_gradient = expected_gradient / lengths[:, None]

    for reference_shift in (0., 100.):
        current = initial.clone().requires_grad_(True)
        reference = (initial.flip(0) + reference_shift).requires_grad_(True)
        loss, metrics = compute_ggpo_loss(
            current, reference, rewards, mask, groups, raw_correctness=rewards,
            beta=0.5, group_size=2, global_batch_size=6, batch_num_tokens=int(mask.sum()),
            dp_size=2, sequence_log_prob_reduction=reduction, reward_kl=False,
        )
        loss.backward()
        torch.testing.assert_close(current.grad, expected_gradient)
        torch.testing.assert_close(loss, (expected_gradient * initial).sum())
        torch.testing.assert_close(metrics["ggpo_adjusted_reward_mean"], rewards.mean())
        assert reference.grad is None


def test_ppo_loss_forwards_ggpo_sequence_mean():
    from verl.trainer.ppo.ggpo import compute_ggpo_loss
    from verl.utils.metric import AggregationType
    from verl.workers.utils.losses import ppo_loss
    from verl.workers.utils.padding import no_padding_2_padding

    current_response_log_probs = torch.tensor(
        [[-0.10, -0.20, -0.30], [-0.40, -0.50, -0.60], [-0.70, -0.80, -0.90], [-0.15, -0.25, -0.35]],
        dtype=torch.float32,
    )
    log_probs = _flat_model_log_probs_from_response_values(current_response_log_probs)
    data = _make_ppo_loss_batch()
    config = _ppo_actor_config(
        loss_mode="ggpo",
        group_size=2,
        ggpo_sequence_log_prob_reduction="mean",
    )

    loss, metrics = ppo_loss(config, {"log_probs": log_probs}, data)

    padded_log_probs = no_padding_2_padding(log_probs, data)
    padded_data = data.select(
        "response_mask", "ref_log_prob", "token_level_scores", "policy_group_ids"
    ).to_padded_tensor()
    sequence_scores = (padded_data["token_level_scores"].float() * padded_data["response_mask"].to(bool)).sum(dim=-1)
    expected, _ = compute_ggpo_loss(
        log_prob=padded_log_probs,
        ref_log_prob=padded_data["ref_log_prob"],
        rewards=sequence_scores,
        raw_correctness=sequence_scores,
        response_mask=padded_data["response_mask"].to(bool),
        group_ids=padded_data["policy_group_ids"],
        beta=config.policy_loss.beta,
        group_size=config.rollout_n,
        global_batch_size=4,
        batch_num_tokens=int(padded_data["response_mask"].sum().item()),
        dp_size=1,
        sequence_log_prob_reduction="mean",
    )

    torch.testing.assert_close(loss, expected, atol=1e-7, rtol=0.0)
    assert {
        "ggpo_reward_mean",
        "ggpo_zero_reward_frac",
        "ggpo_response_length_mean",
        "ggpo_response_length_min",
        "ggpo_response_length_max",
        "ggpo_sequence_log_prob_reduction_is_mean",
    }.isdisjoint(metrics)
    assert metrics["ggpo_adjusted_reward_min"].aggregation is AggregationType.MIN
    assert metrics["ggpo_adjusted_reward_max"].aggregation is AggregationType.MAX
    assert metrics["ggpo_sequence_log_ratio_abs_max"].aggregation is AggregationType.MAX
    assert metrics["ggpo_adjusted_reward_mean_microbatch_p50"].aggregation is AggregationType.MEAN


def test_ppo_loss_rejects_ggpo_with_actor_kl_loss():
    from verl.workers.utils.losses import ppo_loss

    log_probs = _flat_model_log_probs_from_response_values(
        torch.tensor(
            [[-0.10, -0.20, -0.30], [-0.40, -0.50, -0.60], [-0.70, -0.80, -0.90], [-0.15, -0.25, -0.35]],
            dtype=torch.float32,
        )
    )

    with pytest.raises(ValueError, match="GGPO does not support actor use_kl_loss"):
        ppo_loss(
            _ppo_loss_omegaconf_config(loss_mode="ggpo", group_size=2, use_kl_loss=True),
            {"log_probs": log_probs},
            _make_ppo_loss_batch(),
        )


def test_ggpo_padding_metrics_use_the_authoritative_live_schema():
    from verl.trainer.ppo.ggpo import GGPO_METRIC_NAMES
    from verl.workers.utils.losses import _zero_group_reference_metrics

    _, metrics = _zero_group_reference_metrics(torch.zeros(1, requires_grad=True), loss_mode="ggpo")

    assert tuple(metrics) == GGPO_METRIC_NAMES


def test_ppo_loss_vanilla_matches_core_policy_loss():
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    from verl.workers.utils.losses import ppo_loss
    from verl.workers.utils.padding import no_padding_2_padding

    current_response_log_probs = torch.tensor(
        [[-0.30, -0.20, -0.10], [-0.60, -0.50, -0.40], [-0.90, -0.80, -0.70], [-0.35, -0.25, -0.15]],
        dtype=torch.float32,
    )
    log_probs = _flat_model_log_probs_from_response_values(current_response_log_probs)
    data = _make_ppo_loss_batch(old_log_prob_shift=0.05, advantage_scale=1.25)
    config = _ppo_actor_config(loss_mode="vanilla")

    policy_loss, metrics = ppo_loss(config, {"log_probs": log_probs}, data)

    padded_log_probs = no_padding_2_padding(log_probs, data)
    padded_data = data.select("response_mask", "old_log_probs", "advantages").to_padded_tensor()
    expected_pg_loss, expected_pg_metrics = compute_policy_loss_vanilla(
        old_log_prob=padded_data["old_log_probs"],
        log_prob=padded_log_probs,
        advantages=padded_data["advantages"],
        response_mask=padded_data["response_mask"].to(bool),
        loss_agg_mode=config.loss_agg_mode,
        config=config,
    )

    torch.testing.assert_close(policy_loss, expected_pg_loss, atol=1e-7, rtol=0.0)
    assert metrics["actor/pg_loss"].aggregate() == pytest.approx(expected_pg_loss.detach().item(), abs=1e-7)
    for metric_name, expected_value in expected_pg_metrics.items():
        assert metrics[metric_name].aggregate() == pytest.approx(expected_value, abs=1e-7)


def test_prepare_group_reference_actor_update_batch_attaches_group_ids_and_force_group_size(monkeypatch):
    from verl.trainer.ppo.ray_trainer import prepare_group_reference_actor_update_batch
    from verl.workers.utils.padding import left_right_2_no_padding

    def cpu_unpad_input(hidden_states, attention_mask):
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        unpadded = flat_hidden_states.index_select(0, indices)
        lengths = attention_mask.sum(dim=1, dtype=torch.int32)
        cu_seqlens = torch.zeros(lengths.numel() + 1, dtype=torch.int32)
        torch.cumsum(lengths, dim=0, out=cu_seqlens[1:])
        return unpadded, indices, cu_seqlens, int(lengths.max().item())

    monkeypatch.setattr("verl.workers.utils.padding.unpad_input", cpu_unpad_input)

    batch_size = 16
    batch = DataProto.from_single_dict(
        {
            "prompts": torch.zeros(batch_size, 1, dtype=torch.long),
            "responses": torch.zeros(batch_size, 2, dtype=torch.long),
            "input_ids": torch.arange(batch_size * 3, dtype=torch.long).reshape(batch_size, 3),
            "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
            "response_mask": torch.ones(batch_size, 2, dtype=torch.long),
            "position_ids": torch.arange(3, dtype=torch.long).repeat(batch_size, 1),
            "uid": np.array(["a"] * 8 + ["b"] * 8, dtype=object),
            "acc": np.array([0, 1] * 8, dtype=np.float32),
        }
    )

    metadata = prepare_group_reference_actor_update_batch(batch, _ggpo_config(group_size=8))

    assert metadata == {"force_group_size": 8, "micro_batch_size_per_gpu": 2}
    torch.testing.assert_close(batch.batch["policy_group_ids"], torch.tensor([0] * 8 + [1] * 8, dtype=torch.long))
    torch.testing.assert_close(batch.batch["_ggpo_raw_correctness"], torch.tensor([0, 1] * 8, dtype=torch.float32))
    batch_td = batch.to_tensordict()
    assert "policy_group_ids" in batch_td.keys()

    no_padding_batch_td = left_right_2_no_padding(batch_td)
    torch.testing.assert_close(
        no_padding_batch_td["policy_group_ids"],
        torch.tensor([0] * 8 + [1] * 8, dtype=torch.long),
    )
    tu.assign_non_tensor(no_padding_batch_td, use_dynamic_bsz=False, **metadata)
    micro_batches, batch_idx_list = prepare_micro_batches(no_padding_batch_td)
    assert batch_idx_list is None
    assert len(micro_batches) == 1
    assert len(micro_batches[0]) == 16


def test_prepare_group_reference_actor_update_batch_memory_safe_surrogate_uses_per_response_weights():
    from verl.experimental.fully_async_policy.actor_update_padding import (
        LOGICAL_MINI_BATCH_SIZE_META,
        PHYSICAL_MINI_BATCH_SIZE_META,
        REAL_SAMPLE_MASK,
    )
    from verl.trainer.ppo.ggpo import (
        GGPO_SURROGATE_CURRENT_LOG_PROBS,
        GGPO_SURROGATE_METRICS_META,
        GGPO_SURROGATE_WEIGHTS,
    )
    from verl.trainer.ppo.ray_trainer import prepare_group_reference_actor_update_batch

    config = OmegaConf.create(
        {
            "trainer": {"nnodes": 1, "n_gpus_per_node": 2},
            "actor_rollout_ref": {
                "actor": {
                    "ppo_micro_batch_size_per_gpu": 2,
                    "ulysses_sequence_parallel_size": 1,
                    "policy_loss": {
                        "loss_mode": "ggpo",
                        "beta": 0.1,
                        "sequence_log_prob_reduction": "mean",
                        "memory_safe_surrogate": True,
                    },
                },
                "rollout": {"n": 2},
            },
        }
    )
    batch = DataProto.from_single_dict(
        {
            "responses": torch.zeros(6, 2, dtype=torch.long),
            "response_mask": torch.ones(6, 2, dtype=torch.long),
            "token_level_scores": torch.tensor(
                [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            ),
            "ref_log_prob": torch.zeros(6, 2),
            GGPO_SURROGATE_CURRENT_LOG_PROBS: torch.tensor(
                [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4], [-0.4, -0.5], [-0.5, -0.6], [-0.6, -0.7]]
            ),
            REAL_SAMPLE_MASK: torch.tensor([True, True, True, True, False, False]),
            "uid": np.array(["a", "a", "b", "b", "padding", "padding"], dtype=object),
            "acc": np.array([1, 0, 0, 1, 1, 0], dtype=np.float32),
        },
        meta_info={LOGICAL_MINI_BATCH_SIZE_META: 4, PHYSICAL_MINI_BATCH_SIZE_META: 6},
    )

    metadata = prepare_group_reference_actor_update_batch(batch, config)

    assert metadata == {"force_group_size": 1}
    assert GGPO_SURROGATE_CURRENT_LOG_PROBS not in batch.batch
    assert torch.isfinite(batch.batch[GGPO_SURROGATE_WEIGHTS][:4]).all()
    torch.testing.assert_close(batch.batch[GGPO_SURROGATE_WEIGHTS][4:], torch.zeros(2))
    assert "ggpo_memory_safe_surrogate" not in batch.meta_info[GGPO_SURROGATE_METRICS_META]
    assert "ggpo_loss" in batch.meta_info[GGPO_SURROGATE_METRICS_META]
    assert "ggpo_normalized_coefficient_rms" in batch.meta_info[GGPO_SURROGATE_METRICS_META]


def test_prepare_group_reference_actor_update_batch_rejects_partial_public_response_micro_batch():
    from verl.trainer.ppo.ray_trainer import prepare_group_reference_actor_update_batch

    batch = DataProto.from_single_dict(
        {
            "responses": torch.zeros(16, 2, dtype=torch.long),
            "uid": np.array(["a"] * 8 + ["b"] * 8, dtype=object),
        }
    )

    with pytest.raises(ValueError, match="counts responses.*positive multiple"):
        prepare_group_reference_actor_update_batch(batch, _ggpo_config(group_size=8, micro_batch_size=12))


def test_prepare_group_reference_actor_update_batch_leaves_vanilla_batches_unchanged():
    from verl.trainer.ppo.ray_trainer import prepare_group_reference_actor_update_batch

    batch = DataProto.from_single_dict(
        {
            "responses": torch.zeros(2, 2, dtype=torch.long),
            "uid": np.array(["a", "b"], dtype=object),
        }
    )

    metadata = prepare_group_reference_actor_update_batch(batch, _ggpo_config(loss_mode="vanilla"))

    assert metadata == {}
    assert "policy_group_ids" not in batch.batch.keys()


def test_actor_config_rejects_ggpo_micro_batch_with_partial_group():
    config = _ppo_actor_config(loss_mode="ggpo", group_size=8)
    config.ppo_micro_batch_size_per_gpu = 12

    with pytest.raises(ValueError, match="counts responses.*positive multiple"):
        config.validate(n_gpus=4, train_batch_size=32)


def test_actor_config_accepts_ggpo_micro_batch_with_two_complete_groups():
    config = _ppo_actor_config(loss_mode="ggpo", group_size=8)
    config.ppo_mini_batch_size = 32
    config.ppo_micro_batch_size_per_gpu = 16

    config.validate(n_gpus=4, train_batch_size=32)


def test_actor_config_accepts_split_ggpo_micro_batch_with_memory_safe_surrogate():
    config = _ppo_actor_config(loss_mode="ggpo", group_size=8, memory_safe_surrogate=True)
    config.ppo_micro_batch_size_per_gpu = 4

    config.validate(n_gpus=4, train_batch_size=32)


@pytest.mark.parametrize("group_size", [8.5, "8"])
def test_get_policy_group_size_rejects_non_integral_values(group_size):
    from verl.trainer.ppo.ray_trainer import get_policy_group_size

    with pytest.raises(ValueError, match="rollout.n.*integer"):
        get_policy_group_size(_ggpo_config(group_size=group_size))


def test_policy_loss_trainer_compatibility_rejects_ggpo_balance_batch():
    from verl.utils.config import validate_policy_loss_trainer_compatibility

    with pytest.raises(ValueError, match="GGPO.*balance_batch"):
        validate_policy_loss_trainer_compatibility(_ggpo_config(balance_batch=True))


@pytest.mark.parametrize(
    "path",
    [
        "actor_rollout_ref.actor.clip_ratio",
        "actor_rollout_ref.actor.clip_ratio_low",
        "actor_rollout_ref.actor.clip_ratio_high",
        "actor_rollout_ref.actor.clip_ratio_c",
        "actor_rollout_ref.actor.policy_loss.ppo_kl_coef",
        "actor_rollout_ref.actor.policy_loss.clip_cov_lb",
        "actor_rollout_ref.actor.policy_loss.clip_cov_ub",
        "actor_rollout_ref.actor.policy_loss.clip_cov_ratio",
        "actor_rollout_ref.actor.policy_loss.kl_cov_ratio",
        "actor_rollout_ref.actor.loss_agg_mode",
        "actor_rollout_ref.actor.loss_scale_factor",
        "algorithm.adv_estimator",
        "algorithm.gamma",
        "algorithm.lam",
        "algorithm.norm_adv_by_std_in_grpo",
        "algorithm.kl_ctrl.target_kl",
        "algorithm.use_kl_in_reward",
        "algorithm.pf_ppo.reweight_method",
        "algorithm.pf_ppo.weight_pow",
        "algorithm.rollout_correction.rollout_is",
        "algorithm.rollout_correction.rollout_rs",
        "algorithm.rollout_correction.bypass_mode",
        "algorithm.rollout_correction.loss_type",
        "algorithm.rollout_correction.rollout_is_batch_normalize",
    ],
)
def test_policy_loss_trainer_compatibility_rejects_every_inert_ggpo_key(path):
    from verl.utils.config import validate_policy_loss_trainer_compatibility

    config = _ggpo_config()
    # The check is value-agnostic; any non-null value means the inert knob was configured.
    OmegaConf.update(config, path, 0.2, force_add=True)

    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        validate_policy_loss_trainer_compatibility(config)


@pytest.mark.parametrize(
    "path",
    [
        "actor_rollout_ref.actor.loss_scale_factor",
        "algorithm.rollout_correction.rollout_is",
        "algorithm.rollout_correction.rollout_rs",
    ],
)
def test_null_inert_ggpo_keys_are_accepted(path):
    """These three default to null and Hydra cannot delete a null-valued key, so rejecting
    them would leave no way to launch GGPO at all."""
    from verl.utils.config import _validate_ggpo_inert_config_is_absent

    config = _ggpo_config()
    OmegaConf.update(config, path, None, force_add=True)

    _validate_ggpo_inert_config_is_absent(config)


def test_inert_ggpo_keys_remain_valid_outside_ggpo():
    from verl.utils.config import validate_policy_loss_trainer_compatibility

    config = _ggpo_config(loss_mode="vanilla")
    OmegaConf.update(config, "actor_rollout_ref.actor.clip_ratio", 0.2, force_add=True)
    OmegaConf.update(config, "algorithm.adv_estimator", "grpo", force_add=True)
    OmegaConf.update(config, "algorithm.use_kl_in_reward", False, force_add=True)

    validate_policy_loss_trainer_compatibility(config)


def test_policy_loss_trainer_compatibility_rejects_agro_balance_batch():
    from verl.utils.config import validate_policy_loss_trainer_compatibility

    with pytest.raises(ValueError, match="AGRO.*balance_batch"):
        validate_policy_loss_trainer_compatibility(_ggpo_config(loss_mode="agro", balance_batch=True))


def test_policy_loss_trainer_compatibility_allows_vanilla_balance_batch():
    from verl.utils.config import validate_policy_loss_trainer_compatibility

    validate_policy_loss_trainer_compatibility(_ggpo_config(loss_mode="vanilla", balance_batch=True))


def test_policy_loss_trainer_compatibility_rejects_ggpo_v1_trainer():
    from verl.utils.config import validate_policy_loss_trainer_compatibility

    with pytest.raises(ValueError, match="GGPO.*use_v1=False"):
        validate_policy_loss_trainer_compatibility(_ggpo_config(use_v1=True))


def test_ggpo_advantage_bypass_adds_zero_compatibility_tensors():
    from verl.trainer.ppo.ray_trainer import maybe_apply_group_reference_advantage_bypass

    batch = DataProto.from_single_dict(
        {"token_level_rewards": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)}
    )

    assert maybe_apply_group_reference_advantage_bypass(batch, _ggpo_config()) is True
    torch.testing.assert_close(batch.batch["advantages"], torch.zeros_like(batch.batch["token_level_rewards"]))
    torch.testing.assert_close(batch.batch["returns"], torch.zeros_like(batch.batch["token_level_rewards"]))


def test_ggpo_advantage_bypass_leaves_vanilla_batches_unchanged():
    from verl.trainer.ppo.ray_trainer import maybe_apply_group_reference_advantage_bypass

    batch = DataProto.from_single_dict(
        {"token_level_rewards": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)}
    )

    assert maybe_apply_group_reference_advantage_bypass(batch, _ggpo_config(loss_mode="vanilla")) is False
    assert "advantages" not in batch.batch.keys()
    assert "returns" not in batch.batch.keys()


def test_shutdown_stateful_dataloader_workers_joins_and_detaches_iterators():
    from verl.trainer.ppo.ray_trainer import shutdown_stateful_dataloader_workers

    class Iterator:
        def __init__(self):
            self.shutdown_calls = 0

        def _shutdown_workers(self):
            self.shutdown_calls += 1

    class Loader:
        def __init__(self, iterator):
            self._iterator = iterator

    train_iterator = Iterator()
    val_iterator = Iterator()
    train_loader = Loader(train_iterator)
    val_loader = Loader(val_iterator)

    shutdown_stateful_dataloader_workers(train_loader, val_loader, Loader(None))

    assert train_iterator.shutdown_calls == 1
    assert val_iterator.shutdown_calls == 1
    assert train_loader._iterator is None
    assert val_loader._iterator is None


@pytest.mark.parametrize("padded", [False, True])
def test_ggpo_metric_weight_equals_the_group_count_the_diagnostics_counted(padded):
    """The companion-metric identity needs weight == group count per micro-batch.

    ``ppo_loss`` weights every GGPO metric by ``real_response_count //
    group_size`` while ``compute_ggpo_loss`` reports the group count it
    actually reduced over. If those two ever disagreed, the recovered
    ``*_over_all_groups / ggpo_tied_max_reward_subset_group_frac`` ratio would drift
    silently rather than fail, so pin them together here.
    """
    from verl.workers.utils.losses import ppo_loss

    real_response_log_probs = torch.tensor(
        [[-0.10, -0.20, -0.30], [-0.40, -0.50, -0.60], [-0.70, -0.80, -0.90], [-0.15, -0.25, -0.35]],
        dtype=torch.float32,
    )
    if padded:
        log_probs = _flat_model_log_probs_from_response_values(
            torch.cat((real_response_log_probs, torch.zeros(2, 3)), dim=0)
        )
        data = _make_group_padded_ppo_loss_batch()
    else:
        log_probs = _flat_model_log_probs_from_response_values(real_response_log_probs)
        data = _make_ppo_loss_batch()

    _, metrics = ppo_loss(_ppo_actor_config(loss_mode="ggpo", group_size=2), {"log_probs": log_probs}, data)

    ggpo_metrics = {name: metric for name, metric in metrics.items() if name.startswith("ggpo_")}
    assert ggpo_metrics
    observed_group_counts = ggpo_metrics["ggpo_groups_per_microbatch"].values
    assert observed_group_counts == [2.0]
    for name, metric in ggpo_metrics.items():
        assert metric.weights == observed_group_counts, name


def test_ggpo_rejects_a_micro_batch_whose_real_responses_split_a_group():
    from verl.experimental.fully_async_policy.actor_update_padding import REAL_SAMPLE_MASK
    from verl.workers.utils.losses import ppo_loss

    data = _make_group_padded_ppo_loss_batch()
    real_sample_mask = data[REAL_SAMPLE_MASK].clone()
    real_sample_mask[real_sample_mask.nonzero()[-1]] = False
    data[REAL_SAMPLE_MASK] = real_sample_mask
    log_probs = _flat_model_log_probs_from_response_values(torch.zeros(6, 3))

    with pytest.raises(ValueError, match="split a GGPO group"):
        ppo_loss(_ppo_actor_config(loss_mode="ggpo", group_size=2), {"log_probs": log_probs}, data)
