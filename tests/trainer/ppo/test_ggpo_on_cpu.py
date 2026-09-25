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

import pytest
import torch

from verl.trainer.ppo.ggpo import compute_ggpo_loss, compute_ggpo_surrogate_weights


def _historical_exp_ggpo_loss(
    log_prob,
    ref_log_prob,
    rewards,
    response_mask,
    group_ids,
    *,
    beta,
    group_size,
    global_batch_size,
    dp_size,
    sequence_log_prob_reduction="sum",
):
    mask = response_mask.bool()
    current = (log_prob.float() * mask).sum(dim=-1)
    reference = (ref_log_prob.float() * mask).sum(dim=-1)
    if sequence_log_prob_reduction == "mean":
        lengths = mask.sum(dim=-1)
        current = current / lengths
        reference = reference / lengths
    adjusted = rewards.float() - beta * (current - reference)
    grouped = adjusted[torch.argsort(group_ids, stable=True)].reshape(-1, group_size)
    per_group = beta * (torch.logsumexp(grouped / beta, dim=-1) - math.log(group_size))
    per_group = per_group - grouped.mean(dim=-1)
    return per_group.sum() * dp_size / (global_batch_size // group_size)


def _fixture():
    log_prob = torch.tensor(
        [[-0.2, -0.3], [-0.7, -0.1], [-0.4, -0.6], [-0.8, -0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    return (
        log_prob,
        torch.tensor([[-0.4, -0.2], [-0.3, -0.2], [-0.5, -0.4], [-0.6, -0.3]], dtype=torch.float64),
        torch.tensor([1, 0, 0, 1]),
        torch.ones_like(log_prob, dtype=torch.int64),
        torch.tensor([17, 4, 17, 4]),
    )


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_canonical_ggpo_matches_normalized_historical_exp_forward_and_gradient(reduction):
    inputs = _fixture()
    expected_log_prob = inputs[0].detach().clone().requires_grad_(True)
    loss, _ = compute_ggpo_loss(
        *inputs,
        raw_correctness=inputs[2],
        beta=0.25,
        group_size=2,
        global_batch_size=8,
        batch_num_tokens=16,
        dp_size=2,
        sequence_log_prob_reduction=reduction,
    )
    expected = _historical_exp_ggpo_loss(
        expected_log_prob,
        *inputs[1:],
        beta=0.25,
        group_size=2,
        global_batch_size=8,
        dp_size=2,
        sequence_log_prob_reduction=reduction,
    )
    expected = expected / (0.25 * math.sqrt(2 - 1))
    if reduction == "sum":
        expected = expected / 2
    loss.backward()
    expected.backward()

    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(inputs[0].grad, expected_log_prob.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("removed_kwarg", "value"),
    [("objective", "exp"), ("objective_normalization", "beta_sqrt_group_minus_one")],
)
@pytest.mark.parametrize("compute", [compute_ggpo_loss, compute_ggpo_surrogate_weights])
def test_removed_objective_selectors_are_not_accepted(compute, removed_kwarg, value):
    kwargs = {"raw_correctness": _fixture()[2]} if compute is compute_ggpo_loss else {}
    with pytest.raises(TypeError, match=removed_kwarg):
        compute(
            *_fixture(),
            beta=0.25,
            group_size=2,
            global_batch_size=4,
            batch_num_tokens=8,
            dp_size=1,
            **kwargs,
            **{removed_kwarg: value},
        )


def test_canonical_ggpo_always_scales_loss_and_gradient():
    beta = 0.001
    group_size = 16
    ref = torch.linspace(-0.2, 0.2, group_size).unsqueeze(-1)
    rewards = torch.tensor([1.0, 0.0] * (group_size // 2))
    mask = torch.ones(group_size, 1, dtype=torch.bool)
    groups = torch.zeros(group_size, dtype=torch.long)
    expected_log_prob = torch.linspace(-0.4, -0.1, group_size).unsqueeze(-1).requires_grad_(True)
    expected = _historical_exp_ggpo_loss(
        expected_log_prob,
        ref,
        rewards,
        mask,
        groups,
        beta=beta,
        group_size=group_size,
        global_batch_size=group_size,
        dp_size=1,
        sequence_log_prob_reduction="mean",
    )
    expected.backward()
    log_prob = expected_log_prob.detach().clone().requires_grad_(True)
    loss, metrics = compute_ggpo_loss(
        log_prob,
        ref,
        rewards,
        mask,
        groups,
        raw_correctness=rewards,
        beta=beta,
        group_size=group_size,
        global_batch_size=group_size,
        batch_num_tokens=group_size,
        dp_size=1,
        sequence_log_prob_reduction="mean",
    )
    loss.backward()
    scale = beta * math.sqrt(group_size - 1)
    torch.testing.assert_close(loss, expected / scale)
    torch.testing.assert_close(log_prob.grad, expected_log_prob.grad / scale)
    torch.testing.assert_close(metrics["ggpo_loss"], loss.detach())
    torch.testing.assert_close(metrics["ggpo_raw_loss"], expected.detach())


def test_ggpo_padding_tokens_have_zero_gradient_and_inputs_are_detached():
    log_prob = torch.tensor([[-0.2, 10.0], [-0.5, -10.0]], requires_grad=True)
    ref = torch.tensor([[-0.4, 3.0], [-0.3, 4.0]], requires_grad=True)
    rewards = torch.tensor([1.0, 0.0], requires_grad=True)
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    group_ids = torch.tensor([3.0, 3.0], requires_grad=True)
    loss, _ = compute_ggpo_loss(
        log_prob,
        ref,
        rewards,
        mask,
        group_ids,
        raw_correctness=rewards,
        beta=0.25,
        group_size=2,
        global_batch_size=2,
        batch_num_tokens=2,
        dp_size=1,
    )
    loss.backward()
    torch.testing.assert_close(log_prob.grad[:, 1], torch.zeros(2))
    assert ref.grad is rewards.grad is mask.grad is group_ids.grad is None


def test_ggpo_equal_adjusted_rewards_have_zero_loss_and_gradient():
    log_prob = torch.zeros(4, 2, requires_grad=True)
    loss, _ = compute_ggpo_loss(
        log_prob,
        torch.zeros_like(log_prob),
        torch.zeros(4),
        torch.ones_like(log_prob),
        torch.zeros(4, dtype=torch.long),
        raw_correctness=torch.zeros(4),
        beta=0.25,
        group_size=4,
        global_batch_size=4,
        batch_num_tokens=8,
        dp_size=1,
    )
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0)
    torch.testing.assert_close(log_prob.grad, torch.zeros_like(log_prob), atol=1e-7, rtol=0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"beta": 0.0}, "beta must be positive"),
        ({"group_size": 1}, "group_size must be greater than 1"),
        ({"dp_size": 0}, "dp_size must be positive"),
        ({"global_batch_size": 3}, "divisible"),
        ({"sequence_log_prob_reduction": "token_mean"}, "sequence_log_prob_reduction"),
    ],
)
def test_ggpo_rejects_invalid_scalar_configuration(change, message):
    kwargs = dict(beta=0.25, group_size=2, global_batch_size=4, batch_num_tokens=8, dp_size=1)
    kwargs.update(change)
    with pytest.raises(ValueError, match=message):
        inputs = _fixture()
        compute_ggpo_loss(*inputs, raw_correctness=inputs[2], **kwargs)


def test_ggpo_rejects_incomplete_groups_empty_response_and_nonfinite_loss():
    inputs = list(_fixture())
    inputs[4] = torch.tensor([0, 0, 1, 2])
    with pytest.raises(ValueError, match="exactly group_size"):
        compute_ggpo_loss(
            *inputs,
            raw_correctness=inputs[2],
            beta=0.25,
            group_size=2,
            global_batch_size=4,
            batch_num_tokens=8,
            dp_size=1,
        )
    inputs = list(_fixture())
    inputs[3][0] = 0
    with pytest.raises(ValueError, match="at least one valid token"):
        compute_ggpo_loss(
            *inputs,
            raw_correctness=inputs[2],
            beta=0.25,
            group_size=2,
            global_batch_size=4,
            batch_num_tokens=8,
            dp_size=1,
        )
    inputs = list(_fixture())
    inputs[0].data[0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="not finite"):
        compute_ggpo_loss(
            *inputs,
            raw_correctness=inputs[2],
            beta=0.25,
            group_size=2,
            global_batch_size=4,
            batch_num_tokens=8,
            dp_size=1,
        )


def test_ggpo_signed_binary_rewards_keep_group_classification():
    log_prob = torch.zeros((6, 2), dtype=torch.float64, requires_grad=True)
    loss, metrics = compute_ggpo_loss(
        log_prob,
        torch.zeros_like(log_prob),
        torch.tensor([-5, -5, 5, 5, -5, 5], dtype=torch.float64),
        torch.ones_like(log_prob),
        torch.tensor([0, 0, 1, 1, 2, 2]),
        raw_correctness=torch.tensor([0, 0, 1, 1, 0, 1]),
        beta=0.001,
        group_size=2,
        global_batch_size=6,
        batch_num_tokens=12,
        dp_size=1,
        sequence_log_prob_reduction="mean",
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(log_prob.grad).all()
    assert metrics["ggpo_all_zero_group_frac"] == pytest.approx(1 / 3)
    assert metrics["ggpo_all_one_group_frac"] == pytest.approx(1 / 3)
    assert metrics["ggpo_informative_group_frac"] == pytest.approx(1 / 3)
    # The one informative group has a single max-reward response, so the
    # tie-break subset is empty. The family is still emitted, reporting an
    # empty selected population rather than disappearing.
    assert metrics["ggpo_tied_max_reward_subset_group_frac"] == 0
    assert metrics["ggpo_informative_group_frac_tied_max_reward_k_ge_2"] == 0
    assert metrics["ggpo_mean_tied_max_reward_k_in_informative_groups"] == pytest.approx(1.0)


def test_ggpo_constant_group_classification_uses_raw_correctness_with_shaped_rewards():
    log_prob = torch.zeros((4, 1), requires_grad=True)
    _, metrics = compute_ggpo_loss(
        log_prob,
        torch.zeros_like(log_prob),
        torch.tensor([-0.1, -0.2, 0.8, 0.6]),
        torch.ones_like(log_prob),
        torch.tensor([0, 0, 1, 1]),
        raw_correctness=torch.tensor([0, 0, 1, 1]),
        beta=0.5,
        group_size=2,
        global_batch_size=4,
        batch_num_tokens=4,
        dp_size=1,
    )

    assert metrics["ggpo_all_zero_group_frac"] == pytest.approx(0.5)
    assert metrics["ggpo_all_one_group_frac"] == pytest.approx(0.5)
    assert metrics["ggpo_informative_group_frac"] == 0
    # Shaped rewards do not suppress the tie-break diagnostic: inside a
    # tied-max subset the reward cancels whatever its numeric value is.
    assert metrics["ggpo_tied_max_reward_subset_group_frac"] == 0


def test_ggpo_mean_reduction_scales_each_response_by_its_valid_length():
    log_prob = torch.tensor([[-0.2, 9.0, 9.0], [-0.5, -0.4, -0.3]], requires_grad=True)
    expected_log_prob = log_prob.detach().clone().requires_grad_(True)
    ref = torch.tensor([[-0.1, 8.0, 8.0], [-0.3, -0.2, -0.1]])
    rewards = torch.tensor([1.0, 0.0])
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]])
    groups = torch.tensor([4, 4])

    loss, metrics = compute_ggpo_loss(
        log_prob,
        ref,
        rewards,
        mask,
        groups,
        raw_correctness=rewards,
        beta=1.0,
        group_size=2,
        global_batch_size=2,
        batch_num_tokens=4,
        dp_size=1,
        sequence_log_prob_reduction="mean",
    )
    expected = _historical_exp_ggpo_loss(
        expected_log_prob,
        ref,
        rewards,
        mask,
        groups,
        beta=1.0,
        group_size=2,
        global_batch_size=2,
        dp_size=1,
        sequence_log_prob_reduction="mean",
    )
    loss.backward()
    expected.backward()

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(log_prob.grad, expected_log_prob.grad)
    torch.testing.assert_close(log_prob.grad[0, 1:], torch.zeros(2))
    assert metrics["ggpo_raw_sequence_log_ratio_mean_microbatch_std"] > metrics[
        "ggpo_sequence_log_ratio_mean_microbatch_std"
    ]


def _assert_diagnostics_are_finite(metrics):
    for name, value in metrics.items():
        assert torch.isfinite(value), name


def test_ggpo_returns_detached_finite_diagnostics():
    inputs = _fixture()
    _, metrics = compute_ggpo_loss(
        *inputs,
        raw_correctness=inputs[2],
        beta=0.25,
        group_size=2,
        global_batch_size=4,
        batch_num_tokens=8,
        dp_size=1,
    )

    assert metrics
    assert all(not value.requires_grad for value in metrics.values())
    _assert_diagnostics_are_finite(metrics)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values: values.__setitem__(0, values[0].unsqueeze(0)),
        lambda values: values.__setitem__(1, values[1][:, :1]),
        lambda values: values.__setitem__(2, values[2].unsqueeze(-1)),
        lambda values: values.__setitem__(3, values[3][:, :1]),
        lambda values: values.__setitem__(4, values[4].unsqueeze(-1)),
    ],
)
def test_ggpo_rejects_invalid_tensor_shapes(mutate):
    inputs = list(_fixture())
    mutate(inputs)
    with pytest.raises(ValueError, match="shape"):
        compute_ggpo_loss(
            *inputs,
            raw_correctness=inputs[2],
            beta=0.25,
            group_size=2,
            global_batch_size=4,
            batch_num_tokens=8,
            dp_size=1,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (4, torch.tensor([1.1, 1.9, 2.0, 2.0]), "finite integral group_ids"),
        (4, torch.tensor([float("nan"), 1.0, 2.0, 2.0]), "finite integral group_ids"),
        (3, torch.tensor([[2, 1], [1, 1], [1, 1], [1, 1]]), "response_mask must be binary"),
    ],
)
def test_ggpo_rejects_invalid_group_and_mask_values(field, value, message):
    inputs = list(_fixture())
    inputs[field] = value
    with pytest.raises(ValueError, match=message):
        compute_ggpo_loss(
            *inputs,
            raw_correctness=inputs[2],
            beta=0.25,
            group_size=2,
            global_batch_size=4,
            batch_num_tokens=8,
            dp_size=1,
        )


@pytest.mark.parametrize(
    "raw_correctness",
    [torch.tensor([-1.0, 0.0, 0.0, 1.0]), torch.tensor([0.0, float("nan"), 0.0, 1.0])],
)
def test_ggpo_rejects_nonbinary_raw_correctness(raw_correctness):
    inputs = _fixture()
    with pytest.raises(ValueError, match="raw_correctness must be finite and exactly binary"):
        compute_ggpo_loss(
            *inputs,
            raw_correctness=raw_correctness,
            beta=0.25,
            group_size=2,
            global_batch_size=4,
            batch_num_tokens=8,
            dp_size=1,
        )


def test_ggpo_distributed_seqsum_uses_one_global_token_denominator():
    rank0 = _fixture()
    rank1_values = tuple(value.detach().clone() for value in _fixture())
    rank1 = (*rank1_values[:4], rank1_values[4] + 100)
    global_log_prob = torch.cat((rank0[0].detach(), rank1[0])).requires_grad_(True)
    global_inputs = (
        global_log_prob,
        *(torch.cat((rank0[index].detach(), rank1[index])) for index in range(1, 5)),
    )
    global_token_count = int(global_inputs[3].sum().item())
    global_loss, global_metrics = compute_ggpo_loss(
        *global_inputs,
        raw_correctness=global_inputs[2],
        beta=0.25,
        group_size=2,
        global_batch_size=8,
        batch_num_tokens=global_token_count,
        dp_size=1,
    )
    global_loss.backward()

    rank_losses = []
    rank_grads = []
    for rank_inputs in (rank0, rank1):
        rank_log_prob = rank_inputs[0].detach().clone().requires_grad_(True)
        rank_loss, rank_metrics = compute_ggpo_loss(
            rank_log_prob,
            *rank_inputs[1:],
            raw_correctness=rank_inputs[2],
            beta=0.25,
            group_size=2,
            global_batch_size=8,
            batch_num_tokens=global_token_count,
            dp_size=2,
        )
        rank_loss.backward()
        rank_losses.append(rank_loss.detach())
        rank_grads.append(rank_log_prob.grad / 2)
        assert rank_metrics["ggpo_global_valid_token_count"] == global_token_count

    torch.testing.assert_close(torch.stack(rank_losses).mean(), global_loss.detach())
    torch.testing.assert_close(torch.cat(rank_grads), global_log_prob.grad)
    assert global_metrics["ggpo_global_valid_token_count"] == global_token_count


def test_ggpo_seqsum_variable_lengths_use_global_token_mean_not_response_means():
    log_prob = torch.tensor([[-0.2, 7.0, 7.0], [-0.5, -0.4, -0.3]], requires_grad=True)
    ref = torch.tensor([[-0.1, 8.0, 8.0], [-0.3, -0.2, -0.1]])
    rewards = torch.tensor([1.0, 0.0])
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]])
    groups = torch.tensor([0, 0])
    token_count = int(mask.sum().item())

    loss, metrics = compute_ggpo_loss(
        log_prob,
        ref,
        rewards,
        mask,
        groups,
        raw_correctness=rewards,
        beta=0.5,
        group_size=2,
        global_batch_size=2,
        batch_num_tokens=token_count,
        dp_size=1,
        sequence_log_prob_reduction="sum",
    )
    loss.backward()

    with torch.no_grad():
        sequence_ratio = ((log_prob - ref) * mask).sum(dim=-1)
        q = torch.softmax(rewards / 0.5 - sequence_ratio, dim=0)
        coefficients = (2 * q - 1) / math.sqrt(2 - 1)
        expected_gradient = -coefficients[:, None] * mask / token_count
        response_mean_gradient = -coefficients[:, None] * mask / (2 * mask.sum(dim=-1, keepdim=True))
    torch.testing.assert_close(log_prob.grad, expected_gradient)
    assert not torch.allclose(log_prob.grad, response_mean_gradient)
    assert metrics["ggpo_global_valid_token_count"] == token_count
    assert metrics["ggpo_batch_mean_response_length"] == pytest.approx(token_count / 2)
    assert {
        "ggpo_global_token_mean_applied",
        "ggpo_sequence_log_prob_reduction_is_sum",
    }.isdisjoint(metrics)
    assert metrics["ggpo_q_loss_normalization_max_abs_diff"] == 0
    assert metrics["ggpo_q_unchanged_after_loss_normalization"] == 1
    torch.testing.assert_close(
        metrics["ggpo_loss"],
        metrics["ggpo_raw_loss"] * metrics["ggpo_normalized_loss_to_raw_loss_ratio"],
    )


def test_ggpo_is_stable_for_extreme_finite_adjusted_rewards():
    log_prob = torch.tensor([[1e20], [-1e20]], requires_grad=True)
    loss, metrics = compute_ggpo_loss(
        log_prob,
        torch.zeros_like(log_prob),
        torch.tensor([0, 1]),
        torch.ones_like(log_prob),
        torch.tensor([6, 6]),
        raw_correctness=torch.tensor([0, 1]),
        beta=0.125,
        group_size=2,
        global_batch_size=2,
        batch_num_tokens=2,
        dp_size=1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(log_prob.grad).all()
    _assert_diagnostics_are_finite(metrics)


def test_ggpo_tied_max_reward_subset_and_constant_group_diagnostics():
    # One informative group with two max-reward responses and one all-zero
    # constant group. Single-token responses make Delta_i = log_prob - ref.
    deltas = torch.tensor([[0.0, 1.0, 5.0, 7.0], [0.0, 1.0, 2.0, 3.0]])
    log_prob = deltas.reshape(-1, 1).clone().requires_grad_(True)
    ref = torch.zeros_like(log_prob)
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mask = torch.ones_like(ref)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    beta = 0.5

    _, metrics = compute_ggpo_loss(
        log_prob,
        ref,
        rewards,
        mask,
        groups,
        raw_correctness=rewards,
        beta=beta,
        group_size=4,
        global_batch_size=8,
        batch_num_tokens=8,
        dp_size=1,
    )

    subset_q = torch.softmax(torch.tensor([2.0, 1.0]), dim=0)
    centered = torch.tensor([-0.5, 0.5])
    assert metrics["ggpo_tied_max_reward_subset_group_count_mean"] == 1
    assert metrics["ggpo_tied_max_reward_subset_group_frac"] == pytest.approx(0.5)
    assert metrics["ggpo_mean_tied_max_reward_k_in_informative_groups"] == pytest.approx(2.0)
    assert metrics["ggpo_tied_max_reward_subset_delta_spread"] == pytest.approx(0.5)
    assert metrics["ggpo_tied_max_reward_subset_ess_ratio"] == pytest.approx(
        float(1.0 / subset_q.square().sum() / 2.0), rel=1e-6
    )
    assert metrics["ggpo_tied_max_reward_subset_delta_weighted_mean"] == pytest.approx(
        float((subset_q * centered).sum()), rel=1e-6
    )

    constant_q = torch.softmax(-deltas[1], dim=0)
    constant_coefficients = (4 * constant_q - 1) / math.sqrt(3)
    constant_centered = deltas[1] - deltas[1].mean()
    terms = constant_coefficients * constant_centered
    assert metrics["ggpo_const_group_coefficient_rms"] == pytest.approx(
        float(constant_coefficients.square().mean().sqrt()), rel=1e-6
    )
    assert metrics["ggpo_const_group_kl_alignment"] == pytest.approx(
        float(-terms.sum() / (terms.abs().sum() + 1e-12)), rel=1e-6
    )
    assert 0.0 < metrics["ggpo_const_group_squared_mass_share"] < 1.0


# Four single-token groups covering k=1, k=2, k=3 and one constant-reward
# group. With a zero reference and sum reduction, Delta_i is exactly log_prob.
_SUBSET_DELTAS = torch.tensor(
    [
        [0.0, 1.0, 5.0, 7.0],  # k = 2, qualifying
        [0.0, 2.0, 3.0, 4.0],  # k = 1, informative but not qualifying
        [0.0, 1.0, 2.0, 9.0],  # k = 3, qualifying
        [1.0, 2.0, 3.0, 4.0],  # constant rewards, not qualifying
    ]
)
_SUBSET_REWARDS = torch.tensor(
    [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
)
_SUBSET_QUALIFYING_K = (2, 3)


def _subset_reference_values():
    """Hand-derived per-group values for the qualifying groups.

    Inside a max-reward subset the rewards cancel, so q restricted to that
    subset is softmax(-Delta) regardless of beta.
    """
    spreads, ess_ratios, weighted_means = [], [], []
    for row, k in zip((_SUBSET_DELTAS[0], _SUBSET_DELTAS[2]), _SUBSET_QUALIFYING_K):
        subset = row[:k]
        centered = subset - subset.mean()
        q_tilde = torch.softmax(-subset, dim=0)
        spreads.append(float(centered.square().mean().sqrt()))
        ess_ratios.append(float(1.0 / q_tilde.square().sum() / k))
        weighted_means.append(float((q_tilde * centered).sum()))
    mean = lambda values: sum(values) / len(values)  # noqa: E731
    return {
        "ggpo_tied_max_reward_subset_delta_spread": mean(spreads),
        "ggpo_tied_max_reward_subset_ess_ratio": mean(ess_ratios),
        "ggpo_tied_max_reward_subset_delta_weighted_mean": mean(weighted_means),
    }


def _subset_metrics(group_slice=slice(None), *, beta=0.5):
    deltas = _SUBSET_DELTAS[group_slice]
    rewards = _SUBSET_REWARDS[group_slice]
    group_count = deltas.shape[0]
    log_prob = deltas.reshape(-1, 1).clone()
    group_ids = torch.arange(group_count).repeat_interleave(4)
    _, metrics = compute_ggpo_loss(
        log_prob,
        torch.zeros_like(log_prob),
        rewards.reshape(-1).clone(),
        torch.ones_like(log_prob),
        group_ids,
        raw_correctness=rewards.reshape(-1).clone(),
        beta=beta,
        group_size=4,
        global_batch_size=16,
        batch_num_tokens=16,
        dp_size=1,
    )
    return metrics


def test_ggpo_tied_max_reward_subset_companions_recover_the_qualifying_group_mean():
    metrics = _subset_metrics()
    expected = _subset_reference_values()

    assert metrics["ggpo_groups_per_microbatch"] == 4
    assert metrics["ggpo_tied_max_reward_subset_group_count_mean"] == 2
    assert metrics["ggpo_tied_max_reward_subset_group_frac"] == pytest.approx(0.5)
    assert metrics["ggpo_mean_tied_max_reward_k_in_informative_groups"] == pytest.approx(2.0)

    for name, expected_value in expected.items():
        assert metrics[name] == pytest.approx(expected_value, rel=1e-6)
        recovered = metrics[f"{name}_over_all_groups"] / metrics["ggpo_tied_max_reward_subset_group_frac"]
        assert float(recovered) == pytest.approx(expected_value, rel=1e-6)


def test_ggpo_tied_max_reward_subset_metrics_use_selected_population_weight():
    """The companion identity must hold through verl's actual metric reduction.

    ``ppo_loss`` weights conditional metrics by the selected population and
    companion metrics by the micro-batch group count. Micro-batches holding
    different numbers of groups, including an empty selected population, must
    still reduce to the exact global mean over qualifying groups.
    """
    from verl.utils.metric import AggregationType, Metric

    expected = _subset_reference_values()
    micro_batches = [slice(1, 2), slice(0, 1), slice(2, 4)]
    aggregated = {}
    for name in (*expected, "ggpo_tied_max_reward_subset_group_frac"):
        keys = (name, f"{name}_over_all_groups") if name in expected else (name,)
        for key in keys:
            metric = Metric(aggregation=AggregationType.MEAN)
            for group_slice in micro_batches:
                metrics = _subset_metrics(group_slice)
                if key in expected:
                    weight = int(metrics["ggpo_tied_max_reward_subset_group_count_mean"].item())
                else:
                    weight = int(metrics["ggpo_groups_per_microbatch"].item())
                if weight:
                    metric.append(metrics[key], weight=weight)
            aggregated[key] = metric.aggregate()

    assert aggregated["ggpo_tied_max_reward_subset_group_frac"] == pytest.approx(0.5)
    for name, expected_value in expected.items():
        assert aggregated[name] == pytest.approx(expected_value, rel=1e-6)
        recovered = (
            aggregated[f"{name}_over_all_groups"]
            / aggregated["ggpo_tied_max_reward_subset_group_frac"]
        )
        assert recovered == pytest.approx(expected_value, rel=1e-6)


def test_ggpo_empty_tied_max_reward_subset_uses_finite_zero_placeholders():
    metrics = _subset_metrics(slice(1, 2))

    assert metrics["ggpo_tied_max_reward_subset_delta_spread"] == 0
    assert metrics["ggpo_tied_max_reward_subset_ess_ratio"] == 0
    assert metrics["ggpo_tied_max_reward_subset_delta_weighted_mean"] == 0
    assert metrics["ggpo_const_group_kl_alignment"] == 0
    assert metrics["ggpo_tied_max_reward_subset_group_frac"] == 0
    assert metrics["ggpo_tied_max_reward_subset_delta_spread_over_all_groups"] == 0
    assert metrics["ggpo_tied_max_reward_subset_ess_ratio_over_all_groups"] == 0
    assert metrics["ggpo_tied_max_reward_subset_delta_weighted_mean_over_all_groups"] == 0
    assert torch.isfinite(metrics["ggpo_normalized_coefficient_rms"])
    assert torch.isfinite(metrics["ggpo_loss"])


def test_ggpo_tied_max_reward_subset_metrics_survive_underflowing_subset_mass():
    """q restricted to the subset must be renormalized in log space.

    At beta=1e-3 a within-group Delta spread approaching r/beta drives the
    max-reward subset's share of q below float32's smallest normal, so
    dividing the exponentiated q by its subset mass would report a degenerate
    ESS ratio instead of the real one.
    """
    deltas = torch.tensor([[3000.0, 3001.5, 0.0, 0.0]])
    log_prob = deltas.reshape(-1, 1).clone()
    _, metrics = compute_ggpo_loss(
        log_prob,
        torch.zeros_like(log_prob),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        torch.ones_like(log_prob),
        torch.zeros(4, dtype=torch.long),
        raw_correctness=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        beta=1e-3,
        group_size=4,
        global_batch_size=4,
        batch_num_tokens=4,
        dp_size=1,
    )

    subset_q = torch.softmax(-deltas[0, :2], dim=0)
    centered = deltas[0, :2] - deltas[0, :2].mean()
    assert metrics["ggpo_tied_max_reward_subset_delta_spread"] == pytest.approx(0.75)
    assert metrics["ggpo_tied_max_reward_subset_ess_ratio"] == pytest.approx(
        float(1.0 / subset_q.square().sum() / 2.0), rel=1e-4
    )
    assert metrics["ggpo_tied_max_reward_subset_delta_weighted_mean"] == pytest.approx(
        float((subset_q * centered).sum()), rel=1e-4
    )


def test_ggpo_metric_schema_does_not_depend_on_reward_shaping():
    """Shaped rewards must not change which GGPO metrics exist.

    The tied-max-reward family is a KL tie-break diagnostic: within the subset
    whose reward equals the group maximum the reward term cancels exactly, so q
    there is driven purely by exp(-Delta_i). That cancellation holds for any
    reward scale, so the schema is the same under binary and shaped rewards.
    """
    log_prob = torch.zeros((8, 3), requires_grad=True)
    ref = torch.zeros_like(log_prob)
    mask = torch.ones_like(log_prob, dtype=torch.bool)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    acc = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    common = dict(
        beta=0.001,
        group_size=4,
        global_batch_size=8,
        batch_num_tokens=24,
        dp_size=1,
        sequence_log_prob_reduction="sum",
    )

    _, binary = compute_ggpo_loss(
        log_prob, ref, acc.clone(), mask, groups, raw_correctness=acc, **common
    )
    shaped = acc - torch.tensor([0.0, 0.0, 0.3, 0.0, 0.1, 0.0, 0.0, 0.2])
    _, penalized = compute_ggpo_loss(
        log_prob, ref, shaped, mask, groups, raw_correctness=acc, **common
    )

    assert set(binary) == set(penalized)
    assert any("tied_max_reward" in name for name in binary)
