import pytest
import torch

from verl.trainer.ppo.agro import compute_agro_loss


def _inputs(*, requires_grad=True):
    log_prob = torch.tensor(
        [[-0.2, -0.3, 5.0], [-0.7, -0.1, -0.2], [-0.4, -0.6, -0.5], [-0.8, -0.5, 7.0]],
        requires_grad=requires_grad,
    )
    ref = torch.tensor(
        [[-0.4, -0.2, 4.0], [-0.3, -0.2, -0.1], [-0.5, -0.4, -0.3], [-0.6, -0.3, 6.0]]
    )
    rewards = torch.tensor([1.0, 0.0, 0.25, 0.75])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0]])
    groups = torch.tensor([17, 4, 17, 4])
    return log_prob, ref, rewards, mask, groups


def _call(*inputs):
    return compute_agro_loss(
        *inputs,
        beta=0.001,
        group_size=2,
        global_batch_size=4,
        batch_num_tokens=int(inputs[3].sum()),
        dp_size=1,
    )


def test_agro_token_mean_gradient_matches_beta_normalized_variance():
    inputs = _inputs()
    surrogate, metrics = _call(*inputs)
    direct_log_prob = inputs[0].detach().clone().requires_grad_(True)
    mask = inputs[3].bool()
    regularized = inputs[2] - 0.001 * (
        (direct_log_prob * mask).sum(dim=-1) - (inputs[1] * mask).sum(dim=-1)
    )
    grouped = regularized[torch.argsort(inputs[4], stable=True)].reshape(-1, 2)
    centered = grouped - grouped.mean(dim=-1, keepdim=True)
    direct_variance = (centered.square().sum(dim=-1) / 2).mean()
    surrogate.backward()
    direct_variance.backward()

    expected_gradient = direct_log_prob.grad / 0.001 * (len(inputs[0]) / mask.sum())
    torch.testing.assert_close(inputs[0].grad, expected_gradient, atol=2e-9, rtol=2e-5)
    torch.testing.assert_close(metrics["agro_sample_variance"], direct_variance.detach())
    torch.testing.assert_close(inputs[0].grad[~mask], torch.zeros_like(inputs[0].grad[~mask]))


def test_agro_loo_delta_has_exact_n_over_n_minus_one_ratio():
    log_prob, ref, rewards, mask, groups = _inputs()
    _, metrics = _call(log_prob, ref, rewards, mask, groups)
    regularized = rewards - 0.001 * ((log_prob.detach() * mask).sum(-1) - (ref * mask).sum(-1))
    grouped = regularized[torch.argsort(groups, stable=True)].reshape(-1, 2)
    centered = grouped - grouped.mean(-1, keepdim=True)
    loo = grouped - (grouped.sum(-1, keepdim=True) - grouped)
    torch.testing.assert_close(loo, 2 * centered)
    torch.testing.assert_close(metrics["agro_loo_delta_abs_mean"], loo.abs().mean())


def test_agro_reference_policy_changes_regularized_reward_but_has_no_gradient():
    log_prob, ref, rewards, mask, groups = _inputs()
    ref.requires_grad_(True)
    _, first = _call(log_prob, ref, rewards, mask, groups)
    shifted = ref.detach() + mask * 0.25
    loss, second = _call(log_prob, shifted, rewards, mask, groups)
    loss.backward()
    assert ref.grad is None
    assert second["agro_regularized_reward_mean"] > first["agro_regularized_reward_mean"]


def test_agro_constant_regularized_reward_group_has_zero_update():
    log_prob = torch.full((8, 2), -0.5, requires_grad=True)
    loss, metrics = compute_agro_loss(
        log_prob,
        torch.full_like(log_prob, -0.5),
        torch.ones(8),
        torch.ones_like(log_prob),
        torch.zeros(8, dtype=torch.long),
        beta=0.001,
        group_size=8,
        global_batch_size=8,
        batch_num_tokens=16,
        dp_size=1,
    )
    loss.backward()
    torch.testing.assert_close(log_prob.grad, torch.zeros_like(log_prob))
    torch.testing.assert_close(metrics["agro_sample_variance"], torch.tensor(0.0))


@pytest.mark.parametrize(
    ("mutation", "exception", "message"),
    [
        (lambda x: x.__setitem__(0, torch.tensor(float("nan"))), FloatingPointError, "not finite"),
        (lambda x: x.__setitem__(1, torch.tensor(0.0)), ValueError, "at least one"),
    ],
)
def test_agro_fails_on_nonfinite_padding_or_empty_response(mutation, exception, message):
    inputs = list(_inputs())
    if exception is FloatingPointError:
        inputs[0].data[0, 2] = float("nan")
    else:
        inputs[3][1] = 0
    with pytest.raises(exception, match=message):
        _call(*inputs)


def test_agro_accepts_configurable_positive_beta():
    inputs = _inputs()
    loss, metrics = compute_agro_loss(
        *inputs, beta=0.25, group_size=2, global_batch_size=4, batch_num_tokens=int(inputs[3].sum()), dp_size=1,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["agro_sample_variance"])


def test_agro_rejects_group_size_one_and_nonpositive_beta():
    inputs = _inputs()
    with pytest.raises(ValueError, match="greater than 1"):
        compute_agro_loss(
            *inputs, beta=0.001, group_size=1, global_batch_size=4, batch_num_tokens=int(inputs[3].sum()), dp_size=1,
        )
    with pytest.raises(ValueError, match="positive"):
        compute_agro_loss(
            *inputs, beta=0.0, group_size=2, global_batch_size=4, batch_num_tokens=int(inputs[3].sum()), dp_size=1,
        )
