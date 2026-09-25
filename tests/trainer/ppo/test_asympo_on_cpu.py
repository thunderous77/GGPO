from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.ppo.asympo import compute_asympo_loss
from verl.trainer.ppo.core_algos import compute_policy_loss_asympo
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss


def _inputs():
    log_prob = torch.tensor(
        [[-0.2, -0.4, -9.0], [-0.3, -0.6, -0.9], [-0.1, -8.0, -8.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]])
    response_advantages = torch.tensor([2.0, -3.0, 1.0], dtype=torch.float64)
    advantages = response_advantages[:, None].expand_as(log_prob).clone()
    return log_prob, advantages, mask


def test_asympo_near_zero_forward_has_exact_nonzero_response_equal_gradient():
    log_prob, advantages, mask = _inputs()
    # Raw-reward mean centering makes the response advantages sum to zero.
    loss, metrics = compute_asympo_loss(log_prob, advantages, mask)
    loss.backward()
    lengths = mask.sum(-1)
    scales = -(log_prob.detach() * mask).sum(-1) / lengths
    expected = torch.zeros_like(log_prob)
    for row in range(log_prob.shape[0]):
        expected[row, mask[row].bool()] = -advantages[row, 0] / (3 * scales[row] * lengths[row])
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0)
    torch.testing.assert_close(log_prob.grad, expected, atol=1e-7, rtol=1e-6)
    assert torch.count_nonzero(log_prob.grad).item() == int(mask.sum())
    assert metrics["asympo_positive_contribution"] > 0
    assert metrics["asympo_negative_contribution"] < 0


def test_asympo_wrapper_accepts_no_old_log_probs_and_ignores_policy_correction_inputs():
    first_log_prob, advantages, mask = _inputs()
    second_log_prob = first_log_prob.detach().clone().requires_grad_(True)
    first, _ = compute_policy_loss_asympo(
        old_log_prob=None,
        log_prob=first_log_prob,
        advantages=advantages,
        response_mask=mask,
        loss_agg_mode="token-mean",
        config=SimpleNamespace(clip_ratio=0.01),
        rollout_is_weights=torch.zeros_like(first_log_prob),
    )
    second, _ = compute_policy_loss_asympo(
        old_log_prob=torch.full_like(second_log_prob, 100.0),
        log_prob=second_log_prob,
        advantages=advantages,
        response_mask=mask,
        loss_agg_mode="seq-mean-token-sum",
        config=SimpleNamespace(clip_ratio=99.0),
        rollout_is_weights=torch.full_like(second_log_prob, 1e9),
    )
    first.backward()
    second.backward()
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first_log_prob.grad, second_log_prob.grad)


def test_asympo_signed_contribution_metrics_compose_across_microbatches():
    log_prob, advantages, mask = _inputs()
    _, full_metrics = compute_asympo_loss(log_prob, advantages, mask)

    split_metrics = []
    for start, stop in ((0, 2), (2, 3)):
        _, metrics = compute_asympo_loss(
            log_prob[start:stop],
            advantages[start:stop],
            mask[start:stop],
            global_batch_size=log_prob.shape[0],
        )
        split_metrics.append((stop - start, metrics))

    for name in ("asympo_positive_contribution", "asympo_negative_contribution"):
        recombined = sum(size * metrics[name] for size, metrics in split_metrics) / log_prob.shape[0]
        torch.testing.assert_close(recombined, full_metrics[name])
    torch.testing.assert_close(
        full_metrics["asympo_positive_contribution"] + full_metrics["asympo_negative_contribution"],
        full_metrics["asympo_loss"],
    )


def _asympo_ppo_inputs_without_old_log_probs():
    response_log_probs = torch.tensor(
        [[-0.2, -0.4, -0.6], [-0.3, -0.6, -0.9], [-0.1, -0.2, -0.3]], dtype=torch.float32
    )
    response_mask = torch.ones_like(response_log_probs, dtype=torch.long)
    response_advantages = torch.tensor([2.0, -3.0, 1.0], dtype=torch.float32)
    advantages = response_advantages[:, None].expand_as(response_log_probs).clone()
    data = TensorDict(
        {
            "prompts": torch.zeros(3, 1, dtype=torch.long),
            "responses": torch.zeros(3, 3, dtype=torch.long),
            "attention_mask": torch.ones(3, 4, dtype=torch.long),
            "response_mask": response_mask,
            "advantages": advantages,
        },
        batch_size=[3],
    )
    tu.assign_non_tensor(
        data,
        dp_size=1,
        batch_num_tokens=int(response_mask.sum().item()),
        global_batch_size=3,
    )

    flat_log_probs = torch.zeros(12, dtype=torch.float32)
    for row, values in enumerate(response_log_probs):
        flat_log_probs[row * 4 : row * 4 + 3] = values
    return flat_log_probs.requires_grad_(), response_log_probs, advantages, response_mask, data


def test_ppo_loss_routes_asympo_without_behavior_log_probs_and_preserves_extrema():
    flat_log_probs, response_log_probs, advantages, response_mask, data = _asympo_ppo_inputs_without_old_log_probs()
    config = ActorConfig(
        strategy="fsdp",
        ppo_mini_batch_size=3,
        ppo_micro_batch_size_per_gpu=3,
        rollout_n=1,
        policy_loss=PolicyLossConfig(loss_mode="asympo"),
        use_torch_compile=False,
    )

    loss, metrics = ppo_loss(config, {"log_probs": flat_log_probs}, data)
    expected_loss, _ = compute_asympo_loss(response_log_probs, advantages, response_mask)

    assert "old_log_probs" not in data
    torch.testing.assert_close(loss, expected_loss)
    assert metrics["asympo_response_nll_min"].aggregation is AggregationType.MIN
    assert metrics["asympo_response_nll_max"].aggregation is AggregationType.MAX
    assert metrics["asympo_inverse_scale_max"].aggregation is AggregationType.MAX
    loss.backward()
    assert flat_log_probs.grad is not None
    assert torch.isfinite(flat_log_probs.grad).all()


def test_asympo_microbatch_scaling_accumulates_to_one_global_response_mean():
    full_log_prob, advantages, mask = _inputs()
    expected_loss, _ = compute_asympo_loss(full_log_prob, advantages, mask)
    expected_loss.backward()
    expected_gradient = full_log_prob.grad.detach().clone()

    accumulated_log_prob = full_log_prob.detach().clone().requires_grad_(True)
    accumulated_loss = torch.zeros((), dtype=torch.float32, device=accumulated_log_prob.device)
    for row in range(accumulated_log_prob.shape[0]):
        microbatch_loss, metrics = compute_asympo_loss(
            accumulated_log_prob[row : row + 1],
            advantages[row : row + 1],
            mask[row : row + 1],
            global_batch_size=accumulated_log_prob.shape[0],
            dp_size=1,
        )
        accumulated_loss = accumulated_loss + microbatch_loss
        torch.testing.assert_close(metrics["asympo_loss"], advantages[row, 0].float())
    accumulated_loss.backward()

    torch.testing.assert_close(accumulated_loss, expected_loss)
    torch.testing.assert_close(accumulated_log_prob.grad, expected_gradient)


def test_asympo_dp_scaled_local_losses_average_to_one_global_response_mean():
    full_log_prob, advantages, mask = _inputs()
    expected_loss, _ = compute_asympo_loss(full_log_prob, advantages, mask)
    expected_loss.backward()
    expected_gradient = full_log_prob.grad.detach().clone()

    distributed_log_prob = full_log_prob.detach().clone().requires_grad_(True)
    rank_losses = []
    for start, stop in ((0, 2), (2, 3)):
        rank_loss, _ = compute_asympo_loss(
            distributed_log_prob[start:stop],
            advantages[start:stop],
            mask[start:stop],
            global_batch_size=distributed_log_prob.shape[0],
            dp_size=2,
        )
        rank_losses.append(rank_loss)
    distributed_loss = torch.stack(rank_losses).mean()
    distributed_loss.backward()

    torch.testing.assert_close(distributed_loss, expected_loss)
    torch.testing.assert_close(distributed_log_prob.grad, expected_gradient)


@pytest.mark.parametrize(
    ("global_batch_size", "dp_size"),
    [(0, 1), (3, 0), (2, 1), (3.0, 1), (3, True)],
)
def test_asympo_rejects_invalid_global_batch_geometry(global_batch_size, dp_size):
    log_prob, advantages, mask = _inputs()
    with pytest.raises(ValueError, match="global_batch_size|dp_size|global batch geometry"):
        compute_asympo_loss(
            log_prob,
            advantages,
            mask,
            global_batch_size=global_batch_size,
            dp_size=dp_size,
        )


@pytest.mark.parametrize(
    ("mutator", "exception", "message"),
    [
        (lambda logp, adv, mask: mask.__setitem__(0, 0), ValueError, "at least one"),
        (lambda logp, adv, mask: logp.data.__setitem__((0, 2), float("nan")), FloatingPointError, "not finite"),
        (lambda logp, adv, mask: adv.__setitem__((1, 2), float("inf")), FloatingPointError, "not finite"),
        (lambda logp, adv, mask: adv.__setitem__((0, 1), 3.0), ValueError, "response-level advantage"),
        (lambda logp, adv, mask: logp.data.__setitem__((2, 0), 0.0), ValueError, "strictly positive"),
        (lambda logp, adv, mask: logp.data.__setitem__((2, 0), 0.1), ValueError, "strictly positive"),
    ],
)
def test_asympo_fail_fast_conditions(mutator, exception, message):
    log_prob, advantages, mask = _inputs()
    if message == "at least one":
        mask[0] = 0
    else:
        mutator(log_prob, advantages, mask)
    with pytest.raises(exception, match=message):
        compute_asympo_loss(log_prob, advantages, mask)


def test_asympo_rejects_nonbinary_mask_and_shape_mismatch():
    log_prob, advantages, mask = _inputs()
    mask[0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        compute_asympo_loss(log_prob, advantages, mask)
    with pytest.raises(ValueError, match="shapes"):
        compute_asympo_loss(log_prob, advantages[:, :2], torch.ones(3, 2))
