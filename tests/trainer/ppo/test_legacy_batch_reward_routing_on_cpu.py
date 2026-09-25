from types import SimpleNamespace

import numpy as np
import pytest

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _merge_validation_reward_extra_info


def test_legacy_agent_reward_loop_uses_manager_handle_policy():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.reward_loop_manager = SimpleNamespace(
        reward_loop_worker_handles=None,
        reward_loop_workers=["raw-worker-handle"],
    )

    assert trainer._select_reward_loop_worker_handles(enable_agent_reward_loop=True) is None


def test_legacy_colocated_reward_needed_when_rollout_has_no_rm_scores():
    batch_without_scores = SimpleNamespace(batch={})
    batch_with_scores = SimpleNamespace(batch={"rm_scores": object()})

    assert RayPPOTrainer._needs_colocated_reward(batch_without_scores)
    assert not RayPPOTrainer._needs_colocated_reward(batch_with_scores)


def test_validation_reward_extra_info_deduplicates_matching_canonical_diagnostic():
    aggregate = {"reward": [1.0, 0.0]}

    _merge_validation_reward_extra_info(
        aggregate=aggregate,
        reward_extra_info={
            "acc": np.asarray([1.0, 0.0]),
            "parser_success": [1.0, 1.0],
            "valid_response_length": np.asarray([17, 23]),
        },
        canonical_diagnostics={
            "valid_response_length": [17, 23],
            "eos": [1.0, 0.0],
            "truncation": [0.0, 1.0],
        },
    )

    assert aggregate == {
        "reward": [1.0, 0.0],
        "acc": [1.0, 0.0],
        "parser_success": [1.0, 1.0],
        "valid_response_length": [17, 23],
        "eos": [1.0, 0.0],
        "truncation": [0.0, 1.0],
    }


def test_validation_reward_extra_info_rejects_conflicting_canonical_diagnostic():
    with pytest.raises(ValueError, match="canonical validation diagnostic: valid_response_length"):
        _merge_validation_reward_extra_info(
            aggregate={},
            reward_extra_info={"valid_response_length": [17, 22]},
            canonical_diagnostics={"valid_response_length": [17, 23]},
        )
