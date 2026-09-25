import pytest
from omegaconf import OmegaConf

from verl.experimental.separation.utils import create_resource_pool_manager
from verl.experimental.fully_async_policy.rollout_backend.areal_rollouter import _validate_strict_6_2_topology
from verl.trainer.ppo.utils import Role
from verl.utils.config import get_trainer_process_on_nodes


def _config(*, nnodes=2, n_gpus_per_node=4, process_on_nodes=None):
    trainer = {
        "nnodes": nnodes,
        "n_gpus_per_node": n_gpus_per_node,
    }
    if process_on_nodes is not None:
        trainer["process_on_nodes"] = process_on_nodes
    return OmegaConf.create({"trainer": trainer})


def test_legacy_rectangular_trainer_topology_is_unchanged():
    config = _config(nnodes=2, n_gpus_per_node=3)

    assert get_trainer_process_on_nodes(config) == [3, 3]
    manager = create_resource_pool_manager(config, roles=[Role.Actor])
    assert manager.resource_pool_spec == {"trainer_pool": [3, 3]}


def test_two_learner_gpus_are_one_packed_node_and_strict_six_two_layout_validates():
    config = _config(nnodes=1, n_gpus_per_node=2)
    assert get_trainer_process_on_nodes(config) == [2]
    manager = create_resource_pool_manager(config, roles=[Role.Actor, Role.RefPolicy])
    assert manager.resource_pool_spec == {"trainer_pool": [2]}
    _validate_strict_6_2_topology(
        live_gpu_counts=[4, 4], requested_rollout_gpus=6, available_gpus=6
    )
    _validate_strict_6_2_topology(
        live_gpu_counts=[4, 4], requested_rollout_gpus=6, available_gpus=0, initialized=True
    )


@pytest.mark.parametrize(
    ("live_gpu_counts", "requested", "available", "initialized"),
    [([4], 6, 6, False), ([4, 4], 5, 6, False), ([4, 4], 6, 5, False), ([4, 4], 6, 1, True)],
)
def test_strict_six_two_layout_rejects_wrong_resources(live_gpu_counts, requested, available, initialized):
    with pytest.raises(RuntimeError, match="strict 6:2 placement"):
        _validate_strict_6_2_topology(
            live_gpu_counts=live_gpu_counts,
            requested_rollout_gpus=requested,
            available_gpus=available,
            initialized=initialized,
        )


def test_uneven_trainer_topology_reaches_ray_resource_pool_spec():
    config = _config(process_on_nodes=[4, 1])

    assert get_trainer_process_on_nodes(config) == [4, 1]
    manager = create_resource_pool_manager(config, roles=[Role.Actor, Role.RefPolicy])
    assert manager.resource_pool_spec == {"trainer_pool": [4, 1]}
    assert manager.get_n_gpus() == 5


@pytest.mark.parametrize(
    ("process_on_nodes", "message"),
    [
        ([4], "length must equal"),
        ([3, 1], "must equal the maximum"),
        ([4, 0], "positive integers"),
        ([4, 1.5], "positive integers"),
    ],
)
def test_invalid_uneven_trainer_topology_fails_closed(process_on_nodes, message):
    config = _config(process_on_nodes=process_on_nodes)

    with pytest.raises(ValueError, match=message):
        get_trainer_process_on_nodes(config)
