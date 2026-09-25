"""AReaL Ray scheduler construction with an explicit VERL experiment context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .factory import config_get
from .vendored import load_vendored_areal

load_vendored_areal()

from areal.api.cli_args import (  # noqa: E402
    BaseExperimentConfig,
    ClusterSpecConfig,
    InferenceEngineConfig,
    NameResolveConfig,
)
from areal.infra.scheduler.ray import RayScheduler  # noqa: E402


def build_areal_experiment_config(
    config: Any,
    inference_config: InferenceEngineConfig,
) -> BaseExperimentConfig:
    """Build the scheduler context normally supplied by an AReaL launcher."""

    fileroot = Path(inference_config.fileroot).expanduser().resolve()
    name_resolve_root = Path(
        str(config_get(config, "async_training.areal.name_resolve_root", fileroot / "name_resolve"))
    ).expanduser().resolve()
    cluster = ClusterSpecConfig(
        name_resolve=NameResolveConfig(type="nfs", nfs_record_root=str(name_resolve_root)),
        cluster_name="ray",
        fileroot=str(fileroot),
        n_nodes=int(config_get(config, "rollout.nnodes", 1)),
        n_gpus_per_node=int(config_get(config, "rollout.n_gpus_per_node", 1)),
    )
    return BaseExperimentConfig(
        experiment_name=inference_config.experiment_name,
        trial_name=inference_config.trial_name,
        cluster=cluster,
        enable_offload=bool(config_get(config, "async_training.areal.enable_offload", False)),
        tokenizer_path=inference_config.tokenizer_path,
    )


def build_areal_ray_scheduler(
    config: Any,
    inference_config: InferenceEngineConfig,
) -> tuple[BaseExperimentConfig, RayScheduler]:
    """Return a scheduler that always carries the matching experiment config."""

    experiment_config = build_areal_experiment_config(config, inference_config)
    scheduler = RayScheduler(
        startup_timeout=float(config_get(config, "async_training.areal.setup_timeout", 600.0)),
        exp_config=experiment_config,
    )
    return experiment_config, scheduler
