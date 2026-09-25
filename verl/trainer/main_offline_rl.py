# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Hydra entry point for rollout-free training and checkpoint-bound evaluation."""

from __future__ import annotations

import os
import socket
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.offline_rl.evaluate import OfflineMathEvaluator
from verl.trainer.offline_rl.trainer import OfflineRLTrainer, validate_offline_training_config
from verl.trainer.ppo.utils import (
    Role,
    create_rl_dataset,
    create_rl_sampler,
    need_critic,
    need_reference_policy,
)
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


def _model_inputs(config):
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.fs import copy_to_local

    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )
    trust_remote_code = config.data.get("trust_remote_code", False)
    return (
        hf_tokenizer(local_path, trust_remote_code=trust_remote_code),
        hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True),
    )


@ray.remote
class OfflineTaskRunner(BaseTaskRunner):
    def run(self, config):
        print(f"OfflineTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        mode = config.offline_data.mode
        if mode == "evaluate":
            actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
            self.add_reward_model_resource_pool(config)
            self.add_ref_policy_worker(config, actor_rollout_cls)
            validate_config(
                config=config,
                use_reference_policy=need_reference_policy(config),
                use_critic=need_critic(config),
            )
            tokenizer, processor = _model_inputs(config)
            resource_pool_manager = self.init_resource_pool_mgr(config)
            from verl.utils.dataset.rl_dataset import collate_fn

            train_dataset = create_rl_dataset(
                config.data.train_files,
                config.data,
                tokenizer,
                processor,
                is_train=True,
                max_samples=config.data.get("train_max_samples", -1),
            )
            train_sampler = create_rl_sampler(config.data, train_dataset)
            val_dataset = create_rl_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor,
                is_train=False,
                max_samples=config.data.get("val_max_samples", -1),
            )
            trainer = OfflineMathEvaluator(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
            )
        elif mode in {"train", "initialize"}:
            validate_offline_training_config(config, allowed_modes=(mode,))
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.engine_workers import ActorRolloutRefWorker

            use_reference_policy = need_reference_policy(config)
            actor_role = Role.ActorRolloutRef if use_reference_policy else Role.Actor
            self.role_worker_mapping[actor_role] = ray.remote(ActorRolloutRefWorker)
            self.mapping[actor_role] = "global_pool"
            validate_config(config=config, use_reference_policy=use_reference_policy, use_critic=False)
            tokenizer, processor = _model_inputs(config)
            trainer = OfflineRLTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=self.init_resource_pool_mgr(config),
                ray_worker_group_cls=RayWorkerGroup,
            )
        else:
            raise ValueError(f"offline_data.mode must be initialize, train, or evaluate, got {mode!r}")

        trainer.init_workers()
        if mode == "initialize":
            trainer.initialize_checkpoint()
        else:
            trainer.fit()


@hydra.main(config_path="config", config_name="offline_rl_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_ppo(config, task_runner_class=OfflineTaskRunner)


if __name__ == "__main__":
    main()
