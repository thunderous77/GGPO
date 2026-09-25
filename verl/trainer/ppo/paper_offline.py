"""The paper's fixed-pool experiment, sharing the online loss and evaluation path."""

from pathlib import Path

import torch

from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import MetricsAggregator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.rollout_training_mismatch import MISMATCH_AGGREGATION_META


class CollectRollouts(RayPPOTrainer):
    """Save 400 mixed-reward minibatches from the unchanged initial model."""

    def fit(self):
        root = Path(self.config.trainer.offline_pool_dir)
        root.mkdir(parents=True, exist_ok=False)
        iterator = iter(self.train_dataloader)
        epoch = 0
        try:
            for index in range(self.total_training_steps):
                self.global_steps = index + 1
                self.checkpoint_manager.update_weights(0)
                candidate, iterator, epoch = self._next_training_candidate(iterator, epoch)
                batch, iterator, epoch = self._produce_training_batch(candidate, iterator, epoch, True, {}, {}, False)
                torch.save(batch, root / f"batch_{index:04d}.pt", pickle_protocol=4)
                print(f"Saved fixed-pool minibatch {index + 1}/{self.total_training_steps}", flush=True)
        finally:
            self._shutdown_dump_executor()


class OfflineTrainer(RayPPOTrainer):
    """Consume each stored minibatch once; rollout workers only serve evaluation."""

    def _produce_training_batch(
        self,
        batch_dict,
        train_iterator,
        epoch,
        mixed_filter_enabled,
        generation_metrics,
        timing_raw,
        curr_step_profile,
    ):
        root = Path(self.config.trainer.offline_pool_dir)
        block_size = self.config.data.train_batch_size // self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        start = self.global_steps - 1
        # Score the whole block once before the inherited loop consumes its minibatches.
        batch = DataProto.concat([
            torch.load(root / f"batch_{index:04d}.pt", map_location="cpu", weights_only=False)
            for index in range(start, min(start + block_size, self.total_training_steps))
        ])
        self.checkpoint_manager.sleep_replicas()
        return batch, train_iterator, epoch

    def _update_actor(self, batch, update_lr_scheduler=True):
        output = super()._update_actor(batch, update_lr_scheduler=update_lr_scheduler)
        metrics = output.meta_info["metrics"]
        aggregation = output.meta_info[MISMATCH_AGGREGATION_META]
        if getattr(self, "_collection_metrics", None) is None:
            self._collection_metrics = MetricsAggregator(total_gpus=1)
        self._collection_metrics.add_step_metrics({key: metrics[key] for key in aggregation}, aggregation)
        if self.global_steps % 8 == 0:
            # Recompute ESS from all 2,048 sequence ratios, not the mean of minibatch ESS values.
            metrics.update({
                f"offline_block/{key}": value
                for key, value in self._collection_metrics.get_aggregated_metrics().items()
            })
            self._collection_metrics.reset()
        return output
