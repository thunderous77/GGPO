# Copyright 2026 OpenAI
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

"""Loss-masked group padding for fixed-geometry fully-async actor updates."""

from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ggpo import build_policy_group_ids

REAL_SAMPLE_MASK = "_fully_async_actor_update_real_mask"
DUMMY_SAFE_RESPONSE_MASK = "_fully_async_actor_update_dummy_safe_response_mask"
_ORIGINAL_GLOBAL_TOKEN_NUM = "_fully_async_actor_update_original_global_token_num"
_PADDING_META_PREFIX = "_fully_async_actor_update_padding_"
LOGICAL_MINI_BATCH_SIZE_META = f"{_PADDING_META_PREFIX}logical_mini_batch_size"
PHYSICAL_MINI_BATCH_SIZE_META = f"{_PADDING_META_PREFIX}physical_mini_batch_size"


@dataclass(frozen=True)
class ActorUpdatePaddingSpec:
    """Resolved, auditable geometry for complete-group padding."""

    enabled: bool
    logical_groups_per_mini: int = 0
    padding_groups_per_mini: int = 0
    group_size: int = 0
    require_batches: int = 0

    @property
    def logical_responses_per_mini(self) -> int:
        return self.logical_groups_per_mini * self.group_size

    @property
    def physical_responses_per_mini(self) -> int:
        return (self.logical_groups_per_mini + self.padding_groups_per_mini) * self.group_size

    @property
    def logical_responses_per_update(self) -> int:
        return self.logical_responses_per_mini * self.require_batches

    @property
    def physical_responses_per_update(self) -> int:
        return self.physical_responses_per_mini * self.require_batches

    @property
    def padding_responses_per_update(self) -> int:
        return self.padding_groups_per_mini * self.group_size * self.require_batches


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def get_actor_update_padding_spec(config) -> ActorUpdatePaddingSpec:
    """Resolve the optional fully-async actor-update padding configuration."""
    enabled = bool(
        OmegaConf.select(
            config,
            "async_training.actor_update_padding.enabled",
            default=False,
        )
    )
    if not enabled:
        return ActorUpdatePaddingSpec(enabled=False)

    from verl.utils.config import get_fully_async_prompts_per_update

    # Fail before deriving any padding geometry if an explicit trainer budget
    # conflicts with the canonical mini-batch product.
    get_fully_async_prompts_per_update(config)
    logical_groups_per_mini = _positive_int(
        OmegaConf.select(config, "actor_rollout_ref.actor.ppo_mini_batch_size"),
        "actor.ppo_mini_batch_size",
    )
    group_size = _positive_int(
        OmegaConf.select(config, "actor_rollout_ref.rollout.n"),
        "rollout.n",
    )
    require_batches = _positive_int(
        OmegaConf.select(config, "async_training.require_batches"),
        "async_training.require_batches",
    )
    padding_groups_per_mini = _positive_int(
        OmegaConf.select(
            config,
            "async_training.actor_update_padding.groups_per_mini",
            default=1,
        ),
        "async_training.actor_update_padding.groups_per_mini",
    )
    if bool(OmegaConf.select(config, "actor_rollout_ref.actor.shuffle", default=False)):
        raise ValueError("Fully async actor-update padding requires actor.shuffle=false.")

    return ActorUpdatePaddingSpec(
        enabled=True,
        logical_groups_per_mini=logical_groups_per_mini,
        padding_groups_per_mini=padding_groups_per_mini,
        group_size=group_size,
        require_batches=require_batches,
    )


def _validate_complete_uid_groups(batch: DataProto, group_size: int) -> None:
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("Fully async actor-update padding requires non_tensor_batch['uid'].")
    build_policy_group_ids(batch.non_tensor_batch["uid"], group_size=group_size)


def _zero_dummy_scientific_fields(dummy: DataProto) -> None:
    for key in (
        "response_mask",
        "rm_scores",
        "token_level_scores",
        "token_level_rewards",
        "advantages",
        "returns",
        "rollout_is_weights",
    ):
        if dummy.batch is not None and key in dummy.batch:
            dummy.batch[key].zero_()


def pad_fully_async_actor_update(batch: DataProto, config) -> tuple[DataProto, ActorUpdatePaddingSpec]:
    """Interleave loss-masked complete groups after every logical mini-batch.

    The input remains untouched. Padding exists only to make the physical
    dispatch geometry divisible by the effective data-parallel world. The
    scientific loss denominator remains the logical mini-batch size.
    """
    spec = get_actor_update_padding_spec(config)
    if not spec.enabled:
        return batch, spec
    if REAL_SAMPLE_MASK in batch.batch:
        raise ValueError("Actor-update batch is already marked as padded.")
    if len(batch) != spec.logical_responses_per_update:
        raise ValueError(
            "Fully async actor-update padding received the wrong logical batch: "
            f"expected {spec.logical_responses_per_update} responses, got {len(batch)}."
        )
    _validate_complete_uid_groups(batch, spec.group_size)
    if "response_mask" not in batch.batch:
        raise ValueError("Actor-update padding requires batch['response_mask'].")

    chunks: list[DataProto] = []
    nonce = uuid.uuid4().hex
    logical_mini = spec.logical_responses_per_mini
    for mini_index, start in enumerate(range(0, len(batch), logical_mini)):
        stop = start + logical_mini
        real = batch.select_idxs(list(range(start, stop)))
        real.batch[REAL_SAMPLE_MASK] = torch.ones(len(real), dtype=torch.bool, device=real.batch.device)
        real.batch[DUMMY_SAFE_RESPONSE_MASK] = real.batch["response_mask"].clone()
        real.meta_info = {}

        # Reuse the shortest complete group in this logical mini. Its input is
        # valid for a normal forward pass, while remove-padding avoids paying
        # the long-tail token cost of an arbitrary response.
        candidate_starts = list(range(start, stop, spec.group_size))
        if "attention_mask" not in batch.batch:
            raise ValueError("Actor-update padding requires batch['attention_mask'].")
        dummy_start = min(
            candidate_starts,
            key=lambda candidate: int(
                batch.batch["attention_mask"][candidate : candidate + spec.group_size].sum().item()
            ),
        )
        chunks.append(real)
        for padding_index in range(spec.padding_groups_per_mini):
            dummy = batch.select_idxs(list(range(dummy_start, dummy_start + spec.group_size)))
            dummy.batch[DUMMY_SAFE_RESPONSE_MASK] = dummy.batch["response_mask"].clone()
            dummy.batch[REAL_SAMPLE_MASK] = torch.zeros(len(dummy), dtype=torch.bool, device=dummy.batch.device)
            _zero_dummy_scientific_fields(dummy)
            dummy.meta_info = {}
            padding_uid = f"__fully_async_actor_padding__{nonce}_{mini_index}_{padding_index}"
            dummy.non_tensor_batch["uid"] = np.asarray([padding_uid] * spec.group_size, dtype=object)
            chunks.append(dummy)

    padded = DataProto.concat(chunks)
    if len(padded) != spec.physical_responses_per_update:
        raise AssertionError(
            f"Internal actor-update padding error: expected {spec.physical_responses_per_update}, got {len(padded)}."
        )
    marker = padded.batch[REAL_SAMPLE_MASK]
    if int(marker.sum().item()) != spec.logical_responses_per_update:
        raise AssertionError("Actor-update padding marker does not preserve the logical response count.")

    # DataProto.concat compares duplicate metadata with scalar equality. Async
    # trajectory-version metadata intentionally contains numpy arrays, so each
    # temporary chunk is metadata-free and the single authoritative copy is
    # restored only after tensors and non-tensors have been concatenated.
    padded.meta_info = dict(batch.meta_info)
    if "global_token_num" in padded.meta_info:
        padded.meta_info[_ORIGINAL_GLOBAL_TOKEN_NUM] = padded.meta_info["global_token_num"]
        padded.meta_info["global_token_num"] = padded.batch["attention_mask"].sum(dim=-1).tolist()
    padded.meta_info.update(
        {
            f"{_PADDING_META_PREFIX}logical_responses": spec.logical_responses_per_update,
            f"{_PADDING_META_PREFIX}physical_responses": spec.physical_responses_per_update,
            f"{_PADDING_META_PREFIX}padding_responses": spec.padding_responses_per_update,
            LOGICAL_MINI_BATCH_SIZE_META: spec.logical_responses_per_mini,
            PHYSICAL_MINI_BATCH_SIZE_META: spec.physical_responses_per_mini,
        }
    )
    return padded, spec


def select_fully_async_real_samples(batch: DataProto) -> DataProto:
    """Return the exact real-sample view and remove private padding metadata."""
    if batch.batch is None or REAL_SAMPLE_MASK not in batch.batch:
        return batch
    marker = batch.batch[REAL_SAMPLE_MASK]
    if marker.ndim != 1 or marker.dtype != torch.bool:
        raise ValueError(f"{REAL_SAMPLE_MASK} must be a one-dimensional boolean tensor.")
    real = batch.select_idxs(marker)
    real.batch.pop(REAL_SAMPLE_MASK)
    real.batch.pop(DUMMY_SAFE_RESPONSE_MASK)
    real.meta_info = dict(real.meta_info)
    if _ORIGINAL_GLOBAL_TOKEN_NUM in real.meta_info:
        real.meta_info["global_token_num"] = real.meta_info.pop(_ORIGINAL_GLOBAL_TOKEN_NUM)
    for key in list(real.meta_info):
        if key.startswith(_PADDING_META_PREFIX):
            real.meta_info.pop(key)
    return real
