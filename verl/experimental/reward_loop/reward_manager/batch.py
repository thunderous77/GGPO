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

import asyncio
import inspect
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import batch_compute_score


@register("batch")
class BatchRewardManager(RewardManagerBase):
    """Batch math reward manager compatible with the experimental reward loop."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or batch_compute_score
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        outputs = await self.run_batch(data[-1:])
        return outputs[-1]

    async def run_batch(self, data: DataProto) -> list[dict]:
        response_strs, valid_response_lengths = await self._decode_responses(data)
        return await self._score_decoded_batch(data, response_strs, valid_response_lengths)

    def run_batch_sync(self, data: DataProto) -> list[dict]:
        response_strs, valid_response_lengths = self._decode_responses_sync(data)
        return self._score_decoded_batch_sync(data, response_strs, valid_response_lengths)

    async def _score_decoded_batch(
        self, data: DataProto, response_strs: list[str], valid_response_lengths: list[int]
    ) -> list[dict]:
        raw_scores = self._compute_raw_scores(data, response_strs)
        if inspect.isawaitable(raw_scores):
            raw_scores = await raw_scores
        return self._format_outputs(raw_scores, valid_response_lengths)

    def _score_decoded_batch_sync(
        self, data: DataProto, response_strs: list[str], valid_response_lengths: list[int]
    ) -> list[dict]:
        raw_scores = self._compute_raw_scores(data, response_strs)
        if inspect.isawaitable(raw_scores):
            raise TypeError("BatchRewardManager.run_batch_sync requires a synchronous compute_score function")
        return self._format_outputs(raw_scores, valid_response_lengths)

    def _compute_raw_scores(self, data: DataProto, response_strs: list[str]):
        ground_truths = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = [data_item.non_tensor_batch["data_source"] for data_item in data]
        extra_infos = [self._extra_info(data_item) for data_item in data]
        return self.compute_score(
            data_sources=data_sources,
            solution_strs=response_strs,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
        )

    def _format_outputs(self, raw_scores, valid_response_lengths: list[int]) -> list[dict]:
        outputs = []
        for raw_score, valid_response_length in zip(raw_scores, valid_response_lengths, strict=True):
            score, reward_extra_info = self._normalize_score(raw_score)
            reward_extra_info["acc"] = score
            reward_extra_info["valid_response_length"] = int(valid_response_length)
            outputs.append({"reward_score": score, "reward_extra_info": reward_extra_info})
        return outputs

    async def _decode_responses(self, data: DataProto) -> tuple[list[str], list[int]]:
        prompt_length = data.batch["prompts"].size(1)
        response_ids = data.batch["responses"]
        valid_response_lengths = data.batch["attention_mask"][:, prompt_length:].sum(dim=1).tolist()

        response_strs = []
        loop = asyncio.get_running_loop()
        for index, valid_response_length in enumerate(valid_response_lengths):
            valid_response_ids = response_ids[index][:valid_response_length]
            response_str = await loop.run_in_executor(
                None, lambda ids=valid_response_ids: self.tokenizer.decode(ids, skip_special_tokens=True)
            )
            if self.tokenizer.eos_token and response_str.endswith(self.tokenizer.eos_token):
                response_str = response_str[: -len(self.tokenizer.eos_token)]
            response_strs.append(response_str)
        return response_strs, [int(length) for length in valid_response_lengths]

    def _decode_responses_sync(self, data: DataProto) -> tuple[list[str], list[int]]:
        prompt_length = data.batch["prompts"].size(1)
        response_ids = data.batch["responses"]
        valid_response_lengths = data.batch["attention_mask"][:, prompt_length:].sum(dim=1).tolist()

        response_strs = []
        for index, valid_response_length in enumerate(valid_response_lengths):
            valid_response_ids = response_ids[index][:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            if self.tokenizer.eos_token and response_str.endswith(self.tokenizer.eos_token):
                response_str = response_str[: -len(self.tokenizer.eos_token)]
            response_strs.append(response_str)
        return response_strs, [int(length) for length in valid_response_lengths]

    @staticmethod
    def _extra_info(data_item) -> dict[str, Any]:
        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}) or {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(dict(tool_extra_fields))
        extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
        extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})
        return extra_info

    @staticmethod
    def _normalize_score(raw_score) -> tuple[float, dict[str, Any]]:
        if isinstance(raw_score, dict):
            score = float(raw_score["score"])
            reward_extra_info = dict(raw_score)
        else:
            score = float(raw_score)
            reward_extra_info = {}
        return score, reward_extra_info
