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

import builtins

import torch

from verl.utils import attention_utils


def test_cuda_padding_helpers_fall_back_when_flash_attn_is_missing(monkeypatch):
    real_import = builtins.__import__

    def import_without_flash_attn(name, *args, **kwargs):
        if name == "flash_attn.bert_padding":
            raise ModuleNotFoundError("No module named 'flash_attn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("verl.utils.device.is_torch_npu_available", lambda check_device=False: False)
    monkeypatch.setattr(builtins, "__import__", import_without_flash_attn)

    _, pad_input, _, unpad_input = attention_utils._get_attention_functions()

    hidden_states = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    unpadded, indices, cu_seqlens, max_seqlen, used_seqlens = unpad_input(hidden_states, attention_mask)

    assert pad_input.__module__ == "verl.utils.npu_flash_attn_utils"
    assert unpad_input.__module__ == "verl.utils.npu_flash_attn_utils"
    torch.testing.assert_close(cu_seqlens, torch.tensor([0, 2, 5], dtype=torch.int32))
    torch.testing.assert_close(used_seqlens, torch.tensor([2, 3], dtype=torch.int32))
    assert max_seqlen == 3

    repadded = pad_input(unpadded, indices, batch=2, seqlen=4)
    torch.testing.assert_close(repadded[attention_mask.bool()], hidden_states[attention_mask.bool()])
    torch.testing.assert_close(repadded[~attention_mask.bool()], torch.zeros(3, 3))
