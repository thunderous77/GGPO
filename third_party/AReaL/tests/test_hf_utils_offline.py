# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from areal.utils import hf_utils


@pytest.mark.parametrize(
    ("offline", "expected_force_download"),
    [(True, False), (False, True)],
)
def test_tokenizer_force_download_respects_offline_mode(
    monkeypatch, offline: bool, expected_force_download: bool
) -> None:
    calls = []
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)

    def from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return tokenizer

    monkeypatch.setattr(hf_utils, "is_offline_mode", lambda: offline)
    monkeypatch.setattr(
        hf_utils.transformers,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=from_pretrained),
    )
    hf_utils.load_hf_tokenizer.cache_clear()

    assert hf_utils.load_hf_tokenizer("Qwen/example") is tokenizer
    assert calls[0][1]["force_download"] is expected_force_download

    hf_utils.load_hf_tokenizer.cache_clear()
