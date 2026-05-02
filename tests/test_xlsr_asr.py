"""Tests for the XLS-R ASR runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from asr import run_xlsr_asr


class FakeAutoProcessor:
    """Minimal AutoProcessor test double."""

    calls: list[str] = []

    @classmethod
    def from_pretrained(cls, processor_id: str):
        cls.calls.append(processor_id)
        return SimpleNamespace(source=processor_id)


class FakeFeatureExtractor:
    """Minimal Wav2Vec2FeatureExtractor test double."""

    calls: list[str] = []

    @classmethod
    def from_pretrained(cls, processor_id: str):
        cls.calls.append(processor_id)
        return SimpleNamespace(source=processor_id)


class FakeTokenizer:
    """Capture tokenizer initialization arguments."""

    kwargs: dict | None = None
    pad_token_id = 3

    def __init__(self, **kwargs):
        type(self).kwargs = kwargs

    def __len__(self) -> int:
        return 4


class FakeProcessor:
    """Capture combined processor pieces."""

    def __init__(self, *, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer


@pytest.fixture(autouse=True)
def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install fake Transformers classes for import-light processor tests."""
    FakeAutoProcessor.calls = []
    FakeFeatureExtractor.calls = []
    FakeTokenizer.kwargs = None
    monkeypatch.setattr(run_xlsr_asr, "AutoProcessor", FakeAutoProcessor)
    monkeypatch.setattr(run_xlsr_asr, "Wav2Vec2ForCTC", object())
    monkeypatch.setattr(run_xlsr_asr, "Wav2Vec2FeatureExtractor", FakeFeatureExtractor)
    monkeypatch.setattr(run_xlsr_asr, "Wav2Vec2CTCTokenizer", FakeTokenizer)
    monkeypatch.setattr(run_xlsr_asr, "Wav2Vec2Processor", FakeProcessor)


def write_json(path: Path, value: object) -> None:
    """Write compact UTF-8 JSON for local vocab fixtures."""
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_load_processor_uses_pretrained_processor_without_local_tokenizer() -> None:
    processor = run_xlsr_asr.load_processor("open/xls-r", None, None)

    assert processor.source == "open/xls-r"
    assert FakeAutoProcessor.calls == ["open/xls-r"]


def test_load_processor_combines_pretrained_feature_extractor_with_local_tokenizer(
    tmp_path: Path,
) -> None:
    vocab_json = tmp_path / "vocab.json"
    tokenizer_json = tmp_path / "tokenizer.json"
    write_json(vocab_json, {"а": 0, "|": 1, "[UNK]": 2, "[PAD]": 3})
    write_json(tokenizer_json, {"version": "1.0"})

    processor = run_xlsr_asr.load_processor("open/xls-r", vocab_json, tokenizer_json)

    assert processor.feature_extractor.source == "open/xls-r"
    assert FakeFeatureExtractor.calls == ["open/xls-r"]
    assert FakeTokenizer.kwargs == {
        "vocab_file": str(vocab_json),
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "word_delimiter_token": "|",
        "tokenizer_file": str(tokenizer_json),
    }


def test_tokenizer_json_requires_vocab_json(tmp_path: Path) -> None:
    tokenizer_json = tmp_path / "tokenizer.json"
    write_json(tokenizer_json, {"version": "1.0"})

    with pytest.raises(ValueError, match="--tokenizer-json requires --vocab-json"):
        run_xlsr_asr.load_processor("open/xls-r", None, tokenizer_json)


def test_local_tokenizer_special_tokens_supports_angle_bracket_vocab(tmp_path: Path) -> None:
    vocab_json = tmp_path / "vocab.json"
    write_json(vocab_json, {"а": 0, " ": 1, "<unk>": 2, "<pad>": 3})

    assert run_xlsr_asr.local_tokenizer_special_tokens(vocab_json) == {
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "word_delimiter_token": " ",
    }


def test_local_vocab_model_kwargs_sets_ctc_head_shape() -> None:
    processor = SimpleNamespace(tokenizer=FakeTokenizer())

    assert run_xlsr_asr.local_vocab_model_kwargs(processor) == {
        "vocab_size": 4,
        "pad_token_id": 3,
        "ignore_mismatched_sizes": True,
    }
