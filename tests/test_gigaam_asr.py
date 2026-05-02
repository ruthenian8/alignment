"""Tests for the GigaAM ASR runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from asr.asr_common import AudioItem
from asr.run_gigaam_asr import DEFAULT_GIGAAM_MODEL, result_text, transcribe_items


class FakeGigaAMModel:
    """Minimal GigaAM-like model for testing result serialization."""

    def transcribe(self, path: str) -> SimpleNamespace:
        return SimpleNamespace(text=f" text for {Path(path).name} ")


def test_result_text_supports_object_and_string_results() -> None:
    assert result_text(SimpleNamespace(text=" привет ")) == "привет"
    assert result_text(" пока ") == "пока"


def test_transcribe_items_writes_common_asr_rows() -> None:
    items = [AudioItem(path=Path("001.wav"), duration_s=1.2345)]

    rows = transcribe_items(FakeGigaAMModel(), items, DEFAULT_GIGAAM_MODEL)

    assert rows == [
        {
            "path": "001.wav",
            "text": "text for 001.wav",
            "duration_s": 1.234,
            "model_id": "v3_e2e_rnnt",
            "model_type": "gigaam",
        }
    ]
