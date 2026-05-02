"""Tests for optional ASR shared helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from asr import asr_common


class FakeWaveform:
    """Small tensor-like object exposing only the shape needed for duration."""

    def __init__(self, samples: int):
        self.shape = (1, samples)


def test_audio_duration_uses_torchaudio_info_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_info(path: str) -> SimpleNamespace:
        calls.append(path)
        return SimpleNamespace(num_frames=48_000, sample_rate=24_000)

    fake_torchaudio = SimpleNamespace(info=fake_info)
    monkeypatch.setattr(asr_common, "torchaudio", fake_torchaudio)

    assert asr_common.audio_duration_s(Path("sample.wav")) == 2.0
    assert calls == ["sample.wav"]


def test_audio_duration_falls_back_when_torchaudio_info_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load(path: str) -> tuple[FakeWaveform, int]:
        assert path == "sample.wav"
        return FakeWaveform(16_000), 8_000

    fake_torchaudio = SimpleNamespace(load=fake_load)
    monkeypatch.setattr(asr_common, "torchaudio", fake_torchaudio)

    assert asr_common.audio_duration_s(Path("sample.wav")) == 2.0


def test_build_items_keeps_readable_audio_when_torchaudio_info_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load(path: str) -> tuple[FakeWaveform, int]:
        assert path == "sample.m4a"
        return FakeWaveform(44_100), 44_100

    fake_torchaudio = SimpleNamespace(load=fake_load)
    monkeypatch.setattr(asr_common, "torchaudio", fake_torchaudio)

    items = asr_common.build_items([Path("sample.m4a")])

    assert len(items) == 1
    assert items[0].path == Path("sample.m4a")
    assert items[0].duration_s == 1.0
