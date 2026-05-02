"""Tests for utterance-level ASR corpus evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from asr.evaluate_corpus import discover_samples, evaluate_predictions, read_predictions, write_audio_manifest


def test_discover_samples_uses_sibling_txt_references(tmp_path: Path) -> None:
    sample_dir = tmp_path / "pez_001" / "pez_001No1"
    sample_dir.mkdir(parents=True)
    (sample_dir / "001.wav").write_bytes(b"fake")
    (sample_dir / "001.txt").write_text("до\\брый день", encoding="utf-8")
    (sample_dir / "001_orig.txt").write_text("original", encoding="utf-8")
    (sample_dir / "002.wav").write_bytes(b"fake")

    samples = discover_samples(tmp_path)

    assert len(samples) == 1
    assert samples[0].audio_path.name == "001.wav"
    assert samples[0].reference_text == "до\\брый день"


def test_read_predictions_and_evaluate_global_wer(tmp_path: Path) -> None:
    audio_a = tmp_path / "a.wav"
    audio_b = tmp_path / "b.wav"
    audio_a.write_bytes(b"fake")
    audio_b.write_bytes(b"fake")
    (tmp_path / "a.txt").write_text("красный дом", encoding="utf-8")
    (tmp_path / "b.txt").write_text("[Соб.: шум] зеленая луна", encoding="utf-8")
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            [
                json.dumps({"path": str(audio_a), "text": "красный кот"}, ensure_ascii=False),
                json.dumps({"path": str(audio_b), "text": "зеленая луна"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    samples = discover_samples(tmp_path)
    predictions = read_predictions(predictions_path)
    stats, mismatches, rows = evaluate_predictions(samples, predictions)

    assert stats.rows == 2
    assert stats.reference_words == 4
    assert stats.substitutions == 1
    assert stats.wer == 0.25
    assert mismatches[("substitute", "дом", "кот")] == 1
    assert [row["status"] for row in rows] == ["evaluated", "evaluated"]


def test_write_audio_manifest_lists_discovered_paths(tmp_path: Path) -> None:
    audio = tmp_path / "nested" / "001.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"fake")
    audio.with_suffix(".txt").write_text("текст", encoding="utf-8")
    manifest = tmp_path / "manifest.txt"

    write_audio_manifest(discover_samples(tmp_path), manifest)

    assert manifest.read_text(encoding="utf-8").strip() == str(audio)
