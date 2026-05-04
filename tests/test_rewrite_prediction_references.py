"""Tests for one-off ASR prediction reference rewriting."""

import json
from pathlib import Path

from tools.rewrite_prediction_references import read_reference_map, rewrite_prediction_references


def test_rewrite_prediction_references_adds_manifest_reference_path(tmp_path: Path) -> None:
    """Prediction audio paths stay intact while reference_path comes from manifest."""
    audio_path = tmp_path / "samples" / "001.wav"
    text_path = audio_path.with_suffix(".txt")
    relocated_path = audio_path.with_suffix(".asr_relocated.txt")
    audio_path.parent.mkdir(parents=True)
    audio_path.touch()
    text_path.write_text("old", encoding="utf-8")
    relocated_path.write_text("new", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        (f"audio_path,text_path,redacted_path\n{audio_path},{text_path},{relocated_path}\n"),
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"path": str(audio_path), "text": "prediction"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "predictions.refs.jsonl"

    rows, rewritten, fallback = rewrite_prediction_references(
        predictions,
        output,
        reference_map=read_reference_map(manifest),
    )

    copied = json.loads(output.read_text(encoding="utf-8"))
    assert rows == 1
    assert rewritten == 1
    assert fallback == 0
    assert copied["path"] == str(audio_path)
    assert copied["reference_path"] == str(relocated_path)


def test_read_reference_map_falls_back_to_text_path_when_relocated_missing(tmp_path: Path) -> None:
    """Unchanged rows point at the original text path."""
    audio_path = tmp_path / "samples" / "001.wav"
    text_path = audio_path.with_suffix(".txt")
    missing_relocated_path = audio_path.with_suffix(".asr_relocated.txt")
    audio_path.parent.mkdir(parents=True)
    text_path.write_text("old", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        (f"audio_path,text_path,redacted_path\n{audio_path},{text_path},{missing_relocated_path}\n"),
        encoding="utf-8",
    )

    references = read_reference_map(manifest)

    assert references[str(audio_path)] == str(text_path)


def test_rewrite_prediction_references_falls_back_when_manifest_row_missing(tmp_path: Path) -> None:
    """Extra prediction rows can still point at their sibling text reference."""
    audio_path = tmp_path / "samples" / "001.wav"
    audio_path.parent.mkdir(parents=True)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"path": str(audio_path), "text": "prediction"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "predictions.refs.jsonl"

    rows, rewritten, fallback = rewrite_prediction_references(
        predictions,
        output,
        reference_map={},
    )

    copied = json.loads(output.read_text(encoding="utf-8"))
    assert rows == 1
    assert rewritten == 1
    assert fallback == 1
    assert copied["reference_path"] == str(audio_path.with_suffix(".txt"))
