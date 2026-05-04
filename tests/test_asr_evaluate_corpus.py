"""Tests for utterance-level ASR corpus evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr.evaluate_corpus import (
    discover_samples,
    evaluate_predictions,
    main,
    missing_prediction_samples,
    read_predictions,
    run_asr_with_retries,
    write_audio_manifest,
)


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


def test_evaluate_predictions_uses_prediction_reference_path_when_provided(tmp_path: Path) -> None:
    audio_a = tmp_path / "a.wav"
    audio_b = tmp_path / "b.wav"
    audio_a.write_bytes(b"fake")
    audio_b.write_bytes(b"fake")
    audio_a.with_suffix(".txt").write_text("old reference", encoding="utf-8")
    audio_b.with_suffix(".txt").write_text("default reference", encoding="utf-8")
    relocated = tmp_path / "a.asr_relocated.txt"
    relocated.write_text("new reference", encoding="utf-8")
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "path": str(audio_a),
                        "reference_path": str(relocated),
                        "text": "new reference",
                    },
                    ensure_ascii=False,
                ),
                json.dumps({"path": str(audio_b), "text": "default reference"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    samples = discover_samples(tmp_path)
    predictions = read_predictions(predictions_path)
    stats, _mismatches, rows = evaluate_predictions(samples, predictions)

    assert stats.wer == 0
    assert rows[0]["reference_path"] == str(relocated)
    assert rows[0]["reference"] == "new reference"
    assert rows[1]["reference_path"] == str(audio_b.with_suffix(".txt"))
    assert rows[1]["reference"] == "default reference"


def test_main_prefixes_outputs_with_prediction_file_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "001.wav"
    audio.write_bytes(b"fake")
    audio.with_suffix(".txt").write_text("текст", encoding="utf-8")
    predictions = tmp_path / "gigaam_preds.jsonl"
    predictions.write_text(
        json.dumps({"path": str(audio), "text": "текст"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_corpus.py",
            str(tmp_path),
            str(output_dir),
            "--predictions",
            str(predictions),
        ],
    )

    main()

    assert (output_dir / "gigaam_preds.per_utterance.csv").exists()
    assert (output_dir / "gigaam_preds.mismatches.csv").exists()
    assert (output_dir / "gigaam_preds.wer_report.txt").exists()
    assert not (output_dir / "per_utterance.csv").exists()


def test_write_audio_manifest_lists_discovered_paths(tmp_path: Path) -> None:
    audio = tmp_path / "nested" / "001.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"fake")
    audio.with_suffix(".txt").write_text("текст", encoding="utf-8")
    manifest = tmp_path / "manifest.txt"

    write_audio_manifest(discover_samples(tmp_path), manifest)

    assert manifest.read_text(encoding="utf-8").strip() == str(audio)


def test_run_asr_with_retries_recovers_missing_predictions(tmp_path: Path) -> None:
    sample_a = tmp_path / "001.wav"
    sample_b = tmp_path / "002.wav"
    sample_a.write_bytes(b"fake")
    sample_b.write_bytes(b"fake")
    sample_a.with_suffix(".txt").write_text("первый", encoding="utf-8")
    sample_b.with_suffix(".txt").write_text("второй", encoding="utf-8")
    samples = discover_samples(tmp_path)
    manifest = tmp_path / "manifest.txt"
    predictions = tmp_path / "predictions.jsonl"
    write_audio_manifest(samples, manifest)
    script = tmp_path / "fake_asr.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "manifest, output = sys.argv[1:3]",
                "paths = [line.strip() for line in open(manifest, encoding='utf-8') if line.strip()]",
                "if output.endswith('predictions.jsonl'):",
                "    paths = paths[:1]",
                "with open(output, 'w', encoding='utf-8') as f:",
                "    for path in paths:",
                "        row = {'path': path, 'text': path.split('/')[-1]}",
                "        f.write(json.dumps(row, ensure_ascii=False) + '\\n')",
            ]
        ),
        encoding="utf-8",
    )

    recovered = run_asr_with_retries(
        f"python {script} {{manifest}} {{predictions}}",
        input_dir=tmp_path,
        output_dir=tmp_path,
        samples=samples,
        manifest=manifest,
        predictions=predictions,
        path_field="path",
        text_field="text",
        reference_path_field="reference_path",
        retry_missing=1,
    )

    assert missing_prediction_samples(samples, recovered) == []
    assert len(predictions.read_text(encoding="utf-8").strip().splitlines()) == 2
    assert (tmp_path / "missing_predictions_retry_1.txt").read_text(encoding="utf-8").strip() == str(sample_b)


def test_main_fails_after_asr_command_leaves_missing_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_a = tmp_path / "001.wav"
    sample_b = tmp_path / "002.wav"
    sample_a.write_bytes(b"fake")
    sample_b.write_bytes(b"fake")
    sample_a.with_suffix(".txt").write_text("первый", encoding="utf-8")
    sample_b.with_suffix(".txt").write_text("второй", encoding="utf-8")
    script = tmp_path / "fake_asr_one.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "manifest, output = sys.argv[1:3]",
                "path = next(line.strip() for line in open(manifest, encoding='utf-8') if line.strip())",
                "with open(output, 'w', encoding='utf-8') as f:",
                "    f.write(json.dumps({'path': path, 'text': 'ok'}, ensure_ascii=False) + '\\n')",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_corpus.py",
            str(tmp_path),
            str(output_dir),
            "--asr-command",
            f"python {script} {{manifest}} {{predictions}}",
            "--retry-missing",
            "0",
        ],
    )

    with pytest.raises(RuntimeError, match="missing for 1 of 2"):
        main()

    assert (output_dir / "missing_predictions.txt").read_text(encoding="utf-8").strip() == str(sample_b)
