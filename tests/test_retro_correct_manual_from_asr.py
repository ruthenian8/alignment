"""Tests for one-off ASR-based manual transcript correction."""

from pathlib import Path

from tools.retro_correct_manual_from_asr import process_predictions, trim_units


def write_prediction_files(tmp_path: Path, rows: list[tuple[Path, str]]) -> list[Path]:
    """Write identical tiny prediction files for the three-model correction path."""
    prediction_paths = []
    for model in ["gigaam", "mms", "whisper"]:
        prediction_path = tmp_path / f"{model}_preds.jsonl"
        prediction_path.write_text(
            "".join(f'{{"path": "{audio_path}", "text": "{text}"}}\n' for audio_path, text in rows),
            encoding="utf-8",
        )
        prediction_paths.append(prediction_path)
    return prediction_paths


def test_trim_units_removes_only_edges_absent_from_all_models() -> None:
    """Only full leading and trailing sentence units are trimmed."""
    units = ["Noise.", "Shared words remain.", "Tail."]
    predictions = [{"shared", "words"}, {"shared"}, {"remain"}]

    kept, prefix, suffix = trim_units(units, predictions)

    assert kept == ["Shared words remain."]
    assert prefix == ["Noise."]
    assert suffix == ["Tail."]


def test_process_predictions_marks_misaligned_without_redacting(tmp_path: Path) -> None:
    """Rows with zero overlap across all models are flagged but left unchanged."""
    audio_path = tmp_path / "cut_samples" / "pez_001" / "pez_001No0" / "001_SPEAKER_00_00-00-00-001.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.touch()
    text_path = audio_path.with_suffix(".txt")
    text_path.write_text("Ручной текст.\n", encoding="utf-8")

    prediction_paths = write_prediction_files(tmp_path, [(audio_path, "other words")])

    rows = process_predictions(prediction_paths, manifest_path=tmp_path / "manifest.csv")

    assert len(rows) == 1
    assert rows[0]["possibly_misaligned"] is True
    assert rows[0]["changed"] is False
    assert not text_path.with_name(f"{text_path.stem}.asr_redacted.txt").exists()


def test_process_predictions_relocates_suffix_to_next_when_score_improves(tmp_path: Path) -> None:
    """A trimmed suffix can be prepended to the next same-directory transcript."""
    chunk_dir = tmp_path / "cut_samples" / "pez_001" / "pez_001No0"
    chunk_dir.mkdir(parents=True)
    first_audio = chunk_dir / "001_SPEAKER_00_00-00-00-001.wav"
    second_audio = chunk_dir / "002_SPEAKER_00_00-00-00-002.wav"
    first_audio.touch()
    second_audio.touch()
    first_audio.with_suffix(".txt").write_text("Первый текст. Сиротский фрагмент.\n", encoding="utf-8")
    second_audio.with_suffix(".txt").write_text("Второй текст.\n", encoding="utf-8")
    prediction_paths = write_prediction_files(
        tmp_path,
        [
            (first_audio, "первый текст"),
            (second_audio, "сиротский фрагмент второй текст"),
        ],
    )

    rows = process_predictions(
        prediction_paths,
        manifest_path=tmp_path / "manifest.csv",
        relocate_orphans=True,
    )

    first_row, second_row = rows
    assert first_row["removed_suffix_text"] == "Сиротский фрагмент."
    assert first_row["relocated_suffix_to"] == str(second_audio)
    assert first_audio.with_suffix(".asr_redacted.txt").read_text(encoding="utf-8").strip() == "Первый текст."
    assert (
        second_audio.with_suffix(".asr_redacted.txt").read_text(encoding="utf-8").strip()
        == "Сиротский фрагмент. Второй текст."
    )
    assert second_row["received_prefix_from"] == str(first_audio)


def test_process_predictions_does_not_relocate_without_score_gain(tmp_path: Path) -> None:
    """Trimmed spans stay removed if the adjacent ASR text does not support them."""
    chunk_dir = tmp_path / "cut_samples" / "pez_001" / "pez_001No0"
    chunk_dir.mkdir(parents=True)
    first_audio = chunk_dir / "001_SPEAKER_00_00-00-00-001.wav"
    second_audio = chunk_dir / "002_SPEAKER_00_00-00-00-002.wav"
    first_audio.touch()
    second_audio.touch()
    first_audio.with_suffix(".txt").write_text("Первый текст. Лишний фрагмент.\n", encoding="utf-8")
    second_audio.with_suffix(".txt").write_text("Второй текст.\n", encoding="utf-8")
    prediction_paths = write_prediction_files(
        tmp_path,
        [
            (first_audio, "первый текст"),
            (second_audio, "второй текст"),
        ],
    )

    rows = process_predictions(
        prediction_paths,
        manifest_path=tmp_path / "manifest.csv",
        relocate_orphans=True,
    )

    assert rows[0]["relocated_suffix_to"] == ""
    assert first_audio.with_suffix(".asr_redacted.txt").read_text(encoding="utf-8").strip() == "Первый текст."
    assert not second_audio.with_suffix(".asr_redacted.txt").exists()


def test_process_predictions_relocates_short_suffix_phrase_when_enabled(tmp_path: Path) -> None:
    """A comma-delimited edge phrase can move even when the full sentence is kept."""
    chunk_dir = tmp_path / "cut_samples" / "pez_001" / "pez_001No0"
    chunk_dir.mkdir(parents=True)
    first_audio = chunk_dir / "001_SPEAKER_00_00-00-00-001.wav"
    second_audio = chunk_dir / "002_SPEAKER_00_00-00-00-002.wav"
    first_audio.touch()
    second_audio.touch()
    first_audio.with_suffix(".txt").write_text("Первый текст, сиротский фрагмент.\n", encoding="utf-8")
    second_audio.with_suffix(".txt").write_text("Второй текст.\n", encoding="utf-8")
    prediction_paths = write_prediction_files(
        tmp_path,
        [
            (first_audio, "первый текст"),
            (second_audio, "сиротский фрагмент второй текст"),
        ],
    )

    rows = process_predictions(
        prediction_paths,
        manifest_path=tmp_path / "manifest.csv",
        relocate_orphan_phrases=True,
    )

    assert rows[0]["removed_suffix_text"] == "сиротский фрагмент."
    assert rows[0]["relocated_suffix_to"] == str(second_audio)
    assert first_audio.with_suffix(".asr_redacted.txt").read_text(encoding="utf-8").strip() == "Первый текст"
    assert (
        second_audio.with_suffix(".asr_redacted.txt").read_text(encoding="utf-8").strip()
        == "сиротский фрагмент. Второй текст."
    )


def test_process_predictions_keeps_short_phrase_relocation_disabled(tmp_path: Path) -> None:
    """Phrase-level relocation is opt-in."""
    chunk_dir = tmp_path / "cut_samples" / "pez_001" / "pez_001No0"
    chunk_dir.mkdir(parents=True)
    first_audio = chunk_dir / "001_SPEAKER_00_00-00-00-001.wav"
    second_audio = chunk_dir / "002_SPEAKER_00_00-00-00-002.wav"
    first_audio.touch()
    second_audio.touch()
    first_audio.with_suffix(".txt").write_text("Первый текст, сиротский фрагмент.\n", encoding="utf-8")
    second_audio.with_suffix(".txt").write_text("Второй текст.\n", encoding="utf-8")
    prediction_paths = write_prediction_files(
        tmp_path,
        [
            (first_audio, "первый текст"),
            (second_audio, "сиротский фрагмент второй текст"),
        ],
    )

    rows = process_predictions(
        prediction_paths,
        manifest_path=tmp_path / "manifest.csv",
        relocate_orphans=True,
    )

    assert rows[0]["removed_suffix_text"] == ""
    assert rows[0]["changed"] is False
    assert not first_audio.with_suffix(".asr_redacted.txt").exists()
