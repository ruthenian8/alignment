"""Tests for chunk mapping alignment."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from alignment.cli import main
from alignment.io import read_tsv
from alignment.mapping import summary_quality_errors
from alignment.srt import parse_srt


def test_align_map_reads_csv_and_writes_chunk_outputs(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.csv"
    srt_dir = tmp_path / "srt"
    output_dir = tmp_path / "out"
    srt_dir.mkdir()
    mapping.write_text(
        'name,transcript\nchunk001.wav,"[АБ:] до\\брый день. [АБ:] кра\\сный дом."\nchunk002.wav,\n',
        encoding="utf-8",
    )
    (srt_dir / "chunk001.srt").write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "[SPEAKER_00]: добрый день\n\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "[SPEAKER_00]: красный дом\n",
        encoding="utf-8",
    )

    main(
        [
            "align-map",
            str(mapping),
            str(srt_dir),
            str(output_dir),
            "--use-transcript-speakers",
            "--infer-missing-speakers",
        ]
    )

    summary = read_tsv(output_dir / "summary.tsv")
    assert [(row["name"], row["status"]) for row in summary] == [("chunk001", "aligned")]
    assert summary[0]["segments"] == "2"
    assert summary[0]["matched_segments"] == "2"
    assert summary[0]["blank_speakers"] == "0"
    assert summary[0]["matched_blank_speakers"] == "0"
    assert Path(summary[0]["speaker_map"]).name == "chunk001.speaker_map.csv"
    aligned = parse_srt((output_dir / "aligned" / "chunk001.aligned.srt").read_text(encoding="utf-8"))
    assert [segment.speaker for segment in aligned] == ["[АБ]:", "[АБ]:"]
    assert "до\\брый день" in (output_dir / "manual" / "chunk001.manual.txt").read_text(encoding="utf-8")
    assert read_tsv(output_dir / "tables" / "chunk001.aligned.tsv")[0]["index_name"] == "chunk001"
    with (output_dir / "speaker_maps" / "chunk001.speaker_map.csv").open(encoding="utf-8") as file:
        speaker_rows = list(csv.DictReader(file))
    assert [row["whisperx_speaker"] for row in speaker_rows] == ["[SPEAKER_00]:", "[SPEAKER_00]:"]
    assert [row["transcript_speaker"] for row in speaker_rows] == ["[АБ]:", "[АБ]:"]


def test_align_map_uses_speaker_hint_only_for_rows_without_explicit_speakers(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.csv"
    srt_dir = tmp_path / "srt"
    output_dir = tmp_path / "out"
    srt_dir.mkdir()
    mapping.write_text(
        'name,transcript,speaker_hint\nchunk001.wav,"[Что это?] Ручной ответ.",ААК\n',
        encoding="utf-8",
    )
    (srt_dir / "chunk001.srt").write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "[SPEAKER_00]: что это\n\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "[SPEAKER_01]: ручной ответ\n",
        encoding="utf-8",
    )

    main(
        [
            "align-map",
            str(mapping),
            str(srt_dir),
            str(output_dir),
            "--use-transcript-speakers",
            "--infer-missing-speakers",
        ]
    )

    with (output_dir / "speaker_maps" / "chunk001.speaker_map.csv").open(encoding="utf-8") as file:
        speaker_rows = list(csv.DictReader(file))
    assert [row["transcript_speaker"] for row in speaker_rows] == ["[UNK]:", "[ААК]:"]
    assert [row["speaker_source"] for row in speaker_rows] == ["collector_bracket", "speaker_hint"]
    summary = read_tsv(output_dir / "summary.tsv")
    assert summary[0]["matched_blank_speakers"] == "0"


def test_align_map_can_require_diarized_matched_segments(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.csv"
    srt_dir = tmp_path / "srt"
    output_dir = tmp_path / "out"
    srt_dir.mkdir()
    mapping.write_text('name,transcript\nchunk001.wav,"ручной ответ"\n', encoding="utf-8")
    (srt_dir / "chunk001.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: ручной ответ\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="1 matched segments"):
        main(
            [
                "align-map",
                str(mapping),
                str(srt_dir),
                str(output_dir),
                "--use-transcript-speakers",
                "--require-diarized-matches",
            ]
        )

    summary = read_tsv(output_dir / "summary.tsv")
    assert summary[0]["matched_blank_speakers"] == "1"


def test_align_map_guard_rejects_missing_srt_rows(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.csv"
    srt_dir = tmp_path / "srt"
    output_dir = tmp_path / "out"
    srt_dir.mkdir()
    mapping.write_text('name,transcript\nmissing.wav,"[АБ:] ручной ответ"\n', encoding="utf-8")

    with pytest.raises(SystemExit, match="1 mapping rows were not aligned"):
        main(
            [
                "align-map",
                str(mapping),
                str(srt_dir),
                str(output_dir),
                "--use-transcript-speakers",
                "--require-diarized-matches",
            ]
        )

    summary = read_tsv(output_dir / "summary.tsv")
    assert summary[0]["status"] == "missing_srt"


def test_summary_quality_errors_reports_incomplete_and_undiarized_rows() -> None:
    summary = [
        {
            "status": "missing_srt",
            "matched_blank_speakers": "0",
        },
        {
            "status": "aligned",
            "matched_blank_speakers": "2",
        },
    ]

    assert summary_quality_errors(summary) == [
        "1 mapping rows were not aligned",
        "2 matched segments have no transcript-derived speaker",
    ]


def test_guarded_align_map_output_can_be_exported_with_diarized_matches(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.csv"
    srt_dir = tmp_path / "srt"
    aligned_root = tmp_path / "aligned-root"
    audio_root = tmp_path / "audio"
    output_root = tmp_path / "cut_samples"
    corpus_dir = aligned_root / "and_001"
    srt_dir.mkdir()
    (audio_root / "and_001").mkdir(parents=True)
    mapping.write_text('name,transcript\nand_001No1.wav,"[АБ:] ручной ответ"\n', encoding="utf-8")
    (srt_dir / "and_001No1.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: ручной ответ\n",
        encoding="utf-8",
    )
    (audio_root / "and_001" / "and_001No1.wav").write_bytes(b"not real wav")

    main(
        [
            "align-map",
            str(mapping),
            str(srt_dir),
            str(corpus_dir),
            "--use-transcript-speakers",
            "--infer-missing-speakers",
            "--require-diarized-matches",
        ]
    )
    with patch("alignment.export.subprocess.run"):
        main(
            [
                "export-aligned-map",
                str(aligned_root),
                str(audio_root),
                str(output_root),
                "--require-diarized-matches",
            ]
        )

    exported = output_root / "and_001" / "and_001No1" / "001_АБ_00-00-00-000.txt"
    assert exported.read_text(encoding="utf-8") == "ручной ответ"
