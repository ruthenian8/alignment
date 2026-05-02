"""Tests for chunk mapping alignment."""

from __future__ import annotations

from pathlib import Path

from alignment.cli import main
from alignment.io import read_tsv
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
    aligned = parse_srt((output_dir / "aligned" / "chunk001.aligned.srt").read_text(encoding="utf-8"))
    assert [segment.speaker for segment in aligned] == ["[АБ]:", "[АБ]:"]
    assert "до\\брый день" in (output_dir / "manual" / "chunk001.manual.txt").read_text(encoding="utf-8")
    assert read_tsv(output_dir / "tables" / "chunk001.aligned.tsv")[0]["index_name"] == "chunk001"
