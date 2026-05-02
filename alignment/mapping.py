"""Align chunk-to-transcript mapping tables against matching SRT files."""

from __future__ import annotations

import csv
from pathlib import Path

from .align import align_srt_file, write_aligned_tsv
from .io import write_tsv

MAPPING_SUMMARY_COLUMNS = ["name", "srt", "manual", "aligned_srt", "aligned_tsv", "segments", "status"]


def read_mapping_rows(path: Path | str) -> list[dict[str, str]]:
    """Read a CSV or TSV mapping table into dictionaries."""
    input_path = Path(path)
    delimiter = "," if input_path.suffix.lower() == ".csv" else "\t"
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=delimiter))


def chunk_stem(row: dict[str, str]) -> str:
    """Return the chunk stem recorded by a mapping row."""
    return Path(row.get("name", "").strip()).stem


def align_mapping_table(
    mapping_path: Path | str,
    srt_dir: Path | str,
    output_dir: Path | str,
    *,
    use_transcript_speakers: bool = False,
    infer_missing_speakers: bool = False,
) -> list[dict[str, str]]:
    """Align every mapped transcript row to the SRT with the same chunk stem."""
    srt_root = Path(srt_dir)
    output_root = Path(output_dir)
    manual_dir = output_root / "manual"
    aligned_dir = output_root / "aligned"
    table_dir = output_root / "tables"
    summary: list[dict[str, str]] = []

    for row in read_mapping_rows(mapping_path):
        transcript = row.get("transcript", "").strip()
        stem = chunk_stem(row)
        if not transcript or not stem:
            continue

        srt_path = srt_root / f"{stem}.srt"
        manual_path = manual_dir / f"{stem}.manual.txt"
        aligned_srt_path = aligned_dir / f"{stem}.aligned.srt"
        aligned_tsv_path = table_dir / f"{stem}.aligned.tsv"
        if not srt_path.exists():
            summary.append(
                {
                    "name": stem,
                    "srt": str(srt_path),
                    "manual": "",
                    "aligned_srt": "",
                    "aligned_tsv": "",
                    "segments": "0",
                    "status": "missing_srt",
                }
            )
            continue

        manual_path.parent.mkdir(parents=True, exist_ok=True)
        manual_path.write_text(transcript, encoding="utf-8")
        aligned = align_srt_file(
            srt_path,
            transcript,
            aligned_srt_path,
            use_transcript_speakers=use_transcript_speakers,
            infer_missing_speakers=infer_missing_speakers,
        )
        write_aligned_tsv(stem, aligned, aligned_tsv_path)
        summary.append(
            {
                "name": stem,
                "srt": str(srt_path),
                "manual": str(manual_path),
                "aligned_srt": str(aligned_srt_path),
                "aligned_tsv": str(aligned_tsv_path),
                "segments": str(len(aligned)),
                "status": "aligned",
            }
        )

    write_tsv(output_root / "summary.tsv", summary, MAPPING_SUMMARY_COLUMNS)
    return summary
