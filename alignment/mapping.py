"""Align chunk-to-transcript mapping tables against matching SRT files."""

from __future__ import annotations

import csv
from pathlib import Path

from .align import AlignedSegment, align_srt_file, write_aligned_tsv, write_speaker_map
from .io import write_tsv

MAPPING_SUMMARY_COLUMNS = [
    "name",
    "srt",
    "manual",
    "aligned_srt",
    "aligned_tsv",
    "speaker_map",
    "segments",
    "matched_segments",
    "match_ratio",
    "blank_speakers",
    "matched_blank_speakers",
    "status",
]


def read_mapping_rows(path: Path | str) -> list[dict[str, str]]:
    """Read a CSV or TSV mapping table into dictionaries."""
    input_path = Path(path)
    delimiter = "," if input_path.suffix.lower() == ".csv" else "\t"
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=delimiter))


def chunk_stem(row: dict[str, str]) -> str:
    """Return the chunk stem recorded by a mapping row."""
    return Path(row.get("name", "").strip()).stem


def row_speaker_hint(row: dict[str, str]) -> str:
    """Return an optional mapping-provided actual speaker tag."""
    return (
        (
            row.get("transcript_speaker", "")
            or row.get("speaker", "")
            or row.get("speaker_hint", "")
            or row.get("respondent", "")
        )
        .strip()
        .strip("[]:")
    )


def speaker_summary(aligned: list[AlignedSegment]) -> dict[str, str]:
    """Return compact speaker coverage metrics for aligned segments."""
    matched = sum(1 for item in aligned if item.matched)
    blank = sum(1 for item in aligned if not item.transcript_speaker)
    matched_blank = sum(1 for item in aligned if item.matched and not item.transcript_speaker)
    ratio = matched / len(aligned) if aligned else 0.0
    return {
        "segments": str(len(aligned)),
        "matched_segments": str(matched),
        "match_ratio": f"{ratio:.3f}",
        "blank_speakers": str(blank),
        "matched_blank_speakers": str(matched_blank),
    }


def _sample_names(rows: list[dict[str, str]], limit: int = 5) -> str:
    names = [row.get("name", "").strip() for row in rows if row.get("name", "").strip()]
    sample = ", ".join(names[:limit])
    if len(names) > limit:
        sample += f", +{len(names) - limit} more"
    return sample


def summary_quality_errors(summary: list[dict[str, str]], *, min_match_ratio: float = 0.0) -> list[str]:
    """Return quality-gate errors for align-map summary rows."""
    errors = []
    incomplete = [row for row in summary if row["status"] != "aligned"]
    if incomplete:
        by_status = sorted({row["status"] for row in incomplete})
        detail = ", ".join(
            f"{status}: {sum(row['status'] == status for row in incomplete)}" for status in by_status
        )
        sample = _sample_names(incomplete)
        message = f"{len(incomplete)} mapping rows were not aligned ({detail})"
        if sample:
            message += f": {sample}"
        errors.append(message)
    missing = sum(int(row["matched_blank_speakers"]) for row in summary)
    if missing:
        errors.append(f"{missing} matched segments have no transcript-derived speaker")
    if min_match_ratio > 0:
        low_match = [
            row
            for row in summary
            if row["status"] == "aligned"
            and int(row.get("segments", "0") or 0) > 0
            and float(row.get("match_ratio", "0") or 0) < min_match_ratio
        ]
        if low_match:
            sample = _sample_names(low_match)
            message = f"{len(low_match)} aligned rows are below minimum match ratio {min_match_ratio:.3f}"
            if sample:
                message += f": {sample}"
            errors.append(message)
    return errors


def align_mapping_table(
    mapping_path: Path | str,
    srt_dir: Path | str,
    output_dir: Path | str,
    *,
    use_transcript_speakers: bool = False,
    infer_missing_speakers: bool = False,
    allow_leading_transcript_skip: bool = True,
) -> list[dict[str, str]]:
    """Align every mapped transcript row to the SRT with the same chunk stem."""
    srt_root = Path(srt_dir)
    output_root = Path(output_dir)
    manual_dir = output_root / "manual"
    aligned_dir = output_root / "aligned"
    table_dir = output_root / "tables"
    speaker_map_dir = output_root / "speaker_maps"
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
        speaker_map_path = speaker_map_dir / f"{stem}.speaker_map.csv"
        if not srt_path.exists():
            summary.append(
                {
                    "name": stem,
                    "srt": str(srt_path),
                    "manual": "",
                    "aligned_srt": "",
                    "aligned_tsv": "",
                    "speaker_map": "",
                    "segments": "0",
                    "matched_segments": "0",
                    "match_ratio": "0.000",
                    "blank_speakers": "0",
                    "matched_blank_speakers": "0",
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
            fallback_speaker=row_speaker_hint(row),
            allow_leading_transcript_skip=allow_leading_transcript_skip,
        )
        write_aligned_tsv(stem, aligned, aligned_tsv_path)
        write_speaker_map(aligned, speaker_map_path)
        coverage = speaker_summary(aligned)
        summary.append(
            {
                "name": stem,
                "srt": str(srt_path),
                "manual": str(manual_path),
                "aligned_srt": str(aligned_srt_path),
                "aligned_tsv": str(aligned_tsv_path),
                "speaker_map": str(speaker_map_path),
                **coverage,
                "status": "aligned",
            }
        )

    write_tsv(output_root / "summary.tsv", summary, MAPPING_SUMMARY_COLUMNS)
    return summary
