"""Join transcript rows onto active index rows using TSV schemas."""

from __future__ import annotations

from pathlib import Path

from .index_parser import is_continuation_fragment
from .io import JOINED_COLUMNS, parse_bool, read_tsv, write_tsv


def join_rows(
    index_rows: list[dict[str, str]], transcript_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Join transcripts to transcribed index rows without changing row order."""
    output = [dict(row) for row in index_rows]
    for row in output:
        if not parse_bool(row.get("trans")):
            row.update({"transcript": "", "max_speakers": "", "min_speakers": ""})
    targets = [
        row
        for row in output
        if parse_bool(row.get("trans")) and not is_continuation_fragment(row.get("text", ""))
    ]
    if len(targets) < len(transcript_rows):
        targets = [row for row in output if parse_bool(row.get("trans"))]
    for row, transcript in zip(targets, transcript_rows, strict=False):
        row.update(
            {
                "transcript": transcript.get("transcript", ""),
                "max_speakers": transcript.get("max_speakers", ""),
                "min_speakers": transcript.get("min_speakers", ""),
            }
        )
    return output


def join_tsv(index_path: Path | str, transcript_path: Path | str, output_path: Path | str) -> None:
    """Read canonical TSV inputs, join them, and write canonical joined TSV."""
    write_tsv(output_path, join_rows(read_tsv(index_path), read_tsv(transcript_path)), JOINED_COLUMNS)
