"""Join transcript rows onto active index rows using TSV schemas."""

from __future__ import annotations

from pathlib import Path

from .io import JOINED_COLUMNS, parse_bool, read_tsv, write_tsv


def join_rows(
    index_rows: list[dict[str, str]], transcript_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Join transcripts to transcribed index rows without changing row order."""
    output = [dict(row) for row in index_rows]
    transcripts = iter(transcript_rows)
    for row in output:
        if not parse_bool(row.get("trans")):
            row.update({"transcript": "", "max_speakers": "", "min_speakers": ""})
            continue
        transcript = next(transcripts, {})
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
