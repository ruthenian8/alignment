"""Join transcript rows onto active index rows using TSV schemas."""

from __future__ import annotations

from pathlib import Path

from .annotations import bracket_only_spans, is_note_only_transcript_row
from .index_parser import is_continuation_fragment
from .io import JOINED_COLUMNS, parse_bool, read_tsv, write_tsv


def bracketed_meta_spans(text: str) -> list[str]:
    """Return bracket spans when the row has no non-bracket text."""
    return [span.text for span in bracket_only_spans(text)]


def is_meta_commentary_span(text: str) -> bool:
    """Return true for bracketed editorial commentary, not interviewer questions."""
    return is_note_only_transcript_row(f"[{text}]")


def is_meta_commentary_transcript(text: str) -> bool:
    """Return true when a transcript row contains only bracketed commentary."""
    return is_note_only_transcript_row(text)


def transcript_rows_for_join(transcript_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop note-only transcript rows before assigning transcripts to index rows."""
    return [row for row in transcript_rows if not is_meta_commentary_transcript(row.get("transcript", ""))]


def join_rows(
    index_rows: list[dict[str, str]], transcript_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Join transcripts to transcribed index rows without changing row order."""
    transcript_rows = transcript_rows_for_join(transcript_rows)
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
    for index in range(min(len(targets), len(transcript_rows))):
        row = targets[index]
        transcript = transcript_rows[index]
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
