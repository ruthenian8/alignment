"""Parse manual plaintext transcripts into TSV rows."""

from __future__ import annotations

from pathlib import Path

from .align import format_transcript_speaker_marker, speaker_tag_from_line
from .io import TRANSCRIPT_COLUMNS, write_tsv


def parse_transcript_text(text: str) -> list[dict[str, object]]:
    """Parse the repository's block-based manual transcript format."""
    rows: list[dict[str, object]] = []
    blocks = [block for block in text.strip().split("\n\n") if block.strip()]
    for number, block in enumerate(blocks, start=1):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 5:
            continue
        interviewer_count = len([item for item in lines[2].split(",") if item.strip()])
        interviewee_count = len([item for item in lines[-1].split(",") if item.strip()])
        transcript = " ".join(lines[3:-1])
        speaker_tag = speaker_tag_from_line(lines[-1])
        if speaker_tag:
            transcript = f"{format_transcript_speaker_marker(speaker_tag)} {transcript}"
        rows.append(
            {
                "id": number,
                "transcript": transcript,
                "max_speakers": interviewer_count + interviewee_count,
                "min_speakers": interviewee_count,
            }
        )
    return rows


def parse_transcript_file(path: Path | str) -> list[dict[str, object]]:
    """Parse a manual transcript plaintext file."""
    return parse_transcript_text(Path(path).read_text(encoding="utf-8-sig"))


def write_transcript_tsv(input_path: Path | str, output_path: Path | str) -> None:
    """Parse a manual transcript and write canonical transcript TSV."""
    write_tsv(output_path, parse_transcript_file(input_path), TRANSCRIPT_COLUMNS)
