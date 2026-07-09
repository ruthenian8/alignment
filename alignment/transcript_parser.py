"""Parse manual plaintext transcripts into TSV rows."""

from __future__ import annotations

from pathlib import Path

from .align import format_transcript_speaker_marker, speaker_tag_from_line
from .io import TRANSCRIPT_COLUMNS, write_tsv

TABLE_HEADER = ["Текст", "Год", "Село", "Информант", "Собиратель 1", "Собиратель 2", "Программа", "Вопросы"]
NUMBERED_TABLE_HEADER = [
    "№",
    "Текст карточки",
    "Село",
    "Год",
    "№ программы",
    "№№ вопросов",
    "Информант",
    "Собиратели",
]


def speaker_count(text: str) -> int:
    """Count comma/semicolon-separated speaker codes in a metadata field."""
    return len([item for item in text.replace(";", ",").split(",") if item.strip()])


def row_with_speaker(
    number: int, transcript: str, interviewer_count: int, interviewee_text: str
) -> dict[str, object]:
    """Build a transcript row and prepend the recoverable interviewee tag."""
    speaker_tag = speaker_tag_from_line(interviewee_text)
    if speaker_tag:
        transcript = f"{format_transcript_speaker_marker(speaker_tag)} {transcript}"
    interviewee_count = speaker_count(interviewee_text)
    return {
        "id": number,
        "transcript": transcript,
        "max_speakers": interviewer_count + interviewee_count,
        "min_speakers": interviewee_count,
    }


def parse_exported_table_transcript(lines: list[str]) -> list[dict[str, object]]:
    """Parse LibreOffice-exported transcript tables with fixed columns."""
    rows: list[dict[str, object]] = []
    data = lines[len(TABLE_HEADER) :]
    for offset in range(0, len(data) - 7, 8):
        record = data[offset : offset + 8]
        rows.append(
            row_with_speaker(
                len(rows) + 1,
                record[0],
                speaker_count(", ".join(record[4:6])),
                record[3],
            )
        )
    return rows


def parse_numbered_table_transcript(lines: list[str]) -> list[dict[str, object]]:
    """Parse numbered LibreOffice table exports with multiline text cells."""
    rows: list[dict[str, object]] = []
    data = lines[len(NUMBERED_TABLE_HEADER) :]
    starts: list[int] = []
    expected = 1
    for index, line in enumerate(data):
        if line == str(expected):
            starts.append(index)
            expected += 1
    starts.append(len(data))
    for index in range(len(starts) - 1):
        start = starts[index]
        end = starts[index + 1]
        record = data[start + 1 : end]
        if len(record) < 7:
            continue
        transcript = " ".join(record[:-6])
        if not transcript:
            continue
        rows.append(row_with_speaker(len(rows) + 1, transcript, speaker_count(record[-1]), record[-2]))
    return rows


def strip_outer_blank_lines(lines: list[str]) -> list[str]:
    """Remove leading and trailing blanks without collapsing table cells."""
    start = 0
    end = len(lines)
    while start < end and not lines[start]:
        start += 1
    while end > start and not lines[end - 1]:
        end -= 1
    return lines[start:end]


def parse_transcript_text(text: str) -> list[dict[str, object]]:
    """Parse the repository's block-based manual transcript format."""
    table_lines = strip_outer_blank_lines([line.strip() for line in text.splitlines()])
    if table_lines[: len(TABLE_HEADER)] == TABLE_HEADER:
        return parse_exported_table_transcript(table_lines)
    if table_lines[: len(NUMBERED_TABLE_HEADER)] == NUMBERED_TABLE_HEADER:
        return parse_numbered_table_transcript(table_lines)

    rows: list[dict[str, object]] = []
    blocks = [block for block in text.strip().split("\n\n") if block.strip()]
    for number, block in enumerate(blocks, start=1):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 5:
            continue
        transcript = " ".join(lines[3:-1])
        rows.append(row_with_speaker(number, transcript, speaker_count(lines[2]), lines[-1]))
    return rows


def parse_transcript_file(path: Path | str) -> list[dict[str, object]]:
    """Parse a manual transcript plaintext file."""
    return parse_transcript_text(Path(path).read_text(encoding="utf-8-sig"))


def write_transcript_tsv(input_path: Path | str, output_path: Path | str) -> None:
    """Parse a manual transcript and write canonical transcript TSV."""
    write_tsv(output_path, parse_transcript_file(input_path), TRANSCRIPT_COLUMNS)
