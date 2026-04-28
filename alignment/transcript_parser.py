"""Parsing of plaintext transcript files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class TranscriptRecord:
    """A single transcript record.

    Attributes:
        archive_id: Archive identifier string (e.g. ``XXIIа-19``).
        location: Recording location and date string.
        interviewers: List of interviewer names.
        interviewees: List of interviewee names.
        text: Full transcript text (multiple lines joined with space).
    """

    archive_id: str
    location: str
    interviewers: List[str]
    interviewees: List[str]
    text: str


def parse_transcript_file(path: Path) -> List[TranscriptRecord]:
    """Parse a plaintext transcript file into :class:`TranscriptRecord` objects.

    The file format separates records with blank lines (``\\n\\n``). Each
    record has the following structure:

    - Line 0: archive ID
    - Line 1: location / date
    - Line 2: comma-separated interviewer names
    - Lines 3 to -2: transcript text (joined with a single space)
    - Last line: comma-separated interviewee names

    Args:
        path: Path to the plaintext transcript file (UTF-8 encoded).

    Returns:
        List of TranscriptRecord objects, one per record block.
    """
    content = path.read_text(encoding="utf-8").lstrip("\ufeff").rstrip("\n")
    record_strings = content.split("\n\n")
    records: List[TranscriptRecord] = []
    for block in record_strings:
        lines = block.splitlines()
        if len(lines) < 4:
            # Not enough lines to form a valid record
            continue
        archive_id = lines[0].strip()
        location = lines[1].strip()
        interviewers = [n.strip() for n in lines[2].split(",") if n.strip()]
        interviewees = [n.strip() for n in lines[-1].split(",") if n.strip()]
        text_lines = lines[3:-1]
        text = " ".join(line.strip() for line in text_lines if line.strip())
        records.append(
            TranscriptRecord(
                archive_id=archive_id,
                location=location,
                interviewers=interviewers,
                interviewees=interviewees,
                text=text,
            )
        )
    return records


def transcript_records_to_dataframe(records: List[TranscriptRecord]) -> pd.DataFrame:
    """Convert transcript records to a DataFrame.

    Args:
        records: List of TranscriptRecord objects.

    Returns:
        DataFrame with columns: id (1-based), transcript, max_speakers,
        min_speakers.
    """
    rows = []
    for i, rec in enumerate(records, start=1):
        n_interviewers = len(rec.interviewers)
        n_interviewees = len(rec.interviewees)
        rows.append(
            {
                "id": i,
                "transcript": rec.text,
                "max_speakers": n_interviewers + n_interviewees,
                "min_speakers": n_interviewees,
            }
        )
    return pd.DataFrame(rows)
