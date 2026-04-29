"""Shared TSV helpers and schema constants for the alignment pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

INDEX_COLUMNS = ["start", "trans", "cont", "prev", "text", "name"]
TRANSCRIPT_COLUMNS = ["id", "transcript", "max_speakers", "min_speakers"]
JOINED_COLUMNS = INDEX_COLUMNS + ["transcript", "max_speakers", "min_speakers"]
ALIGNED_COLUMNS = [
    "index_name",
    "srt_index",
    "start",
    "end",
    "speaker",
    "srt_text",
    "transcript_text",
    "normalized_text",
    "matched",
    "score",
]
MANIFEST_COLUMNS = [
    "clip_id",
    "audio_path",
    "text_path",
    "text_original_path",
    "start",
    "end",
    "speaker",
    "text",
    "text_original",
]
EMBEDDED_ALIGNMENT_COLUMNS = [
    "pair_index",
    "whisper_text",
    "manual_text",
    "whisper_clean",
    "manual_clean",
    "score",
]


def read_tsv(path: Path | str) -> list[dict[str, str]]:
    """Read a UTF-8 TSV file into dictionaries."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def write_tsv(path: Path | str, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write dictionaries as a UTF-8 TSV file using the supplied schema."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key, "") for key in fieldnames})


def parse_bool(value: object) -> bool:
    """Parse common table boolean values."""
    if isinstance(value, bool):
        return value
    text = "" if value is None else str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}
