"""Reorder joined transcript rows using simple monotonic text similarity."""

from __future__ import annotations

import difflib
import re
import string
from pathlib import Path

from .io import JOINED_COLUMNS, parse_bool, read_tsv, write_tsv

BRACKET_RE = re.compile(r"\[[^\]]*\]")
TIME_RE = re.compile(r"\d{1,2}:\d{2}:\d{2}[,.]\d{3}")


def normalize_for_match(text: str) -> str:
    """Normalize text for rough matching while preserving originals elsewhere."""
    text = BRACKET_RE.sub(" ", text or "")
    text = TIME_RE.sub(" ", text)
    text = text.replace("\\", "").replace("ё", "е").replace("Ё", "Е")
    punctuation = string.punctuation + "«»“”„…—–"
    text = text.translate(str.maketrans({char: " " for char in punctuation}))
    return " ".join(text.lower().split())


def similarity(left: str, right: str) -> float:
    """Return a lightweight token/string similarity score."""
    left_norm = normalize_for_match(left)
    right_norm = normalize_for_match(right)
    if not left_norm or not right_norm:
        return 0.0
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    token_score = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    char_score = difflib.SequenceMatcher(None, left_norm, right_norm).ratio()
    return 0.6 * token_score + 0.4 * char_score


def reorder_rows(rows: list[dict[str, str]], *, max_shift: int = 3) -> list[dict[str, str]]:
    """Reorder transcript fields for active rows when a small shift is detected."""
    active = [index for index, row in enumerate(rows) if parse_bool(row.get("trans"))]
    if len(active) < 2:
        return [dict(row) for row in rows]
    best_shift = 0
    best_score = -1.0
    for shift in range(0, min(max_shift, len(active) - 1) + 1):
        scores = [
            similarity(rows[row]["text"], rows[active[position + shift]].get("transcript", ""))
            for position, row in enumerate(active[: -shift or None])
        ]
        mean = sum(scores) / len(scores) if scores else 0.0
        if mean > best_score:
            best_shift = shift
            best_score = mean
    if best_shift == 0:
        return [dict(row) for row in rows]
    output = [dict(row) for row in rows]
    rotated = active[best_shift:] + active[:best_shift]
    for target, source in zip(active, rotated, strict=True):
        for column in ("transcript", "max_speakers", "min_speakers"):
            output[target][column] = rows[source].get(column, "")
    return output


def reorder_tsv(input_path: Path | str, output_path: Path | str, *, max_shift: int = 3) -> None:
    """Read a joined TSV, reorder transcript fields, and write TSV."""
    write_tsv(output_path, reorder_rows(read_tsv(input_path), max_shift=max_shift), JOINED_COLUMNS)
