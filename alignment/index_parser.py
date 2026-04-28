"""Parsing of plaintext and DOCX audio index files."""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns
_TIME_CODE_RE = re.compile(r"[\d:\.,]{12}")
_TIME_CODE_LINE_RE = re.compile(r"^[\d:\.,]{12}[ \-–]")
_CONTINUATION_RE = re.compile(r"продолж", re.IGNORECASE)


@dataclass
class IndexRow:
    """A single row from an audio index file.

    Attributes:
        start: Start timestamp string (e.g. ``00:00:00.000``).
        trans: Whether this segment has been transcribed.
        cont: Timestamp of the continuation segment (empty string if none).
        prev: Index of the preceding segment for continuations (empty string if none).
        text: Raw description text from the index.
        name: Output audio filename (e.g. ``pez_001No0.wav``).
    """

    start: str
    trans: bool
    cont: str
    prev: str
    text: str
    name: str


def parse_index_plaintext(text: str, base_name: str, suffix: str = ".wav") -> List[IndexRow]:
    """Parse a plaintext index file into :class:`IndexRow` objects.

    The plaintext format has a short header (filename, location, informant,
    collectors) followed by a blank line, then timestamped lines. Lines
    starting with a 12-character timestamp are treated as index entries.
    ``НЕ РАСПИСАНО`` in a line marks it as not transcribed (``trans=False``).
    Continuation timestamps are detected by a trailing timestamp + the word
    ``продолж``.

    After parsing, continuation links are resolved so that each continuation
    row's ``prev`` field points to the index of its predecessor.

    Args:
        text: Raw plaintext content of the index file.
        base_name: Base name for output files (e.g. ``pez_001``).
        suffix: Audio file suffix (default ``.wav``).

    Returns:
        List of IndexRow objects with continuations linked.
    """
    lines = text.splitlines()
    raw_records: List[Dict[str, Any]] = []

    for line in lines:
        stripped = line.strip()
        if not _TIME_CODE_LINE_RE.match(stripped):
            continue

        start_match = _TIME_CODE_RE.match(stripped)
        if not start_match:
            continue
        start_code = start_match.group().replace(",", ".")

        is_transcribed = "РАСПИСАНО" not in stripped

        cont_code = ""
        if _CONTINUATION_RE.search(stripped):
            # Find all timestamps in the line and use the last non-start one
            all_ts = _TIME_CODE_RE.findall(stripped)
            for ts in reversed(all_ts):
                candidate = ts.replace(",", ".")
                if candidate != start_code:
                    cont_code = candidate
                    break

        raw_records.append(
            {
                "start": start_code,
                "trans": is_transcribed,
                "cont": cont_code,
                "prev": "",
                "text": stripped,
            }
        )

    # Assign names
    for idx, rec in enumerate(raw_records):
        rec["name"] = f"{base_name}No{idx}{suffix}"

    # Link continuations: for each record with a cont, find the target start
    start_to_idx = {rec["start"]: i for i, rec in enumerate(raw_records)}
    for i, rec in enumerate(raw_records):
        cont_val = rec["cont"]
        if cont_val:
            target = start_to_idx.get(cont_val)
            if target is not None:
                raw_records[target]["prev"] = str(i)

    return [
        IndexRow(
            start=r["start"],
            trans=r["trans"],
            cont=r["cont"],
            prev=r["prev"],
            text=r["text"],
            name=r["name"],
        )
        for r in raw_records
    ]


def _read_docx_xml(path: Path) -> BeautifulSoup:
    """Read the XML content from a .docx file.

    Args:
        path: Path to the .docx file.

    Returns:
        BeautifulSoup object parsed from ``word/document.xml``.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read as a ZIP archive.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            with archive.open("word/document.xml") as doc_xml:
                content = doc_xml.read()
        return BeautifulSoup(content, "xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise OSError(f"Could not read 'word/document.xml' from {path}") from exc


def parse_index_docx(path: Path) -> List[IndexRow]:
    """Parse a DOCX-format index file into :class:`IndexRow` objects.

    The DOCX is expected to contain timestamped paragraphs in the same
    format as the plaintext variant. Paragraphs are extracted from the
    Word XML, then processed identically to the plaintext parser.

    Args:
        path: Path to the ``.docx`` index file.

    Returns:
        List of IndexRow objects with continuations linked.
    """
    soup = _read_docx_xml(path)
    paragraphs = [p.get_text() for p in soup.find_all("w:p")]
    base_name = path.stem
    suffix = ".wav"
    pseudo_text = "\n".join(paragraphs)
    return parse_index_plaintext(pseudo_text, base_name, suffix)


def index_rows_to_dataframe(rows: List[IndexRow]) -> pd.DataFrame:
    """Convert a list of :class:`IndexRow` objects to a DataFrame.

    Args:
        rows: List of IndexRow objects.

    Returns:
        DataFrame with columns: start, trans, cont, prev, text, name.
    """
    return pd.DataFrame(
        [
            {
                "start": r.start,
                "trans": r.trans,
                "cont": r.cont,
                "prev": r.prev,
                "text": r.text,
                "name": r.name,
            }
            for r in rows
        ]
    )
