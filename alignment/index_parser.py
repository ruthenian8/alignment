"""Parse coarse plaintext or DOCX audio indexes into TSV rows."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from .io import INDEX_COLUMNS, write_tsv
from .srt import normalize_timestamp

TIME_RE = re.compile(r"\d{1,2}:\d{2}:\d{2}[,.]\d{3}")
START_RE = re.compile(rf"^\s*({TIME_RE.pattern})(?:\s*[-–]\s*|\s+)?(.*)")
CONT_RE = re.compile(r"продолж|окончание|начало\s+см", re.IGNORECASE)
INDEX_PREFIX_RE = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*[-–—]\s*")


def is_continuation_fragment(text: str) -> bool:
    """Return True for index descriptions that point back to an earlier start."""
    return "начало см" in INDEX_PREFIX_RE.sub("", text).casefold()


def read_index_text(path: Path | str) -> str:
    """Read plaintext or DOCX index content as UTF-8 text."""
    input_path = Path(path)
    if input_path.suffix.lower() != ".docx":
        return input_path.read_text(encoding="utf-8-sig")
    with zipfile.ZipFile(input_path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for para in root.findall(".//w:p", ns):
        parts = [node.text or "" for node in para.findall(".//w:t", ns)]
        if parts:
            paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


def parse_index_text(text: str, *, audio_stem: str = "audio", suffix: str = ".wav") -> list[dict[str, str]]:
    """Parse a coarse index into canonical index rows."""
    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = START_RE.match(line)
        if match:
            start = normalize_timestamp(match.group(1), decimal=".")
            body = f"{match.group(1)} - {match.group(2).strip()}"
            row = {
                "start": start,
                "trans": str("НЕ РАСПИСАНО" not in line.upper()),
                "cont": "",
                "prev": "",
                "text": body,
                "name": f"{audio_stem}No{len(rows)}{suffix}",
            }
            times = TIME_RE.findall(line)
            if len(times) > 1 and CONT_RE.search(line):
                row["cont"] = normalize_timestamp(times[-1], decimal=".")
            rows.append(row)
            current = row
            continue
        if current is not None:
            current["text"] = f"{current['text']} {line}"
            if "НЕ РАСПИСАНО" in line.upper():
                current["trans"] = "False"
    starts = {row["start"]: index for index, row in enumerate(rows)}
    for index, row in enumerate(rows):
        if row["cont"] in starts:
            rows[starts[row["cont"]]]["prev"] = str(index)
    return rows


def parse_index_file(
    path: Path | str, *, audio_stem: str | None = None, suffix: str = ".wav"
) -> list[dict[str, str]]:
    """Parse an index file into canonical index rows."""
    input_path = Path(path)
    return parse_index_text(
        read_index_text(input_path), audio_stem=audio_stem or input_path.stem, suffix=suffix
    )


def write_index_tsv(
    input_path: Path | str, output_path: Path | str, *, audio_stem: str | None = None
) -> None:
    """Parse an index file and write canonical index TSV."""
    write_tsv(output_path, parse_index_file(input_path, audio_stem=audio_stem), INDEX_COLUMNS)
