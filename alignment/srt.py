"""SRT parsing, timestamp conversion, and formatting."""

from __future__ import annotations

import re
from dataclasses import dataclass

TIMESTAMP_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})")
RANGE_RE = re.compile(rf"({TIMESTAMP_RE.pattern})\s*-->\s*({TIMESTAMP_RE.pattern})")
SPEAKER_RE = re.compile(r"^\s*(\[[^\]]+\]:?)\s*(.*)$")


@dataclass(frozen=True)
class SrtSegment:
    """One subtitle segment with timing, speaker, and text."""

    index: int
    start: str
    end: str
    speaker: str
    text: str


def timestamp_to_ms(timestamp: str) -> int:
    """Convert an SRT timestamp with comma or dot milliseconds to milliseconds."""
    match = TIMESTAMP_RE.fullmatch(timestamp.strip())
    if not match:
        raise ValueError(f"Invalid timestamp: {timestamp!r}")
    hours, minutes, seconds, millis = [int(part) for part in match.groups()]
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def ms_to_timestamp(milliseconds: int, *, decimal: str = ",") -> str:
    """Format milliseconds as an SRT timestamp."""
    if milliseconds < 0:
        raise ValueError("Timestamp cannot be negative")
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{decimal}{millis:03}"


def normalize_timestamp(timestamp: str, *, decimal: str = ".") -> str:
    """Normalize a timestamp to either dot or comma millisecond syntax."""
    return ms_to_timestamp(timestamp_to_ms(timestamp), decimal=decimal)


def parse_srt(text: str) -> list[SrtSegment]:
    """Parse SRT text into segments, preserving multiline text and speaker tags."""
    segments: list[SrtSegment] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        try:
            index = int(lines[0])
        except ValueError:
            continue
        range_match = RANGE_RE.search(lines[1])
        if not range_match:
            continue
        start = normalize_timestamp(range_match.group(1), decimal=",")
        end = normalize_timestamp(range_match.group(6), decimal=",")
        body = lines[2:]
        speaker = ""
        if body:
            speaker_match = SPEAKER_RE.match(body[0])
            if speaker_match:
                speaker = speaker_match.group(1)
                if not speaker.endswith(":"):
                    speaker += ":"
                body[0] = speaker_match.group(2)
        segments.append(SrtSegment(index, start, end, speaker, "\n".join(body).strip()))
    return segments


def format_srt(segments: list[SrtSegment]) -> str:
    """Format segments as SRT text."""
    blocks = []
    for segment in segments:
        prefix = f"{segment.speaker} " if segment.speaker else ""
        blocks.append(f"{segment.index}\n{segment.start} --> {segment.end}\n{prefix}{segment.text}".rstrip())
    return "\n\n".join(blocks) + ("\n" if blocks else "")
