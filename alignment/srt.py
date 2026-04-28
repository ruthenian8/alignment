"""SRT subtitle parsing and formatting utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class SrtSegment:
    """A single SRT subtitle segment.

    Attributes:
        index: The numeric index of the segment.
        start: Start timestamp in ``HH:MM:SS,mmm`` format.
        end: End timestamp in ``HH:MM:SS,mmm`` format.
        speaker: Speaker tag (e.g. ``[SPEAKER_01]:``) or empty string.
        text: The spoken text (without timestamps or index).
    """

    index: int
    start: str
    end: str
    speaker: str
    text: str


# Matches HH:MM:SS,mmm or HH:MM:SS.mmm (WhisperX uses dot sometimes)
_TS_PATTERN = re.compile(r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})\s+-->\s+(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})")
_SPEAKER_PATTERN = re.compile(r"^(\[.*?\]:)\s*(.*)", re.DOTALL)


def _normalise_ts(ts: str) -> str:
    """Convert dot millisecond separator to comma (canonical SRT format)."""
    return ts.replace(".", ",")


def parse_srt(text: str) -> List[SrtSegment]:
    """Parse a raw SRT string into a list of :class:`SrtSegment`.

    Handles both ``HH:MM:SS,mmm`` and ``HH:MM:SS.mmm`` millisecond
    separators. Optional ``[SPEAKER_XX]:`` prefix is extracted into
    the ``speaker`` field. Multi-line segment text is preserved.

    Args:
        text: Raw SRT content as a string.

    Returns:
        List of parsed SrtSegment objects, in order.
    """
    segments: List[SrtSegment] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        if len(lines) < 2:
            continue
        ts_match = _TS_PATTERN.search(lines[1])
        if not ts_match:
            continue
        start = _normalise_ts(ts_match.group(1))
        end = _normalise_ts(ts_match.group(2))

        text_lines = lines[2:]
        speaker = ""
        if text_lines:
            sm = _SPEAKER_PATTERN.match(text_lines[0])
            if sm:
                speaker = sm.group(1)
                text_lines[0] = sm.group(2)
        seg_text = "\n".join(text_lines)
        segments.append(
            SrtSegment(index=index, start=start, end=end, speaker=speaker, text=seg_text)
        )
    return segments


def format_srt(segments: List[SrtSegment]) -> str:
    """Serialise a list of :class:`SrtSegment` back to SRT string.

    Timestamps use the comma millisecond separator. Speaker prefix is
    written on the same line as the text.

    Args:
        segments: List of SrtSegment objects.

    Returns:
        SRT formatted string.
    """
    blocks = []
    for seg in segments:
        speaker_prefix = (seg.speaker + " ") if seg.speaker else ""
        block = f"{seg.index}\n{seg.start} --> {seg.end}\n{speaker_prefix}{seg.text}"
        blocks.append(block)
    return "\n\n".join(blocks) + "\n"
