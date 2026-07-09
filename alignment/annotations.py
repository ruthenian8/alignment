"""Classify bracketed transcript annotations for joining and alignment."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

BRACKET_RE = re.compile(r"\[([^\]]+)\]")
SPEAKER_CODE_RE = re.compile(r"[A-ZА-ЯЁ]{1,6}|\?{3}")
REFERENCE_RE = re.compile(
    r"\bсм\.|[IVXLCDMХVI]+[aа]?-?\d+|граф\.\s*файл|\bфайл\b|нрзб\.\s*\d{1,2}:\d{2}",
    re.IGNORECASE,
)
SPEECH_LABEL_RE = re.compile(
    r"^(?:по[её]т|говорит|говорят одновременно|все|хором|отвечает|спрашивает)\s*:",
    re.IGNORECASE,
)
VISUAL_ACTION_RE = re.compile(
    r"\b(?:показывает|рисует|кивает|сме[её]тся|молчит|мотают|вста[её]т|подходит|бер[её]т|"
    r"клад[её]т|проводит|указывает|начина[её]т|останавливают|спрашивает у|устраивают)\b",
    re.IGNORECASE,
)


class BracketKind(str, Enum):
    """Meaning of a bracketed transcript span."""

    SPEAKER_TAG = "speaker_tag"
    COLLECTOR_UTTERANCE = "collector_utterance"
    SPOKEN_UTTERANCE = "spoken_utterance"
    SPEAKER_NOTE = "speaker_note"
    EDITORIAL_NOTE = "editorial_note"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BracketSpan:
    """One classified bracket span with original string offsets."""

    start: int
    end: int
    text: str
    kind: BracketKind


def speaker_tag_from_text(text: str) -> str:
    """Extract comma-separated speaker codes from a marker-like string."""
    parts = [part.strip() for part in re.split(r"\s*,\s*", text.strip().rstrip(":"))]
    if not parts:
        return ""
    tags = []
    for part in parts:
        if not SPEAKER_CODE_RE.fullmatch(part):
            return ""
        tags.append(part)
    return ", ".join(tags)


def speaker_tag_from_note(text: str) -> str:
    """Extract a speaker code from a short editorial speaker note."""
    code = SPEAKER_CODE_RE.pattern
    patterns = [
        rf"(?<!\w)({code})\s+(?:говорит|спрашивает|отвечает|подхватывает)\b",
        rf"\b(?:говорит|спрашивает|отвечает|подхватывает)(?:\s+\S+){{0,3}}\s+({code})(?!\w)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    body = text.rstrip(":").strip()
    if not re.search(r"[.!?…]", body) and len(body.split()) <= 3:
        match = re.match(rf"({code})(?=\s|,|$)", body)
        if match:
            return match.group(1)
    return ""


def classify_bracket_text(text: str) -> BracketKind:
    """Classify bracket text conservatively, preserving ambiguous utterances."""
    stripped = text.strip()
    if not stripped:
        return BracketKind.EDITORIAL_NOTE
    if speaker_tag_from_text(stripped):
        return BracketKind.SPEAKER_TAG
    if stripped.startswith(("Соб.", "Соб.:")) or "?" in stripped.replace("???", ""):
        return BracketKind.COLLECTOR_UTTERANCE
    if SPEECH_LABEL_RE.match(stripped) or "\\" in stripped:
        return BracketKind.SPOKEN_UTTERANCE
    if speaker_tag_from_note(stripped):
        return BracketKind.SPEAKER_NOTE
    if REFERENCE_RE.search(stripped):
        return BracketKind.EDITORIAL_NOTE
    if VISUAL_ACTION_RE.search(stripped):
        return BracketKind.EDITORIAL_NOTE
    return BracketKind.UNKNOWN


def iter_bracket_spans(text: str) -> list[BracketSpan]:
    """Return classified bracket spans from text."""
    return [
        BracketSpan(match.start(), match.end(), match.group(1).strip(), classify_bracket_text(match.group(1)))
        for match in BRACKET_RE.finditer(text or "")
    ]


def bracket_only_spans(text: str) -> list[BracketSpan]:
    """Return bracket spans only when a row contains no non-bracket text."""
    source = text or ""
    spans: list[BracketSpan] = []
    position = 0
    for match in BRACKET_RE.finditer(source):
        if source[position : match.start()].strip():
            return []
        spans.append(
            BracketSpan(
                match.start(),
                match.end(),
                match.group(1).strip(),
                classify_bracket_text(match.group(1)),
            )
        )
        position = match.end()
    if source[position:].strip():
        return []
    return spans


def is_note_only_transcript_row(text: str) -> bool:
    """Return true when a transcript row contains only removable notes."""
    spans = [span for span in bracket_only_spans(text) if span.kind != BracketKind.SPEAKER_TAG]
    return bool(spans) and all(span.kind == BracketKind.EDITORIAL_NOTE for span in spans)


def should_tokenize_bracket_text(text: str) -> bool:
    """Return true when bracket text should participate in text matching."""
    return classify_bracket_text(text) in {
        BracketKind.COLLECTOR_UTTERANCE,
        BracketKind.SPOKEN_UTTERANCE,
        BracketKind.UNKNOWN,
    }


def is_collector_utterance(text: str) -> bool:
    """Return true when bracket text represents a collector/interviewer utterance."""
    return classify_bracket_text(text) == BracketKind.COLLECTOR_UTTERANCE


def strip_alignment_notes(text: str, speaker_marker: Callable[[str], str]) -> str:
    """Remove editorial notes while preserving speech and compact speaker markers."""

    def replace_note(match: re.Match[str]) -> str:
        marker = match.group(1)
        kind = classify_bracket_text(marker)
        if kind in {BracketKind.COLLECTOR_UTTERANCE, BracketKind.SPOKEN_UTTERANCE, BracketKind.UNKNOWN}:
            return match.group(0)
        tag = speaker_marker(marker)
        return f"[{tag}:]" if tag else ""

    cleaned = re.sub(r"\s+", " ", BRACKET_RE.sub(replace_note, text)).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
