"""Monotonic DP alignment between WhisperX SRT segments and manual transcript text."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace
from pathlib import Path

from .annotations import (
    BRACKET_RE,
    BracketKind,
    classify_bracket_text,
    is_collector_utterance,
    should_tokenize_bracket_text,
    speaker_tag_from_text,
    strip_alignment_notes,
)
from .io import ALIGNED_COLUMNS, SPEAKER_MAP_COLUMNS, write_tsv
from .reorder import normalize_for_match
from .srt import SrtSegment, format_srt, parse_srt

SPEAKER_MARKER_RE = re.compile(r"\[([^\]]{1,300}):\]")
SPEAKER_CODE_RE = re.compile(r"[A-ZА-ЯЁ]{1,6}|\?{3}")
UNKNOWN_SPEAKER = "UNK"


@dataclass(frozen=True)
class TranscriptToken:
    """A token from the original transcript with matching and slice metadata."""

    text: str
    norm: str
    start: int
    end: int


@dataclass(frozen=True)
class AlignedSegment:
    """One SRT segment aligned to a manual transcript span."""

    srt: SrtSegment
    original_speaker: str
    transcript_text: str
    normalized_text: str
    matched: bool
    score: float
    transcript_start: int = -1
    transcript_end: int = -1
    transcript_speaker: str = ""
    speaker_source: str = ""


@dataclass(frozen=True)
class SpeakerBlock:
    """Transcript text range covered by a block-level speaker footer."""

    start: int
    end: int
    tag: str


def tokenize_transcript(text: str) -> list[TranscriptToken]:
    """Tokenize transcript text while preserving original character offsets."""
    skipped_ranges = [
        (match.start(), match.end())
        for match in BRACKET_RE.finditer(text)
        if not should_tokenize_bracket(match.group(1))
    ]
    tokens: list[TranscriptToken] = []
    for match in re.finditer(r"\S+", text):
        if any(match.start() < end and start < match.end() for start, end in skipped_ranges):
            continue
        raw = match.group()
        norm = normalize_for_match(raw.strip("[]"))
        if norm:
            tokens.append(TranscriptToken(raw, norm, match.start(), match.end()))
    return tokens


def token_similarity(left: list[str], right: list[str]) -> float:
    """Compute token F1 similarity for alignment scoring."""
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    common = len(left_set & right_set)
    return 0.0 if common == 0 else (2 * common) / (len(left_set) + len(right_set))


def is_unknown_speaker_bracket(marker_text: str) -> bool:
    """Return true for bracketed interviewer/collector text that should be [UNK]."""
    return is_collector_utterance(marker_text)


def should_tokenize_bracket(marker_text: str) -> bool:
    """Return true for bracketed text whose words should participate in matching."""
    return should_tokenize_bracket_text(marker_text)


def is_short_speaker_action(text: str) -> bool:
    """Return true for compact markers like ``[ХВВ подхватывает:]``."""
    body = text.rstrip(":").strip()
    return not re.search(r"[.!?…]", body) and len(body.split()) <= 3


def speaker_tag_from_editorial_note(text: str) -> str:
    """Extract explicit speaker initials from editorial notes like ``ДС говорит``."""
    code = SPEAKER_CODE_RE.pattern
    patterns = [
        rf"(?<!\w)({code})\s+говорит\b",
        rf"\bговорит(?:\s+\S+){{0,3}}\s+({code})(?!\w)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def speaker_tag_from_common_note(text: str) -> str:
    """Extract a speaker from common short speaker notes without chasing rare comments."""
    stripped = text.strip()
    if classify_bracket_text(stripped) != BracketKind.SPEAKER_NOTE:
        return ""
    if len(stripped.split()) > 8:
        return ""
    code = SPEAKER_CODE_RE.pattern
    match = re.search(
        rf"(?<!\w)({code})\s+(?:говорит|спрашивает|отвечает|подхватывает)\b",
        stripped,
    )
    if match:
        return match.group(1)
    match = re.search(r"\bговорит\b((?:\s+\S+){0,4})", stripped, flags=re.IGNORECASE)
    if match:
        tags = re.findall(code, match.group(1).rstrip(":"))
        if tags:
            return tags[-1]
    return ""


def speaker_tag_from_speaker_text(text: str) -> str:
    """Extract initials from a text that is expected to contain only speaker tags."""
    return speaker_tag_from_text(text)


def speaker_tag_from_marker(marker_text: str) -> str:
    """Extract speaker initials from a bracket marker body."""
    text = marker_text.strip()
    tag = speaker_tag_from_speaker_text(text.rstrip(":"))
    if tag:
        return tag
    if is_unknown_speaker_bracket(marker_text):
        return UNKNOWN_SPEAKER
    if "???" in text:
        return "???"
    tag = speaker_tag_from_common_note(text)
    if tag:
        return tag
    match = re.match(r"([A-ZА-ЯЁ]{1,6})(?=\s|,|$)", text) if is_short_speaker_action(text) else None
    if match:
        return match.group(1)
    return speaker_tag_from_editorial_note(text)


def speaker_note_tag_before_span(transcript: str, start: int) -> str:
    """Find a common speaker note immediately before a transcript span."""
    if start < 0:
        return ""
    search_start = max(0, start - 220)
    prefix = transcript[search_start:start]
    matches = list(BRACKET_RE.finditer(prefix))
    if not matches:
        return ""
    match = matches[-1]
    if prefix[match.end() :].strip():
        return ""
    return speaker_tag_from_common_note(match.group(1))


def speaker_note_tag_at_span_start(transcript: str, start: int) -> str:
    """Find a common speaker note at the beginning of an aligned transcript span."""
    if start < 0:
        return ""
    match = BRACKET_RE.match(transcript[start:])
    if not match:
        return ""
    tag = speaker_tag_from_common_note(match.group(1))
    if not tag:
        return ""
    return tag


def speaker_tag_from_line(line: str) -> str:
    """Extract speaker initials from a standalone transcript speaker line."""
    return speaker_tag_from_speaker_text(line)


def format_speaker_tag(tag: str) -> str:
    """Format transcript speaker initials as an SRT speaker prefix."""
    return f"[{tag}]:"


def format_transcript_speaker_marker(tag: str) -> str:
    """Format speaker initials as a transcript marker."""
    return f"[{tag}:]"


def find_speaker_tag(text: str) -> str:
    """Find the last valid bracketed speaker tag in a text span."""
    tags = [speaker_tag_from_marker(match.group(1)) for match in SPEAKER_MARKER_RE.finditer(text)]
    return next((tag for tag in reversed(tags) if tag), "")


def find_speaker_tag_before_span(transcript: str, start: int) -> str:
    """Find the closest valid speaker marker at or before a transcript span."""
    if start < 0:
        return ""
    search_start = max(0, start - 300)
    search_end = min(len(transcript), start + 300)
    candidates = []
    for match in SPEAKER_MARKER_RE.finditer(transcript[search_start:search_end]):
        absolute_start = search_start + match.start()
        absolute_end = search_start + match.end()
        if absolute_start <= start and (absolute_end <= start or absolute_start == start):
            tag = speaker_tag_from_marker(match.group(1))
            if tag:
                candidates.append(tag)
    return candidates[-1] if candidates else ""


def find_non_note_speaker_tag_before_span(transcript: str, start: int) -> str:
    """Find the closest speaker marker before a span, excluding weak note markers."""
    if start < 0:
        return ""
    search_start = max(0, start - 300)
    search_end = min(len(transcript), start + 300)
    candidates = []
    for match in SPEAKER_MARKER_RE.finditer(transcript[search_start:search_end]):
        absolute_start = search_start + match.start()
        absolute_end = search_start + match.end()
        if absolute_start <= start and (absolute_end <= start or absolute_start == start):
            marker = match.group(1)
            if speaker_tag_from_common_note(marker):
                continue
            tag = speaker_tag_from_marker(marker)
            if tag:
                candidates.append(tag)
    return candidates[-1] if candidates else ""


def leading_speaker_tag(transcript: str) -> str:
    """Return an initial transcript speaker marker when one is present."""
    match = SPEAKER_MARKER_RE.match(transcript.strip())
    if not match:
        return ""
    tag = speaker_tag_from_marker(match.group(1))
    return "" if tag == UNKNOWN_SPEAKER else tag


def unknown_speaker_tag_at_span(transcript: str, start: int, end: int) -> str:
    """Return [UNK] when an aligned span is contained by an unknown-speaker bracket."""
    if start < 0 or end < start:
        return ""
    marker = SPEAKER_MARKER_RE.match(transcript[start:end])
    if marker and speaker_tag_from_marker(marker.group(1)):
        start += marker.end()
        while start < end and transcript[start].isspace():
            start += 1
    search_start = max(0, start - 300)
    search_end = min(len(transcript), start + 300)
    for match in BRACKET_RE.finditer(transcript[search_start:search_end]):
        absolute_start = search_start + match.start()
        absolute_end = search_start + match.end()
        if absolute_start <= start and end <= absolute_end and is_unknown_speaker_bracket(match.group(1)):
            return UNKNOWN_SPEAKER
    return ""


def remove_speaker_markers(text: str) -> str:
    """Remove bracketed speaker markers while keeping other bracketed transcript notes."""

    def replace_marker(match: re.Match[str]) -> str:
        return "" if speaker_tag_from_marker(match.group(1)) else match.group(0)

    return re.sub(r"\s+", " ", SPEAKER_MARKER_RE.sub(replace_marker, text)).strip()


def remove_alignment_notes(text: str) -> str:
    """Remove bracketed editorial notes that should not drive alignment."""
    return strip_alignment_notes(text, speaker_tag_from_marker)


def speaker_blocks_from_transcript(transcript: str) -> list[SpeakerBlock]:
    """Find normal transcript blocks whose final line contains speaker initials."""
    blocks: list[SpeakerBlock] = []
    for match in re.finditer(r"\S.*?(?=\n\s*\n|\Z)", transcript, flags=re.DOTALL):
        block = match.group(0)
        lines = list(re.finditer(r"[^\n]+", block))
        if len(lines) < 2:
            continue
        footer = lines[-1]
        tag = speaker_tag_from_line(footer.group(0))
        if not tag:
            continue
        content_start_line = lines[3] if len(lines) >= 5 else lines[0]
        content_start = match.start() + content_start_line.start()
        content_end = match.start() + footer.start()
        if content_end > content_start:
            blocks.append(SpeakerBlock(content_start, content_end, tag))
    return blocks


def transcript_with_block_speaker_markers(transcript: str) -> str:
    """Convert block-final speaker initials into leading bracket markers for alignment."""
    parts: list[str] = []
    for match in re.finditer(r"\S.*?(?=\n\s*\n|\Z)", transcript, flags=re.DOTALL):
        block = match.group(0)
        lines = list(re.finditer(r"[^\n]+", block))
        if len(lines) < 2:
            parts.append(block.strip())
            continue
        footer = lines[-1]
        tag = speaker_tag_from_line(footer.group(0))
        if not tag:
            parts.append(block.strip())
            continue
        content_start = lines[3].start() if len(lines) >= 5 else lines[0].start()
        content = block[content_start : footer.start()].strip()
        if content:
            parts.append(f"{format_transcript_speaker_marker(tag)} {content}")
    return "\n\n".join(parts)


def speaker_tag_from_blocks(blocks: list[SpeakerBlock], start: int) -> str:
    """Return the block-level speaker tag for a transcript offset."""
    for block in blocks:
        if block.start <= start < block.end:
            return block.tag
    return ""


def align_segments(
    srt_segments: list[SrtSegment],
    transcript: str,
    *,
    max_span: int = 25,
    skip_penalty: float = 0.8,
    similarity_threshold: float = 0.3,
    allow_leading_transcript_skip: bool = False,
) -> list[AlignedSegment]:
    """Align SRT segments to contiguous manual transcript spans monotonically."""
    tokens = tokenize_transcript(transcript)
    n = len(srt_segments)
    m = len(tokens)
    srt_tokens = [normalize_for_match(segment.text).split() for segment in srt_segments]
    dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    prev: list[list[tuple[int, bool, float] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    if allow_leading_transcript_skip:
        for j in range(m + 1):
            dp[0][j] = 0.0
    else:
        dp[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(m + 1):
            if dp[i - 1][j] + skip_penalty < dp[i][j]:
                dp[i][j] = dp[i - 1][j] + skip_penalty
                prev[i][j] = (j, False, 0.0)
            for start in range(max(0, j - max_span), j):
                if dp[i - 1][start] == float("inf"):
                    continue
                span = [token.norm for token in tokens[start:j]]
                score = token_similarity(srt_tokens[i - 1], span)
                if score < similarity_threshold:
                    continue
                cost = dp[i - 1][start] + (1.0 - score)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    prev[i][j] = (start, True, score)
    end = min(range(m + 1), key=lambda col: dp[n][col])
    spans: list[tuple[int, int, bool, float]] = [(0, 0, False, 0.0)] * n
    i = n
    j = end
    while i > 0:
        cell = prev[i][j]
        if cell is None:
            start, matched, score = j, False, 0.0
        else:
            start, matched, score = cell
        spans[i - 1] = (start, j, matched, score)
        if matched:
            j = start
        i -= 1
    aligned: list[AlignedSegment] = []
    for segment, (start, end, matched, score) in zip(srt_segments, spans, strict=True):
        if matched and end > start:
            span_text = transcript[tokens[start].start : tokens[end - 1].end]
            normalized = normalize_for_match(span_text)
            span_start = tokens[start].start
            span_end = tokens[end - 1].end
        else:
            span_text = ""
            normalized = ""
            span_start = -1
            span_end = -1
        aligned.append(
            AlignedSegment(
                segment,
                segment.speaker,
                span_text,
                normalized,
                matched and bool(span_text),
                score,
                span_start,
                span_end,
            )
        )
    return aligned


def aligned_speaker_tag(
    item: AlignedSegment, transcript: str, speaker_blocks: list[SpeakerBlock]
) -> tuple[str, str]:
    """Infer the transcript speaker tag and source for one aligned segment."""
    if not item.matched or item.transcript_end < 0:
        return "", ""
    tag = unknown_speaker_tag_at_span(transcript, item.transcript_start, item.transcript_end)
    if tag == UNKNOWN_SPEAKER:
        return tag, "collector_bracket"
    tag = find_non_note_speaker_tag_before_span(transcript, item.transcript_start)
    if tag:
        return tag, "preceding_marker"
    tag = speaker_note_tag_at_span_start(transcript, item.transcript_start)
    if tag:
        return tag, "speaker_note"
    tag = speaker_note_tag_before_span(transcript, item.transcript_start)
    if tag:
        return tag, "speaker_note"
    tag = speaker_tag_from_blocks(speaker_blocks, item.transcript_start)
    if tag:
        return tag, "block_footer"
    return "", ""


def apply_transcript_speakers(
    aligned: list[AlignedSegment],
    transcript: str,
    *,
    infer_missing: bool = False,
    fallback_speaker: str = "",
) -> list[AlignedSegment]:
    """Update SRT speaker prefixes from bracketed speaker tags in the transcript.

    When ``infer_missing`` is true, a speaker tag found in one aligned span is
    carried forward until another explicit bracket tag appears.
    """
    output: list[AlignedSegment] = []
    speaker_blocks = speaker_blocks_from_transcript(transcript)
    last_tag = leading_speaker_tag(transcript) if infer_missing else ""
    for item in aligned:
        if not item.matched:
            output.append(item)
            continue
        tag, source = aligned_speaker_tag(item, transcript, speaker_blocks)
        if tag:
            if tag != UNKNOWN_SPEAKER:
                last_tag = tag
        elif infer_missing:
            tag = last_tag
            source = "carried_forward_prev" if tag else ""
        if not tag and fallback_speaker and item.matched:
            tag = fallback_speaker
            source = "speaker_hint"
        if tag:
            item = replace(
                item,
                srt=SrtSegment(
                    item.srt.index,
                    item.srt.start,
                    item.srt.end,
                    format_speaker_tag(tag),
                    item.srt.text,
                ),
                transcript_text=remove_speaker_markers(item.transcript_text),
                transcript_speaker=format_speaker_tag(tag),
                speaker_source=source,
            )
        output.append(item)
    return output


def aligned_to_srt(aligned: list[AlignedSegment], *, fallback_to_srt: bool = True) -> str:
    """Serialize aligned rows as SRT, preserving timing and speaker tags."""
    segments = []
    for item in aligned:
        text = item.transcript_text if item.matched else (item.srt.text if fallback_to_srt else "")
        segments.append(SrtSegment(item.srt.index, item.srt.start, item.srt.end, item.srt.speaker, text))
    return format_srt(segments)


def aligned_to_rows(index_name: str, aligned: list[AlignedSegment]) -> list[dict[str, object]]:
    """Convert aligned segments to canonical aligned TSV rows."""
    return [
        {
            "index_name": index_name,
            "srt_index": item.srt.index,
            "start": item.srt.start,
            "end": item.srt.end,
            "speaker": item.srt.speaker,
            "srt_text": item.srt.text,
            "transcript_text": item.transcript_text,
            "normalized_text": item.normalized_text,
            "matched": item.matched,
            "score": f"{item.score:.3f}",
        }
        for item in aligned
    ]


def aligned_to_speaker_map_rows(aligned: list[AlignedSegment]) -> list[dict[str, object]]:
    """Convert aligned segments to generic speaker-map rows."""
    return [
        {
            "srt_index": item.srt.index,
            "start": item.srt.start,
            "end": item.srt.end,
            "whisperx_speaker": item.original_speaker,
            "transcript_speaker": item.transcript_speaker,
            "speaker_source": item.speaker_source,
            "matched": item.matched,
            "score": f"{item.score:.3f}",
        }
        for item in aligned
    ]


def align_srt_file(
    srt_path: Path | str,
    transcript_text: str,
    output_srt: Path | str | None = None,
    *,
    use_transcript_speakers: bool = False,
    infer_missing_speakers: bool = False,
    fallback_speaker: str = "",
    allow_leading_transcript_skip: bool = False,
) -> list[AlignedSegment]:
    """Align one SRT file to transcript text and optionally write merged SRT."""
    alignment_transcript = (
        transcript_with_block_speaker_markers(transcript_text) if use_transcript_speakers else transcript_text
    )
    alignment_transcript = remove_alignment_notes(alignment_transcript)
    aligned = align_segments(
        parse_srt(Path(srt_path).read_text(encoding="utf-8-sig")),
        alignment_transcript,
        allow_leading_transcript_skip=allow_leading_transcript_skip,
    )
    if use_transcript_speakers:
        aligned = apply_transcript_speakers(
            aligned,
            alignment_transcript,
            infer_missing=infer_missing_speakers,
            fallback_speaker=fallback_speaker,
        )
    if output_srt is not None:
        Path(output_srt).parent.mkdir(parents=True, exist_ok=True)
        Path(output_srt).write_text(aligned_to_srt(aligned), encoding="utf-8")
    return aligned


def write_aligned_tsv(index_name: str, aligned: list[AlignedSegment], output_path: Path | str) -> None:
    """Write aligned segments to canonical TSV."""
    write_tsv(output_path, aligned_to_rows(index_name, aligned), ALIGNED_COLUMNS)


def write_speaker_map(aligned: list[AlignedSegment], output_path: Path | str) -> None:
    """Write generic alignment-time speaker provenance rows."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SPEAKER_MAP_COLUMNS)
        writer.writeheader()
        writer.writerows(aligned_to_speaker_map_rows(aligned))
