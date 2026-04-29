"""Monotonic DP alignment between WhisperX SRT segments and manual transcript text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .io import ALIGNED_COLUMNS, write_tsv
from .reorder import normalize_for_match
from .srt import SrtSegment, format_srt, parse_srt


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
    transcript_text: str
    normalized_text: str
    matched: bool
    score: float


def tokenize_transcript(text: str) -> list[TranscriptToken]:
    """Tokenize transcript text while preserving original character offsets."""
    tokens: list[TranscriptToken] = []
    for match in re.finditer(r"\S+", text):
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


def align_segments(
    srt_segments: list[SrtSegment],
    transcript: str,
    *,
    max_span: int = 25,
    skip_penalty: float = 0.8,
    similarity_threshold: float = 0.3,
) -> list[AlignedSegment]:
    """Align SRT segments to contiguous manual transcript spans monotonically."""
    tokens = tokenize_transcript(transcript)
    n = len(srt_segments)
    m = len(tokens)
    srt_tokens = [normalize_for_match(segment.text).split() for segment in srt_segments]
    dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    prev: list[list[tuple[int, bool, float] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
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
        else:
            span_text = ""
            normalized = ""
        aligned.append(AlignedSegment(segment, span_text, normalized, matched and bool(span_text), score))
    return aligned


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


def align_srt_file(
    srt_path: Path | str, transcript_text: str, output_srt: Path | str | None = None
) -> list[AlignedSegment]:
    """Align one SRT file to transcript text and optionally write merged SRT."""
    aligned = align_segments(parse_srt(Path(srt_path).read_text(encoding="utf-8-sig")), transcript_text)
    if output_srt is not None:
        Path(output_srt).parent.mkdir(parents=True, exist_ok=True)
        Path(output_srt).write_text(aligned_to_srt(aligned), encoding="utf-8")
    return aligned


def write_aligned_tsv(index_name: str, aligned: list[AlignedSegment], output_path: Path | str) -> None:
    """Write aligned segments to canonical TSV."""
    write_tsv(output_path, aligned_to_rows(index_name, aligned), ALIGNED_COLUMNS)
