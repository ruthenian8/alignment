"""DP-based alignment of SRT segments to manual transcripts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

from alignment.srt import SrtSegment, parse_srt


@dataclass
class T2Token:
    """A single token from the transcript with normalised form and offsets.

    Attributes:
        text: Original token text.
        norm: Normalised token used for matching.
        start: Start character index in original transcript.
        end: End character index (exclusive).
    """

    text: str
    norm: str
    start: int
    end: int


@dataclass
class AlignedSegment:
    """An SRT segment with aligned manual transcript text.

    Attributes:
        index: SRT segment numeric index.
        start: Start timestamp.
        end: End timestamp.
        speaker: Speaker tag (or empty string).
        original_text: Original SRT (WhisperX) text.
        transcript_text: Aligned manual transcript span.
        matched: Whether this segment was successfully aligned.
    """

    index: int
    start: str
    end: str
    speaker: str
    original_text: str
    transcript_text: str
    matched: bool


def normalize_text_for_match(text: str) -> str:
    """Normalise text for token matching.

    Lowercases, removes diacritics, strips punctuation and stress markers,
    converts ё→е, and collapses whitespace.

    Args:
        text: Input text.

    Returns:
        Normalised string suitable for matching.
    """
    text = text.replace("\\", "")
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = unicodedata.normalize("NFD", text).lower()
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[\.,!?…—–:;\(\)\[\]\"'«»]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_transcript(t2_text: str) -> List[T2Token]:
    """Tokenise transcript text into :class:`T2Token` objects.

    Splits on whitespace while preserving original character offsets.
    Bracketed content is kept in the token stream; normalised form is
    computed on the inner content only.

    Args:
        t2_text: Raw transcript text.

    Returns:
        List of T2Token objects in order.
    """
    tokens: List[T2Token] = []
    for m in re.finditer(r"\S+", t2_text):
        raw = m.group()
        start, end = m.start(), m.end()
        raw_inner = raw
        if raw.startswith("[") and raw.endswith("]"):
            raw_inner = raw[1:-1]
        norm = normalize_text_for_match(raw_inner)
        tokens.append(T2Token(text=raw, norm=norm, start=start, end=end))
    return tokens


def _normalise_segment_text(text: str) -> List[str]:
    """Normalise SRT segment text into a list of tokens."""
    norm = normalize_text_for_match(text)
    if not norm:
        return []
    return norm.split()


def compute_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute F1-score similarity between two token lists.

    Defined as 2|A ∩ B| / (|A| + |B|). Returns 0 if either list is empty.

    Args:
        tokens_a: First token list.
        tokens_b: Second token list.

    Returns:
        Similarity score in [0, 1].
    """
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    common = len(set_a & set_b)
    if common == 0:
        return 0.0
    return (2.0 * common) / (len(set_a) + len(set_b))


def align_segments(
    srt_segments: List[SrtSegment],
    t2_tokens: List[T2Token],
    min_span: int = 1,
    max_span: int = 25,
    skip_penalty: float = 0.8,
    similarity_threshold: float = 0.3,
    length_penalty: float = 0.02,
) -> Tuple[List[int], List[float]]:
    """Align SRT segments to transcript token spans using dynamic programming.

    Enforces monotonic alignment. Each SRT segment is either matched to a
    contiguous span of transcript tokens or skipped (incurring the skip
    penalty). The DP minimises total cost.

    Args:
        srt_segments: List of SRT segments to align.
        t2_tokens: Tokenised transcript.
        min_span: Minimum span length in tokens.
        max_span: Maximum span length in tokens.
        skip_penalty: Cost of skipping an SRT segment.
        similarity_threshold: Minimum similarity to accept a match.
        length_penalty: Per-token penalty for longer spans.

    Returns:
        Tuple of (boundaries, costs). boundaries[i] is the token index
        where segment i ends; costs[i] is the DP cost at segment i.
    """
    n = len(srt_segments)
    m = len(t2_tokens)
    srt_norms = [_normalise_segment_text(seg.text) for seg in srt_segments]

    dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    prev: List[List[Optional[Tuple[int, int, float, bool]]]] = [
        [None] * (m + 1) for _ in range(n + 1)
    ]
    dp[0][0] = 0.0
    prev[0][0] = (0, 0, 0.0, True)

    for i in range(1, n + 1):
        for j in range(0, m + 1):
            if dp[i - 1][j] != float("inf"):
                cost = dp[i - 1][j] + skip_penalty
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    prev[i][j] = (j, j, cost, False)
            for span_len in range(min_span, max_span + 1):
                k = j - span_len
                if k < 0:
                    break
                if dp[i - 1][k] == float("inf"):
                    continue
                span_tokens = [t2_tokens[t].norm for t in range(k, j)]
                sim = compute_similarity(srt_norms[i - 1], span_tokens)
                if sim < similarity_threshold:
                    continue
                cost = dp[i - 1][k] + (1.0 - sim) + length_penalty * span_len
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    prev[i][j] = (k, j, cost, True)

    best_j = 0
    best_cost = float("inf")
    for j in range(0, m + 1):
        if dp[n][j] < best_cost:
            best_cost = dp[n][j]
            best_j = j

    boundaries = [0] * (n + 1)
    costs = [0.0] * n
    i, j = n, best_j
    boundaries[n] = j
    while i > 0:
        cell = prev[i][j]
        if cell is None:
            k = j
            matched = False
        else:
            k, j_end, cost, matched = cell
        boundaries[i - 1] = k
        costs[i - 1] = dp[i][j]
        if cell is not None:
            if matched:
                j = k
        i -= 1
    return boundaries, costs


def align_srt_to_transcript(
    srt_text: str,
    transcript_text: str,
    *,
    min_span: int = 1,
    max_span: int = 25,
    skip_penalty: float = 0.8,
    similarity_threshold: float = 0.3,
    length_penalty: float = 0.02,
) -> List[AlignedSegment]:
    """Align an SRT file to a manual transcript using DP.

    Parses the SRT, tokenises the transcript, runs the DP alignment, and
    returns structured :class:`AlignedSegment` objects preserving original
    transcript text in aligned spans.

    Args:
        srt_text: Raw SRT content.
        transcript_text: Manual transcript text.
        min_span: Minimum token span length.
        max_span: Maximum token span length.
        skip_penalty: Cost of skipping an SRT segment.
        similarity_threshold: Minimum similarity to accept a match.
        length_penalty: Per-token penalty for longer spans.

    Returns:
        List of AlignedSegment objects, one per SRT segment.
    """
    srt_segments = parse_srt(srt_text)
    t2_tokens = tokenize_transcript(transcript_text)
    if not srt_segments:
        return []

    boundaries, costs = align_segments(
        srt_segments,
        t2_tokens,
        min_span=min_span,
        max_span=max_span,
        skip_penalty=skip_penalty,
        similarity_threshold=similarity_threshold,
        length_penalty=length_penalty,
    )

    result: List[AlignedSegment] = []
    for i, seg in enumerate(srt_segments):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        matched = end_idx > start_idx
        if matched:
            span_start = t2_tokens[start_idx].start
            span_end = t2_tokens[end_idx - 1].end
            transcript_span = transcript_text[span_start:span_end]
        else:
            transcript_span = ""
        result.append(
            AlignedSegment(
                index=seg.index,
                start=seg.start,
                end=seg.end,
                speaker=seg.speaker,
                original_text=seg.text,
                transcript_text=transcript_span,
                matched=matched,
            )
        )
    return result


def merge_srt_and_transcript(
    srt_text: str,
    t2_text: str,
    *,
    min_span: int = 1,
    max_span: int = 25,
    skip_penalty: float = 0.8,
    similarity_threshold: float = 0.3,
    length_penalty: float = 0.02,
) -> str:
    """Merge an SRT with a detailed manual transcript, returning SRT string.

    A convenience wrapper around :func:`align_srt_to_transcript` that
    reconstructs an SRT string with transcript text substituted in for
    each matched segment.

    Args:
        srt_text: Input SRT content.
        t2_text: Manual transcript text.
        min_span: Minimum token span length.
        max_span: Maximum token span length.
        skip_penalty: Cost of skipping an SRT segment.
        similarity_threshold: Minimum similarity to accept a match.
        length_penalty: Per-token penalty for longer spans.

    Returns:
        SRT string with transcript text inserted where matched.
    """
    aligned = align_srt_to_transcript(
        srt_text,
        t2_text,
        min_span=min_span,
        max_span=max_span,
        skip_penalty=skip_penalty,
        similarity_threshold=similarity_threshold,
        length_penalty=length_penalty,
    )
    blocks: List[str] = []
    for seg in aligned:
        content = seg.transcript_text if seg.matched else seg.original_text
        speaker_prefix = (seg.speaker + " ") if seg.speaker else ""
        blocks.append(f"{seg.index}\n{seg.start} --> {seg.end}\n{speaker_prefix}{content}")
    return "\n\n".join(blocks) + "\n"
