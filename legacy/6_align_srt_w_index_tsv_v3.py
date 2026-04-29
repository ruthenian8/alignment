"""
This module implements a redesigned alignment algorithm for merging SRT
transcriptions (``t1``) with more detailed transcripts (``t2``). It is based
on the design described in the accompanying plan:

* Robust parsing of SRT segments, keeping timestamps and speaker IDs.
* Tokenisation and normalisation of the free‑form transcript while
  preserving original text and bracketed asides.
* A monotonic dynamic programming (DP) alignment between SRT segments
  and contiguous spans of transcript tokens. The DP ensures that
  assignments do not cross and can skip segments or tokens with
  configurable penalties.
* A reconstruction phase that preserves SRT timestamps and inserts
  corresponding transcript spans, falling back to SRT text when no good
  match is found.

The public API exposes a single function :func:`merge_srt_and_transcript`
which takes the raw SRT and transcript text and returns a merged SRT
string. See the ``__main__`` block for a simple command line interface
and test harness.

The alignment algorithm uses a similarity measure based on token
overlap (F1 score). Various parameters (minimum and maximum span
lengths, skip penalties and similarity thresholds) are exposed to allow
empirical tuning. A small grid search can be performed by passing
multiple parameter sets to the ``evaluate_params`` helper.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Dict, Any


@dataclass
class SrtSegment:
    """Representation of a single SRT segment.

    Attributes
    ----------
    index: int
        The numeric index of the segment.
    start: str
        Start timestamp (hh:mm:ss,ms) as found in the SRT.
    end: str
        End timestamp (hh:mm:ss,ms) as found in the SRT.
    speaker: str
        Speaker tag (e.g. ``[SPEAKER_01]``) or empty string.
    text: str
        The spoken text in this segment (without timestamps or index).
    """
    index: int
    start: str
    end: str
    speaker: str
    text: str

    def header(self) -> str:
        """Return the SRT header line: index, timestamp and speaker."""
        parts = [str(self.index), f"{self.start} --> {self.end}"]
        if self.speaker:
            parts.append(self.speaker)
        return "\n".join(parts)

    def to_srt_block(self, content: str) -> str:
        """Construct a complete SRT block replacing the text with `content`."""
        return f"{self.index}\n{self.start} --> {self.end}\n{self.speaker}{content}\n"


def parse_srt(srt_text: str) -> List[SrtSegment]:
    """Parse a raw SRT string into a list of :class:`SrtSegment`.

    The SRT format expected is similar to the provided example: each
    block begins with a numeric index, followed by a timestamp line,
    then one or more lines of text. Speaker tags (e.g. ``[SPEAKER_00]:``)
    are treated as part of the first text line. Segments are separated
    by one or more blank lines.
    """
    segments: List[SrtSegment] = []
    # split by double newlines to isolate blocks
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        # first line is index
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue  # skip malformed
        # second line contains timestamps
        ts = lines[1].strip()
        # handle possible trailing speaker on this line (rare); prefer pattern HH:MM:SS,ms --> HH:MM:SS,ms
        m = re.match(r"(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)", ts)
        if not m:
            continue
        start, end = m.group(1), m.group(2)
        # accumulate remaining lines as text
        text_lines = lines[2:]
        # If the first line has a speaker tag at beginning, separate it
        speaker = ""
        if text_lines:
            first = text_lines[0]
            # speaker tag pattern [SPEAKER_00]: or similar
            sm = re.match(r"(\[.*?\]:)\s*(.*)", first)
            if sm:
                speaker = sm.group(1)
                text_lines[0] = sm.group(2)
        text = "\n".join(text_lines)
        segments.append(SrtSegment(index=index, start=start, end=end, speaker=speaker, text=text))
    return segments


@dataclass
class T2Token:
    """A single token from the transcript, with normalised form and original text and offsets."""
    text: str  # original token text
    norm: str  # normalised token used for matching
    start: int  # start character index in original transcript
    end: int  # end character index (exclusive)


def normalize_text_for_match(text: str) -> str:
    """Normalise text for matching.

    Lowercases, removes diacritics, strips common punctuation and stress
    markers, converts ё→е and reduces multiple whitespace. Filler words
    can be filtered at a later stage if desired.
    """
    # Replace stress markers and escaped characters
    text = text.replace("\\", "")  # remove backslashes used for stress
    # Replace ё with е
    text = text.replace("ё", "е").replace("Ё", "Е")
    # Lowercase and remove accents/diacritics (fold accents)
    text = unicodedata.normalize('NFD', text).lower()
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    # Remove punctuation except hyphens and apostrophes which may be meaningful
    text = re.sub(r"[\.,!?…—–:;\(\)\[\]\"'«»]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_transcript(t2_text: str) -> List[T2Token]:
    """Tokenise the transcript into :class:`T2Token` objects.

    This simple tokenizer splits on whitespace but preserves the original
    offsets of each token. Bracketed content (e.g. ``[А люди отличаются?]``)
    is kept as part of the surrounding token stream; the normalised form
    of a bracketed token is computed on its content only, so that
    bracketed asides still participate in alignment.
    """
    tokens: List[T2Token] = []
    # iterate through whitespace separated tokens, capturing offsets
    pos = 0
    for m in re.finditer(r"\S+", t2_text):
        raw = m.group()
        start, end = m.start(), m.end()
        # Remove surrounding brackets for normalisation but leave raw as is
        raw_inner = raw
        if raw.startswith("[") and raw.endswith("]"):
            raw_inner = raw[1:-1]
        norm = normalize_text_for_match(raw_inner)
        tokens.append(T2Token(text=raw, norm=norm, start=start, end=end))
    return tokens


def normalise_segment_text(text: str) -> List[str]:
    """Normalise an SRT segment's text into a list of tokens for matching."""
    norm = normalize_text_for_match(text)
    if not norm:
        return []
    return norm.split()


def compute_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute similarity between two token lists based on the F1 score.

    The similarity is defined as 2*|A ∩ B| / (|A| + |B|). If either list
    is empty, similarity is zero. The cost used in DP will be 1 - sim.
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
    """Align SRT segments to contiguous spans of transcript tokens using DP.

    Parameters
    ----------
    srt_segments : list of SrtSegment
        The list of SRT segments.
    t2_tokens : list of T2Token
        The tokenised transcript.
    min_span, max_span : int
        Minimum and maximum length (in tokens) of a matched span. These
        heuristics prevent pathological spans that are too short or too long.
    skip_penalty : float
        Penalty added when skipping an SRT segment (i.e. not matching it
        to any transcript span). Should be non‑negative. A higher value
        discourages skipping and encourages matching.
    similarity_threshold : float
        Minimum similarity for a span to be considered a valid match.
        Spans below this threshold will incur the skip penalty rather
        than being matched.
    length_penalty : float
        Penalty per token of span length; longer spans pay more.

    Returns
    -------
    boundaries : list of int
        A list of boundaries ``b[0], b[1], ..., b[n]`` where ``b[i]`` is
        the token index in ``t2_tokens`` at which the ``i``‑th SRT segment
        ends. The first boundary is 0 and the last is len(t2_tokens).
    costs : list of float
        DP cost values associated with each segment.
    """
    n = len(srt_segments)
    m = len(t2_tokens)
    # Precompute normalised tokens for each SRT segment
    srt_norms = [normalise_segment_text(seg.text) for seg in srt_segments]

    # DP table: dp[i][j] = minimal cost aligning first i segments to first j tokens
    # We use float('inf') to denote impossible states.
    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    prev: List[List[Optional[Tuple[int, int, float, bool]]]] = [
        [None] * (m + 1) for _ in range(n + 1)
    ]
    # base case: 0 segments aligned to 0 tokens cost zero
    dp[0][0] = 0.0
    prev[0][0] = (0, 0, 0.0, True)
    # Fill DP
    for i in range(1, n + 1):
        for j in range(0, m + 1):
            # Option 1: skip this SRT segment (do not consume any transcript tokens)
            # Only possible if previous state exists at (i-1, j)
            if dp[i - 1][j] != float('inf'):
                cost = dp[i - 1][j] + skip_penalty
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    prev[i][j] = (j, j, cost, False)  # False indicates skipped
            # Option 2: match this SRT segment to a span ending at j
            # Consider spans of length L = j - k within [min_span, max_span]
            # and k >= 0
            for span_len in range(min_span, max_span + 1):
                k = j - span_len
                if k < 0:
                    break
                if dp[i - 1][k] == float('inf'):
                    continue
                span_tokens = [t2_tokens[t].norm for t in range(k, j)]
                sim = compute_similarity(srt_norms[i - 1], span_tokens)
                # If similarity below threshold, treat as skip (apply skip_penalty)
                if sim < similarity_threshold:
                    continue
                # cost is previous cost minus similarity (higher sim -> lower cost) plus length penalty
                cost = dp[i - 1][k] + (1.0 - sim) + length_penalty * span_len
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    prev[i][j] = (k, j, cost, True)
    # Now select final state with minimal cost at (n, j) where j ranges 0..m
    # We favour states that consume all tokens (j=m) but we allow partial consumption
    best_j = 0
    best_cost = float('inf')
    for j in range(0, m + 1):
        if dp[n][j] < best_cost:
            best_cost = dp[n][j]
            best_j = j
    # Reconstruct boundaries
    boundaries = [0] * (n + 1)
    costs = [0.0] * n
    i, j = n, best_j
    boundaries[n] = j
    while i > 0:
        cell = prev[i][j]
        if cell is None:
            # Should not happen
            k = j
            matched = False
        else:
            k, j_end, cost, matched = cell
            k = k  # starting index of span
        boundaries[i - 1] = k
        costs[i - 1] = dp[i][j]
        # For skip, j remains the same; for match, set j = k
        if cell is not None:
            if matched:
                j = k
            else:
                # skip: j unchanged
                pass
        i -= 1
    return boundaries, costs


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
    """Merge an SRT transcript with a more detailed transcript.

    Parameters
    ----------
    srt_text : str
        Input SRT containing timestamps, speaker tags and Whisper text.
    t2_text : str
        A more detailed transcript containing stress marks, bracketed asides,
        hesitations, etc.
    min_span, max_span, skip_penalty, similarity_threshold, length_penalty
        Parameters controlling the DP alignment. See :func:`align_segments`.

    Returns
    -------
    merged_srt : str
        An SRT string where each original SRT segment's text is replaced by
        the best matching span from the transcript, using original
        timestamps. If no suitable span is found, the original segment text
        is preserved.
    """
    srt_segments = parse_srt(srt_text)
    t2_tokens = tokenize_transcript(t2_text)
    if not srt_segments:
        return srt_text
    boundaries, costs = align_segments(
        srt_segments,
        t2_tokens,
        min_span=min_span,
        max_span=max_span,
        skip_penalty=skip_penalty,
        similarity_threshold=similarity_threshold,
        length_penalty=length_penalty,
    )
    # Build merged SRT
    merged_blocks: List[str] = []
    for i, seg in enumerate(srt_segments):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        # Determine if this segment was matched
        matched = end_idx > start_idx
        # If matched, reconstruct text from t2 tokens preserving raw text
        if matched:
            # Concatenate raw token texts with spaces. We preserve original spacing by
            # taking substrings from t2_text rather than joining with fixed spaces.
            # Use char offsets to extract the substring from the original transcript.
            span_start = t2_tokens[start_idx].start
            span_end = t2_tokens[end_idx - 1].end
            new_text = t2_text[span_start:span_end]
            # Prepend a space if original text had leading newline(s)
            content = new_text
        else:
            # No match: keep original text
            content = seg.text
        # The speaker tag and colon should already be in seg.speaker; ensure spacing
        # Insert a space after the speaker tag if necessary
        speaker_prefix = seg.speaker
        if speaker_prefix and not speaker_prefix.endswith(' '):
            speaker_prefix += ' '
        merged_blocks.append(
            f"{seg.index}\n{seg.start} --> {seg.end}\n{speaker_prefix}{content}\n"
        )
    return "\n".join(merged_blocks)


def evaluate_params(
    srt_text: str,
    t2_text: str,
    param_grid: Iterable[Dict[str, Any]],
    metric: str = "unmatched",
) -> Tuple[Dict[str, Any], int]:
    """Evaluate multiple parameter sets and return the best according to a metric.

    Parameters
    ----------
    srt_text, t2_text : str
        Input texts as in :func:`merge_srt_and_transcript`.
    param_grid : iterable of dict
        Each dict should contain parameters supported by
        :func:`merge_srt_and_transcript` (min_span, max_span, skip_penalty,
        similarity_threshold, length_penalty).
    metric : str
        Metric to minimise. Currently supports ``"unmatched"`` which counts
        how many segments did not get matched.

    Returns
    -------
    best_params : dict
        The parameter set that yields the minimal metric.
    best_metric : int
        The value of the metric for the best parameter set.
    """
    best_params: Dict[str, Any] = {}
    best_score: Optional[int] = None
    for params in param_grid:
        # Perform merge and also collect alignment boundaries
        srt_segments = parse_srt(srt_text)
        t2_tokens = tokenize_transcript(t2_text)
        if not srt_segments:
            continue
        boundaries, _ = align_segments(
            srt_segments,
            t2_tokens,
            min_span=params.get('min_span', 1),
            max_span=params.get('max_span', 25),
            skip_penalty=params.get('skip_penalty', 0.8),
            similarity_threshold=params.get('similarity_threshold', 0.3),
            length_penalty=params.get('length_penalty', 0.02),
        )
        # Count unmatched segments as those with zero-length span
        unmatched = sum(1 for i in range(len(srt_segments)) if boundaries[i + 1] == boundaries[i])
        score = unmatched
        if best_score is None or score < best_score:
            best_score = score
            best_params = params
    return best_params, best_score if best_score is not None else 0


if __name__ == "__main__":
    # Example usage with provided t1 and t2
    import argparse
    parser = argparse.ArgumentParser(description="Merge SRT and transcript using DP alignment")
    parser.add_argument("--srt", required=False, help="Path to SRT file or raw string")
    parser.add_argument("--transcript", required=False, help="Path to transcript file or raw string")
    parser.add_argument("--grid", action="store_true", help="Run grid search for parameters")
    args = parser.parse_args()
    if not args.srt or not args.transcript:
        # Use hardcoded example if no paths provided
        example_srt = """1\n00:00:00,571 --> 00:00:03,074\n[SPEAKER_01]: Гораздо мягче.\n\n2\n00:00:03,154 --> 00:00:03,996\n[SPEAKER_00]: А люди?\n"""
        example_srt += """\n3\n00:00:04,116 --> 00:00:06,779\n[SPEAKER_01]: Люди?\n\n4\n00:00:06,819 --> 00:00:12,587\n[SPEAKER_01]: Люди другие, обычаи другие.\n"""
        example_t2 = "[А люди отличаются?] Лю\\ди други\\е. Обы\\чаи други\\е."
        srt_text = example_srt
        t2_text = example_t2
    else:
        # load from files
        with open(args.srt, 'r', encoding='utf-8') as f:
            srt_text = f.read()
        with open(args.transcript, 'r', encoding='utf-8') as f:
            t2_text = f.read()
    if args.grid:
        grid = [
            {'min_span': 1, 'max_span': 15, 'skip_penalty': sp, 'similarity_threshold': st, 'length_penalty': lp}
            for sp in [0.5, 0.8, 1.0]
            for st in [0.2, 0.3, 0.4]
            for lp in [0.01, 0.02, 0.05]
        ]
        best_params, score = evaluate_params(srt_text, t2_text, grid)
        print("Best params:", best_params)
        print("Unmatched count:", score)
        merged = merge_srt_and_transcript(srt_text, t2_text, **best_params)
    else:
        merged = merge_srt_and_transcript(srt_text, t2_text, min_span=1, max_span=20, skip_penalty=0.8, similarity_threshold=0.3, length_penalty=0.02)
    print(merged)