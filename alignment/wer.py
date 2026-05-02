"""WER utilities for aligned TSV outputs."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .io import parse_bool, read_tsv
from .reorder import normalize_for_match


@dataclass(frozen=True)
class WerStats:
    """Global WER counts computed over aligned row pairs."""

    rows: int
    reference_words: int
    substitutions: int
    deletions: int
    insertions: int
    wer: float


BRACKET_CONTENT_RE = re.compile(r"\[[^\]]*\]")


def normalize_for_wer(text: str) -> str:
    """Normalize text for WER, dropping bracketed notes and speaker tags."""
    text = BRACKET_CONTENT_RE.sub(" ", text)
    text = re.sub(r"^[^\[]*\]\s*", " ", text)
    text = re.sub(r"\[[^\[]*$", " ", text)
    return normalize_for_match(text)


def aligned_tsv_rows_for_wer(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return aligned TSV rows that should contribute to WER."""
    output = []
    for row in rows:
        try:
            score = float(row.get("score", "0") or 0)
        except ValueError:
            score = 0.0
        if score > 0 and parse_bool(row.get("matched")):
            output.append(row)
    return output


def edit_operations(reference: list[str], hypothesis: list[str]) -> list[tuple[str, str, str]]:
    """Return edit operations between reference and hypothesis word tokens."""
    n = len(reference)
    m = len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "insert"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                choices = [(dp[i - 1][j - 1], "equal")]
            else:
                choices = [(dp[i - 1][j - 1] + 1, "substitute")]
            choices.extend(
                [
                    (dp[i - 1][j] + 1, "delete"),
                    (dp[i][j - 1] + 1, "insert"),
                ]
            )
            dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])

    operations: list[tuple[str, str, str]] = []
    i = n
    j = m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "equal":
            operations.append((op, reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif op == "substitute":
            operations.append((op, reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif op == "delete":
            operations.append((op, reference[i - 1], ""))
            i -= 1
        else:
            operations.append(("insert", "", hypothesis[j - 1]))
            j -= 1
    operations.reverse()
    return operations


def compute_wer(rows: list[dict[str, str]]) -> tuple[WerStats, Counter[tuple[str, str, str]]]:
    """Compute global WER and mismatch counts from aligned TSV rows."""
    filtered = aligned_tsv_rows_for_wer(rows)
    reference_words: list[str] = []
    hypothesis_words: list[str] = []
    for row in filtered:
        reference_words.extend(normalize_for_wer(row.get("transcript_text", "")).split())
        hypothesis_words.extend(normalize_for_wer(row.get("srt_text", "")).split())

    substitutions = deletions = insertions = 0
    mismatches: Counter[tuple[str, str, str]] = Counter()
    for op, reference, hypothesis in edit_operations(reference_words, hypothesis_words):
        if op == "substitute":
            substitutions += 1
            mismatches[(op, reference, hypothesis)] += 1
        elif op == "delete":
            deletions += 1
            mismatches[(op, reference, "<del>")] += 1
        elif op == "insert":
            insertions += 1
            mismatches[(op, "<ins>", hypothesis)] += 1

    errors = substitutions + deletions + insertions
    total = len(reference_words)
    stats = WerStats(
        rows=len(filtered),
        reference_words=total,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        wer=0.0 if total == 0 else errors / total,
    )
    return stats, mismatches


def compute_wer_from_tsv(path: Path | str) -> tuple[WerStats, Counter[tuple[str, str, str]]]:
    """Compute WER from a post-alignment TSV file."""
    return compute_wer(read_tsv(path))


def format_wer_report(stats: WerStats, mismatches: Counter[tuple[str, str, str]], *, top: int = 20) -> str:
    """Format global WER stats and the most common mismatch operations."""
    lines = [
        f"rows\t{stats.rows}",
        f"reference_words\t{stats.reference_words}",
        f"wer\t{stats.wer:.4f}",
        f"substitutions\t{stats.substitutions}",
        f"deletions\t{stats.deletions}",
        f"insertions\t{stats.insertions}",
        "",
        "count\ttype\treference\thypothesis",
    ]
    for (op, reference, hypothesis), count in mismatches.most_common(top):
        lines.append(f"{count}\t{op}\t{reference}\t{hypothesis}")
    return "\n".join(lines)
