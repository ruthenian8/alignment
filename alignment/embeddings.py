"""Optional embedding-based helpers for dialect transcript alignment."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .io import EMBEDDED_ALIGNMENT_COLUMNS, write_tsv
from .reorder import normalize_for_match
from .srt import parse_srt


class EmbeddingModel(Protocol):
    """Minimal interface expected from sentence embedding models."""

    def encode(self, sentences: list[str], **kwargs: object) -> list[object]:
        """Encode sentences into numeric vectors."""


@dataclass(frozen=True)
class ManualSegment:
    """A manual transcript segment marked as dialect or interviewer text."""

    text: str
    is_dialect: bool


@dataclass(frozen=True)
class DialectExtraction:
    """Dialect-only text extracted from SRT and manual transcript sources."""

    whisper_text: str
    manual_text: str
    stats: dict[str, int]


@dataclass(frozen=True)
class EmbeddedAlignedPair:
    """One embedding-aligned pair preserving original and cleaned text."""

    whisper_text: str
    manual_text: str
    whisper_clean: str
    manual_clean: str
    score: float


def parse_manual_annotation(text: str) -> list[ManualSegment]:
    """Split manual transcript into dialect text and bracketed interviewer prompts."""
    segments: list[ManualSegment] = []
    for part in re.split(r"(\[.*?\])", text):
        part = part.strip()
        if not part:
            continue
        if part.startswith("[") and part.endswith("]"):
            content = part[1:-1].strip()
            if content and not content.startswith(("...", "…", "Соб.")):
                segments.append(ManualSegment(content, is_dialect=False))
        else:
            segments.append(ManualSegment(part, is_dialect=True))
    return segments


def clean_text_for_embedding(text: str) -> str:
    """Clean stress and spacing markers while keeping text readable for embeddings."""
    text = re.sub(r"\\", "", text)
    text = re.sub(r"[́̀̂̃̄]", "", text)
    text = text.replace("_", "")
    return re.sub(r"\s+", " ", text).strip()


def similarity_ratio(left: str, right: str) -> float:
    """Return a lightweight normalized string similarity."""
    import difflib

    left_norm = normalize_for_match(left)
    right_norm = normalize_for_match(right)
    if not left_norm or not right_norm:
        return 0.0
    return difflib.SequenceMatcher(None, left_norm, right_norm).ratio()


def extract_dialect_text(srt_text: str, manual_text: str, *, threshold: float = 0.65) -> DialectExtraction:
    """Remove bracketed interviewer prompts and matching SRT interviewer segments."""
    srt_segments = parse_srt(srt_text)
    manual_segments = parse_manual_annotation(manual_text)
    standard_segments = [segment for segment in manual_segments if not segment.is_dialect]
    standard_srt_indices: set[int] = set()
    for standard in standard_segments:
        best_index = -1
        best_score = 0.0
        for index, srt_segment in enumerate(srt_segments):
            score = similarity_ratio(standard.text, srt_segment.text)
            if score > best_score:
                best_index = index
                best_score = score
        if best_index >= 0 and best_score >= threshold:
            standard_srt_indices.add(best_index)

    whisper_dialect = [
        segment.text for index, segment in enumerate(srt_segments) if index not in standard_srt_indices
    ]
    manual_dialect = [segment.text for segment in manual_segments if segment.is_dialect]
    return DialectExtraction(
        whisper_text=" ".join(whisper_dialect),
        manual_text=" ".join(manual_dialect),
        stats={
            "whisper_total": len(srt_segments),
            "whisper_dialect": len(whisper_dialect),
            "whisper_standard": len(standard_srt_indices),
            "manual_total": len(manual_segments),
            "manual_dialect": len(manual_dialect),
            "manual_standard": len(standard_segments),
        },
    )


def segment_text_with_pauses(text: str, *, max_words: int = 30, join_short_words: int = 5) -> list[str]:
    """Segment text on sentence and pause punctuation while avoiding long fragments."""
    segments: list[str] = []
    current = ""
    for part in re.split(r"[.!?…]+", text):
        part = part.strip()
        if not part:
            continue
        current_words = len(current.split())
        part_words = len(part.split())
        if not current:
            current = part
        elif current_words + part_words > max_words:
            segments.append(current)
            current = part
        elif current_words < join_short_words:
            current = f"{current} {part}"
        else:
            segments.append(current)
            current = part
    if current:
        segments.append(current)
    return segments


def prepare_text_for_embedding(segments: list[str], *, join_short: bool = True) -> list[tuple[str, str]]:
    """Return `(original, cleaned)` segment pairs for embedding alignment."""
    pairs = [(segment, clean_text_for_embedding(segment)) for segment in segments]
    pairs = [(original, clean) for original, clean in pairs if clean]
    if not join_short:
        return pairs
    output: list[tuple[str, str]] = []
    index = 0
    while index < len(pairs):
        original, clean = pairs[index]
        if len(clean.split()) < 3 and index + 1 < len(pairs):
            next_original, next_clean = pairs[index + 1]
            output.append((f"{original} {next_original}", f"{clean} {next_clean}"))
            index += 2
        else:
            output.append((original, clean))
            index += 1
    return output


def _load_sentence_transformer(model_name: str) -> EmbeddingModel:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Embedding alignment requires sentence-transformers. Install it or pass a model with encode()."
        ) from exc
    return SentenceTransformer(model_name)


def _vector_to_floats(vector: object) -> list[float]:
    if hasattr(vector, "tolist"):
        return [float(value) for value in vector.tolist()]
    return [float(value) for value in vector]  # type: ignore[union-attr]


def cosine_similarity(left: object, right: object) -> float:
    """Compute cosine similarity for numeric embedding vectors."""
    left_values = _vector_to_floats(left)
    right_values = _vector_to_floats(right)
    numerator = sum(a * b for a, b in zip(left_values, right_values, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def align_pairs_with_embeddings(
    whisper_text: str,
    manual_text: str,
    *,
    model: EmbeddingModel | None = None,
    model_name: str = "sentence-transformers/LaBSE",
    threshold: float = 0.5,
    max_skip: int = 4,
    join_short: bool = True,
) -> list[EmbeddedAlignedPair]:
    """Align segmented texts monotonically using sentence embeddings."""
    model = model or _load_sentence_transformer(model_name)
    whisper_pairs = prepare_text_for_embedding(segment_text_with_pauses(whisper_text), join_short=join_short)
    manual_pairs = prepare_text_for_embedding(segment_text_with_pauses(manual_text), join_short=join_short)
    if not whisper_pairs or not manual_pairs:
        return []

    whisper_vectors = model.encode([clean for _, clean in whisper_pairs], show_progress_bar=False)
    manual_vectors = model.encode([clean for _, clean in manual_pairs], show_progress_bar=False)

    results: list[EmbeddedAlignedPair] = []
    manual_start = 0
    for whisper_index, (whisper_original, whisper_clean) in enumerate(whisper_pairs):
        best_index = -1
        best_score = 0.0
        search_end = min(len(manual_pairs), manual_start + max_skip + 1)
        for manual_index in range(manual_start, search_end):
            score = cosine_similarity(whisper_vectors[whisper_index], manual_vectors[manual_index])
            if score > best_score:
                best_score = score
                best_index = manual_index
        if best_index >= 0 and best_score >= threshold:
            manual_original, manual_clean = manual_pairs[best_index]
            results.append(
                EmbeddedAlignedPair(
                    whisper_text=whisper_original,
                    manual_text=manual_original,
                    whisper_clean=whisper_clean,
                    manual_clean=manual_clean,
                    score=best_score,
                )
            )
            manual_start = best_index + 1
    return results


def embedded_pairs_to_rows(pairs: list[EmbeddedAlignedPair]) -> list[dict[str, object]]:
    """Convert embedding-aligned pairs to TSV rows."""
    return [
        {
            "pair_index": index,
            "whisper_text": pair.whisper_text,
            "manual_text": pair.manual_text,
            "whisper_clean": pair.whisper_clean,
            "manual_clean": pair.manual_clean,
            "score": f"{pair.score:.3f}",
        }
        for index, pair in enumerate(pairs, start=1)
    ]


def write_embedded_alignment_tsv(pairs: list[EmbeddedAlignedPair], output_path: Path | str) -> None:
    """Write embedding-aligned pairs to TSV."""
    write_tsv(output_path, embedded_pairs_to_rows(pairs), EMBEDDED_ALIGNMENT_COLUMNS)
