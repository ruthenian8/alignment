"""Retroactively trim manual cut-sample text using multiple ASR predictions.

The script is intentionally conservative: it only removes full leading or
trailing sentence-like units when every supplied ASR prediction has no
normalized word overlap with that unit. Interior text is never removed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    from alignment.wer import normalize_for_wer
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from alignment.wer import normalize_for_wer


SENTENCE_RE = re.compile(r".+?(?:[.!?…]+(?:[\"»”)\]]+)?|$)", re.DOTALL)
DEFAULT_SUFFIX = ".asr_redacted"


@dataclass
class CorrectionDecision:
    """One manual text correction before manifest serialization."""

    audio_path: Path
    text_path: Path
    source_text: str
    redacted_text: str
    removed_prefix: list[str]
    removed_suffix: list[str]
    possibly_misaligned: bool
    relocated_prefix_to: Path | None = None
    relocated_suffix_to: Path | None = None
    received_prefix_from: list[Path] = field(default_factory=list)
    received_suffix_from: list[Path] = field(default_factory=list)


def normalized_words(text: str) -> set[str]:
    """Return normalized words used for overlap checks."""
    return set(normalize_for_wer(text).split())


def split_sentence_units(text: str) -> list[str]:
    """Split text into sentence-like units while preserving original spelling."""
    units = [match.group(0).strip() for match in SENTENCE_RE.finditer(text) if match.group(0).strip()]
    return units or ([text.strip()] if text.strip() else [])


def read_predictions(path: Path) -> dict[Path, str]:
    """Read JSONL predictions keyed by their referenced audio path."""
    predictions: dict[Path, str] = {}
    with path.open(encoding="utf-8-sig") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("path"):
                predictions[Path(str(row["path"]))] = str(row.get("text", "")).strip()
    return predictions


def model_name(path: Path) -> str:
    """Derive a compact model label from a prediction filename."""
    return path.name.split("_preds", 1)[0]


def common_audio_paths(predictions_by_model: dict[str, dict[Path, str]]) -> list[Path]:
    """Return audio paths that are present in every model prediction file."""
    model_paths = [set(predictions) for predictions in predictions_by_model.values()]
    if not model_paths:
        return []
    return sorted(set.intersection(*model_paths))


def trim_units(
    units: list[str], prediction_word_sets: list[set[str]]
) -> tuple[list[str], list[str], list[str]]:
    """Trim full leading and trailing units absent from every prediction."""
    start = 0
    end = len(units)
    while start < end and unit_absent_from_all(units[start], prediction_word_sets):
        start += 1
    while end > start and unit_absent_from_all(units[end - 1], prediction_word_sets):
        end -= 1
    return units[start:end], units[:start], units[end:]


def unit_absent_from_all(unit: str, prediction_word_sets: list[set[str]]) -> bool:
    """Return true when a text unit has no word overlap with any prediction."""
    words = normalized_words(unit)
    return bool(words) and all(not (words & prediction_words) for prediction_words in prediction_word_sets)


def overlap_score(text: str, prediction_word_sets: list[set[str]]) -> int:
    """Return total normalized word overlap across model predictions."""
    words = normalized_words(text)
    return sum(len(words & prediction_words) for prediction_words in prediction_word_sets)


def join_units(units: list[str]) -> str:
    """Join sentence-like units into one text line."""
    return " ".join(unit for unit in units if unit).strip()


def redacted_path_for(source_path: Path, suffix: str) -> Path:
    """Return sibling redacted text path for a source reference file."""
    return source_path.with_name(f"{source_path.stem}{suffix}{source_path.suffix}")


def correction_row(
    *,
    audio_path: Path,
    text_path: Path,
    redacted_path: Path,
    source_text: str,
    redacted_text: str,
    removed_prefix: list[str],
    removed_suffix: list[str],
    predictions_by_model: dict[str, dict[Path, str]],
    possibly_misaligned: bool,
    relocated_prefix_to: Path | None = None,
    relocated_suffix_to: Path | None = None,
    received_prefix_from: list[Path] | None = None,
    received_suffix_from: list[Path] | None = None,
) -> dict[str, object]:
    """Build one manifest row for a correction decision."""
    manual_words = normalized_words(source_text)
    overlap_by_model = {
        name: len(manual_words & normalized_words(predictions[audio_path]))
        for name, predictions in predictions_by_model.items()
    }
    changed = source_text != redacted_text
    row: dict[str, object] = {
        "audio_path": str(audio_path),
        "text_path": str(text_path),
        "redacted_path": str(redacted_path),
        "changed": changed,
        "possibly_misaligned": possibly_misaligned,
        "removed_prefix_sentences": len(removed_prefix),
        "removed_suffix_sentences": len(removed_suffix),
        "removed_prefix_text": " ".join(removed_prefix),
        "removed_suffix_text": " ".join(removed_suffix),
        "relocated_prefix_to": str(relocated_prefix_to) if relocated_prefix_to else "",
        "relocated_suffix_to": str(relocated_suffix_to) if relocated_suffix_to else "",
        "received_prefix_from": " ".join(str(path) for path in received_prefix_from or []),
        "received_suffix_from": " ".join(str(path) for path in received_suffix_from or []),
        "source_text": source_text,
        "redacted_text": redacted_text,
    }
    for name, overlap in overlap_by_model.items():
        row[f"{name}_word_overlap"] = overlap
        row[f"{name}_prediction"] = predictions_by_model[name][audio_path]
    return row


def build_decision(
    audio_path: Path,
    predictions_by_model: dict[str, dict[Path, str]],
) -> CorrectionDecision | None:
    """Build the conservative trim decision for one audio path."""
    text_path = audio_path.with_suffix(".txt")
    if not text_path.exists():
        return None
    source_text = text_path.read_text(encoding="utf-8-sig").strip()
    units = split_sentence_units(source_text)
    prediction_word_sets = [
        normalized_words(predictions[audio_path]) for predictions in predictions_by_model.values()
    ]
    manual_words = normalized_words(source_text)
    possibly_misaligned = bool(manual_words) and all(
        not (manual_words & prediction_words) for prediction_words in prediction_word_sets
    )
    if possibly_misaligned:
        kept_units, removed_prefix, removed_suffix = units, [], []
    else:
        kept_units, removed_prefix, removed_suffix = trim_units(units, prediction_word_sets)
    return CorrectionDecision(
        audio_path=audio_path,
        text_path=text_path,
        source_text=source_text,
        redacted_text=join_units(kept_units),
        removed_prefix=removed_prefix,
        removed_suffix=removed_suffix,
        possibly_misaligned=possibly_misaligned,
    )


def relocate_orphan_spans(
    decisions: list[CorrectionDecision],
    predictions_by_model: dict[str, dict[Path, str]],
) -> None:
    """Move trimmed edge spans to adjacent files when ASR overlap improves."""
    by_audio = {decision.audio_path: decision for decision in decisions}
    ordered_paths = [decision.audio_path for decision in decisions]
    for index, decision in enumerate(decisions):
        if decision.removed_prefix and index > 0:
            previous = by_audio[ordered_paths[index - 1]]
            maybe_relocate_span(
                span=join_units(decision.removed_prefix),
                source=decision,
                target=previous,
                target_side="suffix",
                predictions_by_model=predictions_by_model,
            )
        if decision.removed_suffix and index + 1 < len(ordered_paths):
            following = by_audio[ordered_paths[index + 1]]
            maybe_relocate_span(
                span=join_units(decision.removed_suffix),
                source=decision,
                target=following,
                target_side="prefix",
                predictions_by_model=predictions_by_model,
            )


def maybe_relocate_span(
    *,
    span: str,
    source: CorrectionDecision,
    target: CorrectionDecision,
    target_side: str,
    predictions_by_model: dict[str, dict[Path, str]],
) -> None:
    """Relocate one span if it improves the adjacent target score."""
    if not span or source.text_path.parent != target.text_path.parent:
        return
    prediction_word_sets = [
        normalized_words(predictions[target.audio_path]) for predictions in predictions_by_model.values()
    ]
    current_score = overlap_score(target.redacted_text, prediction_word_sets)
    candidate_text = (
        join_units([span, target.redacted_text])
        if target_side == "prefix"
        else join_units([target.redacted_text, span])
    )
    if overlap_score(candidate_text, prediction_word_sets) <= current_score:
        return
    target.redacted_text = candidate_text
    if target_side == "prefix":
        source.relocated_suffix_to = target.audio_path
        target.received_prefix_from.append(source.audio_path)
    else:
        source.relocated_prefix_to = target.audio_path
        target.received_suffix_from.append(source.audio_path)


def process_predictions(
    prediction_files: list[Path],
    *,
    manifest_path: Path,
    suffix: str = DEFAULT_SUFFIX,
    write_unchanged: bool = False,
    relocate_orphans: bool = False,
) -> list[dict[str, object]]:
    """Write redacted manual texts and return manifest rows."""
    predictions_by_model = {model_name(path): read_predictions(path) for path in prediction_files}
    decisions = [
        decision
        for audio_path in common_audio_paths(predictions_by_model)
        if (decision := build_decision(audio_path, predictions_by_model)) is not None
    ]
    if relocate_orphans:
        relocate_orphan_spans(decisions, predictions_by_model)

    rows: list[dict[str, object]] = []
    for decision in decisions:
        redacted_path = redacted_path_for(decision.text_path, suffix)
        if decision.source_text != decision.redacted_text or write_unchanged:
            redacted_path.write_text(
                decision.redacted_text + ("\n" if decision.redacted_text else ""),
                encoding="utf-8",
            )
        elif redacted_path.exists():
            redacted_path.unlink()
        rows.append(
            correction_row(
                audio_path=decision.audio_path,
                text_path=decision.text_path,
                redacted_path=redacted_path,
                source_text=decision.source_text,
                redacted_text=decision.redacted_text,
                removed_prefix=decision.removed_prefix,
                removed_suffix=decision.removed_suffix,
                predictions_by_model=predictions_by_model,
                possibly_misaligned=decision.possibly_misaligned,
                relocated_prefix_to=decision.relocated_prefix_to,
                relocated_suffix_to=decision.relocated_suffix_to,
                received_prefix_from=decision.received_prefix_from,
                received_suffix_from=decision.received_suffix_from,
            )
        )

    write_manifest(manifest_path, rows)
    return rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    """Write correction decisions as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prediction_files",
        nargs="*",
        type=Path,
        default=[
            Path("build/rewritten-preds/gigaam_preds.cut_samples_srt_speakers.jsonl"),
            Path("build/rewritten-preds/mms_preds.cut_samples_srt_speakers.jsonl"),
            Path("build/rewritten-preds/whisper_preds.cut_samples_srt_speakers.jsonl"),
        ],
        help="Rewritten ASR prediction JSONL files. Defaults to build/rewritten-preds outputs.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("build/rewritten-preds/manual_asr_redactions.csv"),
        help="CSV manifest of correction decisions.",
    )
    parser.add_argument(
        "--suffix",
        default=DEFAULT_SUFFIX,
        help="Suffix inserted before .txt for redacted manual files.",
    )
    parser.add_argument(
        "--write-unchanged",
        action="store_true",
        help="Also write redacted copies for unchanged source text files.",
    )
    parser.add_argument(
        "--relocate-orphan-spans",
        action="store_true",
        help=(
            "Move trimmed prefixes/suffixes to adjacent same-directory transcripts when ASR overlap improves."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the retroactive correction procedure."""
    args = parse_args(argv)
    rows = process_predictions(
        args.prediction_files,
        manifest_path=args.manifest,
        suffix=args.suffix,
        write_unchanged=args.write_unchanged,
        relocate_orphans=args.relocate_orphan_spans,
    )
    changed = sum(row["changed"] is True for row in rows)
    misaligned = sum(row["possibly_misaligned"] is True for row in rows)
    print(f"processed {len(rows)} files; changed {changed}; possibly misaligned {misaligned}")
    print(f"manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
