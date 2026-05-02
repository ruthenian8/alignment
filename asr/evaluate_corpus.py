"""Run optional ASR inference and evaluate utterance-level corpus WER."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    from alignment.wer import WerStats, edit_operations, format_wer_report, normalize_for_wer
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from alignment.wer import WerStats, edit_operations, format_wer_report, normalize_for_wer


@dataclass(frozen=True)
class CorpusSample:
    """One utterance-level audio file with its sibling reference text."""

    audio_path: Path
    reference_path: Path
    reference_text: str


def discover_samples(input_dir: Path | str, *, glob_pattern: str = "*.wav") -> list[CorpusSample]:
    """Find audio files that have sibling ``.txt`` reference files."""
    samples: list[CorpusSample] = []
    for audio_path in sorted(Path(input_dir).rglob(glob_pattern)):
        if not audio_path.is_file():
            continue
        reference_path = audio_path.with_suffix(".txt")
        if not reference_path.exists():
            continue
        samples.append(
            CorpusSample(
                audio_path=audio_path,
                reference_path=reference_path,
                reference_text=reference_path.read_text(encoding="utf-8-sig").strip(),
            )
        )
    return samples


def write_audio_manifest(samples: list[CorpusSample], output_path: Path | str) -> None:
    """Write one audio path per line for the model-specific ASR scripts."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(sample.audio_path) for sample in samples) + "\n", encoding="utf-8")


def read_predictions(
    path: Path | str, *, path_field: str = "path", text_field: str = "text"
) -> dict[str, str]:
    """Read JSONL or CSV ASR predictions keyed by resolved audio path."""
    input_path = Path(path)
    rows: list[dict[str, object]] = []
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open(encoding="utf-8-sig") as file:
            rows = [json.loads(line) for line in file if line.strip()]
    elif input_path.suffix.lower() == ".csv":
        with input_path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    else:
        raise ValueError("Predictions must be .jsonl or .csv")

    predictions: dict[str, str] = {}
    for row in rows:
        if not row.get(path_field):
            continue
        key = str(Path(str(row[path_field])).resolve())
        predictions[key] = str(row.get(text_field, "")).strip()
    return predictions


def sample_wer(reference: str, hypothesis: str) -> tuple[int, int, int, int, Counter[tuple[str, str, str]]]:
    """Return reference length, edit counts, and mismatch counts for one sample."""
    reference_words = normalize_for_wer(reference).split()
    hypothesis_words = normalize_for_wer(hypothesis).split()
    substitutions = deletions = insertions = 0
    mismatches: Counter[tuple[str, str, str]] = Counter()
    for op, ref_word, hyp_word in edit_operations(reference_words, hypothesis_words):
        if op == "substitute":
            substitutions += 1
            mismatches[(op, ref_word, hyp_word)] += 1
        elif op == "delete":
            deletions += 1
            mismatches[(op, ref_word, "<del>")] += 1
        elif op == "insert":
            insertions += 1
            mismatches[(op, "<ins>", hyp_word)] += 1
    return len(reference_words), substitutions, deletions, insertions, mismatches


def evaluate_predictions(
    samples: list[CorpusSample], predictions: dict[str, str]
) -> tuple[WerStats, Counter[tuple[str, str, str]], list[dict[str, object]]]:
    """Compute global and per-utterance WER for corpus predictions."""
    rows: list[dict[str, object]] = []
    mismatches: Counter[tuple[str, str, str]] = Counter()
    reference_words = substitutions = deletions = insertions = evaluated = 0
    for sample in samples:
        key = str(sample.audio_path.resolve())
        if key not in predictions:
            rows.append(
                {
                    "path": str(sample.audio_path),
                    "reference_path": str(sample.reference_path),
                    "reference": sample.reference_text,
                    "prediction": "",
                    "reference_words": 0,
                    "substitutions": 0,
                    "deletions": 0,
                    "insertions": 0,
                    "wer": "",
                    "status": "missing_prediction",
                }
            )
            continue

        prediction = predictions[key]
        ref_count, subs, dels, ins, sample_mismatches = sample_wer(sample.reference_text, prediction)
        errors = subs + dels + ins
        evaluated += 1
        reference_words += ref_count
        substitutions += subs
        deletions += dels
        insertions += ins
        mismatches.update(sample_mismatches)
        rows.append(
            {
                "path": str(sample.audio_path),
                "reference_path": str(sample.reference_path),
                "reference": sample.reference_text,
                "prediction": prediction,
                "reference_words": ref_count,
                "substitutions": subs,
                "deletions": dels,
                "insertions": ins,
                "wer": 0.0 if ref_count == 0 else errors / ref_count,
                "status": "evaluated",
            }
        )

    total_errors = substitutions + deletions + insertions
    stats = WerStats(
        rows=evaluated,
        reference_words=reference_words,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        wer=0.0 if reference_words == 0 else total_errors / reference_words,
    )
    return stats, mismatches, rows


def write_csv(path: Path | str, rows: list[dict[str, object]]) -> None:
    """Write dictionaries as UTF-8 CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_mismatches(path: Path | str, mismatches: Counter[tuple[str, str, str]]) -> None:
    """Write common WER edit operations as CSV."""
    rows = [
        {"count": count, "type": op, "reference": reference, "hypothesis": hypothesis}
        for (op, reference, hypothesis), count in mismatches.most_common()
    ]
    write_csv(path, rows)


def run_asr_command(command_template: str, *, input_dir: Path, manifest: Path, predictions: Path) -> None:
    """Run a model-specific ASR command with common corpus placeholders."""
    command = command_template.format(
        input_dir=str(input_dir),
        manifest=str(manifest),
        predictions=str(predictions),
        output=str(predictions),
    )
    subprocess.run(shlex.split(command), check=True)


def parse_args() -> argparse.Namespace:
    """Parse corpus ASR evaluation arguments."""
    parser = argparse.ArgumentParser(
        description="Run ASR over utterance-level cut samples and compute corpus WER."
    )
    parser.add_argument(
        "input_dir", type=Path, help="Corpus root containing .wav files with sibling .txt files."
    )
    parser.add_argument("output_dir", type=Path, help="Directory for manifest, predictions, and WER outputs.")
    parser.add_argument("--glob", default="*.wav", help="Recursive audio glob under input_dir.")
    parser.add_argument("--predictions", type=Path, help="Existing .jsonl or .csv predictions to evaluate.")
    parser.add_argument(
        "--asr-command",
        help=(
            "Optional inference command template. Available placeholders: "
            "{input_dir}, {manifest}, {predictions}, {output}."
        ),
    )
    parser.add_argument("--prediction-path-field", default="path", help="Prediction row audio path field.")
    parser.add_argument("--prediction-text-field", default="text", help="Prediction row transcript field.")
    parser.add_argument("--top", type=int, default=50, help="Mismatch count shown in wer_report.txt.")
    parser.add_argument("--limit", type=int, help="Optional cap on discovered samples.")
    return parser.parse_args()


def main() -> None:
    """Run optional inference and write corpus WER outputs."""
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(args.input_dir, glob_pattern=args.glob)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError("No .wav files with sibling .txt references found")

    manifest_path = output_dir / "audio_manifest.txt"
    predictions_path = args.predictions or output_dir / "predictions.jsonl"
    write_audio_manifest(samples, manifest_path)
    if args.asr_command:
        run_asr_command(
            args.asr_command,
            input_dir=args.input_dir,
            manifest=manifest_path,
            predictions=predictions_path,
        )
    if not predictions_path.exists():
        raise RuntimeError("No predictions file found. Provide --predictions or --asr-command.")

    predictions = read_predictions(
        predictions_path,
        path_field=args.prediction_path_field,
        text_field=args.prediction_text_field,
    )
    stats, mismatches, rows = evaluate_predictions(samples, predictions)
    write_csv(output_dir / "per_utterance.csv", rows)
    write_mismatches(output_dir / "mismatches.csv", mismatches)
    report = format_wer_report(stats, mismatches, top=args.top)
    (output_dir / "wer_report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
