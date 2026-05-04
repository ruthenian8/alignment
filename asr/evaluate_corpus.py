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


@dataclass(frozen=True)
class AsrPrediction:
    """One ASR prediction row with an optional reference override."""

    text: str
    reference_path: Path | None = None


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


def prediction_key(path: Path | str) -> str:
    """Return the canonical path key used to join samples and predictions."""
    return str(Path(path).resolve())


def read_prediction_rows(path: Path | str) -> list[dict[str, object]]:
    """Read JSONL or CSV ASR prediction rows without interpreting their schema."""
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open(encoding="utf-8-sig") as file:
            return [json.loads(line) for line in file if line.strip()]
    if input_path.suffix.lower() == ".csv":
        with input_path.open(encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))
    raise ValueError("Predictions must be .jsonl or .csv")


def read_predictions(
    path: Path | str,
    *,
    path_field: str = "path",
    text_field: str = "text",
    reference_path_field: str = "reference_path",
) -> dict[str, AsrPrediction]:
    """Read JSONL or CSV ASR predictions keyed by resolved audio path."""
    predictions: dict[str, AsrPrediction] = {}
    for row in read_prediction_rows(path):
        if not row.get(path_field):
            continue
        key = prediction_key(str(row[path_field]))
        reference_path = row.get(reference_path_field)
        predictions[key] = AsrPrediction(
            text=str(row.get(text_field, "")).strip(),
            reference_path=Path(str(reference_path)) if reference_path else None,
        )
    return predictions


def missing_prediction_samples(
    samples: list[CorpusSample], predictions: dict[str, AsrPrediction]
) -> list[CorpusSample]:
    """Return samples with no matching ASR prediction row."""
    return [sample for sample in samples if prediction_key(sample.audio_path) not in predictions]


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
    samples: list[CorpusSample], predictions: dict[str, AsrPrediction]
) -> tuple[WerStats, Counter[tuple[str, str, str]], list[dict[str, object]]]:
    """Compute global and per-utterance WER for corpus predictions."""
    rows: list[dict[str, object]] = []
    mismatches: Counter[tuple[str, str, str]] = Counter()
    reference_words = substitutions = deletions = insertions = evaluated = 0
    for sample in samples:
        key = prediction_key(sample.audio_path)
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
        reference_path = prediction.reference_path or sample.reference_path
        reference_text = (
            reference_path.read_text(encoding="utf-8-sig").strip()
            if prediction.reference_path
            else sample.reference_text
        )
        ref_count, subs, dels, ins, sample_mismatches = sample_wer(reference_text, prediction.text)
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
                "reference_path": str(reference_path),
                "reference": reference_text,
                "prediction": prediction.text,
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


def prefixed_output_path(output_dir: Path, prefix: str, filename: str) -> Path:
    """Return an evaluation output path, optionally prefixed by prediction file name."""
    return output_dir / (f"{prefix}.{filename}" if prefix else filename)


def run_asr_command(command_template: str, *, input_dir: Path, manifest: Path, predictions: Path) -> None:
    """Run a model-specific ASR command with common corpus placeholders."""
    command = command_template.format(
        input_dir=str(input_dir),
        manifest=str(manifest),
        predictions=str(predictions),
        output=str(predictions),
    )
    subprocess.run(shlex.split(command), check=True)


def append_prediction_file(target: Path, addition: Path) -> None:
    """Append a retry prediction file to the main prediction file."""
    if not addition.exists():
        return
    if target.suffix.lower() != addition.suffix.lower():
        raise ValueError("Retry predictions must use the same file extension as the main predictions")
    if target.suffix.lower() == ".jsonl":
        with target.open("a", encoding="utf-8") as output, addition.open(encoding="utf-8-sig") as input_file:
            for line in input_file:
                if line.strip():
                    output.write(line if line.endswith("\n") else line + "\n")
        return
    if target.suffix.lower() == ".csv":
        rows = read_prediction_rows(target) + read_prediction_rows(addition)
        fieldnames = list(dict.fromkeys(field for row in rows for field in row))
        with target.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return
    raise ValueError("Predictions must be .jsonl or .csv")


def run_asr_with_retries(
    command_template: str,
    *,
    input_dir: Path,
    output_dir: Path,
    samples: list[CorpusSample],
    manifest: Path,
    predictions: Path,
    path_field: str,
    text_field: str,
    reference_path_field: str,
    retry_missing: int,
) -> dict[str, AsrPrediction]:
    """Run ASR and retry missing prediction rows with smaller manifests."""
    run_asr_command(command_template, input_dir=input_dir, manifest=manifest, predictions=predictions)
    if not predictions.exists():
        raise RuntimeError("ASR command completed but did not create a predictions file")

    current = read_predictions(
        predictions,
        path_field=path_field,
        text_field=text_field,
        reference_path_field=reference_path_field,
    )
    for attempt in range(1, retry_missing + 1):
        missing = missing_prediction_samples(samples, current)
        if not missing:
            break
        retry_manifest = output_dir / f"missing_predictions_retry_{attempt}.txt"
        retry_predictions = output_dir / f"predictions.retry{attempt}{predictions.suffix}"
        write_audio_manifest(missing, retry_manifest)
        run_asr_command(
            command_template,
            input_dir=input_dir,
            manifest=retry_manifest,
            predictions=retry_predictions,
        )
        append_prediction_file(predictions, retry_predictions)
        current = read_predictions(
            predictions,
            path_field=path_field,
            text_field=text_field,
            reference_path_field=reference_path_field,
        )
    return current


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
    parser.add_argument(
        "--prediction-reference-path-field",
        default="reference_path",
        help="Optional prediction row field containing a reference text path.",
    )
    parser.add_argument(
        "--retry-missing",
        type=int,
        default=1,
        help="When running --asr-command, retry missing prediction rows this many times.",
    )
    parser.add_argument(
        "--allow-missing-predictions",
        action="store_true",
        help="Do not fail when predictions are still missing after retries.",
    )
    parser.add_argument("--top", type=int, default=50, help="Mismatch count shown in the WER report.")
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
    output_prefix = predictions_path.stem if args.predictions else ""
    write_audio_manifest(samples, manifest_path)
    if args.asr_command:
        predictions = run_asr_with_retries(
            args.asr_command,
            input_dir=args.input_dir,
            output_dir=output_dir,
            samples=samples,
            manifest=manifest_path,
            predictions=predictions_path,
            path_field=args.prediction_path_field,
            text_field=args.prediction_text_field,
            reference_path_field=args.prediction_reference_path_field,
            retry_missing=args.retry_missing,
        )
    else:
        if not predictions_path.exists():
            raise RuntimeError("No predictions file found. Provide --predictions or --asr-command.")
        predictions = read_predictions(
            predictions_path,
            path_field=args.prediction_path_field,
            text_field=args.prediction_text_field,
            reference_path_field=args.prediction_reference_path_field,
        )
    missing = missing_prediction_samples(samples, predictions)
    if missing:
        missing_path = prefixed_output_path(output_dir, output_prefix, "missing_predictions.txt")
        write_audio_manifest(missing, missing_path)
        if args.asr_command and not args.allow_missing_predictions:
            raise RuntimeError(
                f"ASR predictions are missing for {len(missing)} of {len(samples)} samples after "
                f"{args.retry_missing} retry attempt(s). See {missing_path}."
            )
    stats, mismatches, rows = evaluate_predictions(samples, predictions)
    write_csv(prefixed_output_path(output_dir, output_prefix, "per_utterance.csv"), rows)
    write_mismatches(prefixed_output_path(output_dir, output_prefix, "mismatches.csv"), mismatches)
    report = format_wer_report(stats, mismatches, top=args.top)
    prefixed_output_path(output_dir, output_prefix, "wer_report.txt").write_text(
        report + "\n", encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
