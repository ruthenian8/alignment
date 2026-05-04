"""Copy ASR prediction files with manifest-derived reference paths.

This one-off helper keeps prediction ``path`` fields pointed at audio files and
adds a ``reference_path`` field from the ASR-relocation manifest. Rows with a
written relocated reference use that file; unchanged rows fall back to the
original manifest ``text_path``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def read_reference_map(manifest_path: Path) -> dict[str, str]:
    """Read audio path to reference path mapping from a relocation manifest."""
    references: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            audio_path = row.get("audio_path", "")
            if not audio_path:
                continue
            redacted_path = Path(row.get("redacted_path", ""))
            text_path = row.get("text_path", "")
            reference_path = str(redacted_path) if redacted_path.exists() else text_path
            if not reference_path:
                raise ValueError(f"Manifest row has no usable reference for {audio_path}")
            references[str(Path(audio_path))] = reference_path
    return references


def rewrite_prediction_references(
    input_path: Path,
    output_path: Path,
    *,
    reference_map: dict[str, str],
    path_field: str = "path",
    reference_field: str = "reference_path",
) -> tuple[int, int, int]:
    """Copy one JSONL prediction file while adding reference paths."""
    rows = rewritten = fallback = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8-sig") as source, output_path.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            audio_path = row.get(path_field)
            if not audio_path:
                raise KeyError(f"{input_path}:{line_number} does not contain field {path_field!r}")
            reference_path = reference_map.get(str(Path(str(audio_path))))
            if reference_path is None:
                reference_path = str(Path(str(audio_path)).with_suffix(".txt"))
                fallback += 1
            row[reference_field] = reference_path
            rewritten += 1
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows, rewritten, fallback


def prediction_files(input_dir: Path, patterns: list[str]) -> list[Path]:
    """Return prediction files selected by glob patterns."""
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(input_dir.glob(pattern)))
    return list(dict.fromkeys(files))


def output_name(input_path: Path, output_dir: Path, suffix: str) -> Path:
    """Return output path for a copied prediction file."""
    return output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prediction_files",
        nargs="*",
        type=Path,
        help="Prediction JSONL files. Defaults to *_preds.cut_samples_srt_speakers.jsonl.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("build/rewritten-preds/manual_asr_redactions.relocated.csv"),
        help="ASR relocation manifest produced by retro_correct_manual_from_asr.py.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("build/rewritten-preds"),
        help="Directory used when prediction files are not passed explicitly.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/rewritten-preds"),
        help="Directory for copied prediction files.",
    )
    parser.add_argument(
        "--suffix",
        default=".asr_relocated_refs",
        help="Suffix added before .jsonl for copied prediction files.",
    )
    parser.add_argument("--path-field", default="path", help="Prediction row audio path field.")
    parser.add_argument("--reference-field", default="reference_path", help="Output reference path field.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Copy prediction files with relocated reference paths."""
    args = parse_args(argv)
    inputs = args.prediction_files or prediction_files(
        args.input_dir,
        ["*_preds.cut_samples_srt_speakers.jsonl"],
    )
    if not inputs:
        raise RuntimeError(f"No prediction files found in {args.input_dir}")
    reference_map = read_reference_map(args.manifest)
    for input_path in inputs:
        output_path = output_name(input_path, args.output_dir, args.suffix)
        rows, rewritten, fallback = rewrite_prediction_references(
            input_path,
            output_path,
            reference_map=reference_map,
            path_field=args.path_field,
            reference_field=args.reference_field,
        )
        print(
            f"{input_path} -> {output_path}: wrote {rewritten}/{rows} references; "
            f"fallback_to_sibling_txt={fallback}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
