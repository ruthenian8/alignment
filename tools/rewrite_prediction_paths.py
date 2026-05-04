"""Rewrite ASR prediction paths from one cut-sample layout to another.

This is a one-off helper for prediction files created against
``cut_samples/<corpus>/<chunk>/<number>_<speaker>_<timestamp>.wav`` when the
target tree has different order numbers but the same per-chunk timestamps.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def timestamp_from_name(path: Path) -> str:
    """Extract the terminal timestamp from a cut-sample filename."""
    stem_parts = path.stem.rsplit("_", 1)
    if len(stem_parts) != 2:
        raise ValueError(f"Cannot parse timestamp from {path}")
    return stem_parts[-1]


def cut_sample_key(path: str | Path) -> tuple[str, str, str]:
    """Return ``(corpus, chunk, timestamp)`` for a cut-sample path."""
    sample_path = Path(path)
    if len(sample_path.parts) < 3:
        raise ValueError(f"Path does not include corpus/chunk directories: {path}")
    return sample_path.parts[-3], sample_path.parts[-2], timestamp_from_name(sample_path)


def build_target_index(target_root: Path) -> dict[tuple[str, str, str], Path]:
    """Index target WAV files by corpus, chunk, and timestamp."""
    grouped: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for wav_path in sorted(target_root.rglob("*.wav")):
        grouped[cut_sample_key(wav_path)].append(wav_path)

    duplicates = {key: paths for key, paths in grouped.items() if len(paths) > 1}
    if duplicates:
        examples = "\n".join(f"{key}: {paths[:3]}" for key, paths in list(duplicates.items())[:10])
        raise ValueError(f"Target timestamps are not unique within terminal directories:\n{examples}")

    return {key: paths[0] for key, paths in grouped.items()}


def display_path(target_path: Path, target_root: Path, target_prefix: str) -> str:
    """Return the rewritten path with a caller-selected display prefix."""
    relative_path = target_path.relative_to(target_root)
    if target_prefix:
        return str(Path(target_prefix) / relative_path)
    return str(target_path)


def rewrite_prediction_file(
    input_path: Path,
    output_path: Path,
    *,
    target_root: Path,
    target_index: dict[tuple[str, str, str], Path],
    target_prefix: str,
    path_field: str,
) -> tuple[int, int]:
    """Copy a JSONL prediction file while replacing its audio paths."""
    rows = rewritten = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8-sig") as source, output_path.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            if path_field not in row:
                raise KeyError(f"{input_path}:{line_number} does not contain field {path_field!r}")
            key = cut_sample_key(str(row[path_field]))
            if key not in target_index:
                raise KeyError(f"{input_path}:{line_number} has no target match for {key}")
            row[path_field] = display_path(target_index[key], target_root, target_prefix)
            rewritten += 1
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows, rewritten


def output_name(input_path: Path, output_dir: Path, suffix: str) -> Path:
    """Return a stable output path for the rewritten prediction copy."""
    return output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_files", nargs="+", type=Path, help="JSONL prediction files to rewrite.")
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("build/cut_samples-srt-speakers"),
        help="Local target cut-sample tree used to resolve timestamp matches.",
    )
    parser.add_argument(
        "--target-prefix",
        default="build/cut_samples-srt-speakers",
        help=(
            "Path prefix written into output JSONL rows. "
            "Use an empty string to write local target-root paths."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/rewritten-preds"),
        help="Directory for rewritten prediction file copies.",
    )
    parser.add_argument("--path-field", default="path", help="JSON field containing the audio path.")
    parser.add_argument("--suffix", default=".cut_samples_srt_speakers", help="Suffix added before .jsonl.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Rewrite all requested prediction files."""
    args = parse_args(argv)
    target_root = args.target_root
    target_index = build_target_index(target_root)
    for input_path in args.prediction_files:
        output_path = output_name(input_path, args.output_dir, args.suffix)
        rows, rewritten = rewrite_prediction_file(
            input_path,
            output_path,
            target_root=target_root,
            target_index=target_index,
            target_prefix=args.target_prefix,
            path_field=args.path_field,
        )
        print(f"{input_path} -> {output_path}: rewrote {rewritten}/{rows} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
