"""Populate the segment-validation annotation project from sampled cut files."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


def read_rows(sample_csv: Path) -> list[dict[str, str]]:
    """Read selected validation rows from CSV."""
    with sample_csv.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def media_name(row: dict[str, str]) -> str:
    """Return a collision-safe flat media filename."""
    audio_path = Path(row["audio_path"])
    folder = row["folder"]
    chunk = row["chunk"]
    return f"{folder}__{chunk}__{audio_path.name}"


def populate_annotation_project(
    sample_csv: Path,
    cut_root: Path,
    project_root: Path,
) -> tuple[int, int]:
    """Copy selected audio and write the annotation JSONL file."""
    rows = read_rows(sample_csv)
    files_dir = project_root / "data" / "files"
    jsonl_path = project_root / "data" / "seg_validation.jsonl"
    files_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    copied = 0
    with jsonl_path.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows, start=1):
            source_audio = cut_root / row["audio_path"]
            if not source_audio.exists():
                raise FileNotFoundError(source_audio)
            target_name = media_name(row)
            target_audio = files_dir / target_name
            shutil.copy2(source_audio, target_audio)
            copied += 1
            item = {
                "id": f"seg_{index:04d}",
                "audio_url": f"media/{target_name}",
                "transcript": row["text"],
                "folder": row["folder"],
                "chunk": row["chunk"],
                "source_audio_path": row["audio_path"],
                "source_caption_path": row["caption_path"],
                "source_original_caption_path": row["original_caption_path"],
                "word_count": int(row["word_count"]),
            }
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(rows), copied


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-csv", type=Path, default=Path("build/manual-validation-samples/set_b.csv"))
    parser.add_argument("--cut-root", type=Path, default=Path("build/cut_samples-srt-speakers"))
    parser.add_argument("--project-root", type=Path, default=Path("validation_anno"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run annotation project population."""
    args = parse_args(argv)
    rows, copied = populate_annotation_project(args.sample_csv, args.cut_root, args.project_root)
    print(f"wrote {rows} JSONL rows and copied {copied} audio files into {args.project_root / 'data/files'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
