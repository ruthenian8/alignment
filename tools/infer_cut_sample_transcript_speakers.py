"""Infer transcript speaker tags for an existing cut-sample tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alignment.speakers import (  # noqa: E402
    process_cut_sample_tree,
    summarize_speaker_maps,
    write_raw_transcript_inventory,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut-root", type=Path, default=Path("build/cut_samples-srt-speakers"))
    parser.add_argument(
        "--aligned-root",
        type=Path,
        default=Path("build/align-map-wx-transcripts-srt-speakers"),
    )
    parser.add_argument("--output-name", default="speaker_map.csv")
    parser.add_argument(
        "--prefer-table-speakers",
        action="store_true",
        help="Use non-WhisperX speaker tags already present in aligned TSV speaker columns.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("build/speaker-inference-summary"),
        help="Directory for coverage summary CSV files.",
    )
    parser.add_argument(
        "--raw-transcript-root",
        type=Path,
        default=Path("raw_transcript"),
        help="Optional raw transcript root for transcript-level speaker inventory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run transcript speaker inference for a cut-sample tree."""
    args = parse_args(argv)
    chunks, rows, missing = process_cut_sample_tree(
        args.cut_root,
        args.aligned_root,
        args.output_name,
        prefer_table_speakers=args.prefer_table_speakers,
    )
    metrics = summarize_speaker_maps(args.cut_root, args.summary_dir, args.output_name)
    inventory_rows = 0
    if args.raw_transcript_root.exists():
        inventory_rows = write_raw_transcript_inventory(
            args.raw_transcript_root,
            args.summary_dir / "raw_transcript_speaker_inventory.csv",
        )
    print(f"wrote {args.output_name} for {chunks} chunk dirs, {rows} audio rows; skipped {missing} dirs")
    print(
        "summary: "
        f"{metrics['inferred_rows']}/{metrics['rows']} inferred, "
        f"{metrics['blank_rows']} blank "
        f"({metrics['matched_blank_rows']} matched, {metrics['unmatched_blank_rows']} unmatched), "
        f"{metrics['unknown_rows']} unknown"
    )
    if inventory_rows:
        print(f"wrote raw transcript speaker inventory for {inventory_rows} text files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
