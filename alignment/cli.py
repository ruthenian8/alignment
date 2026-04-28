"""Command-line interface for the alignment pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", stream=sys.stderr)


def cmd_parse_index(args: argparse.Namespace) -> None:
    """Parse a plaintext index file and write a TSV."""
    from alignment.index_parser import index_rows_to_dataframe, parse_index_plaintext
    from alignment.io import write_tsv

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".tsv")
    base_name = args.base_name or input_path.stem
    suffix = args.suffix or ".wav"

    text = input_path.read_text(encoding="utf-8")
    rows = parse_index_plaintext(text, base_name, suffix)
    df = index_rows_to_dataframe(rows)
    write_tsv(df, output_path)
    logger.info("Wrote %d index rows to %s", len(df), output_path)


def cmd_parse_transcript(args: argparse.Namespace) -> None:
    """Parse a plaintext transcript file and write a TSV."""
    from alignment.io import write_tsv
    from alignment.transcript_parser import parse_transcript_file, transcript_records_to_dataframe

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".tsv")

    records = parse_transcript_file(input_path)
    df = transcript_records_to_dataframe(records)
    write_tsv(df, output_path)
    logger.info("Wrote %d transcript records to %s", len(df), output_path)


def cmd_join(args: argparse.Namespace) -> None:
    """Join an index TSV with a transcript TSV."""
    from alignment.io import read_tsv, write_tsv
    from alignment.join import join_index_and_transcripts

    index_df = read_tsv(Path(args.index))
    transcript_df = read_tsv(Path(args.transcript))
    result = join_index_and_transcripts(index_df, transcript_df)
    output_path = Path(args.output)
    write_tsv(result, output_path)
    logger.info("Wrote joined TSV to %s", output_path)


def cmd_reorder(args: argparse.Namespace) -> None:
    """Reorder transcript column in a joined TSV."""
    from alignment.io import read_tsv, write_tsv
    from alignment.reorder import reorder_dataframe

    df = read_tsv(Path(args.input))
    result = reorder_dataframe(df, char_weight=args.char_weight, max_shift=args.max_shift)
    output_path = Path(args.output) if args.output else Path(args.input)
    write_tsv(result, output_path)
    logger.info("Wrote reordered TSV to %s", output_path)


def cmd_align_srt(args: argparse.Namespace) -> None:
    """Align an SRT file to a transcript text file."""
    import pandas as pd

    from alignment.align import align_srt_to_transcript

    srt_text = Path(args.srt).read_text(encoding="utf-8")
    transcript_text = Path(args.transcript).read_text(encoding="utf-8")
    aligned = align_srt_to_transcript(srt_text, transcript_text)

    rows = [
        {
            "index": seg.index,
            "start": seg.start,
            "end": seg.end,
            "speaker": seg.speaker,
            "original_text": seg.original_text,
            "transcript_text": seg.transcript_text,
            "matched": seg.matched,
        }
        for seg in aligned
    ]
    df = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    matched = sum(1 for seg in aligned if seg.matched)
    logger.info("Aligned %d/%d segments, wrote to %s", matched, len(aligned), output_path)


def cmd_export_corpus(args: argparse.Namespace) -> None:
    """Cut audio clips by aligned SRT and write corpus manifest."""
    import pandas as pd

    from alignment.export import ExportSegment, cut_audio_segments, write_manifest
    from alignment.srt import parse_srt

    input_audio = Path(args.audio)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.tsv"

    if args.aligned_tsv:
        df = pd.read_csv(Path(args.aligned_tsv), sep="\t", encoding="utf-8")
        segments = [
            ExportSegment(
                index=int(row["index"]),
                start=str(row["start"]),
                end=str(row["end"]),
                speaker=str(row.get("speaker", "")),
                text=str(row.get("transcript_text", "") or row.get("original_text", "")),
                text_clean=str(row.get("original_text", "")),
            )
            for _, row in df.iterrows()
        ]
    else:
        srt_text = Path(args.srt).read_text(encoding="utf-8")
        srt_segs = parse_srt(srt_text)
        segments = [
            ExportSegment(
                index=seg.index,
                start=seg.start,
                end=seg.end,
                speaker=seg.speaker,
                text=seg.text,
                text_clean=seg.text,
            )
            for seg in srt_segs
        ]

    audio_paths = cut_audio_segments(input_audio, segments, output_dir)
    write_manifest(segments, audio_paths, manifest_path)
    logger.info("Exported %d clips to %s", len(segments), output_dir)


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run the full pipeline end-to-end."""
    logger.info("run-all: running full pipeline")
    args_parse_index = argparse.Namespace(
        input=args.index,
        output=str(Path(args.work_dir) / "index.tsv"),
        base_name=args.base_name,
        suffix=args.suffix or ".wav",
    )
    cmd_parse_index(args_parse_index)

    args_parse_transcript = argparse.Namespace(
        input=args.transcript,
        output=str(Path(args.work_dir) / "transcript.tsv"),
    )
    cmd_parse_transcript(args_parse_transcript)

    args_join = argparse.Namespace(
        index=str(Path(args.work_dir) / "index.tsv"),
        transcript=str(Path(args.work_dir) / "transcript.tsv"),
        output=str(Path(args.work_dir) / "joined.tsv"),
    )
    cmd_join(args_join)

    logger.info(
        "run-all: pipeline complete (SRT alignment step requires SRT files; "
        "run align-srt separately)"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="alignment",
        description="Alignment pipeline for speech corpus production.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("parse-index", help="Parse plaintext index file to TSV")
    p_index.add_argument("input", help="Path to plaintext index file")
    p_index.add_argument("--output", "-o", help="Output TSV path (default: input with .tsv)")
    p_index.add_argument("--base-name", help="Base name for output audio files")
    p_index.add_argument("--suffix", help="Audio file suffix (default: .wav)")
    p_index.set_defaults(func=cmd_parse_index)

    p_trans = sub.add_parser("parse-transcript", help="Parse plaintext transcript file to TSV")
    p_trans.add_argument("input", help="Path to plaintext transcript file")
    p_trans.add_argument("--output", "-o", help="Output TSV path (default: input with .tsv)")
    p_trans.set_defaults(func=cmd_parse_transcript)

    p_join = sub.add_parser("join", help="Join index TSV and transcript TSV")
    p_join.add_argument("index", help="Path to index TSV")
    p_join.add_argument("transcript", help="Path to transcript TSV")
    p_join.add_argument("output", help="Path to output joined TSV")
    p_join.set_defaults(func=cmd_join)

    p_reorder = sub.add_parser("reorder", help="Reorder transcript column in joined TSV")
    p_reorder.add_argument("input", help="Path to joined TSV")
    p_reorder.add_argument("--output", "-o", help="Output TSV path (default: overwrite input)")
    p_reorder.add_argument("--char-weight", type=float, default=0.7)
    p_reorder.add_argument("--max-shift", type=int, default=3)
    p_reorder.set_defaults(func=cmd_reorder)

    p_align = sub.add_parser("align-srt", help="Align SRT to transcript text")
    p_align.add_argument("srt", help="Path to SRT file")
    p_align.add_argument("transcript", help="Path to transcript text file")
    p_align.add_argument("output", help="Path to output aligned TSV")
    p_align.set_defaults(func=cmd_align_srt)

    p_export = sub.add_parser("export-corpus", help="Cut audio by aligned SRT and write manifest")
    p_export.add_argument("audio", help="Path to source audio file")
    p_export.add_argument("output_dir", help="Directory for output clips")
    p_export.add_argument("--srt", help="Path to SRT file (if not using aligned TSV)")
    p_export.add_argument("--aligned-tsv", help="Path to aligned TSV from align-srt")
    p_export.add_argument(
        "--manifest", help="Path for manifest TSV (default: output_dir/manifest.tsv)"
    )
    p_export.set_defaults(func=cmd_export_corpus)

    p_all = sub.add_parser("run-all", help="Run full pipeline")
    p_all.add_argument("index", help="Path to plaintext index file")
    p_all.add_argument("transcript", help="Path to plaintext transcript file")
    p_all.add_argument(
        "--work-dir", default="build", help="Working directory for intermediate files"
    )
    p_all.add_argument("--base-name", help="Base name for audio files")
    p_all.add_argument("--suffix", help="Audio file suffix")
    p_all.set_defaults(func=cmd_run_all)

    return parser


def main() -> None:
    """Entry point for the alignment CLI."""
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))
    args.func(args)
