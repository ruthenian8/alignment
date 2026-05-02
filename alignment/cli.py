"""Command-line interface for the alignment pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .align import align_srt_file, write_aligned_tsv
from .embeddings import align_pairs_with_embeddings, extract_dialect_text, write_embedded_alignment_tsv
from .export import export_srt_files
from .index_parser import write_index_tsv
from .join import join_tsv
from .mapping import align_mapping_table
from .reorder import reorder_tsv
from .transcript_parser import write_transcript_tsv
from .wer import compute_wer_from_tsv, format_wer_report


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(prog="alignment", description="Build aligned speech-corpus data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_index = subparsers.add_parser("parse-index", help="Parse plaintext or DOCX index into TSV.")
    parse_index.add_argument("input", type=Path, help="Plaintext or DOCX index path.")
    parse_index.add_argument("output", type=Path, help="Output index TSV path.")
    parse_index.add_argument("--audio-stem", help="Stem used for generated chunk audio names.")

    parse_transcript = subparsers.add_parser("parse-transcript", help="Parse manual transcript into TSV.")
    parse_transcript.add_argument("input", type=Path, help="Manual plaintext transcript path.")
    parse_transcript.add_argument("output", type=Path, help="Output transcript TSV path.")

    join = subparsers.add_parser("join", help="Join transcript TSV onto index TSV.")
    join.add_argument("index_tsv", type=Path, help="Canonical index TSV path.")
    join.add_argument("transcript_tsv", type=Path, help="Canonical transcript TSV path.")
    join.add_argument("output", type=Path, help="Output joined TSV path.")

    reorder = subparsers.add_parser("reorder", help="Reorder transcript fields in a joined TSV.")
    reorder.add_argument("input", type=Path, help="Joined TSV path.")
    reorder.add_argument("output", type=Path, help="Output reordered TSV path.")
    reorder.add_argument("--max-shift", type=int, default=3, help="Maximum active-row shift to test.")

    align = subparsers.add_parser("align-srt", help="Align one SRT file with one transcript text file.")
    align.add_argument("srt", type=Path, help="WhisperX SRT file.")
    align.add_argument("transcript", type=Path, help="Manual transcript text file.")
    align.add_argument("output_srt", type=Path, help="Output aligned SRT path.")
    align.add_argument("--output-tsv", type=Path, help="Optional output aligned TSV path.")
    align.add_argument("--index-name", default="", help="Index/chunk name recorded in aligned TSV.")
    align.add_argument(
        "--use-transcript-speakers",
        action="store_true",
        help="Replace SRT speaker codes with bracketed speaker tags from the manual transcript.",
    )
    align.add_argument(
        "--infer-missing-speakers",
        action="store_true",
        help="Carry the last bracketed transcript speaker tag forward across following aligned segments.",
    )

    align_map = subparsers.add_parser(
        "align-map",
        help="Align every transcript row in a chunk mapping CSV/TSV to matching SRT files.",
    )
    align_map.add_argument("mapping", type=Path, help="CSV/TSV table with name and transcript columns.")
    align_map.add_argument("srt_dir", type=Path, help="Directory containing SRT files named after chunks.")
    align_map.add_argument(
        "output_dir", type=Path, help="Directory for manual, aligned, and summary outputs."
    )
    align_map.add_argument(
        "--use-transcript-speakers",
        action="store_true",
        help="Replace SRT speaker codes with bracketed speaker tags from the manual transcript.",
    )
    align_map.add_argument(
        "--infer-missing-speakers",
        action="store_true",
        help="Carry the last bracketed transcript speaker tag forward across following aligned segments.",
    )

    export = subparsers.add_parser("export-corpus", help="Cut audio clips and write text plus manifest.")
    export.add_argument("audio", type=Path, help="Input audio chunk.")
    export.add_argument("original_srt", type=Path, help="SRT preserving original/manual text.")
    export.add_argument("clean_srt", type=Path, help="SRT with cleaned output text.")
    export.add_argument("output_dir", type=Path, help="Directory for clip wav/txt files.")
    export.add_argument("manifest", type=Path, help="Output manifest TSV path.")

    align_embeddings = subparsers.add_parser(
        "align-embeddings",
        help="Optionally align dialect-only SRT/manual text pairs with sentence embeddings.",
    )
    align_embeddings.add_argument("srt", type=Path, help="WhisperX SRT file.")
    align_embeddings.add_argument("transcript", type=Path, help="Manual transcript text file.")
    align_embeddings.add_argument("output", type=Path, help="Output embedded alignment TSV path.")
    align_embeddings.add_argument(
        "--model-name",
        default="sentence-transformers/LaBSE",
        help="SentenceTransformer model name loaded only for this command.",
    )
    align_embeddings.add_argument("--threshold", type=float, default=0.5, help="Minimum cosine score.")
    align_embeddings.add_argument(
        "--no-join-short",
        action="store_true",
        help="Keep very short text segments separate instead of joining them before embedding.",
    )
    align_embeddings.add_argument(
        "--standard-threshold",
        type=float,
        default=0.65,
        help="Similarity threshold for removing bracketed interviewer prompts from SRT.",
    )

    wer = subparsers.add_parser(
        "wer",
        help="Compute global WER from an aligned TSV, skipping unmatched or zero-score rows.",
    )
    wer.add_argument("aligned_tsv", type=Path, help="Post-alignment TSV path.")
    wer.add_argument("--top", type=int, default=20, help="Number of common mismatches to print.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the alignment command-line interface."""
    args = build_parser().parse_args(argv)
    if args.command == "parse-index":
        write_index_tsv(args.input, args.output, audio_stem=args.audio_stem)
    elif args.command == "parse-transcript":
        write_transcript_tsv(args.input, args.output)
    elif args.command == "join":
        join_tsv(args.index_tsv, args.transcript_tsv, args.output)
    elif args.command == "reorder":
        reorder_tsv(args.input, args.output, max_shift=args.max_shift)
    elif args.command == "align-srt":
        transcript = args.transcript.read_text(encoding="utf-8-sig")
        aligned = align_srt_file(
            args.srt,
            transcript,
            args.output_srt,
            use_transcript_speakers=args.use_transcript_speakers,
            infer_missing_speakers=args.infer_missing_speakers,
        )
        if args.output_tsv:
            write_aligned_tsv(args.index_name or args.srt.stem, aligned, args.output_tsv)
    elif args.command == "align-map":
        align_mapping_table(
            args.mapping,
            args.srt_dir,
            args.output_dir,
            use_transcript_speakers=args.use_transcript_speakers,
            infer_missing_speakers=args.infer_missing_speakers,
        )
    elif args.command == "export-corpus":
        export_srt_files(args.audio, args.original_srt, args.clean_srt, args.output_dir, args.manifest)
    elif args.command == "align-embeddings":
        extraction = extract_dialect_text(
            args.srt.read_text(encoding="utf-8-sig"),
            args.transcript.read_text(encoding="utf-8-sig"),
            threshold=args.standard_threshold,
        )
        pairs = align_pairs_with_embeddings(
            extraction.whisper_text,
            extraction.manual_text,
            model_name=args.model_name,
            threshold=args.threshold,
            join_short=not args.no_join_short,
        )
        write_embedded_alignment_tsv(pairs, args.output)
    elif args.command == "wer":
        stats, mismatches = compute_wer_from_tsv(args.aligned_tsv)
        print(format_wer_report(stats, mismatches, top=args.top))


if __name__ == "__main__":
    main()
