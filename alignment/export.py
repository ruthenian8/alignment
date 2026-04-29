"""Export aligned SRT rows as corpus clips, text files, and a manifest."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .audio import build_cut_command
from .io import MANIFEST_COLUMNS, write_tsv
from .srt import SrtSegment, normalize_timestamp, parse_srt


def safe_time(timestamp: str) -> str:
    """Make a timestamp safe for deterministic filenames."""
    return normalize_timestamp(timestamp, decimal=".").replace(":", "-").replace(".", "-")


def clip_id(segment: SrtSegment) -> str:
    """Build a stable clip identifier from SRT index, speaker, and start time."""
    speaker = segment.speaker.strip("[]:") or "UNKNOWN"
    return f"{segment.index:03}_{speaker}_{safe_time(segment.start)}"


def export_segments(
    input_audio: Path | str,
    original_srt: str,
    clean_srt: str,
    output_dir: Path | str,
    *,
    run: bool = True,
) -> list[dict[str, str]]:
    """Cut audio clips and write original/clean text files from paired SRT strings."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    original_segments = parse_srt(original_srt)
    clean_segments = parse_srt(clean_srt)
    clean_by_index = {segment.index: segment for segment in clean_segments}
    manifest = []
    for segment in original_segments:
        clean = clean_by_index.get(segment.index, segment)
        base = clip_id(segment)
        audio_path = output / f"{base}.wav"
        text_path = output / f"{base}.txt"
        original_text_path = output / f"{base}_orig.txt"
        command = build_cut_command(
            input_audio,
            audio_path,
            normalize_timestamp(segment.start, decimal="."),
            normalize_timestamp(segment.end, decimal="."),
        )
        if run:
            subprocess.run(command, check=True)
        text_path.write_text(clean.text, encoding="utf-8")
        original_text_path.write_text(segment.text, encoding="utf-8")
        manifest.append(
            {
                "clip_id": base,
                "audio_path": str(audio_path),
                "text_path": str(text_path),
                "text_original_path": str(original_text_path),
                "start": normalize_timestamp(segment.start, decimal="."),
                "end": normalize_timestamp(segment.end, decimal="."),
                "speaker": segment.speaker,
                "text": clean.text,
                "text_original": segment.text,
            }
        )
    return manifest


def export_srt_files(
    input_audio: Path | str,
    original_srt_path: Path | str,
    clean_srt_path: Path | str,
    output_dir: Path | str,
    manifest_path: Path | str,
) -> None:
    """Export clips from paired SRT files and write the manifest TSV."""
    manifest = export_segments(
        input_audio,
        Path(original_srt_path).read_text(encoding="utf-8-sig"),
        Path(clean_srt_path).read_text(encoding="utf-8-sig"),
        output_dir,
    )
    write_tsv(manifest_path, manifest, MANIFEST_COLUMNS)
