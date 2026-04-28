"""Corpus clip export: cut audio segments and write manifest."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class ExportSegment:
    """A segment ready for corpus export.

    Attributes:
        index: Segment index (used in naming).
        start: Start timestamp (``HH:MM:SS,mmm``).
        end: End timestamp (``HH:MM:SS,mmm``).
        speaker: Speaker tag (or empty string).
        text: Original transcript text.
        text_clean: Cleaned/normalised text for the corpus.
    """

    index: int
    start: str
    end: str
    speaker: str
    text: str
    text_clean: str


def _safe_time(t: str) -> str:
    """Convert timestamp to a filename-safe string."""
    return t.replace(":", "-").replace(",", "-").replace(".", "-")


def make_segment_basename(seg: ExportSegment) -> str:
    """Return the base filename (without extension) for a segment.

    Format: ``{index:03d}_{speaker}_{safe_start}``

    Args:
        seg: Export segment.

    Returns:
        Base filename string.
    """
    speaker = seg.speaker.strip("[]:")
    return f"{seg.index:03d}_{speaker}_{_safe_time(seg.start)}"


def cut_audio_segments(
    input_audio: Path,
    segments: List[ExportSegment],
    output_dir: Path,
) -> List[Path]:
    """Cut audio clips for each segment using ffmpeg.

    For each segment, writes a ``.wav`` file to ``output_dir``. Also writes
    ``{base}_orig.txt`` with the original text and ``{base}.txt`` with the
    clean text.

    Args:
        input_audio: Path to the source audio file.
        segments: List of ExportSegment objects.
        output_dir: Directory to write output files.

    Returns:
        List of paths to the written audio files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_paths: List[Path] = []

    for seg in segments:
        base = make_segment_basename(seg)
        audio_path = output_dir / f"{base}.wav"
        text_path = output_dir / f"{base}_orig.txt"
        text_clean_path = output_dir / f"{base}.txt"

        start = seg.start.replace(",", ".")
        end = seg.end.replace(",", ".")

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            start,
            "-to",
            end,
            "-i",
            str(input_audio),
            "-map",
            "0:a",
            "-c:a",
            "copy",
            str(audio_path),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        text_path.write_text(seg.text, encoding="utf-8")
        text_clean_path.write_text(seg.text_clean, encoding="utf-8")
        audio_paths.append(audio_path)

    return audio_paths


def write_manifest(
    segments: List[ExportSegment],
    audio_paths: List[Path],
    output_path: Path,
) -> None:
    """Write a TSV manifest file for the exported corpus clips.

    Args:
        segments: List of ExportSegment objects.
        audio_paths: Corresponding list of audio file paths.
        output_path: Path to write the manifest TSV.
    """
    rows = []
    for seg, audio_path in zip(segments, audio_paths):
        rows.append(
            {
                "audio_path": str(audio_path),
                "text": seg.text,
                "text_clean": seg.text_clean,
                "speaker": seg.speaker,
                "start": seg.start,
                "end": seg.end,
                "index": seg.index,
            }
        )
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
