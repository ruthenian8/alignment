"""Audio metadata and ffmpeg command helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def ffprobe_duration(path: Path | str) -> float:
    """Return audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def build_cut_command(input_audio: Path | str, output_audio: Path | str, start: str, end: str) -> list[str]:
    """Build the ffmpeg command used to cut one audio clip."""
    return [
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
        str(output_audio),
    ]
