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
    input_path = Path(input_audio)
    output_path = Path(output_audio)
    if output_path.suffix.lower() != ".wav":
        raise ValueError("Corpus audio clips must be written as .wav files.")
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        start,
        "-to",
        end,
        "-i",
        str(input_path),
        "-map",
        "0:a",
    ]
    if input_path.suffix.lower() == output_path.suffix.lower():
        command.extend(["-c:a", "copy"])
    else:
        command.extend(["-c:a", "pcm_s16le"])
    command.append(str(output_path))
    return command
