"""Audio cutting and splitting utilities using ffmpeg/ffprobe."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

from alignment.index_parser import IndexRow

logger = logging.getLogger(__name__)


def cut_audio(
    input_path: Path,
    start: str,
    end: Optional[str],
    output_path: Path,
) -> None:
    """Cut a segment from an audio file using ffmpeg.

    Args:
        input_path: Source audio file path.
        start: Start timestamp (e.g. ``00:00:05.000``).
        end: End timestamp, or ``None`` to cut to end of file.
        output_path: Destination file path.
    """
    cmd = ["ffmpeg", "-y", "-ss", start, "-i", str(input_path)]
    if end is not None:
        cmd.extend(["-to", end])
    cmd.extend(["-c", "copy", "-avoid_negative_ts", "1", str(output_path)])
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def get_audio_duration(path: Path) -> float:
    """Return the duration of an audio file in seconds using ffprobe.

    Args:
        path: Path to the audio file.

    Returns:
        Duration in seconds as a float.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    for stream in info.get("streams", []):
        duration = stream.get("duration")
        if duration is not None:
            return float(duration)
    return 0.0


def split_audio_by_index(
    audio_path: Path,
    rows: List[IndexRow],
    output_dir: Path,
) -> None:
    """Split an audio file into segments based on index rows.

    Each row's ``start`` timestamp defines the beginning of a segment;
    the next row's ``start`` defines the end (last segment runs to EOF).

    Args:
        audio_path: Source audio file.
        rows: List of IndexRow objects defining segment boundaries.
        output_dir: Directory to write output segment files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    starts = [r.start for r in rows]
    names = [r.name for r in rows]
    for i, (name, start) in enumerate(zip(names, starts)):
        end = starts[i + 1] if i < len(starts) - 1 else None
        output_path = output_dir / name
        logger.info("Cutting %s -> %s", name, output_path)
        cut_audio(audio_path, start, end, output_path)


def concat_continuations(rows: List[IndexRow], work_dir: Path) -> List[IndexRow]:
    """Concatenate continuation segments into their predecessor files.

    Rows with a non-empty ``prev`` pointer are merged (via ffmpeg concat)
    into the segment at that index. The continuation rows are then removed
    from the returned list.

    Args:
        rows: List of IndexRow objects (with ``prev`` links set).
        work_dir: Directory containing the split audio files.

    Returns:
        Cleaned list of IndexRow objects with continuations removed.
    """
    rows_by_idx = {i: r for i, r in enumerate(rows)}
    temp_list = work_dir / "concat_list.txt"

    for i in sorted(rows_by_idx.keys(), reverse=True):
        row = rows_by_idx[i]
        if not row.prev:
            continue
        try:
            prev_idx = int(row.prev)
        except ValueError:
            continue
        prev_row = rows_by_idx.get(prev_idx)
        if prev_row is None:
            continue

        prev_path = work_dir / prev_row.name
        curr_path = work_dir / row.name
        if not prev_path.exists() or not curr_path.exists():
            logger.warning("Missing files for concat: %s + %s", prev_row.name, row.name)
            continue

        with open(temp_list, "w", encoding="utf-8") as f:
            f.write(f"file '{prev_path.resolve()}'\n")
            f.write(f"file '{curr_path.resolve()}'\n")

        temp_out = work_dir / f"temp_{prev_row.name}"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(temp_list),
            "-c",
            "copy",
            str(temp_out),
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            temp_out.replace(prev_path)
        except subprocess.CalledProcessError:
            logger.error("ffmpeg failed to concat %s", row.name)

    if temp_list.exists():
        temp_list.unlink()

    clean_rows = [r for r in rows if not r.prev]
    valid_names = {r.name for r in clean_rows}
    suffix = rows[0].name.rsplit(".", 1)[-1] if rows else "wav"
    for f in work_dir.iterdir():
        if f.suffix == f".{suffix}" and f.name not in valid_names:
            f.unlink()

    return clean_rows
