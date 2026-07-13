"""Build an HF-style local layout for the ``and`` collection.

The script reads only ``raw_audio_plus_indices/and`` and ``raw_transcript/and``,
then writes derived files under ``build/and``.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alignment.audio import build_cut_command, ffprobe_duration  # noqa: E402
from alignment.index_parser import parse_index_file  # noqa: E402
from alignment.io import INDEX_COLUMNS, TRANSCRIPT_COLUMNS, write_tsv  # noqa: E402
from alignment.srt import timestamp_to_ms  # noqa: E402
from alignment.transcript_parser import parse_transcript_file  # noqa: E402

MAPPING_PATH = ROOT / "mapping2.txt"
RAW_AUDIO_INDEX_ROOT = ROOT / "raw_audio_plus_indices" / "and"
RAW_TRANSCRIPT_ROOT = ROOT / "raw_transcript" / "and"
OUTPUT_ROOT = ROOT / "build" / "and"


@dataclass(frozen=True)
class Recording:
    target: str
    source_stem: str


def read_and_mapping(path: Path) -> list[Recording]:
    """Read ``and`` rows from mapping2.txt."""
    recordings: list[Recording] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=2)
            if len(parts) == 3 and parts[0] == "and":
                recordings.append(Recording(target=parts[1], source_stem=parts[2]))
    return recordings


def find_transcript(recording: Recording) -> Path | None:
    """Find the best matching raw transcript for a mapped recording."""
    direct = RAW_TRANSCRIPT_ROOT / f"{recording.target}.txt"
    if direct.exists():
        return direct

    source_name = f"{recording.source_stem.rstrip('_')}_txt.txt"
    for candidate in RAW_TRANSCRIPT_ROOT.glob("*.txt"):
        if candidate.name.casefold() == source_name.casefold():
            return candidate
    return None


def copy_inputs(recording: Recording, manifest_rows: list[dict[str, str]]) -> tuple[Path, Path, Path | None]:
    """Copy raw audio, index, and transcript inputs into the output layout."""
    audio_source = RAW_AUDIO_INDEX_ROOT / f"{recording.target}.wav"
    index_source = RAW_AUDIO_INDEX_ROOT / f"{recording.target}.txt"
    transcript_source = find_transcript(recording)

    audio_output = OUTPUT_ROOT / "audio" / f"{recording.target}.wav"
    index_output = OUTPUT_ROOT / "indices" / f"{recording.target}.txt"
    transcript_output = OUTPUT_ROOT / "transcripts" / f"{recording.target}.txt"

    audio_output.parent.mkdir(parents=True, exist_ok=True)
    index_output.parent.mkdir(parents=True, exist_ok=True)
    transcript_output.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(audio_source, audio_output)
    shutil.copy2(index_source, index_output)
    if transcript_source:
        shutil.copy2(transcript_source, transcript_output)
    else:
        transcript_output = None
        manifest_rows.append(
            {
                "recording": recording.target,
                "status": "missing_transcript",
                "path": "",
                "detail": f"no transcript found for {recording.source_stem}",
            }
        )

    return audio_output, index_output, transcript_output


def write_tables(
    recording: Recording, index_path: Path, transcript_path: Path | None
) -> list[dict[str, str]]:
    """Write parsed index and transcript TSV tables."""
    index_rows = parse_index_file(index_path, audio_stem=recording.target)
    write_tsv(OUTPUT_ROOT / "index_tables" / f"{recording.target}.tsv", index_rows, INDEX_COLUMNS)

    if transcript_path:
        transcript_rows = parse_transcript_file(transcript_path)
        write_tsv(
            OUTPUT_ROOT / "transcript_tables" / f"{recording.target}.tsv",
            transcript_rows,
            TRANSCRIPT_COLUMNS,
        )
    return index_rows


def cut_index_audio(
    recording: Recording, audio_path: Path, index_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Cut one source recording into index-defined WAV chunks."""
    duration = ffprobe_duration(audio_path)
    clip_rows: list[dict[str, str]] = []
    output_dir = OUTPUT_ROOT / "cut_audio" / recording.target
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(index_rows):
        start = row["start"]
        end = (
            index_rows[index + 1]["start"] if index + 1 < len(index_rows) else seconds_to_timestamp(duration)
        )
        output_path = output_dir / row["name"]
        command = build_cut_command(audio_path, output_path, start, end)
        subprocess.run(command, check=True)
        clip_rows.append({**row, "clip_path": str(output_path), "end": end})

    with (output_dir / f"{recording.target}.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [*INDEX_COLUMNS, "end", "clip_path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clip_rows)
    return clip_rows


def seconds_to_timestamp(seconds: float) -> str:
    """Format seconds as an SRT-style timestamp accepted by ffmpeg."""
    millis = round(seconds * 1000)
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def verify_clips(recording: Recording, clip_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Verify that all expected clips exist and have positive duration."""
    rows: list[dict[str, str]] = []
    for row in clip_rows:
        clip_path = Path(row["clip_path"])
        try:
            duration = ffprobe_duration(clip_path)
        except subprocess.CalledProcessError as exc:
            rows.append(
                {
                    "recording": recording.target,
                    "clip": clip_path.name,
                    "status": "ffprobe_failed",
                    "duration": "",
                    "detail": str(exc.returncode),
                }
            )
            continue
        rows.append(
            {
                "recording": recording.target,
                "clip": clip_path.name,
                "status": "ok" if clip_path.exists() and duration > 0 else "bad_duration",
                "duration": f"{duration:.3f}",
                "detail": "",
            }
        )
    return rows


def summarize_verification(
    recording: Recording, audio_path: Path, clip_rows: list[dict[str, str]]
) -> dict[str, str]:
    """Summarize source duration against the indexed span covered by clips."""
    source_duration = ffprobe_duration(audio_path)
    first_start = timestamp_to_ms(clip_rows[0]["start"]) / 1000 if clip_rows else 0.0
    expected_covered = max(0.0, source_duration - first_start)
    clip_duration = sum(ffprobe_duration(row["clip_path"]) for row in clip_rows)
    return {
        "recording": recording.target,
        "clips": str(len(clip_rows)),
        "source_duration": f"{source_duration:.3f}",
        "first_index_start": f"{first_start:.3f}",
        "expected_covered_duration": f"{expected_covered:.3f}",
        "sum_clip_duration": f"{clip_duration:.3f}",
        "delta": f"{clip_duration - expected_covered:.3f}",
    }


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write CSV manifest rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build ``build/and`` from raw ``and`` files."""
    manifest_rows: list[dict[str, str]] = []
    verification_rows: list[dict[str, str]] = []
    verification_summary_rows: list[dict[str, str]] = []
    for recording in read_and_mapping(MAPPING_PATH):
        audio_path, index_path, transcript_path = copy_inputs(recording, manifest_rows)
        index_rows = write_tables(recording, index_path, transcript_path)
        clip_rows = cut_index_audio(recording, audio_path, index_rows)
        verification_rows.extend(verify_clips(recording, clip_rows))
        verification_summary_rows.append(summarize_verification(recording, audio_path, clip_rows))

    write_manifest(
        OUTPUT_ROOT / "manifest.csv",
        manifest_rows,
        ["recording", "status", "path", "detail"],
    )
    write_manifest(
        OUTPUT_ROOT / "verification.csv",
        verification_rows,
        ["recording", "clip", "status", "duration", "detail"],
    )
    write_manifest(
        OUTPUT_ROOT / "verification_summary.csv",
        verification_summary_rows,
        [
            "recording",
            "clips",
            "source_duration",
            "first_index_start",
            "expected_covered_duration",
            "sum_clip_duration",
            "delta",
        ],
    )


if __name__ == "__main__":
    main()
