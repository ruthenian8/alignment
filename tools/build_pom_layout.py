"""Build a local processing layout for the indexed ``pom`` collection."""

# ruff: noqa: E402

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alignment.audio import build_cut_command, ffprobe_duration
from alignment.index_parser import parse_index_file
from alignment.io import INDEX_COLUMNS, JOINED_COLUMNS, TRANSCRIPT_COLUMNS, write_tsv
from alignment.join import join_rows
from alignment.reorder import reorder_rows
from alignment.srt import timestamp_to_ms
from alignment.transcript_parser import parse_transcript_file

RAW_AUDIO_INDEX_ROOT = ROOT / "raw_audio_plus_indices" / "pom"
RAW_TRANSCRIPT_ROOT = ROOT / "raw_transcript" / "pom"
OUTPUT_ROOT = ROOT / "build" / "pom"


@dataclass(frozen=True)
class Recording:
    """One indexed POM recording."""

    target: str


def read_recordings() -> list[Recording]:
    """Read indexed POM recordings that have audio, index, and transcript files."""
    recordings: list[Recording] = []
    for audio_path in sorted(RAW_AUDIO_INDEX_ROOT.glob("pom_*.wav")):
        target = audio_path.stem
        if (RAW_AUDIO_INDEX_ROOT / f"{target}.txt").exists() and (
            RAW_TRANSCRIPT_ROOT / f"{target}.txt"
        ).exists():
            recordings.append(Recording(target=target))
    return recordings


def copy_inputs(recording: Recording) -> tuple[Path, Path, Path]:
    """Copy raw indexed inputs into the build layout."""
    audio_source = RAW_AUDIO_INDEX_ROOT / f"{recording.target}.wav"
    index_source = RAW_AUDIO_INDEX_ROOT / f"{recording.target}.txt"
    transcript_source = RAW_TRANSCRIPT_ROOT / f"{recording.target}.txt"

    audio_output = OUTPUT_ROOT / "audio" / audio_source.name
    index_output = OUTPUT_ROOT / "indices" / index_source.name
    transcript_output = OUTPUT_ROOT / "transcripts" / transcript_source.name

    for output in (audio_output, index_output, transcript_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    if not audio_output.exists():
        shutil.copy2(audio_source, audio_output)
    shutil.copy2(index_source, index_output)
    shutil.copy2(transcript_source, transcript_output)
    return audio_output, index_output, transcript_output


def write_tables(recording: Recording, index_path: Path, transcript_path: Path) -> list[dict[str, str]]:
    """Write parsed, joined, and reordered TSV tables."""
    index_rows = parse_index_file(index_path, audio_stem=recording.target)
    transcript_rows = parse_transcript_file(transcript_path)
    joined_rows = join_rows(index_rows, transcript_rows)
    reordered_rows = reorder_rows(joined_rows)

    write_tsv(OUTPUT_ROOT / "index_tables" / f"{recording.target}.tsv", index_rows, INDEX_COLUMNS)
    write_tsv(
        OUTPUT_ROOT / "transcript_tables" / f"{recording.target}.tsv",
        transcript_rows,
        TRANSCRIPT_COLUMNS,
    )
    write_tsv(OUTPUT_ROOT / "joined" / f"{recording.target}.joined.tsv", joined_rows, JOINED_COLUMNS)
    write_tsv(
        OUTPUT_ROOT / "joined" / f"{recording.target}.reordered.tsv",
        reordered_rows,
        JOINED_COLUMNS,
    )
    return index_rows


def seconds_to_timestamp(seconds: float) -> str:
    """Format seconds as an SRT-style timestamp accepted by ffmpeg."""
    millis = round(seconds * 1000)
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


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
        if timestamp_to_ms(end) <= timestamp_to_ms(start):
            clip_rows.append(
                {
                    **row,
                    "clip_path": "",
                    "end": end,
                    "cut_status": "skipped_non_positive_duration",
                }
            )
            continue
        if not output_path.exists():
            subprocess.run(build_cut_command(audio_path, output_path, start, end), check=True)
        clip_rows.append({**row, "clip_path": str(output_path), "end": end, "cut_status": "ok"})

    with (output_dir / f"{recording.target}.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [*INDEX_COLUMNS, "end", "clip_path", "cut_status"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clip_rows)
    return clip_rows


def verify_clips(recording: Recording, clip_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Verify that all expected clips exist and have positive duration."""
    rows: list[dict[str, str]] = []
    for row in clip_rows:
        if not row["clip_path"]:
            rows.append(
                {
                    "recording": recording.target,
                    "clip": row["name"],
                    "status": row.get("cut_status", "skipped"),
                    "duration": "",
                    "detail": f"{row['start']}..{row['end']}",
                }
            )
            continue
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
    clip_duration = sum(ffprobe_duration(row["clip_path"]) for row in clip_rows if row["clip_path"])
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
    """Build ``build/pom`` from indexed raw POM files."""
    manifest_rows = [
        {
            "recording": "pom_011",
            "status": "missing_transcript",
            "path": "raw_audio_plus_indices/pom/Jarenga/Tmp_ne_zapisannoe.wma",
            "detail": "no transcript found during POM indexing",
        }
    ]
    verification_rows: list[dict[str, str]] = []
    verification_summary_rows: list[dict[str, str]] = []
    for recording in read_recordings():
        print(recording.target, flush=True)
        audio_path, index_path, transcript_path = copy_inputs(recording)
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
