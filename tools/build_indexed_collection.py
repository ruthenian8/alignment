"""Build indexed collection layouts from raw audio, index, and transcript files."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
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

AUDIO_EXTENSIONS = {".wav", ".wma", ".mp3", ".m4a"}
DOC_EXTENSIONS = {".doc", ".docx"}


@dataclass(frozen=True)
class SourceRecording:
    """One raw audio/index pair and optional matching transcript."""

    target: str
    audio: Path
    index_doc: Path
    transcript: Path | None


def normalized_stem(path: Path) -> str:
    """Return a case-insensitive key with transcript suffixes removed."""
    stem = path.stem.strip()
    if stem.lower().endswith("_txt"):
        stem = stem[:-4]
    return stem.strip().rstrip("_").lower()


def transcript_key_map(root: Path) -> dict[str, Path]:
    """Map normalized transcript source keys to TXT files."""
    mapping: dict[str, Path] = {}
    for path in sorted(root.rglob("*.txt")):
        key = normalized_stem(path)
        if key and key not in mapping:
            mapping[key] = path
    return mapping


def read_sources(
    audio_root: Path, transcript_root: Path, prefix: str
) -> tuple[list[SourceRecording], list[dict[str, str]]]:
    """Find raw audio/index/transcript sources in a collection folder."""
    docs = {
        normalized_stem(path): path
        for path in sorted(audio_root.rglob("*"))
        if path.suffix.lower() in DOC_EXTENSIONS
    }
    transcripts = transcript_key_map(transcript_root)
    rows: list[SourceRecording] = []
    manifest: list[dict[str, str]] = []
    audio_paths = [path for path in sorted(audio_root.rglob("*")) if path.suffix.lower() in AUDIO_EXTENSIONS]
    for audio in audio_paths:
        key = normalized_stem(audio)
        doc = docs.get(key)
        transcript = transcripts.get(key)
        if not doc:
            manifest.append(
                {
                    "recording": "",
                    "status": "missing_index",
                    "path": audio.as_posix(),
                    "detail": "no matching index DOC/DOCX",
                }
            )
            continue
        target = f"{prefix}_{len(rows) + 1:03d}"
        if not transcript:
            manifest.append(
                {
                    "recording": target,
                    "status": "missing_transcript",
                    "path": audio.as_posix(),
                    "detail": "no matching transcript TXT",
                }
            )
            continue
        rows.append(SourceRecording(target, audio, doc, transcript))
    return rows, manifest


def export_doc_to_txt(source: Path, output_dir: Path) -> Path:
    """Export one DOC/DOCX index file to TXT using LibreOffice."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{source.stem}.txt"
    if output.exists():
        output.unlink()
    subprocess.run(
        [
            "libreoffice",
            "-env:UserInstallation=file:///tmp/libreoffice-indexed-collection",
            "--headless",
            "--convert-to",
            "txt",
            "--outdir",
            output_dir.as_posix(),
            source.as_posix(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def convert_audio_to_wav(source: Path, target: Path) -> None:
    """Convert raw source audio to mono PCM WAV."""
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            source.as_posix(),
            "-map",
            "0:a",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            target.as_posix(),
        ],
        check=True,
    )


def copy_inputs(recording: SourceRecording, output_root: Path) -> tuple[Path, Path, Path]:
    """Write indexed audio, index, and transcript files under the build layout."""
    audio_output = output_root / "audio" / f"{recording.target}.wav"
    index_output = output_root / "indices" / f"{recording.target}.txt"
    transcript_output = output_root / "transcripts" / f"{recording.target}.txt"

    converted_index = export_doc_to_txt(recording.index_doc, output_root / "_converted_indices")
    convert_audio_to_wav(recording.audio, audio_output)
    index_output.parent.mkdir(parents=True, exist_ok=True)
    transcript_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(converted_index, index_output)
    shutil.copyfile(recording.transcript, transcript_output)
    return audio_output, index_output, transcript_output


def write_tables(
    recording: SourceRecording, index_path: Path, transcript_path: Path, output_root: Path
) -> list[dict[str, str]]:
    """Write parsed, joined, and reordered TSV tables."""
    index_rows = parse_index_file(index_path, audio_stem=recording.target)
    transcript_rows = parse_transcript_file(transcript_path)
    joined_rows = join_rows(index_rows, transcript_rows)
    reordered_rows = reorder_rows(joined_rows)

    write_tsv(output_root / "index_tables" / f"{recording.target}.tsv", index_rows, INDEX_COLUMNS)
    write_tsv(
        output_root / "transcript_tables" / f"{recording.target}.tsv",
        transcript_rows,
        TRANSCRIPT_COLUMNS,
    )
    write_tsv(output_root / "joined" / f"{recording.target}.joined.tsv", joined_rows, JOINED_COLUMNS)
    write_tsv(output_root / "joined" / f"{recording.target}.reordered.tsv", reordered_rows, JOINED_COLUMNS)
    return index_rows


def seconds_to_timestamp(seconds: float) -> str:
    """Format seconds as an SRT-style timestamp accepted by ffmpeg."""
    millis = round(seconds * 1000)
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def cut_index_audio(
    recording: SourceRecording, audio_path: Path, index_rows: list[dict[str, str]], output_root: Path
) -> list[dict[str, str]]:
    """Cut one indexed source recording into index-defined WAV chunks."""
    duration = ffprobe_duration(audio_path)
    output_dir = output_root / "cut_audio" / recording.target
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_rows: list[dict[str, str]] = []

    for index, row in enumerate(index_rows):
        start = row["start"]
        end = (
            index_rows[index + 1]["start"] if index + 1 < len(index_rows) else seconds_to_timestamp(duration)
        )
        output_path = output_dir / row["name"]
        if timestamp_to_ms(end) <= timestamp_to_ms(start):
            clip_rows.append(
                {**row, "clip_path": "", "end": end, "cut_status": "skipped_non_positive_duration"}
            )
            continue
        subprocess.run(build_cut_command(audio_path, output_path, start, end), check=True)
        clip_rows.append({**row, "clip_path": str(output_path), "end": end, "cut_status": "ok"})

    with (output_dir / f"{recording.target}.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [*INDEX_COLUMNS, "end", "clip_path", "cut_status"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clip_rows)
    return clip_rows


def verify_clips(recording: SourceRecording, clip_rows: list[dict[str, str]]) -> list[dict[str, str]]:
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
        duration = ffprobe_duration(clip_path)
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
    recording: SourceRecording, audio_path: Path, clip_rows: list[dict[str, str]]
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


def build_collection(audio_root: Path, transcript_root: Path, output_root: Path, prefix: str) -> None:
    """Build one collection layout under the requested output root."""
    sources, manifest_rows = read_sources(audio_root, transcript_root, prefix)
    verification_rows: list[dict[str, str]] = []
    verification_summary_rows: list[dict[str, str]] = []
    for recording in sources:
        print(recording.target, flush=True)
        audio_path, index_path, transcript_path = copy_inputs(recording, output_root)
        index_rows = write_tables(recording, index_path, transcript_path, output_root)
        clip_rows = cut_index_audio(recording, audio_path, index_rows, output_root)
        verification_rows.extend(verify_clips(recording, clip_rows))
        verification_summary_rows.append(summarize_verification(recording, audio_path, clip_rows))

    write_manifest(output_root / "manifest.csv", manifest_rows, ["recording", "status", "path", "detail"])
    write_manifest(
        output_root / "verification.csv",
        verification_rows,
        ["recording", "clip", "status", "duration", "detail"],
    )
    write_manifest(
        output_root / "verification_summary.csv",
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
    shutil.rmtree(output_root / "_converted_indices", ignore_errors=True)


def main() -> None:
    """Run the indexed collection builder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", help="Collection prefix, for example uht.")
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--transcript-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    collection = args.collection.lower()
    build_collection(
        args.audio_root or ROOT / "raw_audio_plus_indices" / collection,
        args.transcript_root or ROOT / "raw_transcript" / collection,
        args.output_root or ROOT / "build" / collection,
        collection,
    )


if __name__ == "__main__":
    main()
