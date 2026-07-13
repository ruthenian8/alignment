"""Normalize and rename raw audio files according to mapping2.txt.

This is a one-time migration helper for ``raw_audio_plus_indices``. It scans
three-letter collection folders, skips ``pez``, and writes normalized WAV copies
beside the original audio files.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav", ".wma"}


@dataclass(frozen=True)
class MappingRow:
    folder: str
    target_stem: str
    source_stem: str


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int | None
    channels: int | None


def read_mapping(path: Path) -> dict[tuple[str, str], MappingRow]:
    """Read mapping rows keyed by ``(folder, source_stem)``."""
    rows: dict[tuple[str, str], MappingRow] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=2)
            if len(parts) != 3:
                raise ValueError(f"Invalid mapping line {line_no}: {raw_line!r}")
            folder, target_stem, source_stem = parts
            rows[(folder, source_stem)] = MappingRow(folder, target_stem, source_stem)
    return rows


def iter_audio_files(root: Path) -> list[Path]:
    """Return audio files in three-letter subfolders except ``pez``."""
    files: list[Path] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or len(folder.name) != 3 or folder.name == "pez":
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                files.append(path)
    return files


def probe_audio(path: Path) -> AudioInfo:
    """Return the first audio stream channel count and sample rate."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        return AudioInfo(sample_rate=None, channels=None)
    stream = streams[0]
    sample_rate = stream.get("sample_rate")
    return AudioInfo(
        sample_rate=int(sample_rate) if sample_rate else None,
        channels=stream.get("channels"),
    )


def probe_audio_or_none(path: Path) -> AudioInfo | None:
    """Probe audio metadata, returning ``None`` for unreadable files."""
    try:
        return probe_audio(path)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def same_audio_shape(source: AudioInfo, destination: AudioInfo) -> bool:
    """Return whether output preserved channel count and sample rate."""
    return source.channels == destination.channels and source.sample_rate == destination.sample_rate


def normalize_audio(source: Path, destination: Path, *, overwrite: bool) -> None:
    """Create a normalized WAV copy without changing channel count or sample rate."""
    audio_info = probe_audio(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-af",
        "loudnorm=I=-16:LRA=11:TP=-1.5",
    ]
    if audio_info.channels:
        command.extend(["-ac", str(audio_info.channels)])
    if audio_info.sample_rate:
        command.extend(["-ar", str(audio_info.sample_rate)])
    command.extend(["-map_metadata", "-1", "-acodec", "pcm_s16le", str(destination)])
    subprocess.run(command, check=True)


def manifest_row(
    *,
    status: str,
    source: Path,
    destination: Path | None,
    folder: str,
    source_stem: str,
    target_stem: str,
    reason: str = "",
) -> dict[str, str]:
    """Build one manifest row."""
    return {
        "status": status,
        "source_path": str(source),
        "destination_path": "" if destination is None else str(destination),
        "folder": folder,
        "source_stem": source_stem,
        "target_stem": target_stem,
        "reason": reason,
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a CSV manifest describing converted, unmatched, and skipped files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["status", "source_path", "destination_path", "folder", "source_stem", "target_stem", "reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def process_one(
    source: Path,
    mapping: dict[tuple[str, str], MappingRow],
    target_keys: set[tuple[str, str]],
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, str] | None:
    """Process one source audio file and return its manifest row."""
    folder = source.parent.name
    if (folder, source.stem) in target_keys and (folder, source.stem) not in mapping:
        return None

    mapping_row = mapping.get((folder, source.stem))
    if mapping_row is None:
        return manifest_row(
            status="unmatched_audio",
            source=source,
            destination=None,
            folder=folder,
            source_stem=source.stem,
            target_stem="",
            reason="no mapping row for folder and source stem",
        )

    destination = source.with_name(f"{mapping_row.target_stem}.wav")
    if destination.exists() and not overwrite:
        source_info = probe_audio_or_none(source)
        destination_info = probe_audio_or_none(destination)
        if destination_info is None:
            return manifest_row(
                status="existing_output_invalid",
                source=source,
                destination=destination,
                folder=folder,
                source_stem=source.stem,
                target_stem=mapping_row.target_stem,
                reason="destination exists but ffprobe cannot read it; pass --overwrite to rebuild",
            )
        if source_info and not same_audio_shape(source_info, destination_info):
            return manifest_row(
                status="existing_output_format_mismatch",
                source=source,
                destination=destination,
                folder=folder,
                source_stem=source.stem,
                target_stem=mapping_row.target_stem,
                reason=(
                    "destination channel count or sample rate differs from source; "
                    f"source={source_info.channels}ch/{source_info.sample_rate}Hz, "
                    f"destination={destination_info.channels}ch/{destination_info.sample_rate}Hz"
                ),
            )
        return manifest_row(
            status="skipped_existing_output",
            source=source,
            destination=destination,
            folder=folder,
            source_stem=source.stem,
            target_stem=mapping_row.target_stem,
            reason="destination exists; pass --overwrite to replace",
        )

    try:
        if not dry_run:
            normalize_audio(source, destination, overwrite=overwrite)
    except subprocess.CalledProcessError as exc:
        return manifest_row(
            status="conversion_failed",
            source=source,
            destination=destination,
            folder=folder,
            source_stem=source.stem,
            target_stem=mapping_row.target_stem,
            reason=f"ffmpeg exited with status {exc.returncode}",
        )

    return manifest_row(
        status="would_convert" if dry_run else "converted",
        source=source,
        destination=destination,
        folder=folder,
        source_stem=source.stem,
        target_stem=mapping_row.target_stem,
    )


def process(
    *,
    root: Path,
    mapping_path: Path,
    manifest_path: Path,
    dry_run: bool,
    overwrite: bool,
    jobs: int,
) -> None:
    """Process mapped audio files and persist a manifest."""
    mapping = read_mapping(mapping_path)
    target_keys = {(row.folder, row.target_stem) for row in mapping.values()}
    sources = iter_audio_files(root)
    manifest_rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        futures = [
            executor.submit(process_one, source, mapping, target_keys, dry_run=dry_run, overwrite=overwrite)
            for source in sources
        ]
        for future in as_completed(futures):
            row = future.result()
            if row is not None:
                manifest_rows.append(row)

    manifest_rows.sort(key=lambda row: row["source_path"])

    write_manifest(manifest_path, manifest_rows)


def main() -> None:
    """Run the one-time raw audio normalization helper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("raw_audio_plus_indices"))
    parser.add_argument("--mapping", type=Path, default=Path("mapping2.txt"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("build/raw_audio_mapping_normalization_manifest.csv"),
    )
    parser.add_argument("--dry-run", action="store_true", help="Only write the manifest; do not run ffmpeg.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing normalized WAV files.")
    parser.add_argument(
        "--jobs", type=int, default=4, help="Number of ffmpeg conversions to run concurrently."
    )
    args = parser.parse_args()

    process(
        root=args.root,
        mapping_path=args.mapping,
        manifest_path=args.manifest,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
