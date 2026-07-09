"""Prepare indexed POM audio, index, and transcript files."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

AUDIO_EXTENSIONS = {".wma", ".wav", ".mp3", ".m4a"}
DOC_EXTENSIONS = {".doc", ".docx"}


def normalized_stem(path: Path) -> str:
    """Return a case-insensitive stem key with transcript suffixes removed."""
    stem = path.stem.strip()
    if stem.lower().endswith("_txt"):
        stem = stem[:-4]
    if stem.lower().endswith(".txt"):
        stem = stem[:-4]
    return stem.strip().rstrip("_").lower()


def export_doc_to_txt(path: Path, *, dry_run: bool) -> Path:
    """Export one DOC/DOCX file to TXT with LibreOffice."""
    output = path.with_suffix(".txt")
    if dry_run or output.exists():
        return output
    command = [
        "libreoffice",
        "-env:UserInstallation=file:///tmp/libreoffice-pom-layout",
        "--headless",
        "--convert-to",
        "txt",
        "--outdir",
        path.parent.as_posix(),
        path.as_posix(),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output


def transcript_key_map(transcript_root: Path) -> dict[str, Path]:
    """Map normalized transcript source keys to TXT files."""
    mapping: dict[str, Path] = {}
    paths = [
        *transcript_root.glob("*.txt"),
        *(path.with_suffix(".txt") for path in transcript_root.glob("*.doc")),
        *(path.with_suffix(".txt") for path in transcript_root.glob("*.docx")),
    ]
    paths = sorted(set(paths))
    for path in paths:
        key = normalized_stem(path)
        if key and key not in mapping:
            mapping[key] = path
    for path in paths:
        for part in normalized_stem(path).split("_"):
            if part and part not in mapping:
                mapping[part] = path
    return mapping


def source_pairs(audio_root: Path) -> list[tuple[Path, Path]]:
    """Return audio/index pairs from village subfolders."""
    pairs: list[tuple[Path, Path]] = []
    for audio_path in sorted(audio_root.glob("*/*")):
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        key = normalized_stem(audio_path)
        docs = [
            path
            for path in sorted(audio_path.parent.glob("*"))
            if path.suffix.lower() in DOC_EXTENSIONS and normalized_stem(path) == key
        ]
        if docs:
            pairs.append((audio_path, docs[0]))
    return pairs


def ffmpeg_convert_to_wav(source: Path, target: Path, *, dry_run: bool) -> None:
    """Convert source audio to mono WAV."""
    if dry_run or target.exists():
        return
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source.as_posix(),
        "-ac",
        "1",
        target.as_posix(),
    ]
    subprocess.run(command, check=True)


def copy_text(source: Path, target: Path, *, dry_run: bool) -> None:
    """Copy one UTF-8-ish text file to the indexed target path."""
    if dry_run:
        return
    shutil.copyfile(source, target)


def prepare_layout(audio_root: Path, transcript_root: Path, *, dry_run: bool = False) -> list[dict[str, str]]:
    """Create indexed POM WAV, index TXT, and transcript TXT files."""
    docs = [
        *audio_root.rglob("*.doc"),
        *audio_root.rglob("*.docx"),
        *transcript_root.glob("*.doc"),
        *transcript_root.glob("*.docx"),
    ]
    for doc in sorted(docs):
        export_doc_to_txt(doc, dry_run=dry_run)

    transcripts = transcript_key_map(transcript_root)
    manifest: list[dict[str, str]] = []
    for number, (audio_path, doc_path) in enumerate(source_pairs(audio_root), start=1):
        indexed = f"pom_{number:03d}"
        index_txt = doc_path.with_suffix(".txt")
        transcript_path = transcripts.get(normalized_stem(audio_path))
        audio_target = audio_root / f"{indexed}.wav"
        index_target = audio_root / f"{indexed}.txt"
        transcript_target = transcript_root / f"{indexed}.txt"
        status = "ok" if transcript_path else "missing_transcript"
        if transcript_path:
            print(f"{indexed}\t{audio_path}", flush=True)
            ffmpeg_convert_to_wav(audio_path, audio_target, dry_run=dry_run)
            copy_text(index_txt, index_target, dry_run=dry_run)
            copy_text(transcript_path, transcript_target, dry_run=dry_run)
        manifest.append(
            {
                "id": indexed,
                "status": status,
                "source_audio": audio_path.as_posix(),
                "source_index": doc_path.as_posix(),
                "source_transcript": transcript_path.as_posix() if transcript_path else "",
                "indexed_audio": audio_target.as_posix() if transcript_path else "",
                "indexed_index": index_target.as_posix() if transcript_path else "",
                "indexed_transcript": transcript_target.as_posix() if transcript_path else "",
            }
        )
    return manifest


def write_manifest(path: Path, rows: list[dict[str, str]], *, dry_run: bool) -> None:
    """Write the layout manifest."""
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "id",
            "status",
            "source_audio",
            "source_index",
            "source_transcript",
            "indexed_audio",
            "indexed_index",
            "indexed_transcript",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the POM layout preparation utility."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, default=Path("raw_audio_plus_indices/pom"))
    parser.add_argument("--transcript-root", type=Path, default=Path("raw_transcript/pom"))
    parser.add_argument("--manifest", type=Path, default=Path("build/pom_layout_manifest.tsv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = prepare_layout(args.audio_root, args.transcript_root, dry_run=args.dry_run)
    write_manifest(args.manifest, rows, dry_run=args.dry_run)
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    for status, count in sorted(statuses.items()):
        print(f"{status}\t{count}")


if __name__ == "__main__":
    main()
