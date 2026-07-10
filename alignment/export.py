"""Export aligned SRT rows as corpus clips, text files, and manifests."""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
from pathlib import Path

from .audio import build_cut_command
from .io import MANIFEST_COLUMNS, write_tsv
from .srt import SrtSegment, normalize_timestamp, parse_srt

STRESS_MARK_RE = re.compile(r"[\\_\u0300\u0301]")


def safe_time(timestamp: str) -> str:
    """Make a timestamp safe for deterministic filenames."""
    return normalize_timestamp(timestamp, decimal=".").replace(":", "-").replace(".", "-")


def clip_id(segment: SrtSegment) -> str:
    """Build a stable clip identifier from SRT index, speaker, and start time."""
    speaker = clean_speaker_code(segment.speaker)
    return f"{segment.index:03}_{speaker}_{safe_time(segment.start)}"


def clean_speaker_code(speaker: str) -> str:
    """Return a speaker code suitable for cut-sample filenames."""
    code = speaker.strip().strip("[]:")
    if not code:
        return "UNKNOWN"
    if set(code) == {"?"}:
        return "UNK"
    code = re.sub(r"\s*,\s*", "-", code)
    code = re.sub(r"\s+", "-", code)
    code = re.sub(r"[^0-9A-ZА-ЯЁa-zа-яё?_-]+", "", code)
    return code.replace("?", "UNK") or "UNKNOWN"


def normalize_caption_text(text: str) -> str:
    """Normalize a caption for ASR references while preserving readable punctuation."""
    text = STRESS_MARK_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def speaker_map_by_index(path: Path) -> dict[int, str]:
    """Read transcript-derived speakers from a speaker map."""
    if not path.exists():
        return {}
    speakers = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            matched = str(row.get("matched", "")).strip().lower() in {"true", "1", "yes", "y"}
            if not matched:
                continue
            speaker = row.get("transcript_speaker", "").strip()
            if speaker:
                speakers[int(row["srt_index"])] = speaker
    return speakers


def has_matched_blank_speakers(path: Path) -> bool:
    """Return true when a speaker map has matched rows without transcript speakers."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            matched = str(row.get("matched", "")).strip().lower() in {"true", "1", "yes", "y"}
            if matched and not row.get("transcript_speaker", "").strip():
                return True
    return False


def speaker_map_indices(path: Path) -> set[int]:
    """Return SRT indices represented in a speaker map."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {int(row["srt_index"]) for row in csv.DictReader(file)}


def apply_speaker_map(segments: list[SrtSegment], speakers: dict[int, str]) -> list[SrtSegment]:
    """Return SRT segments with transcript-derived speakers applied where available."""
    if not speakers:
        return segments
    return [
        SrtSegment(
            segment.index,
            segment.start,
            segment.end,
            speakers.get(segment.index, segment.speaker),
            segment.text,
        )
        for segment in segments
    ]


def _export_srt_segments(
    input_audio: Path | str,
    segments: list[SrtSegment],
    output_dir: Path | str,
    *,
    text_by_index: dict[int, str] | None = None,
    run: bool = True,
) -> list[dict[str, str]]:
    """Cut audio and write paired normalized/original text files for SRT segments."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for segment in segments:
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
        text = text_by_index.get(segment.index, segment.text) if text_by_index is not None else segment.text
        text_path.write_text(text, encoding="utf-8")
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
                "text": text,
                "text_original": segment.text,
            }
        )
    return manifest


def export_segments(
    input_audio: Path | str,
    original_srt: str,
    clean_srt: str,
    output_dir: Path | str,
    *,
    run: bool = True,
) -> list[dict[str, str]]:
    """Cut audio clips and write original/clean text files from paired SRT strings."""
    original_segments = parse_srt(original_srt)
    clean_segments = parse_srt(clean_srt)
    clean_text_by_index = {segment.index: segment.text for segment in clean_segments}
    return _export_srt_segments(
        input_audio,
        original_segments,
        output_dir,
        text_by_index=clean_text_by_index,
        run=run,
    )


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


def export_aligned_srt(
    input_audio: Path | str,
    aligned_srt_path: Path | str,
    output_dir: Path | str,
    *,
    speaker_map_path: Path | str | None = None,
    run: bool = True,
) -> list[dict[str, str]]:
    """Cut one aligned SRT into wav, normalized txt, and original _orig.txt files."""
    segments = parse_srt(Path(aligned_srt_path).read_text(encoding="utf-8-sig"))
    if speaker_map_path is not None:
        segments = apply_speaker_map(segments, speaker_map_by_index(Path(speaker_map_path)))
    clean_text_by_index = {segment.index: normalize_caption_text(segment.text) for segment in segments}
    return _export_srt_segments(
        input_audio,
        segments,
        output_dir,
        text_by_index=clean_text_by_index,
        run=run,
    )


def export_aligned_srt_tree(
    aligned_root: Path | str,
    audio_root: Path | str,
    output_root: Path | str,
    manifest_path: Path | str | None = None,
    *,
    require_diarized_matches: bool = False,
    run: bool = True,
) -> list[dict[str, str]]:
    """Export a tree of ``corpus/aligned/*.aligned.srt`` files like ``cut_samples``."""
    aligned_base = Path(aligned_root)
    audio_base = Path(audio_root)
    output_base = Path(output_root)
    rows = []
    for aligned_srt in sorted(aligned_base.glob("*/aligned/*.aligned.srt")):
        corpus = aligned_srt.parent.parent.name
        chunk = aligned_srt.name.removesuffix(".aligned.srt")
        audio_path = audio_base / corpus / f"{chunk}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing chunk audio for {aligned_srt}: {audio_path}")
        speaker_map = aligned_base / corpus / "speaker_maps" / f"{chunk}.speaker_map.csv"
        if require_diarized_matches:
            if not speaker_map.exists():
                raise ValueError(f"Missing speaker map for diarization guard: {speaker_map}")
            segments = parse_srt(aligned_srt.read_text(encoding="utf-8-sig"))
            missing_indices = {segment.index for segment in segments} - speaker_map_indices(speaker_map)
            if missing_indices:
                missing_text = ", ".join(str(index) for index in sorted(missing_indices))
                raise ValueError(f"Speaker map lacks rows for SRT indices {missing_text}: {speaker_map}")
            if has_matched_blank_speakers(speaker_map):
                raise ValueError(f"Matched segments without transcript speakers in {speaker_map}")
        rows.extend(
            export_aligned_srt(
                audio_path,
                aligned_srt,
                output_base / corpus / chunk,
                speaker_map_path=speaker_map,
                run=run,
            )
        )
        if speaker_map.exists():
            target_dir = output_base / corpus / chunk
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(speaker_map, target_dir / "speaker_map.csv")
    if manifest_path is not None:
        write_tsv(manifest_path, rows, MANIFEST_COLUMNS)
    return rows
