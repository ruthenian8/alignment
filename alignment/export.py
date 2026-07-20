"""Export aligned SRT rows as corpus clips, text files, and manifests."""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .audio import build_cut_command
from .io import MANIFEST_COLUMNS, write_tsv
from .srt import SrtSegment, normalize_timestamp, parse_srt

STRESS_MARK_RE = re.compile(r"[\\_\u0300\u0301]")


@dataclass(frozen=True)
class ExportPlan:
    """One validated clip export operation."""

    segment: SrtSegment
    clip_id: str
    audio_path: Path
    text_path: Path
    original_text_path: Path
    text: str
    command: list[str]

    def manifest_row(self) -> dict[str, str]:
        """Return the manifest row represented by this plan."""
        return {
            "clip_id": self.clip_id,
            "audio_path": str(self.audio_path),
            "text_path": str(self.text_path),
            "text_original_path": str(self.original_text_path),
            "start": normalize_timestamp(self.segment.start, decimal="."),
            "end": normalize_timestamp(self.segment.end, decimal="."),
            "speaker": self.segment.speaker,
            "text": self.text,
            "text_original": self.segment.text,
        }


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


def speaker_map_matched_indices(path: Path) -> set[int]:
    """Return SRT indices marked as aligned to manual transcript text."""
    matched_indices = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            matched = str(row.get("matched", "")).strip().lower() in {"true", "1", "yes", "y"}
            if matched:
                matched_indices.add(int(row["srt_index"]))
    return matched_indices


def quality_failure_chunks(path: Path | str) -> set[str]:
    """Read chunk names that failed a corpus-level quality audit."""
    chunks = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            name = row.get("name", "").strip()
            if name:
                chunks.add(name)
    return chunks


def summary_match_ratios(path: Path) -> dict[str, float]:
    """Read per-chunk alignment match ratios from an align-map summary."""
    ratios = {}
    if not path.exists():
        return ratios
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            try:
                ratio = row.get("match_ratio", "")
                if ratio:
                    ratios[row["name"]] = float(ratio)
                else:
                    segments = int(row.get("segments", "0") or 0)
                    matched = int(row.get("matched_segments", "0") or 0)
                    ratios[row["name"]] = matched / segments if segments else 0.0
            except (TypeError, ValueError):
                ratios[row["name"]] = 0.0
    return ratios


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


def _sample(items: list[str], limit: int = 5) -> str:
    sample = ", ".join(items[:limit])
    if len(items) > limit:
        sample += f", +{len(items) - limit} more"
    return sample


def _segments_by_unique_index(segments: list[SrtSegment], label: str) -> dict[int, SrtSegment]:
    """Index segments while rejecting duplicate SRT indices."""
    indexed: dict[int, SrtSegment] = {}
    duplicates: set[int] = set()
    for segment in segments:
        if segment.index in indexed:
            duplicates.add(segment.index)
        indexed[segment.index] = segment
    if duplicates:
        values = [str(index) for index in sorted(duplicates)]
        raise ValueError(f"duplicate {label} SRT indices: {_sample(values)}")
    return indexed


def _plan_srt_segments(
    input_audio: Path | str,
    segments: list[SrtSegment],
    output_dir: Path | str,
    *,
    text_by_index: dict[int, str] | None = None,
) -> list[ExportPlan]:
    """Build side-effect-free export plans for SRT segments."""
    output = Path(output_dir)
    plans = []
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
        text = text_by_index.get(segment.index, segment.text) if text_by_index is not None else segment.text
        plans.append(
            ExportPlan(
                segment=segment,
                clip_id=base,
                audio_path=audio_path,
                text_path=text_path,
                original_text_path=original_text_path,
                text=text,
                command=command,
            )
        )
    return plans


def _validate_export_plans(plans: list[ExportPlan]) -> None:
    """Reject collisions and conflicting deterministic reruns before writes."""
    fields = {
        "clip IDs": [str(plan.audio_path.parent / plan.clip_id) for plan in plans],
        "audio paths": [str(plan.audio_path) for plan in plans],
        "text paths": [str(plan.text_path) for plan in plans],
        "original-text paths": [str(plan.original_text_path) for plan in plans],
    }
    failures = []
    for label, values in fields.items():
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            failures.append(f"duplicate {label}: {_sample(duplicates)}")
    for plan in plans:
        for path, expected in (
            (plan.text_path, plan.text),
            (plan.original_text_path, plan.segment.text),
        ):
            if path.exists() and path.read_text(encoding="utf-8") != expected:
                failures.append(f"existing caption conflicts with planned text: {path}")
    if failures:
        raise ValueError("; ".join(failures))


def _execute_export_plans(plans: list[ExportPlan], *, run: bool = True) -> list[dict[str, str]]:
    """Execute validated export plans and return manifest rows."""
    for plan in plans:
        plan.audio_path.parent.mkdir(parents=True, exist_ok=True)
        if run:
            subprocess.run(plan.command, check=True)
        plan.text_path.write_text(plan.text, encoding="utf-8")
        plan.original_text_path.write_text(plan.segment.text, encoding="utf-8")
    return [plan.manifest_row() for plan in plans]


def _export_srt_segments(
    input_audio: Path | str,
    segments: list[SrtSegment],
    output_dir: Path | str,
    *,
    text_by_index: dict[int, str] | None = None,
    run: bool = True,
) -> list[dict[str, str]]:
    """Validate and export paired normalized/original text for SRT segments."""
    plans = _plan_srt_segments(
        input_audio,
        segments,
        output_dir,
        text_by_index=text_by_index,
    )
    _validate_export_plans(plans)
    return _execute_export_plans(plans, run=run)


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
    original_by_index = _segments_by_unique_index(original_segments, "original")
    clean_by_index = _segments_by_unique_index(clean_segments, "clean")
    missing = sorted(original_by_index.keys() - clean_by_index.keys())
    unexpected = sorted(clean_by_index.keys() - original_by_index.keys())
    if missing or unexpected:
        parts = ["paired SRT index mismatch"]
        if missing:
            parts.append(f"missing clean indices: {_sample([str(index) for index in missing])}")
        if unexpected:
            parts.append(f"unexpected clean indices: {_sample([str(index) for index in unexpected])}")
        raise ValueError("; ".join(parts))
    clean_text_by_index = {index: segment.text for index, segment in clean_by_index.items()}
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
    matched_only: bool = False,
    run: bool = True,
) -> list[dict[str, str]]:
    """Cut one aligned SRT into wav, normalized txt, and original _orig.txt files."""
    segments = parse_srt(Path(aligned_srt_path).read_text(encoding="utf-8-sig"))
    _segments_by_unique_index(segments, "aligned")
    if matched_only:
        if speaker_map_path is None:
            raise ValueError("--matched-only export requires a speaker map")
        matched_indices = speaker_map_matched_indices(Path(speaker_map_path))
        segments = [segment for segment in segments if segment.index in matched_indices]
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
    corpora: set[str] | None = None,
    require_diarized_matches: bool = False,
    matched_only: bool = False,
    min_match_ratio: float = 0.0,
    exclude_quality_failures: Path | str | None = None,
    run: bool = True,
) -> list[dict[str, str]]:
    """Export a tree of ``corpus/aligned/*.aligned.srt`` files like ``cut_samples``."""
    aligned_base = Path(aligned_root)
    audio_base = Path(audio_root)
    output_base = Path(output_root)
    plans: list[ExportPlan] = []
    speaker_map_copies: list[tuple[Path, Path]] = []
    ratio_cache: dict[str, dict[str, float]] = {}
    excluded_chunks = quality_failure_chunks(exclude_quality_failures) if exclude_quality_failures else set()
    aligned_files = []
    for aligned_srt in sorted(aligned_base.glob("*/aligned/*.aligned.srt")):
        corpus = aligned_srt.parent.parent.name
        if corpora is not None and corpus not in corpora:
            continue
        if aligned_srt.name.removesuffix(".aligned.srt") in excluded_chunks:
            continue
        aligned_files.append((corpus, aligned_srt))

    if min_match_ratio > 0:
        missing_ratios = []
        low_ratios = []
        for corpus, aligned_srt in aligned_files:
            chunk = aligned_srt.name.removesuffix(".aligned.srt")
            ratios = ratio_cache.setdefault(
                corpus,
                summary_match_ratios(aligned_base / corpus / "summary.tsv"),
            )
            if chunk not in ratios:
                missing_ratios.append(chunk)
            elif ratios[chunk] < min_match_ratio:
                low_ratios.append(f"{chunk}={ratios[chunk]:.3f}")
        if missing_ratios or low_ratios:
            parts = []
            if missing_ratios:
                parts.append(f"missing summary match ratio for {_sample(missing_ratios)}")
            if low_ratios:
                parts.append(
                    f"{len(low_ratios)} chunks below minimum match ratio {min_match_ratio:.3f}: "
                    f"{_sample(low_ratios)}"
                )
            raise ValueError("; ".join(parts))

    for corpus, aligned_srt in aligned_files:
        chunk = aligned_srt.name.removesuffix(".aligned.srt")
        audio_path = audio_base / corpus / f"{chunk}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing chunk audio for {aligned_srt}: {audio_path}")
        speaker_map = aligned_base / corpus / "speaker_maps" / f"{chunk}.speaker_map.csv"
        segments = parse_srt(aligned_srt.read_text(encoding="utf-8-sig"))
        _segments_by_unique_index(segments, f"{chunk} aligned")
        if require_diarized_matches:
            if not speaker_map.exists():
                raise ValueError(f"Missing speaker map for diarization guard: {speaker_map}")
            missing_indices = {segment.index for segment in segments} - speaker_map_indices(speaker_map)
            if missing_indices:
                missing_text = ", ".join(str(index) for index in sorted(missing_indices))
                raise ValueError(f"Speaker map lacks rows for SRT indices {missing_text}: {speaker_map}")
            if has_matched_blank_speakers(speaker_map):
                raise ValueError(f"Matched segments without transcript speakers in {speaker_map}")
        if matched_only:
            if not speaker_map.exists():
                raise ValueError("--matched-only export requires a speaker map")
            matched_indices = speaker_map_matched_indices(speaker_map)
            segments = [segment for segment in segments if segment.index in matched_indices]
        if speaker_map.exists():
            segments = apply_speaker_map(segments, speaker_map_by_index(speaker_map))
        clean_text_by_index = {segment.index: normalize_caption_text(segment.text) for segment in segments}
        target_dir = output_base / corpus / chunk
        plans.extend(
            _plan_srt_segments(
                audio_path,
                segments,
                target_dir,
                text_by_index=clean_text_by_index,
            )
        )
        if speaker_map.exists():
            speaker_map_copies.append((speaker_map, target_dir))
    _validate_export_plans(plans)
    rows = _execute_export_plans(plans, run=run)
    for speaker_map, target_dir in speaker_map_copies:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(speaker_map, target_dir / "speaker_map.csv")
    if manifest_path is not None:
        write_tsv(manifest_path, rows, MANIFEST_COLUMNS)
    return rows
