"""Infer actual speaker tags from aligned transcript rows."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from .align import (
    UNKNOWN_SPEAKER,
    find_non_note_speaker_tag_before_span,
    find_speaker_tag,
    format_speaker_tag,
    is_unknown_speaker_bracket,
    remove_alignment_notes,
    speaker_blocks_from_transcript,
    speaker_note_tag_at_span_start,
    speaker_note_tag_before_span,
    speaker_tag_from_blocks,
    speaker_tag_from_marker,
    tokenize_transcript,
    transcript_with_block_speaker_markers,
    unknown_speaker_tag_at_span,
)
from .io import parse_bool, read_tsv

CLIP_RE = re.compile(r"^(?P<index>\d{3})_(?P<speaker>.+)_(?P<timestamp>\d{2}-\d{2}-\d{2}-\d{3})$")
LEADING_CONTEXT_SPEAKER_RE = re.compile(
    r"^\s*\[((?:[A-ZА-ЯЁ](?=[:\],])|[A-ZА-ЯЁ]{2,6}(?=[:\s,\]])|\?{3}))[^\]]*\]"
)
SPEAKER_MAP_COLUMNS = [
    "audio_file",
    "text_file",
    "text_original_file",
    "srt_index",
    "timestamp",
    "whisperx_speaker",
    "transcript_speaker",
    "speaker_source",
    "aligned_matched",
    "alignment_score",
]
SUMMARY_COLUMNS = ["metric", "value"]
INVENTORY_COLUMNS = [
    "transcript_path",
    "stem_speakers",
    "leading_context_speaker",
    "bracket_speakers",
    "status",
]


def clip_parts(path: Path) -> tuple[int, str, str]:
    """Return SRT index, WhisperX speaker code, and filename timestamp."""
    match = CLIP_RE.fullmatch(path.stem)
    if not match:
        raise ValueError(f"Cannot parse cut-sample filename: {path}")
    return int(match.group("index")), match.group("speaker"), match.group("timestamp")


def token_positions(tokens: list[str]) -> dict[str, list[int]]:
    """Index normalized transcript tokens by token text."""
    positions: dict[str, list[int]] = defaultdict(list)
    for index, token in enumerate(tokens):
        positions[token].append(index)
    return positions


def find_token_span(
    tokens: list[str],
    positions: dict[str, list[int]],
    wanted: list[str],
    cursor: int,
) -> tuple[int, int] | None:
    """Find a normalized row span in transcript token order."""
    if not wanted:
        return None
    candidates = positions.get(wanted[0], [])
    starts = [index for index in candidates if index >= cursor]
    starts.extend(index for index in candidates if max(0, cursor - 100) <= index < cursor)
    starts.extend(index for index in candidates if index < max(0, cursor - 100))
    seen = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        end = start + len(wanted)
        if tokens[start:end] == wanted:
            return start, end
    return None


def leading_context_speaker(transcript: str) -> str:
    """Infer a carried speaker from an initial editorial speaker note."""
    match = LEADING_CONTEXT_SPEAKER_RE.match(transcript)
    return match.group(1) if match else ""


def row_speaker_tag(text: str) -> str:
    """Infer a speaker tag directly from one aligned transcript row."""
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        marker = stripped[1:-1]
        if is_unknown_speaker_bracket(marker):
            return UNKNOWN_SPEAKER
    return find_speaker_tag(text)


def table_speaker_tag(text: str) -> str:
    """Return an already inferred aligned-table speaker tag, if present."""
    speaker = text.strip()
    if re.fullmatch(r"\[[^\]]+\]:", speaker) and not speaker.startswith("[SPEAKER_"):
        return speaker.strip("[]:")
    return ""


def explicit_row_speaker(
    row: dict[str, str],
    transcript: str,
    tokens: list,
    normalized_tokens: list[str],
    positions: dict[str, list[int]],
    speaker_blocks: list,
    cursor: int,
    *,
    prefer_table_speakers: bool,
) -> tuple[str, str, int]:
    """Infer an explicit speaker tag for one aligned TSV row."""
    tag = table_speaker_tag(row.get("speaker", "")) if prefer_table_speakers else ""
    source = "aligned_table" if tag else "unmatched"
    if not tag:
        tag = row_speaker_tag(row.get("transcript_text", ""))
        source = "row_marker" if tag else source

    wanted = row.get("normalized_text", "").split()
    span = find_token_span(normalized_tokens, positions, wanted, cursor)
    if not span:
        return tag, source, cursor

    start_token, end_token = span
    start_char = tokens[start_token].start
    end_char = tokens[end_token - 1].end
    cursor = end_token
    span_tag = unknown_speaker_tag_at_span(transcript, start_char, end_char)
    if span_tag == UNKNOWN_SPEAKER:
        return span_tag, "collector_bracket", cursor
    if tag:
        return tag, source, cursor

    tag = find_non_note_speaker_tag_before_span(transcript, start_char)
    if tag:
        return tag, "preceding_marker", cursor
    tag = speaker_note_tag_at_span_start(transcript, start_char) or speaker_note_tag_before_span(
        transcript, start_char
    )
    if tag:
        return tag, "speaker_note", cursor
    tag = speaker_tag_from_blocks(speaker_blocks, start_char)
    if tag:
        return tag, "block_footer", cursor
    return "", "unmatched", cursor


def fill_speaker_gaps(rows: list[dict[str, str]], leading_tag: str) -> list[dict[str, str]]:
    """Fill safe blank speaker spans inside a chunk."""
    filled = [dict(row) for row in rows]
    anchor_indices = [index for index, row in enumerate(filled) if row["tag"]]

    if leading_tag:
        first_anchor = anchor_indices[0] if anchor_indices else len(filled)
        for row in filled[:first_anchor]:
            row["tag"] = leading_tag
            row["source"] = "leading_context"

    anchor_indices = [index for index, row in enumerate(filled) if row["tag"]]
    for left, right in zip(anchor_indices, anchor_indices[1:], strict=False):
        left_tag = filled[left]["tag"]
        right_tag = filled[right]["tag"]
        if left_tag == right_tag and left_tag != UNKNOWN_SPEAKER:
            for row in filled[left + 1 : right]:
                if not row["tag"]:
                    row["tag"] = left_tag
                    row["source"] = "bridged_same_speaker"

    current = ""
    saw_actual_anchor = False
    for row in filled:
        if row["tag"]:
            if row["tag"] == UNKNOWN_SPEAKER:
                current = ""
            else:
                current = row["tag"]
                saw_actual_anchor = True
        elif current:
            row["tag"] = current
            row["source"] = "carried_forward_prev"
        elif leading_tag and not saw_actual_anchor:
            row["tag"] = leading_tag
            row["source"] = "leading_context"
    return filled


def infer_table_speakers(
    table_rows: list[dict[str, str]],
    raw_transcript: str,
    *,
    prefer_table_speakers: bool = False,
) -> dict[int, dict[str, str]]:
    """Infer actual speaker tags for aligned TSV rows."""
    transcript = transcript_with_block_speaker_markers(raw_transcript)
    transcript = remove_alignment_notes(transcript)
    tokens = tokenize_transcript(transcript)
    normalized_tokens = [token.norm for token in tokens]
    positions = token_positions(normalized_tokens)
    speaker_blocks = speaker_blocks_from_transcript(transcript)

    cursor = 0
    inferred_rows = []
    for row in table_rows:
        tag, source, cursor = explicit_row_speaker(
            row,
            transcript,
            tokens,
            normalized_tokens,
            positions,
            speaker_blocks,
            cursor,
            prefer_table_speakers=prefer_table_speakers,
        )
        inferred_rows.append(
            {
                "srt_index": int(row["srt_index"]),
                "tag": tag,
                "source": source,
                "matched": row.get("matched", ""),
                "score": row.get("score", ""),
            }
        )

    filled_rows = fill_speaker_gaps(inferred_rows, leading_context_speaker(raw_transcript))
    return {
        row["srt_index"]: {
            "transcript_speaker": format_speaker_tag(row["tag"]) if row["tag"] else "",
            "speaker_source": row["source"],
            "aligned_matched": row["matched"],
            "alignment_score": row["score"],
        }
        for row in filled_rows
    }


def infer_chunk_rows(
    table_path: Path,
    manual_path: Path,
    cut_dir: Path,
    *,
    prefer_table_speakers: bool = False,
) -> list[dict[str, str]]:
    """Infer transcript speaker rows for one terminal cut-sample directory."""
    inferred_by_index = infer_table_speakers(
        read_tsv(table_path),
        manual_path.read_text(encoding="utf-8-sig"),
        prefer_table_speakers=prefer_table_speakers,
    )
    output_rows = []
    for wav_path in sorted(cut_dir.glob("*.wav")):
        srt_index, whisperx_speaker, timestamp = clip_parts(wav_path)
        inferred = inferred_by_index.get(srt_index, {})
        base = wav_path.with_suffix("")
        output_rows.append(
            {
                "audio_file": wav_path.name,
                "text_file": f"{base.name}.txt",
                "text_original_file": f"{base.name}_orig.txt",
                "srt_index": str(srt_index),
                "timestamp": timestamp,
                "whisperx_speaker": f"[{whisperx_speaker}]:",
                "transcript_speaker": inferred.get("transcript_speaker", ""),
                "speaker_source": inferred.get("speaker_source", "missing_aligned_row"),
                "aligned_matched": inferred.get("aligned_matched", ""),
                "alignment_score": inferred.get("alignment_score", ""),
            }
        )
    return output_rows


def write_speaker_map(path: Path, rows: list[dict[str, str]]) -> None:
    """Write inferred speaker rows to CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPEAKER_MAP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def process_cut_sample_tree(
    cut_root: Path,
    aligned_root: Path,
    output_name: str,
    *,
    prefer_table_speakers: bool = False,
) -> tuple[int, int, int]:
    """Write one inferred speaker CSV in each terminal cut-sample directory."""
    chunks = rows = missing = 0
    for cut_dir in sorted(path for path in cut_root.glob("*/*") if path.is_dir()):
        corpus = cut_dir.parent.name
        chunk = cut_dir.name
        table_path = aligned_root / corpus / "tables" / f"{chunk}.aligned.tsv"
        manual_path = aligned_root / corpus / "manual" / f"{chunk}.manual.txt"
        if not table_path.exists() or not manual_path.exists():
            missing += 1
            continue
        chunk_rows = infer_chunk_rows(
            table_path,
            manual_path,
            cut_dir,
            prefer_table_speakers=prefer_table_speakers,
        )
        write_speaker_map(cut_dir / output_name, chunk_rows)
        chunks += 1
        rows += len(chunk_rows)
    return chunks, rows, missing


def summarize_speaker_maps(
    cut_root: Path, summary_dir: Path, output_name: str = "speaker_map.csv"
) -> dict[str, int]:
    """Summarize inferred speaker map coverage into CSV files."""
    source_counts: Counter[str] = Counter()
    speaker_counts: Counter[str] = Counter()
    total = missing = matched_blank = unmatched_blank = unk = same_as_whisperx = 0
    for path in cut_root.glob(f"*/*/{output_name}"):
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                total += 1
                speaker = row["transcript_speaker"]
                matched = parse_bool(row.get("aligned_matched", row.get("matched", "")))
                if not speaker:
                    missing += 1
                    if matched:
                        matched_blank += 1
                    else:
                        unmatched_blank += 1
                if speaker == "[UNK]:":
                    unk += 1
                if speaker and speaker == row["whisperx_speaker"]:
                    same_as_whisperx += 1
                source_counts[row["speaker_source"]] += 1
                speaker_counts[speaker] += 1

    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "rows": total,
        "inferred_rows": total - missing,
        "blank_rows": missing,
        "matched_blank_rows": matched_blank,
        "unmatched_blank_rows": unmatched_blank,
        "unknown_rows": unk,
        "same_as_whisperx_rows": same_as_whisperx,
    }
    write_counter_csv(summary_dir / "speaker_sources.csv", source_counts, "speaker_source")
    write_counter_csv(summary_dir / "speakers.csv", speaker_counts, "speaker")
    with (summary_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows({"metric": key, "value": value} for key, value in metrics.items())
    return metrics


def write_counter_csv(path: Path, counts: Counter[str], field: str) -> None:
    """Write a frequency table CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field, "count"])
        writer.writeheader()
        for key, count in counts.most_common():
            writer.writerow({field: key, "count": count})


def speaker_codes_from_stem(path: Path) -> str:
    """Extract candidate speaker codes from a raw transcript filename stem."""
    stem = path.stem.removesuffix("_txt")
    if "_" in stem:
        stem = stem.split("_", 1)[0]
    codes = [part.strip() for part in re.split(r"&|,", stem) if part.strip()]
    return ", ".join(codes)


def bracket_speakers_from_text(text: str) -> str:
    """List unique speaker tags explicitly recoverable from bracket markers."""
    speakers = []
    for match in re.finditer(r"\[([^\]]+)\]", text):
        tag = speaker_tag_from_marker(match.group(1))
        if tag and tag != UNKNOWN_SPEAKER and tag not in speakers:
            speakers.append(tag)
    return ", ".join(speakers)


def write_raw_transcript_inventory(transcript_root: Path, output_path: Path) -> int:
    """Write a transcript-level speaker inventory for raw transcript text files."""
    rows = []
    for transcript_path in sorted(transcript_root.rglob("*.txt")):
        try:
            text = transcript_path.read_text(encoding="utf-8-sig")
            rows.append(
                {
                    "transcript_path": str(transcript_path),
                    "stem_speakers": speaker_codes_from_stem(transcript_path),
                    "leading_context_speaker": leading_context_speaker(text),
                    "bracket_speakers": bracket_speakers_from_text(text),
                    "status": "ok",
                }
            )
        except UnicodeDecodeError:
            rows.append(
                {
                    "transcript_path": str(transcript_path),
                    "stem_speakers": speaker_codes_from_stem(transcript_path),
                    "leading_context_speaker": "",
                    "bracket_speakers": "",
                    "status": "decode_error",
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
