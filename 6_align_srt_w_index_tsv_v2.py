"""
Utility to align manual transcripts to time-coded SRT files.

This script reads a CSV metadata file where each row refers to an audio file
and optionally contains a manual transcript in the ``transcript`` column.  For
each row where the boolean ``trans`` column is True and a transcript is
present, the corresponding ``.srt`` file is loaded from the same directory.

Each SRT file consists of numbered segments with start and end time codes and a
speaker placeholder such as ``[SPEAKER_00]:`` or ``[SPEAKER_01]:``.  The
transcripts contain rich markup in square brackets: stage directions,
speaker labels (e.g. ``[МВ:]`` or ``[ДС:]``) and narrative descriptions.  This
script attempts to match each phrase in the transcript to the appropriate SRT
segment, replace the placeholder speaker tag with the correct label from the
transcript, and preserve the exact wording (including stress marks) from the
transcript.

Alignment relies on simple heuristics adapted from the provided
``align_whisper_w_real_v1.py``:  transcripts are split into a sequence of
segments consisting either of bracketed markup or free text delimited by
sentence‑ending punctuation.  Bracket segments that end with a colon (e.g.
``[МВ:]``) are treated as speaker labels and automatically combined with the
subsequent text segment.  For matching, all punctuation, bracketed content
and stress marks are stripped and the ``rapidfuzz`` library is used to score
similarity between segments.  The algorithm walks through the SRT segments
in order, attempting to align each one with the next transcript segment.
Short look‑ahead searches are used to skip over unmatchable items.  When a
transcript segment does not contain an explicit speaker label the last
assigned label is propagated forward.  If no transcript label is available
the original ``SPEAKER_xx`` tag from the SRT is retained.

Usage::

    python align_srt_transcripts.py path/to/pez_XXX.csv

The script will write a new SRT file for each processed row using the same
basename but with ``_aligned.srt`` appended.  Rows where ``trans`` is False
or where the transcript is missing are skipped, and the original SRT file
will remain untouched.
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import pandas as pd
from rapidfuzz import fuzz


def parse_srt(file_path: str) -> List[Dict[str, object]]:
    """Parse a simple SRT file into a list of segment dictionaries.

    Each dictionary contains the keys:
      - index: the ordinal number of the segment
      - start: start timestamp as a string (``HH:MM:SS,ms``)
      - end: end timestamp as a string (``HH:MM:SS,ms``)
      - speaker: the placeholder speaker tag (e.g. ``SPEAKER_00``) or None
      - text: the concatenated text of the segment after removing the
        placeholder and any line breaks
      - orig_lines: the original list of text lines (including the
        placeholder) for fallback use
    """
    segments: List[Dict[str, object]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    i = 0
    while i < len(lines):
        # Skip blank lines
        if not lines[i].strip():
            i += 1
            continue
        # Segment index line
        try:
            idx = int(lines[i].strip())
        except ValueError:
            # Malformed index; skip this line
            i += 1
            continue
        i += 1
        if i >= len(lines):
            break
        # Time code line
        time_line = lines[i].strip()
        i += 1
        if "-->" not in time_line:
            # Missing time codes; skip this segment
            continue
        start_time, end_time = [t.strip() for t in time_line.split("-->")]
        # Collect all following non‑blank lines as the text
        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].rstrip("\n"))
            i += 1
        # Extract speaker placeholder and the remainder of the text
        speaker = None
        text = ""
        if text_lines:
            first_line = text_lines[0]
            m = re.match(r"\[(SPEAKER_\d+)\]:\s*(.*)", first_line)
            if m:
                speaker = m.group(1)
                rest = [m.group(2)] + text_lines[1:]
            else:
                # No explicit speaker in first line
                rest = text_lines
            text = " ".join(rest).strip()
        # Append segment dict
        segments.append({
            "index": idx,
            "start": start_time,
            "end": end_time,
            "speaker": speaker,
            "text": text,
            "orig_lines": text_lines,
        })
        # Skip blank line separating segments
        i += 1
    return segments


def prepare_transcript_segments(transcript: str) -> Tuple[List[str], List[str]]:
    """Split a transcript into a list of logical segments and return both the
    original segments and a normalized version for matching.

    The transcript may contain square‑bracket markup for stage directions
    and speaker labels.  We first break the transcript into a sequence of
    bracketed chunks and free text separated by sentence‑ending punctuation.
    Next, we combine a bracketed chunk that ends with a colon (e.g. ``[MV:]``)
    with the following non‑bracket text so that speaker labels stay with
    their utterances.  Finally, we produce a normalized version of each
    combined segment by stripping out bracketed markup, punctuation and
    stress marks and converting to lowercase.  The normalized list is used
    solely for fuzzy matching and does not affect the original strings.
    """
    text = transcript
    # Step 1: extract bracketed segments and free text segments
    segments: List[str] = []
    pos = 0
    # Find all bracketed spans
    for m in re.finditer(r"\[[^\]]+\]", text):
        # Add any free text before this bracketed span, split by sentence punctuation
        if m.start() > pos:
            outside = text[pos:m.start()]
            outside_parts = re.split(r"(?<=[.!?])\s+", outside)
            for part in outside_parts:
                if part.strip():
                    segments.append(part.strip())
        # Append the bracketed span itself
        segments.append(m.group(0).strip())
        pos = m.end()
    # Add any trailing free text after the last bracketed span
    if pos < len(text):
        outside = text[pos:]
        outside_parts = re.split(r"(?<=[.!?])\s+", outside)
        for part in outside_parts:
            if part.strip():
                segments.append(part.strip())
    # Step 2: combine speaker labels with the following utterance
    combined: List[str] = []
    i = 0
    # Pattern for bracketed labels that end with a colon, e.g. [МВ:] or [Соб.:]
    label_pattern = re.compile(r"^\[[^\]]*?:\]$")
    while i < len(segments):
        seg = segments[i]
        if label_pattern.match(seg) and i + 1 < len(segments) and not segments[i + 1].startswith("["):
            # Merge the label with the following non‑bracket segment
            combined.append(f"{seg} {segments[i + 1]}")
            i += 2
        else:
            combined.append(seg)
            i += 1
    # Step 3: produce normalized versions for matching
    #
    # For each combined segment, we differentiate between bracketed labels
    # (segments that end with a colon inside the brackets) and bracketed
    # utterances/questions (segments that lack a colon).  Labels are removed
    # entirely for normalization, whereas bracketed questions have their
    # outer brackets stripped but the text inside retained.  All segments
    # then have punctuation and stress marks removed and are lower‑cased.
    norm_list: List[str] = []
    label_re = re.compile(r"^\[[^\]]*?:\]$")
    for seg in combined:
        if label_re.match(seg):
            # Bracketed label: drop the entire bracket
            core = ""
        elif seg.startswith("[") and seg.endswith("]"):
            # Bracketed utterance without colon: strip brackets but keep content
            core = seg[1:-1]
        else:
            core = seg
        # Remove stress marks
        clean = core.replace("\\", "")
        # Remove punctuation
        clean = re.sub(r"[\.,!?;:—\"\(\)…]", " ", clean)
        # Collapse whitespace
        clean = re.sub(r"\s+", " ", clean)
        norm_list.append(clean.strip().lower())
    return combined, norm_list


def normalize_srt_text(text: str) -> str:
    """Normalize SRT text for comparison by removing markup, punctuation and stress marks."""
    clean = re.sub(r"\[[^\[\]]+\]", "", text)
    clean = clean.replace("\\", "")
    clean = re.sub(r"[\.,!?;:—\"\(\)…]", " ", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip().lower()


def right_condition(norm_srt: str, norm_trans: str, alpha: int = 55, beta: int = 65) -> bool:
    """Check whether two normalized strings match sufficiently.

    We compute the token set ratio and the partial ratio using rapidfuzz and
    require either to exceed their respective thresholds.  Lowering these
    values increases the number of candidate matches and makes the alignment
    more aggressive.
    """
    if not norm_srt or not norm_trans:
        return False
    return fuzz.token_set_ratio(norm_srt, norm_trans) >= alpha or fuzz.partial_ratio(norm_srt, norm_trans) >= beta


def align_segments(
    segments: List[Dict[str, object]],
    trans_norm: List[str],
    alpha: int = 50,
    beta: int = 60,
    retry: int = 4,
) -> Dict[int, List[int]]:
    """Align SRT segments to transcript segments based on fuzzy matching.

    Returns a mapping from SRT segment indices (0‑based) to lists of
    transcript indices.  The algorithm scans both lists in order,
    attempting to align each SRT segment with the next unmatched transcript
    segment.  If a direct match fails it will look ahead up to ``retry``
    positions in the transcript to find a match.  When a match is found,
    the pointer into the transcript list advances beyond the matched
    segment(s).  Unmatched SRT segments will map to an empty list.
    """
    aligned: Dict[int, List[int]] = {i: [] for i in range(len(segments))}
    num = 0
    for i, seg in enumerate(segments):
        norm_srt = normalize_srt_text(seg.get("text", ""))
        matched_any = False
        # Match as many consecutive transcript segments as possible
        while num < len(trans_norm) and right_condition(norm_srt, trans_norm[num], alpha, beta):
            aligned[i].append(num)
            matched_any = True
            num += 1
        if not matched_any:
            # Try looking ahead a few transcript segments for a match
            for shift in range(1, retry + 1):
                if num + shift < len(trans_norm) and right_condition(norm_srt, trans_norm[num + shift], alpha, beta):
                    aligned[i].append(num + shift)
                    num = num + shift + 1
                    matched_any = True
                    break
        # If no match is found the aligned entry remains empty and the transcript
        # pointer does not advance.  Remaining transcript segments at the end
        # will be discarded.
    return aligned


def compose_final_segments(
    segments: List[Dict[str, object]],
    trans_list: List[str],
    aligned: Dict[int, List[int]],
) -> Tuple[List[Dict[str, object]], set]:
    """Construct final SRT segments with updated text and speaker tags.

    ``segments`` is the list returned by ``parse_srt``.  ``trans_list`` is
    the list of original transcript segments returned by
    ``prepare_transcript_segments``.  ``aligned`` maps SRT indices to
    transcript indices.  Returns a list of new segment dicts with an
    additional ``new_text`` key and a set of transcript indices that were
    used.

    When constructing the new text, any bracketed speaker label at the
    beginning of a transcript segment (e.g. ``[МВ:]``) is extracted and
    used to replace the placeholder ``SPEAKER_xx``.  If no explicit label is
    found, the label from the previous aligned segment is propagated.
    """
    new_segments: List[Dict[str, object]] = []
    used_indices: set = set()
    # Pattern that matches short bracketed speaker labels, including unknown
    # ones such as [???:] or [?:].  We capture the characters before the
    # colon (up to 8 letters, dots or question marks) and allow optional
    # whitespace between the label and colon.  This pattern intentionally
    # does not require the closing bracket to occur immediately after the
    # colon so that constructs like ``[Соб.: Вот, ...]`` are recognized as
    # containing a speaker label ``Соб.``.  Longer bracketed strings (e.g.
    # descriptions or compound labels) will not match and will be treated
    # as part of the utterance.
    speaker_pattern = re.compile(r"^\[([A-Za-zА-Яа-яЁё\.\?]{1,8})\s*:")
    last_speaker = None  # propagate speaker when missing
    for i, seg in enumerate(segments):
        trans_indices = aligned.get(i, [])
        if trans_indices:
            # Gather the original transcript strings for this segment
            lines = [trans_list[idx] for idx in trans_indices]
            speaker_bracket = None
            stripped_lines: List[str] = []
            for line in lines:
                # Attempt to extract a short speaker label from the beginning of the line.
                # The `speaker_pattern` captures up to 8 characters before a colon.  If a
                # label is found and we haven't yet set a speaker for this segment, extract
                # the label and compute the remainder of the utterance.  The remainder may
                # be located either inside the same bracket (after the colon) or after the
                # closing bracket.  This handles constructs like ``[ДС:] Я ...`` and
                # ``[Соб.: Вот, кто-то рассказывает.]``.
                m = speaker_pattern.match(line)
                if m and not speaker_bracket:
                    label = m.group(1)
                    # Standardize the speaker bracket; include a colon and closing bracket
                    speaker_bracket = f"[{label}:]"
                    # Locate the colon and the first closing bracket after it
                    colon_idx = line.find(':', 1)
                    bracket_end_idx = line.find(']', colon_idx) if colon_idx != -1 else -1
                    remainder_parts: List[str] = []
                    if colon_idx != -1:
                        # Content inside the same bracket after the colon up to the closing bracket
                        if bracket_end_idx != -1:
                            inner = line[colon_idx + 1:bracket_end_idx].strip()
                            if inner:
                                remainder_parts.append(inner)
                            # Append any text after the closing bracket
                            tail = line[bracket_end_idx + 1:].strip()
                            if tail:
                                remainder_parts.append(tail)
                        else:
                            # No closing bracket; treat everything after the colon as remainder
                            tail = line[colon_idx + 1:].strip()
                            if tail:
                                remainder_parts.append(tail)
                    else:
                        # Fallback: no colon found; keep the rest of the line
                        tail = line[m.end():].strip()
                        if tail:
                            remainder_parts.append(tail)
                    remainder = " ".join(remainder_parts).strip()
                else:
                    # No speaker label extracted; the entire line is the remainder
                    remainder = line.strip()
                # If the remainder is still enclosed in a single pair of brackets, strip them
                if remainder.startswith('[') and remainder.endswith(']') and len(remainder) > 2:
                    remainder = remainder[1:-1].strip()
                stripped_lines.append(remainder)
            if not speaker_bracket:
                # No explicit speaker label in transcripts; reuse last label or fallback to SRT placeholder
                if last_speaker:
                    speaker_bracket = last_speaker
                elif seg.get("speaker"):
                    speaker_bracket = f"[{seg['speaker']}]"
                else:
                    speaker_bracket = ""
            # Update last_speaker for subsequent segments if a label was found
            last_speaker = speaker_bracket if speaker_bracket else last_speaker
            # Construct the content by joining all stripped lines with spaces
            content = " ".join([t.strip() for t in stripped_lines if t.strip()])
            # Prepend the speaker tag
            if speaker_bracket:
                # If the bracket already contains a colon (e.g. [МВ:]) keep it; otherwise append a colon
                prefix = speaker_bracket if ":" in speaker_bracket else f"{speaker_bracket}:"
                new_text = f"{prefix} {content}".strip() if content else prefix
            else:
                new_text = content
            new_seg = seg.copy()
            new_seg["new_text"] = new_text.strip()
            new_segments.append(new_seg)
            for idx in trans_indices:
                used_indices.add(idx)
        else:
            # No aligned transcript; preserve original lines exactly
            new_seg = seg.copy()
            if seg.get("orig_lines"):
                new_seg["new_text"] = "\n".join(seg["orig_lines"])
            else:
                new_seg["new_text"] = ""
            new_segments.append(new_seg)
    return new_segments, used_indices


def write_srt(segments: List[Dict[str, object]], output_path: str) -> None:
    """Write a list of SRT segments to a file.

    Each segment dict should have the keys ``index``, ``start``, ``end`` and
    ``new_text``.  If ``new_text`` contains embedded newlines they will
    produce multiple subtitle lines.  A blank line is written between
    segments as required by the SRT format.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg['index']}\n")
            f.write(f"{seg['start']} --> {seg['end']}\n")
            # ``new_text`` may contain newlines if multiple lines are desired
            text = seg.get("new_text", "")
            if text:
                # ensure consistent line breaks
                if isinstance(text, str):
                    lines = text.split("\n")
                else:
                    lines = [str(text)]
                for line in lines:
                    f.write(f"{line}\n")
            # blank line between segments
            f.write("\n")


def process_csv(csv_path: str, output_dir: str) -> None:
    """Process the provided CSV file and align transcripts with their SRT files.

    For each row in the CSV where ``trans`` is True and a non‑empty
    ``transcript`` is present, the script locates the SRT file by replacing
    the ``.wav`` extension in the ``name`` column with ``.srt``.  The SRT
    file is parsed, aligned and re‑written to ``<basename>_aligned.srt`` in
    the specified output directory.  Rows where ``trans`` is False or the
    ``name`` column is missing are skipped.  The original SRT files are
    never modified.
    """
    df = pd.read_csv(csv_path)
    # Determine directory of CSV and output directory
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)
    for _, row in df.iterrows():
        trans_flag = row.get("trans")
        transcript = row.get("transcript")
        wav_name = row.get("name")
        if pd.isna(trans_flag) or not trans_flag:
            # Skip rows without transcripts
            continue
        if not isinstance(transcript, str) or not transcript.strip():
            # Skip if transcript is missing or empty
            continue
        if not isinstance(wav_name, str) or not wav_name.strip():
            # Skip if name column is missing
            continue
        # Locate the corresponding SRT file
        srt_name = os.path.splitext(wav_name)[0] + ".srt"
        srt_path = os.path.join(base_dir, srt_name)
        if not os.path.isfile(srt_path):
            # Skip if SRT does not exist
            print(f"Warning: SRT file not found for {wav_name}, expected {srt_path}", file=sys.stderr)
            continue
        # Parse SRT
        segments = parse_srt(srt_path)
        # Prepare transcript
        trans_list, trans_norm = prepare_transcript_segments(transcript)
        # Align
        aligned = align_segments(segments, trans_norm)
        # Compose new segments
        new_segments, used = compose_final_segments(segments, trans_list, aligned)
        # Write output
        output_srt = os.path.join(output_dir, os.path.splitext(srt_name)[0] + "_aligned.srt")
        write_srt(new_segments, output_srt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align transcripts to SRT files.")
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the CSV metadata file (e.g. pez_011.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="aligned_srt",
        help="Directory where aligned SRT files will be saved",
    )
    args = parser.parse_args()
    process_csv(args.csv_path, args.output_dir)


if __name__ == "__main__":
    main()
