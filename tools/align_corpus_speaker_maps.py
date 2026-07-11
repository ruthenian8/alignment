"""Run alignment with speaker-map output for known local corpus layouts."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alignment.align import find_speaker_tag, speaker_tag_from_line  # noqa: E402
from alignment.mapping import align_mapping_table, summary_quality_errors  # noqa: E402
from alignment.reorder import normalize_for_match  # noqa: E402

INDEXED_CORPORA = ("and", "pom", "uht")
LATIN_INITIALS = {
    "a": "А",
    "b": "Б",
    "v": "В",
    "g": "Г",
    "d": "Д",
    "e": "Е",
    "z": "З",
    "i": "И",
    "j": "Й",
    "k": "К",
    "l": "Л",
    "m": "М",
    "n": "Н",
    "o": "О",
    "p": "П",
    "r": "Р",
    "s": "С",
    "t": "Т",
    "u": "У",
    "f": "Ф",
    "h": "Х",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        choices=["pez", *INDEXED_CORPORA],
        help="Corpus family to process. May be repeated. Defaults to pez, and, pom, uht.",
    )
    parser.add_argument(
        "--name",
        action="append",
        help="Optional mapping table name to process, for example pez_001 or and_004.",
    )
    parser.add_argument("--hf-root", type=Path, default=Path("hf-repo/wx_transcripts"))
    parser.add_argument("--build-root", type=Path, default=Path("build"))
    parser.add_argument("--raw-transcript-root", type=Path, default=Path("raw_transcript"))
    parser.add_argument("--output-root", type=Path, default=Path("build/aligned-with-speaker-maps"))
    parser.add_argument("--use-transcript-speakers", action="store_true", default=True)
    parser.add_argument(
        "--no-use-transcript-speakers",
        action="store_false",
        dest="use_transcript_speakers",
    )
    parser.add_argument("--infer-missing-speakers", action="store_true", default=True)
    parser.add_argument(
        "--no-infer-missing-speakers",
        action="store_false",
        dest="infer_missing_speakers",
    )
    parser.add_argument(
        "--require-diarized-matches",
        action="store_true",
        help="Fail when mapping rows are missing or matched segments lack transcript speakers.",
    )
    return parser.parse_args(argv)


def text_blocks_with_speakers(text: str) -> list[tuple[str, str]]:
    """Return transcript block text with its final-line speaker tag."""
    blocks = []
    for match in re.finditer(r"\S.*?(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
        block = match.group(0).strip()
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        tag = speaker_tag_from_line(lines[-1])
        if tag:
            blocks.append((block, tag))
    return blocks


def best_block_speaker(transcript: str, blocks: list[tuple[str, str]]) -> str:
    """Find the source block speaker for a chunk transcript when the match is unambiguous."""
    normalized = normalize_for_match(transcript)
    if not normalized:
        return ""
    matches = []
    for block, tag in blocks:
        block_normalized = normalize_for_match(block)
        if normalized in block_normalized or block_normalized in normalized:
            matches.append((tag, len(block_normalized)))
    tags = {tag for tag, _ in matches}
    if len(tags) == 1:
        return next(iter(tags))
    if matches:
        matches.sort(key=lambda item: item[1])
        return matches[0][0]
    return ""


def write_mapping_with_speaker_hints(mapping: Path, raw_transcript: Path, output_path: Path) -> Path:
    """Write a mapping copy with speaker_hint values recovered from raw transcript blocks."""
    if not raw_transcript.exists():
        return mapping
    delimiter = "," if mapping.suffix.lower() == ".csv" else "\t"
    with mapping.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    blocks = text_blocks_with_speakers(raw_transcript.read_text(encoding="utf-8-sig"))
    if "speaker_hint" not in fieldnames:
        fieldnames.append("speaker_hint")
    for row in rows:
        row["speaker_hint"] = row.get("speaker_hint", "") or best_block_speaker(
            row.get("transcript", ""), blocks
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def trailing_speaker_hint(text: str) -> str:
    """Return a trailing speaker code from one-line table text when present."""
    line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    tag = speaker_tag_from_line(line)
    if tag:
        return tag
    match = re.search(r"(?:^|\s)([A-ZА-ЯЁ]{1,6}|\?{3})\s*$", line)
    return match.group(1) if match else ""


def transcript_speaker_hint(text: str, candidates: list[str] | None = None) -> str:
    """Return an explicit speaker hint from a mapping transcript cell."""
    explicit = find_speaker_tag(text) or trailing_speaker_hint(text)
    if explicit:
        return explicit
    if not candidates:
        return ""
    present = [
        candidate
        for candidate in candidates
        if re.search(rf"(?<![А-ЯЁA-Z]){re.escape(candidate)}(?![А-ЯЁA-Z])", text)
    ]
    if len(present) == 1:
        return present[0]
    return candidates[0] if len(candidates) == 1 else ""


def transliterate_initials(code: str) -> str:
    """Return a Cyrillic version of a Latin speaker-code stem."""
    result = []
    index = 0
    lowered = code.lower()
    while index < len(code):
        if lowered.startswith(("ja", "ya"), index):
            result.append("Я")
            index += 2
        elif lowered.startswith(("ju", "yu"), index):
            result.append("Ю")
            index += 2
        elif lowered.startswith(("jo", "yo"), index):
            result.append("Ё")
            index += 2
        else:
            result.append(LATIN_INITIALS.get(lowered[index], code[index].upper()))
            index += 1
    return "".join(result)


def speaker_candidates_from_source(path: str) -> list[str]:
    """Return respondent speaker candidates from an original transcript path."""
    if not path:
        return []
    stem = Path(path).stem
    if stem.lower().endswith("_txt"):
        stem = stem[:-4]
    if stem.lower().endswith(".txt"):
        stem = stem[:-4]
    if "_" in stem:
        stem = stem.split("_", 1)[0]
    candidates = []
    for part in re.split(r"&|,", stem):
        code = part.strip()
        if not code:
            continue
        candidate = transliterate_initials(code) if re.search(r"[A-Za-z]", code) else code.upper()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def indexed_speaker_candidates(corpus: str, build_root: Path) -> dict[str, list[str]]:
    """Load respondent candidates from a local layout manifest when available."""
    manifest = build_root / f"{corpus}_layout_manifest.tsv"
    if not manifest.exists():
        return {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file, delimiter="\t")
        return {
            row["id"]: speaker_candidates_from_source(row.get("source_transcript", ""))
            for row in rows
            if row.get("id")
        }


def write_indexed_mapping_with_speaker_hints(
    mapping: Path, output_path: Path, candidates: list[str] | None = None
) -> Path:
    """Write a mapping copy with speaker hints carried through indexed transcript rows."""
    with mapping.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "speaker_hint" not in fieldnames:
        fieldnames.append("speaker_hint")

    current_hint = ""
    for row in rows:
        transcript = row.get("transcript", "")
        explicit = transcript_speaker_hint(transcript, candidates)
        if explicit:
            current_hint = explicit
        row["speaker_hint"] = row.get("speaker_hint", "") or (current_hint if transcript else "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def pez_jobs(hf_root: Path, raw_transcript_root: Path, output_root: Path) -> list[tuple[str, Path, Path]]:
    """Return PEZ mapping/SRT jobs from the Hugging Face-style layout."""
    jobs = []
    for mapping in sorted(hf_root.glob("pez_*/pez_*.csv")):
        name = mapping.parent.name
        srt_dir = mapping.parent
        raw_transcript = raw_transcript_root / "pez" / f"{name}.txt"
        mapping = write_mapping_with_speaker_hints(
            mapping,
            raw_transcript,
            output_root / name / f"{name}.speaker_hints.csv",
        )
        jobs.append((name, mapping, srt_dir))
    return jobs


def indexed_jobs(
    corpus: str, build_root: Path, hf_root: Path, output_root: Path
) -> list[tuple[str, Path, Path]]:
    """Return jobs for build/<corpus>/joined tables with discovered SRT directories."""
    jobs = []
    speaker_candidates = indexed_speaker_candidates(corpus, build_root)
    for mapping in sorted((build_root / corpus / "joined").glob(f"{corpus}_*.reordered.tsv")):
        stem = mapping.stem.removesuffix(".reordered")
        candidates = [
            build_root / corpus / "srt" / stem,
            build_root / corpus / "wx_transcripts" / stem,
            build_root / corpus / "srt",
            build_root / corpus / "wx_transcripts",
            hf_root / stem,
        ]
        srt_dir = next((candidate for candidate in candidates if candidate.exists()), None)
        if srt_dir is not None:
            mapping = write_indexed_mapping_with_speaker_hints(
                mapping,
                output_root / stem / f"{stem}.speaker_hints.tsv",
                speaker_candidates.get(stem, []),
            )
            jobs.append((stem, mapping, srt_dir))
    return jobs


def main(argv: list[str] | None = None) -> int:
    """Align selected corpora and write alignment-time speaker maps."""
    args = parse_args(argv)
    corpora = args.corpus or ["pez", *INDEXED_CORPORA]
    jobs: list[tuple[str, Path, Path]] = []
    if "pez" in corpora:
        jobs.extend(pez_jobs(args.hf_root, args.raw_transcript_root, args.output_root))
    for corpus in INDEXED_CORPORA:
        if corpus in corpora:
            jobs.extend(indexed_jobs(corpus, args.build_root, args.hf_root, args.output_root))
    if args.name:
        names = set(args.name)
        jobs = [job for job in jobs if job[0] in names]

    aligned = missing = 0
    for name, mapping, srt_dir in jobs:
        print(f"{name}: aligning {mapping} against {srt_dir}", flush=True)
        summary = align_mapping_table(
            mapping,
            srt_dir,
            args.output_root / name,
            use_transcript_speakers=args.use_transcript_speakers,
            infer_missing_speakers=args.infer_missing_speakers,
        )
        if args.require_diarized_matches:
            errors = summary_quality_errors(summary)
            if errors:
                raise SystemExit(f"{name}: {'; '.join(errors)}")
        aligned += sum(row["status"] == "aligned" for row in summary)
        missing += sum(row["status"] != "aligned" for row in summary)
        print(f"{name}: {aligned} aligned so far, {missing} missing so far", flush=True)

    print(f"processed {len(jobs)} mapping tables; aligned {aligned}; missing {missing}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
