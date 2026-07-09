"""Select stratified cut-sample files for manual validation.

This one-time helper samples canonical ``.wav`` + ``.txt`` pairs from a
cut-sample tree. It uses ``*_orig.txt`` only for filtering dialect-marked
captions, because normalized captions may intentionally remove markup such as
backslashes.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

WORD_RE = re.compile(r"[\w+]+", re.UNICODE)


@dataclass(frozen=True)
class Candidate:
    """One selectable audio-caption pair."""

    folder: str
    chunk: str
    audio_path: Path
    caption_path: Path
    original_caption_path: Path | None
    word_count: int
    text: str


def canonical_caption(path: Path) -> bool:
    """Return whether ``path`` is the normalized caption paired with audio."""
    return (
        path.suffix == ".txt"
        and not path.name.endswith("_orig.txt")
        and ".asr_" not in path.name
        and path.name != "speaker_map.csv"
    )


def word_count(text: str) -> int:
    """Count coarse word-like tokens for phrase-length filtering."""
    return len(WORD_RE.findall(text))


def relative(path: Path, root: Path) -> str:
    """Return a stable POSIX path relative to ``root`` when possible."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def read_text(path: Path) -> str:
    """Read UTF-8 text with BOM tolerance."""
    return path.read_text(encoding="utf-8-sig").strip()


def collect_candidates(
    root: Path,
    *,
    min_words: int,
    max_words: int,
    require_dialect_marker: bool,
) -> list[Candidate]:
    """Collect canonical audio-caption pairs that satisfy validation filters."""
    candidates: list[Candidate] = []
    for caption_path in sorted(root.rglob("*.txt")):
        if not canonical_caption(caption_path):
            continue
        audio_path = caption_path.with_suffix(".wav")
        if not audio_path.exists():
            continue
        try:
            folder = caption_path.relative_to(root).parts[0]
            chunk = caption_path.relative_to(root).parts[1]
        except IndexError:
            continue
        text = read_text(caption_path)
        count = word_count(text)
        if count < min_words or count > max_words:
            continue
        original_caption_path = caption_path.with_name(f"{caption_path.stem}_orig.txt")
        original_text = read_text(original_caption_path) if original_caption_path.exists() else text
        if require_dialect_marker and "\\" not in original_text:
            continue
        candidates.append(
            Candidate(
                folder=folder,
                chunk=chunk,
                audio_path=audio_path,
                caption_path=caption_path,
                original_caption_path=original_caption_path if original_caption_path.exists() else None,
                word_count=count,
                text=text,
            )
        )
    return candidates


def allocate_counts(group_sizes: dict[str, int], target: int) -> dict[str, int]:
    """Allocate a target sample size proportionally across groups."""
    total = sum(group_sizes.values())
    if target > total:
        raise ValueError(f"Requested {target} samples, but only {total} candidates passed filters")
    raw = {group: target * size / total for group, size in group_sizes.items()}
    allocation = {group: min(group_sizes[group], int(value)) for group, value in raw.items()}

    remaining = target - sum(allocation.values())
    order = sorted(
        group_sizes,
        key=lambda group: (raw[group] - int(raw[group]), group_sizes[group], group),
        reverse=True,
    )
    while remaining:
        changed = False
        for group in order:
            if allocation[group] >= group_sizes[group]:
                continue
            allocation[group] += 1
            remaining -= 1
            changed = True
            if remaining == 0:
                break
        if not changed:
            raise ValueError("Could not allocate all requested samples")
    return allocation


def stratified_sample(candidates: list[Candidate], target: int, *, seed: int) -> list[Candidate]:
    """Return a deterministic proportional stratified sample by top-level folder."""
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.folder].append(candidate)
    allocation = allocate_counts({group: len(rows) for group, rows in groups.items()}, target)

    rng = random.Random(seed)
    selected: list[Candidate] = []
    for group in sorted(groups):
        rows = groups[group][:]
        rng.shuffle(rows)
        selected.extend(rows[: allocation[group]])
    selected.sort(key=lambda row: (row.folder, row.chunk, row.audio_path.name))
    return selected


def write_sample(path: Path, rows: list[Candidate], *, root: Path, set_name: str) -> None:
    """Write selected filenames and compact metadata as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "set",
                "folder",
                "chunk",
                "audio_path",
                "caption_path",
                "original_caption_path",
                "word_count",
                "text",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "set": set_name,
                    "folder": row.folder,
                    "chunk": row.chunk,
                    "audio_path": relative(row.audio_path, root),
                    "caption_path": relative(row.caption_path, root),
                    "original_caption_path": (
                        relative(row.original_caption_path, root) if row.original_caption_path else ""
                    ),
                    "word_count": row.word_count,
                    "text": row.text,
                }
            )


def write_summary(
    path: Path, candidates: list[Candidate], set_a: list[Candidate], set_b: list[Candidate]
) -> None:
    """Write per-folder candidate and sample counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate_counts: dict[str, int] = defaultdict(int)
    set_a_counts: dict[str, int] = defaultdict(int)
    set_b_counts: dict[str, int] = defaultdict(int)
    for row in candidates:
        candidate_counts[row.folder] += 1
    for row in set_a:
        set_a_counts[row.folder] += 1
    for row in set_b:
        set_b_counts[row.folder] += 1
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["folder", "candidates", "set_a", "set_b"])
        writer.writeheader()
        for folder in sorted(candidate_counts):
            writer.writerow(
                {
                    "folder": folder,
                    "candidates": candidate_counts[folder],
                    "set_a": set_a_counts[folder],
                    "set_b": set_b_counts[folder],
                }
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("build/cut_samples-srt-speakers"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/manual-validation-samples"))
    parser.add_argument("--set-a-size", type=int, default=500)
    parser.add_argument("--set-b-extra", type=int, default=500)
    parser.add_argument("--min-words", type=int, default=1)
    parser.add_argument(
        "--max-words",
        type=int,
        default=12,
        help="Maximum normalized-caption word count for phrase-length samples.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--include-unmarked",
        action="store_true",
        help="Do not require a backslash dialect marker in *_orig.txt or the caption text.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Select set A and superset B, then write CSV manifests."""
    args = parse_args(argv)
    if args.set_a_size <= 0 or args.set_b_extra <= 0:
        raise ValueError("Sample sizes must be positive")
    candidates = collect_candidates(
        args.root,
        min_words=args.min_words,
        max_words=args.max_words,
        require_dialect_marker=not args.include_unmarked,
    )
    set_a = stratified_sample(candidates, args.set_a_size, seed=args.seed)
    set_b = stratified_sample(candidates, args.set_a_size + args.set_b_extra, seed=args.seed)
    set_a_paths = {row.audio_path for row in set_a}
    if not set_a_paths.issubset({row.audio_path for row in set_b}):
        raise RuntimeError("Set B is not a superset of set A")

    write_sample(args.output_dir / "set_a.csv", set_a, root=args.root, set_name="A")
    write_sample(args.output_dir / "set_b.csv", set_b, root=args.root, set_name="B")
    write_summary(args.output_dir / "summary.csv", candidates, set_a, set_b)
    print(f"candidates={len(candidates)} set_a={len(set_a)} set_b={len(set_b)} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
