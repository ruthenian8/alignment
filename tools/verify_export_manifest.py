"""Verify exported corpus manifests against alignment quality gates."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 TSV file into dictionaries."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def failure_chunks(path: Path | None) -> set[str]:
    """Return chunk names listed in an optional quality-failure TSV."""
    if path is None:
        return set()
    return {row["name"] for row in read_tsv(path) if row.get("name")}


def summary_rows(summary_root: Path, corpora: set[str] | None = None) -> list[dict[str, str]]:
    """Return all align-map summary rows below a root."""
    rows = []
    for path in sorted(summary_root.glob("*/summary.tsv")):
        if corpora is not None and path.parent.name not in corpora:
            continue
        rows.extend(read_tsv(path))
    return rows


def summary_failures(
    rows: list[dict[str, str]],
    *,
    excluded_chunks: set[str],
    min_match_ratio: float = 0.0,
) -> list[str]:
    """Return failures for kept aligned chunks that do not meet final-output quality."""
    matched_blank = 0
    low_match = []
    for row in rows:
        if row.get("name") in excluded_chunks:
            continue
        matched_blank += int(row.get("matched_blank_speakers", "0") or 0)
        ratio_text = row.get("match_ratio", "")
        if min_match_ratio > 0 and ratio_text:
            try:
                ratio = float(ratio_text)
            except ValueError:
                ratio = 0.0
            if ratio < min_match_ratio:
                low_match.append(f"{row.get('name', '<unknown>')}={ratio:.3f}")
    failures = []
    if matched_blank:
        failures.append(f"{matched_blank} kept matched summary rows have blank transcript speakers")
    if low_match:
        failures.append(
            f"{len(low_match)} kept chunks are below minimum match ratio {min_match_ratio:.3f}: "
            f"{sample(low_match)}"
        )
    return failures


def sample(items: list[str], limit: int = 5) -> str:
    """Return a compact sample of failure labels."""
    text = ", ".join(items[:limit])
    if len(items) > limit:
        text += f", +{len(items) - limit} more"
    return text


def manifest_failures(
    manifest_rows: list[dict[str, str]],
    *,
    excluded_chunks: set[str],
    expected_rows: int | None = None,
    check_files: bool = False,
) -> list[str]:
    """Return human-readable manifest invariant failures."""
    failures = []
    if expected_rows is not None and len(manifest_rows) != expected_rows:
        failures.append(f"manifest rows {len(manifest_rows)} != expected {expected_rows}")
    blank_speakers = sum(not row.get("speaker", "").strip() for row in manifest_rows)
    if blank_speakers:
        failures.append(f"{blank_speakers} manifest rows have blank speakers")
    whisperx_speakers = sum(row.get("speaker", "").strip().startswith("[SPEAKER_") for row in manifest_rows)
    if whisperx_speakers:
        failures.append(f"{whisperx_speakers} manifest rows keep WhisperX speaker codes")
    failed_rows = sum(
        bool(excluded_chunks & set(Path(row.get("audio_path", "")).parts)) for row in manifest_rows
    )
    if failed_rows:
        failures.append(f"{failed_rows} manifest rows come from excluded quality-failure chunks")
    if check_files:
        failures.extend(file_failures(manifest_rows))
    return failures


def file_failures(manifest_rows: list[dict[str, str]]) -> list[str]:
    """Return failures for missing or stale files referenced by the manifest."""
    missing_audio = missing_text = stale_text = missing_original = stale_original = 0
    for row in manifest_rows:
        audio_value = row.get("audio_path", "")
        audio_path = Path(audio_value)
        if not audio_value or not audio_path.exists():
            missing_audio += 1

        text_value = row.get("text_path", "")
        text_path = Path(text_value)
        if not text_value or not text_path.exists():
            missing_text += 1
        elif text_path.read_text(encoding="utf-8") != row.get("text", ""):
            stale_text += 1

        original_value = row.get("text_original_path", "")
        original_path = Path(original_value)
        if not original_value or not original_path.exists():
            missing_original += 1
        elif original_path.read_text(encoding="utf-8") != row.get("text_original", ""):
            stale_original += 1

    failures = []
    if missing_audio:
        failures.append(f"{missing_audio} manifest audio files are missing")
    if missing_text:
        failures.append(f"{missing_text} manifest text files are missing")
    if stale_text:
        failures.append(f"{stale_text} manifest text files differ from manifest text")
    if missing_original:
        failures.append(f"{missing_original} manifest original-text files are missing")
    if stale_original:
        failures.append(f"{stale_original} manifest original-text files differ from manifest text_original")
    return failures


def verify_manifest(
    manifest_path: Path,
    *,
    summary_root: Path | None = None,
    quality_failures: Path | None = None,
    corpora: set[str] | None = None,
    check_files: bool = False,
    min_match_ratio: float = 0.0,
) -> tuple[dict[str, int], list[str]]:
    """Verify an exported manifest and return metrics plus failures."""
    rows = read_tsv(manifest_path)
    excluded = failure_chunks(quality_failures)
    summaries = summary_rows(summary_root, corpora) if summary_root else []
    expected = None
    if summary_root:
        expected = sum(
            int(row.get("matched_segments", "0") or 0) for row in summaries if row.get("name") not in excluded
        )
    failures = manifest_failures(
        rows,
        excluded_chunks=excluded,
        expected_rows=expected,
        check_files=check_files,
    )
    if summary_root:
        failures.extend(
            summary_failures(summaries, excluded_chunks=excluded, min_match_ratio=min_match_ratio)
        )
    metrics = {
        "manifest_rows": len(rows),
        "excluded_chunks": len(excluded),
        "blank_speakers": sum(not row.get("speaker", "").strip() for row in rows),
        "whisperx_speakers": sum(row.get("speaker", "").strip().startswith("[SPEAKER_") for row in rows),
    }
    if expected is not None:
        metrics["expected_rows"] = expected
    return metrics, failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Exported manifest TSV path.")
    parser.add_argument(
        "--summary-root",
        type=Path,
        help="Aligned output root containing corpus summary.tsv files.",
    )
    parser.add_argument(
        "--quality-failures",
        type=Path,
        help="Optional quality_failures.tsv whose chunks must be absent from the manifest.",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        help="Only count this corpus directory from --summary-root. May be repeated.",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Also verify referenced audio/text files exist and text files match manifest fields.",
    )
    parser.add_argument(
        "--min-match-ratio",
        type=float,
        default=0.0,
        help="Reject kept summary rows whose match_ratio is below this threshold.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run manifest verification."""
    args = parse_args(argv)
    metrics, failures = verify_manifest(
        args.manifest,
        summary_root=args.summary_root,
        quality_failures=args.quality_failures,
        corpora=set(args.corpus) if args.corpus else None,
        check_files=args.check_files,
        min_match_ratio=args.min_match_ratio,
    )
    for key, value in metrics.items():
        print(f"{key}\t{value}")
    if failures:
        for failure in failures:
            print(f"ERROR\t{failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
