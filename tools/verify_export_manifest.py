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


def expected_exported_rows(
    summary_root: Path, excluded_chunks: set[str], corpora: set[str] | None = None
) -> int:
    """Return matched aligned rows excluding known quality-failure chunks."""
    total = 0
    for row in summary_rows(summary_root, corpora):
        if row.get("name") in excluded_chunks:
            continue
        total += int(row.get("matched_segments", "0") or 0)
    return total


def manifest_failures(
    manifest_rows: list[dict[str, str]],
    *,
    excluded_chunks: set[str],
    expected_rows: int | None = None,
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
    return failures


def verify_manifest(
    manifest_path: Path,
    *,
    summary_root: Path | None = None,
    quality_failures: Path | None = None,
    corpora: set[str] | None = None,
) -> tuple[dict[str, int], list[str]]:
    """Verify an exported manifest and return metrics plus failures."""
    rows = read_tsv(manifest_path)
    excluded = failure_chunks(quality_failures)
    expected = expected_exported_rows(summary_root, excluded, corpora) if summary_root else None
    failures = manifest_failures(rows, excluded_chunks=excluded, expected_rows=expected)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run manifest verification."""
    args = parse_args(argv)
    metrics, failures = verify_manifest(
        args.manifest,
        summary_root=args.summary_root,
        quality_failures=args.quality_failures,
        corpora=set(args.corpus) if args.corpus else None,
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
