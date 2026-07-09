"""Consolidate human-evaluation annotations into item-level scores."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SEGMENT_SCORES = {
    "Полностью неверное аудио": 1,
    "Верная середина, расхождение в начале и конце": 2,
    "Расхождение в конце": 3,
    "Расхождение в начале": 4,
    "Полностью верное аудио": 5,
}
USABILITY_SCORES = {"Да": 1, "Нет": 0}
QUESTION_FIELDS = ("segment_issues", "usability", "audio_quality")


def scalar(value: Any) -> Any:
    """Return the parsed scalar from Potato/MACE typed values."""
    if isinstance(value, dict) and "parsedValue" in value:
        return value["parsedValue"]
    return value


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_mace(path: Path) -> tuple[str, dict[str, str], dict[str, float]]:
    """Read MACE aggregate labels and entropy by item ID."""
    data = read_json(path)
    entropy = {item_id: float(scalar(value)) for item_id, value in data.get("label_entropy", {}).items()}
    return data["schema_name"], data.get("predicted_labels", {}), entropy


def clean_label(value: Any) -> str:
    """Return a stable string label for parquet values."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def majority(labels: list[str]) -> tuple[str, int, int]:
    """Return deterministic majority label, winning count, and labeled total."""
    labels = [label for label in labels if label]
    if not labels:
        return "", 0, 0
    counts = Counter(labels)
    label, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return label, count, len(labels)


def read_annotations(path: Path) -> dict[str, dict[str, Any]]:
    """Read per-annotation parquet rows and aggregate them by item."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Reading annotations.parquet requires pandas with a parquet backend") from exc

    frame = pd.read_parquet(path)
    required = {"instance_id", "user_id", *QUESTION_FIELDS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

    annotations: dict[str, dict[str, Any]] = {}
    for item_id, group in frame.groupby("instance_id", sort=True):
        item: dict[str, Any] = {
            "id": item_id,
            "annotation_count": len(group),
            "annotators": ";".join(sorted(clean_label(value) for value in group["user_id"].dropna())),
        }
        for field in QUESTION_FIELDS:
            label, count, total = majority([clean_label(value) for value in group[field].tolist()])
            item[f"{field}_raw_label"] = label
            item[f"{field}_raw_count"] = count
            item[f"{field}_raw_total"] = total
        annotations[item_id] = item
    return annotations


def score(question: str, label: str) -> int | str:
    """Return numeric score for a final label when available."""
    if question == "segment_issues":
        return SEGMENT_SCORES.get(label, "")
    if question == "usability":
        return USABILITY_SCORES.get(label, "")
    if question == "audio_quality" and label.isdigit():
        return int(label)
    return ""


def consolidated_rows(input_dir: Path) -> list[dict[str, Any]]:
    """Build compact item-level rows from raw and MACE exports."""
    annotations = read_annotations(input_dir / "annotations.parquet")
    mace: dict[str, dict[str, str]] = {}
    entropy: dict[str, dict[str, float]] = {}
    for path in sorted(input_dir.glob("*.mace-aggr.json")):
        schema, labels, label_entropy = read_mace(path)
        mace[schema] = labels
        entropy[schema] = label_entropy

    rows: list[dict[str, Any]] = []
    for item_id, raw in sorted(annotations.items()):
        row: dict[str, Any] = {
            "id": item_id,
            "annotation_count": raw.get("annotation_count", ""),
            "annotators": raw.get("annotators", ""),
            "segment_issues_raw_label": raw.get("segment_issues_raw_label", ""),
            "segment_issues_raw_count": raw.get("segment_issues_raw_count", ""),
            "segment_issues_raw_total": raw.get("segment_issues_raw_total", ""),
            "segment_issues_label": "",
            "segment_issues_score": "",
            "segment_issues_source": "",
            "segment_issues_entropy": "",
            "usability_raw_label": raw.get("usability_raw_label", ""),
            "usability_raw_count": raw.get("usability_raw_count", ""),
            "usability_raw_total": raw.get("usability_raw_total", ""),
            "usability_label": "",
            "usability_score": "",
            "usability_source": "",
            "usability_entropy": "",
            "audio_quality_raw_label": raw.get("audio_quality_raw_label", ""),
            "audio_quality_raw_count": raw.get("audio_quality_raw_count", ""),
            "audio_quality_raw_total": raw.get("audio_quality_raw_total", ""),
            "audio_quality_label": "",
            "audio_quality_score": "",
            "audio_quality_source": "",
        }
        for question in QUESTION_FIELDS:
            final_label = mace.get(question, {}).get(item_id) or raw.get(f"{question}_raw_label", "")
            if not final_label:
                continue
            row[f"{question}_label"] = final_label
            row[f"{question}_score"] = score(question, final_label)
            row[f"{question}_source"] = "mace" if item_id in mace.get(question, {}) else "parquet_majority"
            if question in entropy and item_id in entropy[question]:
                row[f"{question}_entropy"] = entropy[question][item_id]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def proportion_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute final-label proportions for each question."""
    output: list[dict[str, Any]] = []
    questions = [
        ("segment_issues", "segment_issues_label"),
        ("usability", "usability_label"),
        ("audio_quality", "audio_quality_label"),
    ]
    for question, field in questions:
        labels = [str(row[field]) for row in rows if row[field] != ""]
        counts = Counter(labels)
        total = sum(counts.values())
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            output.append(
                {
                    "question": question,
                    "label": label,
                    "count": count,
                    "proportion": round(count / total, 6) if total else "",
                    "n_labeled": total,
                }
            )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("human_eval"))
    parser.add_argument("--scores", type=Path, default=Path("human_eval/consolidated_scores.csv"))
    parser.add_argument("--summary", type=Path, default=Path("human_eval/label_proportions.csv"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write consolidated scores and label-proportion summary."""
    args = parse_args(argv)
    rows = consolidated_rows(args.input_dir)
    summary = proportion_rows(rows)
    write_csv(args.scores, rows)
    write_csv(args.summary, summary)
    print(f"wrote {len(rows)} rows to {args.scores}")
    print(f"wrote {len(summary)} summary rows to {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
