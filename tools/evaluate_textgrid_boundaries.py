"""Evaluate TextGrid word boundaries against a reference TextGrid tree."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

WORD_RE = re.compile(r"[\w+]+", re.UNICODE)


@dataclass(frozen=True)
class Interval:
    """One labeled TextGrid interval."""

    xmin: float
    xmax: float
    text: str

    @property
    def middle(self) -> float:
        """Return the interval midpoint."""
        return (self.xmin + self.xmax) / 2


@dataclass(frozen=True)
class Match:
    """One aligned reference/prediction interval pair."""

    ref_index: int
    pred_index: int
    distance: int


def read_text(path: Path) -> str:
    """Read a TextGrid that may be UTF-8 or UTF-16."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8-sig", errors="replace")


def parse_textgrid(path: Path) -> dict[str, list[Interval]]:
    """Parse IntervalTier intervals from a long-form Praat TextGrid."""
    tiers: dict[str, list[Interval]] = {}
    current_tier: str | None = None
    current_interval: dict[str, str] | None = None
    in_interval = False

    for raw_line in read_text(path).splitlines():
        line = raw_line.strip()
        if line.startswith("name = "):
            current_tier = unquote(line.split("=", 1)[1].strip())
            tiers.setdefault(current_tier, [])
            current_interval = None
            in_interval = False
            continue
        if current_tier and line.startswith("intervals ["):
            current_interval = {}
            in_interval = True
            continue
        if not current_tier or not in_interval or current_interval is None or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        current_interval[key] = value
        if key == "text":
            tiers[current_tier].append(
                Interval(
                    xmin=float(current_interval["xmin"]),
                    xmax=float(current_interval["xmax"]),
                    text=unquote(value).strip(),
                )
            )
            current_interval = None
            in_interval = False
    return tiers


def unquote(value: str) -> str:
    """Remove Praat quotes and unescape doubled quotes."""
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('""', '"')
    return value


def word_intervals(path: Path, preferred_tiers: tuple[str, ...]) -> list[Interval]:
    """Return non-empty word intervals from the first available tier."""
    tiers = parse_textgrid(path)
    for tier_name in preferred_tiers:
        if tier_name in tiers:
            return [interval for interval in tiers[tier_name] if normalize_word(interval.text)]
    available = ", ".join(sorted(tiers))
    raise ValueError(f"{path} does not contain any of {preferred_tiers}; available tiers: {available}")


def normalize_word(text: str) -> str:
    """Normalize a TextGrid word label for matching."""
    tokens = WORD_RE.findall(text.lower().replace("ё", "е"))
    return "".join(tokens)


def edit_distance_at_most(a: str, b: str, limit: int) -> int | None:
    """Return edit distance when it is at most ``limit``, otherwise ``None``."""
    if abs(len(a) - len(b)) > limit:
        return None
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        row_min = i
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return None
        previous = current
    distance = previous[-1]
    return distance if distance <= limit else None


def word_distance(ref: Interval, pred: Interval, *, fuzzy_distance: int) -> int | None:
    """Return normalized word distance when labels are an allowed match."""
    ref_word = normalize_word(ref.text)
    pred_word = normalize_word(pred.text)
    if not ref_word or not pred_word:
        return None
    return edit_distance_at_most(ref_word, pred_word, fuzzy_distance)


def align_words(ref_words: list[Interval], pred_words: list[Interval], *, fuzzy_distance: int) -> list[Match]:
    """Monotonically align reference and predicted word intervals."""
    ref_count = len(ref_words)
    pred_count = len(pred_words)
    costs = [[0] * (pred_count + 1) for _ in range(ref_count + 1)]
    moves = [[""] * (pred_count + 1) for _ in range(ref_count + 1)]
    for i in range(1, ref_count + 1):
        costs[i][0] = i
        moves[i][0] = "up"
    for j in range(1, pred_count + 1):
        costs[0][j] = j
        moves[0][j] = "left"

    for i in range(1, ref_count + 1):
        for j in range(1, pred_count + 1):
            distance = word_distance(ref_words[i - 1], pred_words[j - 1], fuzzy_distance=fuzzy_distance)
            substitution_cost = 0 if distance is not None else 2
            candidates = [
                (costs[i - 1][j - 1] + substitution_cost, "diag"),
                (costs[i - 1][j] + 1, "up"),
                (costs[i][j - 1] + 1, "left"),
            ]
            costs[i][j], moves[i][j] = min(candidates, key=lambda item: item[0])

    matches: list[Match] = []
    i, j = ref_count, pred_count
    while i > 0 or j > 0:
        move = moves[i][j]
        if move == "diag":
            distance = word_distance(ref_words[i - 1], pred_words[j - 1], fuzzy_distance=fuzzy_distance)
            if distance is not None:
                matches.append(Match(i - 1, j - 1, distance))
            i -= 1
            j -= 1
        elif move == "up":
            i -= 1
        else:
            j -= 1
    return list(reversed(matches))


def reference_key(path: Path, root: Path) -> tuple[str, str]:
    """Return the file matching key shared by both trees."""
    relative = path.relative_to(root)
    return relative.parent.name, path.name


def prediction_key(path: Path, root: Path) -> tuple[str, str]:
    """Return the prediction file matching key, ignoring the corpus directory."""
    relative = path.relative_to(root)
    return relative.parent.name, path.name


def evaluate_pair(
    *,
    ref_path: Path,
    pred_path: Path,
    tolerance: float,
    fuzzy_distance: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate one reference/prediction TextGrid pair."""
    ref_words = word_intervals(ref_path, ("words_text", "words"))
    pred_words = word_intervals(pred_path, ("words", "words_text"))
    matches = align_words(ref_words, pred_words, fuzzy_distance=fuzzy_distance)
    word_rows: list[dict[str, object]] = []
    left_matches = right_matches = middle_matches = 0

    for match in matches:
        ref = ref_words[match.ref_index]
        pred = pred_words[match.pred_index]
        left_delta = pred.xmin - ref.xmin
        right_delta = pred.xmax - ref.xmax
        middle_delta = pred.middle - ref.middle
        left_ok = abs(left_delta) <= tolerance
        right_ok = abs(right_delta) <= tolerance
        middle_ok = abs(middle_delta) <= tolerance
        left_matches += int(left_ok)
        right_matches += int(right_ok)
        middle_matches += int(middle_ok)
        word_rows.append(
            {
                "reference_path": str(ref_path),
                "prediction_path": str(pred_path),
                "ref_index": match.ref_index + 1,
                "pred_index": match.pred_index + 1,
                "ref_word": ref.text,
                "pred_word": pred.text,
                "word_edit_distance": match.distance,
                "ref_xmin": f"{ref.xmin:.6f}",
                "ref_xmax": f"{ref.xmax:.6f}",
                "pred_xmin": f"{pred.xmin:.6f}",
                "pred_xmax": f"{pred.xmax:.6f}",
                "left_delta_s": f"{left_delta:.6f}",
                "right_delta_s": f"{right_delta:.6f}",
                "middle_delta_s": f"{middle_delta:.6f}",
                "left_border_matches": left_ok,
                "right_border_matches": right_ok,
                "middle_matches": middle_ok,
            }
        )

    matched = len(matches)
    file_row = {
        "reference_path": str(ref_path),
        "prediction_path": str(pred_path),
        "ref_words": len(ref_words),
        "pred_words": len(pred_words),
        "matched_words": matched,
        "left_border_matches": left_matches,
        "right_border_matches": right_matches,
        "middle_matches": middle_matches,
        "left_accuracy": ratio(left_matches, matched),
        "right_accuracy": ratio(right_matches, matched),
        "middle_accuracy": ratio(middle_matches, matched),
        "alignment_accuracy_percent": percent(left_matches + right_matches, len(pred_words) * 2),
        "word_recall": ratio(matched, len(ref_words)),
        "word_precision": ratio(matched, len(pred_words)),
    }
    return file_row, word_rows


def ratio(numerator: int, denominator: int) -> str:
    """Return a stable decimal ratio string."""
    return "" if denominator == 0 else f"{numerator / denominator:.6f}"


def percent(numerator: int, denominator: int) -> str:
    """Return a stable percentage string."""
    return "" if denominator == 0 else f"{100 * numerator / denominator:.2f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], missing: int) -> dict[str, object]:
    """Build corpus-level counts and rates."""
    totals = {
        "files": len(rows),
        "missing_predictions": missing,
        "ref_words": sum(int(row["ref_words"]) for row in rows),
        "pred_words": sum(int(row["pred_words"]) for row in rows),
        "matched_words": sum(int(row["matched_words"]) for row in rows),
        "left_border_matches": sum(int(row["left_border_matches"]) for row in rows),
        "right_border_matches": sum(int(row["right_border_matches"]) for row in rows),
        "middle_matches": sum(int(row["middle_matches"]) for row in rows),
    }
    matched = int(totals["matched_words"])
    totals["left_accuracy"] = ratio(int(totals["left_border_matches"]), matched)
    totals["right_accuracy"] = ratio(int(totals["right_border_matches"]), matched)
    totals["middle_accuracy"] = ratio(int(totals["middle_matches"]), matched)
    totals["alignment_accuracy_percent"] = percent(
        int(totals["left_border_matches"]) + int(totals["right_border_matches"]),
        int(totals["pred_words"]) * 2,
    )
    totals["word_recall"] = ratio(matched, int(totals["ref_words"]))
    totals["word_precision"] = ratio(matched, int(totals["pred_words"]))
    return totals


def markdown_summary(exact: dict[str, object], fuzzy: dict[str, object]) -> str:
    """Return a Markdown summary table."""
    lines = [
        "# TextGrid Boundary Evaluation",
        "",
        "| word matching | files | missing files | ref words | pred words | matched words | "
        "Alignment Accuracy (%) | left acc | right acc | middle acc | recall | precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in (("exact", exact), ("fuzzy_ed1", fuzzy)):
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(summary["files"]),
                    str(summary["missing_predictions"]),
                    str(summary["ref_words"]),
                    str(summary["pred_words"]),
                    str(summary["matched_words"]),
                    str(summary["alignment_accuracy_percent"]),
                    str(summary["left_accuracy"]),
                    str(summary["right_accuracy"]),
                    str(summary["middle_accuracy"]),
                    str(summary["word_recall"]),
                    str(summary["word_precision"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def run_evaluation(
    args: argparse.Namespace, *, fuzzy_distance: int
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate all matching TextGrids for one word matching setting."""
    pred_index = {
        prediction_key(path, args.prediction_root): path
        for path in sorted(args.prediction_root.rglob("*.TextGrid"))
    }
    file_rows: list[dict[str, object]] = []
    word_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for ref_path in sorted(args.reference_root.rglob("*.TextGrid")):
        key = reference_key(ref_path, args.reference_root)
        pred_path = pred_index.get(key)
        if pred_path is None:
            missing_rows.append({"reference_path": str(ref_path), "chunk": key[0], "filename": key[1]})
            continue
        file_row, rows = evaluate_pair(
            ref_path=ref_path,
            pred_path=pred_path,
            tolerance=args.tolerance_ms / 1000,
            fuzzy_distance=fuzzy_distance,
        )
        file_rows.append(file_row)
        word_rows.extend(rows)
    return summarize(file_rows, len(missing_rows)), file_rows, word_rows, missing_rows


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=Path("test_textgrids"))
    parser.add_argument("--prediction-root", type=Path, default=Path("hf-repo/aligned"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/textgrid-eval"))
    parser.add_argument("--tolerance-ms", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    """Run exact and edit-distance-1 TextGrid evaluations."""
    args = parse_args()
    exact_summary, exact_files, exact_words, exact_missing = run_evaluation(args, fuzzy_distance=0)
    fuzzy_summary, fuzzy_files, fuzzy_words, fuzzy_missing = run_evaluation(args, fuzzy_distance=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "file_metrics_exact.csv", exact_files)
    write_csv(args.output_dir / "word_metrics_exact.csv", exact_words)
    write_csv(args.output_dir / "missing_exact.csv", exact_missing)
    write_csv(args.output_dir / "file_metrics_fuzzy_ed1.csv", fuzzy_files)
    write_csv(args.output_dir / "word_metrics_fuzzy_ed1.csv", fuzzy_words)
    write_csv(args.output_dir / "missing_fuzzy_ed1.csv", fuzzy_missing)
    write_csv(
        args.output_dir / "summary.csv",
        [exact_summary | {"word_matching": "exact"}, fuzzy_summary | {"word_matching": "fuzzy_ed1"}],
    )
    summary_md = markdown_summary(exact_summary, fuzzy_summary)
    (args.output_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    print(summary_md)


if __name__ == "__main__":
    main()
