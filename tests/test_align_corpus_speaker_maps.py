"""Tests for corpus speaker-map helper script logic."""

import csv
from pathlib import Path

from tools.align_corpus_speaker_maps import (
    best_block_speaker,
    format_quality_failures,
    pez_jobs,
    quality_failure_rows,
    text_blocks_with_speakers,
)


def test_best_block_speaker_tolerates_small_transcript_differences():
    raw = """
VIII-15 в
Пежма
АБМ, БИС
[Как вели купленную корову домой?» Ой, там то\\жо при\\говор есть, а во\\т и забы\\ла при\\говор-то.
КЛП
""".strip()
    transcript = "[Как вели купленную корову домой?] Ой, там то\\жо при\\говор есть, а во\\т забы\\ла."

    assert best_block_speaker(transcript, text_blocks_with_speakers(raw)) == "КЛП"


def test_pez_jobs_accepts_flat_raw_transcript_root(tmp_path: Path) -> None:
    hf_root = tmp_path / "wx"
    mapping_dir = hf_root / "pez_001"
    mapping_dir.mkdir(parents=True)
    (mapping_dir / "pez_001.csv").write_text(
        'name,transcript\npez_001No0.wav,"ручной ответ"\n',
        encoding="utf-8",
    )
    raw_root = tmp_path / "transcripts"
    raw_root.mkdir()
    (raw_root / "pez_001.txt").write_text(
        "I-1\nPlace\nAB\nручной ответ\nААК\n",
        encoding="utf-8",
    )

    jobs = pez_jobs(hf_root, raw_root, tmp_path / "out")

    assert len(jobs) == 1
    with jobs[0][1].open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["speaker_hint"] == "ААК"


def test_format_quality_failures_caps_batch_output() -> None:
    message = format_quality_failures([f"pez_{index:03}: low coverage" for index in range(7)], limit=3)

    assert message == "pez_000: low coverage; pez_001: low coverage; pez_002: low coverage; +4 more"


def test_quality_failure_rows_are_row_level_and_typed() -> None:
    rows = quality_failure_rows(
        "pom_001",
        [
            {
                "name": "pom_001No1",
                "status": "missing_srt",
                "segments": "0",
                "match_ratio": "0.000",
                "matched_blank_speakers": "0",
            },
            {
                "name": "pom_001No2",
                "status": "aligned",
                "segments": "10",
                "match_ratio": "0.100",
                "matched_blank_speakers": "2",
            },
            {
                "name": "pom_001No3",
                "status": "aligned",
                "segments": "10",
                "match_ratio": "0.900",
                "matched_blank_speakers": "0",
            },
        ],
        min_match_ratio=0.2,
    )

    assert rows == [
        {
            "job": "pom_001",
            "name": "pom_001No1",
            "reason": "missing_srt",
            "value": "",
            "threshold": "",
            "status": "missing_srt",
        },
        {
            "job": "pom_001",
            "name": "pom_001No2",
            "reason": "matched_blank_speakers",
            "value": "2",
            "threshold": "0",
            "status": "aligned",
        },
        {
            "job": "pom_001",
            "name": "pom_001No2",
            "reason": "low_match_ratio",
            "value": "0.100",
            "threshold": "0.200",
            "status": "aligned",
        },
    ]
