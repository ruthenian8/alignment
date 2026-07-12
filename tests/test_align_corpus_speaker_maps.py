"""Tests for corpus speaker-map helper script logic."""

import csv
from pathlib import Path

from tools.align_corpus_speaker_maps import best_block_speaker, pez_jobs, text_blocks_with_speakers


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
