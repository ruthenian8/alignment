"""Optional end-to-end checks against the linked Hugging Face repository."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from alignment.align import transcript_with_block_speaker_markers
from alignment.cli import main
from alignment.io import read_tsv
from alignment.srt import parse_srt

HF_ROOT = Path(__file__).resolve().parents[1] / "hf-repo"


@pytest.mark.skipif(not HF_ROOT.exists(), reason="hf-repo link is not available")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for clip export")
def test_hf_repo_pez_001_pipeline_smoke(tmp_path: Path) -> None:
    """Run a real HF fixture through the supported stages without writing to hf-repo."""
    index_tsv = tmp_path / "pez_001.index.tsv"
    transcript_tsv = tmp_path / "pez_001.transcript.tsv"
    joined_tsv = tmp_path / "pez_001.joined.tsv"
    reordered_tsv = tmp_path / "pez_001.reordered.tsv"
    manual_text = tmp_path / "pez_001No0.manual.txt"
    aligned_srt = tmp_path / "pez_001No0.aligned.srt"
    aligned_tsv = tmp_path / "pez_001No0.aligned.tsv"
    manifest_tsv = tmp_path / "manifest.tsv"

    main(["parse-index", str(HF_ROOT / "indices" / "pez_001.txt"), str(index_tsv), "--audio-stem", "pez_001"])
    main(["parse-transcript", str(HF_ROOT / "transcripts" / "pez_001.txt"), str(transcript_tsv)])
    main(["join", str(index_tsv), str(transcript_tsv), str(joined_tsv)])
    main(["reorder", str(joined_tsv), str(reordered_tsv)])

    joined_rows = read_tsv(reordered_tsv)
    assert len(joined_rows) == 38
    active_rows = [row for row in joined_rows if row["trans"].lower() == "true"]
    assert len(active_rows) == 24
    assert len([row for row in active_rows if row["transcript"]]) == 22
    assert all(row["transcript"] == "" for row in joined_rows if row["trans"].lower() != "true")

    manual_text.write_text(joined_rows[0]["transcript"], encoding="utf-8")
    main(
        [
            "align-srt",
            str(HF_ROOT / "wx_transcripts" / "pez_001" / "pez_001No0.srt"),
            str(manual_text),
            str(aligned_srt),
            "--output-tsv",
            str(aligned_tsv),
            "--index-name",
            "pez_001No0",
            "--use-transcript-speakers",
            "--infer-missing-speakers",
        ]
    )

    aligned_segments = parse_srt(aligned_srt.read_text(encoding="utf-8"))
    assert len(aligned_segments) == 8
    assert aligned_segments[0].speaker == "[UNK]:"
    assert all(segment.speaker == "[ААК]:" for segment in aligned_segments[1:])
    assert "Часовня" in aligned_segments[0].text
    assert "часо\\вня" in aligned_segments[1].text

    main(
        [
            "export-corpus",
            str(HF_ROOT / "cut_audio" / "pez_001" / "pez_001No0.wav"),
            str(HF_ROOT / "wx_transcripts" / "pez_001" / "pez_001No0.srt"),
            str(aligned_srt),
            str(tmp_path / "clips"),
            str(manifest_tsv),
        ]
    )

    manifest_rows = read_tsv(manifest_tsv)
    assert len(manifest_rows) == 8
    assert Path(manifest_rows[0]["audio_path"]).exists()
    assert Path(manifest_rows[0]["text_path"]).read_text(encoding="utf-8") == aligned_segments[0].text
    assert "Вот эта часовня ваша" in Path(manifest_rows[0]["text_original_path"]).read_text(encoding="utf-8")


@pytest.mark.skipif(not HF_ROOT.exists(), reason="hf-repo link is not available")
def test_hf_repo_pez_011_speaker_replacement_smoke(tmp_path: Path) -> None:
    """Verify real pez_011 speaker tags, collector questions, and ??? tags."""
    blocks = [
        block
        for block in (HF_ROOT / "transcripts" / "pez_011.txt")
        .read_text(encoding="utf-8-sig")
        .strip()
        .split("\n\n")
        if block.strip()
    ]
    assert "[???:]" in blocks[0]
    assert "[М:]" in blocks[1]
    assert "[ЛД:]" in blocks[1]
    assert "[UNK:]" not in transcript_with_block_speaker_markers(blocks[0])
    assert "[МВ, ???:]" in transcript_with_block_speaker_markers(blocks[0])
    assert "[ДГ, ДС, ЛД, МВ, М, ???:]" in transcript_with_block_speaker_markers(blocks[1])

    transcript = tmp_path / "pez_011No0.txt"
    aligned_srt = tmp_path / "pez_011No0.srt"
    transcript.write_text(blocks[0], encoding="utf-8")
    main(
        [
            "align-srt",
            str(HF_ROOT / "wx_transcripts" / "pez_011" / "pez_011No0.srt"),
            str(transcript),
            str(aligned_srt),
            "--use-transcript-speakers",
            "--infer-missing-speakers",
        ]
    )

    speakers = {segment.speaker for segment in parse_srt(aligned_srt.read_text(encoding="utf-8"))}
    assert "[UNK]:" in speakers
    assert "[МВ, ???]:" in speakers
