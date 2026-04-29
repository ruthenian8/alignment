from pathlib import Path
from unittest.mock import patch

from alignment.align import align_segments, aligned_to_srt
from alignment.audio import build_cut_command
from alignment.export import export_segments
from alignment.srt import parse_srt


def test_alignment_is_monotonic_and_preserves_original_transcript_text():
    srt = parse_srt(
        """
1
00:00:00,000 --> 00:00:01,000
[SPEAKER_00]: добрый день

2
00:00:01,000 --> 00:00:02,000
[SPEAKER_01]: красный дом
""".strip()
    )
    transcript = "до\\брый день кра\\сный дом"
    aligned = align_segments(srt, transcript, max_span=3, similarity_threshold=0.2)
    assert [item.transcript_text for item in aligned] == ["до\\брый день", "кра\\сный дом"]
    assert all(item.matched for item in aligned)
    assert "до\\брый день" in aligned_to_srt(aligned)


def test_alignment_marks_skipped_segments_explicitly():
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: unrelated\n")
    aligned = align_segments(srt, "ручной текст", similarity_threshold=0.9)
    assert aligned[0].matched is False
    assert aligned[0].transcript_text == ""


def test_export_builds_deterministic_names_and_ffmpeg_commands(tmp_path: Path):
    original = "1\n00:00:00,000 --> 00:00:01,250\n[SPEAKER_00]: original\n"
    clean = "1\n00:00:00,000 --> 00:00:01,250\n[SPEAKER_00]: clean\n"
    with patch("alignment.export.subprocess.run") as run:
        manifest = export_segments("input.wav", original, clean, tmp_path)
    base = "001_SPEAKER_00_00-00-00-000"
    assert manifest[0]["clip_id"] == base
    assert (tmp_path / f"{base}.txt").read_text(encoding="utf-8") == "clean"
    assert (tmp_path / f"{base}_orig.txt").read_text(encoding="utf-8") == "original"
    assert run.call_args.args[0] == build_cut_command(
        "input.wav", tmp_path / f"{base}.wav", "00:00:00.000", "00:00:01.250"
    )
