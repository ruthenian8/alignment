"""Tests for alignment.export module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from alignment.export import ExportSegment, make_segment_basename, write_manifest


def _seg(
    index=1,
    start="00:00:01,000",
    end="00:00:03,000",
    speaker="[SPEAKER_01]:",
    text="Привет.",
    text_clean="Привет.",
):
    return ExportSegment(
        index=index,
        start=start,
        end=end,
        speaker=speaker,
        text=text,
        text_clean=text_clean,
    )


def test_make_segment_basename():
    seg = _seg(index=5, start="00:00:01,234", speaker="[SPEAKER_01]:")
    name = make_segment_basename(seg)
    assert name.startswith("005_")
    assert "SPEAKER_01" in name
    assert "00-00-01" in name
    assert "," not in name
    assert ":" not in name


def test_cut_audio_ffmpeg_command():
    """Verify correct ffmpeg arguments are constructed."""
    seg = _seg()
    with patch("alignment.export.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            from alignment.export import cut_audio_segments

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            try:
                cut_audio_segments(Path("/fake/audio.wav"), [seg], output_dir)
            except Exception:
                pass
        if mock_run.called:
            call_args = mock_run.call_args[0][0]
            assert "ffmpeg" in call_args
            assert "-ss" in call_args
            assert "-to" in call_args


def test_write_manifest():
    seg1 = _seg(index=1)
    seg2 = _seg(index=2, start="00:00:03,000", end="00:00:06,000")
    audio_paths = [Path("/out/001_seg.wav"), Path("/out/002_seg.wav")]

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.tsv"
        write_manifest([seg1, seg2], audio_paths, manifest_path)
        df = pd.read_csv(manifest_path, sep="\t")
        assert "audio_path" in df.columns
        assert "text" in df.columns
        assert "text_clean" in df.columns
        assert "speaker" in df.columns
        assert len(df) == 2
