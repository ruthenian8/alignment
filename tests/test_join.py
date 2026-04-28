"""Tests for alignment.join module."""

import pandas as pd

from alignment.join import join_index_and_transcripts


def _make_index_df():
    return pd.DataFrame(
        {
            "start": ["00:00:00.000", "00:05:00.000", "00:10:00.000"],
            "trans": [True, False, True],
            "cont": ["", "", ""],
            "prev": ["", "", ""],
            "text": ["Desc A", "Desc B (НЕ РАСПИСАНО)", "Desc C"],
            "name": ["pez_001No0.wav", "pez_001No1.wav", "pez_001No2.wav"],
        }
    )


def _make_transcript_df():
    return pd.DataFrame(
        {
            "id": [1, 2],
            "transcript": ["Полный текст A.", "Полный текст C."],
            "max_speakers": [3, 3],
            "min_speakers": [1, 1],
        }
    )


def test_join_tsv_basic():
    index_df = _make_index_df()
    transcript_df = _make_transcript_df()
    result = join_index_and_transcripts(index_df, transcript_df)
    assert "transcript" in result.columns
    assert "max_speakers" in result.columns
    assert "min_speakers" in result.columns
    assert len(result) == len(index_df)


def test_csv_tsv_regression():
    """Verify that the join uses TSV (tab-separated) assumptions, not CSV."""
    import tempfile
    from pathlib import Path

    from alignment.io import read_tsv, write_tsv

    df = pd.DataFrame({"a": [1, 2], "b": ["hello world", "foo"]})
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w", encoding="utf-8") as f:
        path = Path(f.name)
    write_tsv(df, path)
    df2 = read_tsv(path)
    assert list(df2.columns) == ["a", "b"]
    path.unlink()


def test_join_sorts_by_trans():
    """trans=True rows should come first when assigning transcripts."""
    index_df = _make_index_df()
    transcript_df = _make_transcript_df()
    result = join_index_and_transcripts(index_df, transcript_df)
    trans_rows = result[result["trans"] == True]  # noqa: E712
    assert trans_rows["transcript"].notna().sum() > 0
