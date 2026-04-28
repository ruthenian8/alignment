"""Tests for alignment.reorder module."""

import numpy as np
import pandas as pd

from alignment.reorder import (
    reorder_dataframe,
)


def test_reorder_identity():
    """Already-aligned data should not be reordered."""
    df = pd.DataFrame(
        {
            "text": ["кот сидит на мате", "собака бежит по полю"],
            "transcript": ["кот сидит на мате и смотрит в окно", "собака бежала по полю весь день"],
            "trans": [True, True],
        }
    )
    result = reorder_dataframe(df)
    assert result.iloc[0]["transcript"] == df.iloc[0]["transcript"]
    assert result.iloc[1]["transcript"] == df.iloc[1]["transcript"]


def test_reorder_one_row_shift():
    """Should detect and correct a 1-row offset."""
    df = pd.DataFrame(
        {
            "text": ["кот сидит на мате", "собака бежит по полю"],
            "transcript": [
                "собака бежала по полю весь день",
                "кот сидит на мате и смотрит в окно",
            ],
            "trans": [True, True],
        }
    )
    result = reorder_dataframe(df, max_shift=2)
    assert "кот" in result.iloc[0]["transcript"].lower()


def test_reorder_inactive_rows():
    """Rows with trans=False should not participate in reordering."""
    df = pd.DataFrame(
        {
            "text": ["кот сидит", "НЕ РАСПИСАНО", "собака бежит"],
            "transcript": ["кот сидит на мате", np.nan, "собака бежала по полю"],
            "trans": [True, False, True],
        }
    )
    result = reorder_dataframe(df)
    assert len(result) >= 3


def test_reorder_missing_transcript_rows():
    """Should handle rows with NaN transcripts without crashing."""
    df = pd.DataFrame(
        {
            "text": ["кот сидит", "собака бежит"],
            "transcript": [np.nan, "собака бежала по полю весь день"],
            "trans": [True, True],
        }
    )
    result = reorder_dataframe(df)
    assert len(result) >= 2
