"""Joining index and transcript DataFrames."""

from __future__ import annotations

import pandas as pd


def join_index_and_transcripts(
    index_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join index rows with transcript rows.

    Sorts the index DataFrame by ``trans`` descending then ``start``
    ascending, assigns the first N transcript rows to the first N active
    (``trans=True``) index rows, and returns the joined DataFrame.

    This function uses TSV-compatible DataFrames throughout, fixing the
    CSV/TSV mismatch present in older pipeline scripts.

    Args:
        index_df: Index DataFrame with columns including ``trans`` and ``start``.
        transcript_df: Transcript DataFrame with transcript columns.

    Returns:
        Joined DataFrame with all index columns plus transcript columns
        (``transcript``, ``max_speakers``, ``min_speakers``).
    """
    df = index_df.copy()
    df = df.sort_values(["trans", "start"], ascending=[False, True])
    n_trans = transcript_df.shape[0]
    # Assign transcript data to the first n_trans rows
    trans_indexed = transcript_df.copy()
    trans_indexed.index = df.index[:n_trans]
    result = df.join(trans_indexed[["transcript", "max_speakers", "min_speakers"]])
    return result
