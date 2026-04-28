"""TSV I/O utilities and schema constants."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Column schema constants
INDEX_COLUMNS = ["start", "trans", "cont", "prev", "text", "name"]

TRANSCRIPT_COLUMNS = ["id", "transcript", "max_speakers", "min_speakers"]

JOINED_COLUMNS = INDEX_COLUMNS + ["transcript", "max_speakers", "min_speakers"]

ALIGNED_COLUMNS = [
    "index",
    "start",
    "end",
    "speaker",
    "original_text",
    "transcript_text",
    "matched",
]

MANIFEST_COLUMNS = ["audio_path", "text", "text_clean", "speaker", "start", "end", "index"]


def read_tsv(path: Path) -> pd.DataFrame:
    """Read a TSV file into a DataFrame.

    Args:
        path: Path to the TSV file (UTF-8 encoded, tab-separated).

    Returns:
        DataFrame with the file contents.
    """
    return pd.read_csv(path, sep="\t", encoding="utf-8")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to a TSV file.

    Args:
        df: DataFrame to write.
        path: Destination path (will be created or overwritten).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")
