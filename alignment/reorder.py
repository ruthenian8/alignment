"""Transcript reordering to align descriptions with transcripts."""

from __future__ import annotations

import math
import re
import string
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(text: str) -> str:
    """Normalise text for similarity comparison.

    Removes interviewer prompts in square brackets, stress markers
    (backslashes), time codes, punctuation, and extra whitespace.
    Converts to lowercase.

    Args:
        text: Input text string. Non-string values return empty string.

    Returns:
        Normalised lowercase string.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.replace("\\", "")
    text = re.sub(r"\d{1,2}:\d{2}:\d{2}[\.,]\d{3}", " ", text)
    text = re.sub("[{}]".format(re.escape(string.punctuation)), " ", text)
    text = " ".join(text.split())
    return text.lower()


def build_similarity_matrix(
    texts: List[str],
    transcripts: List[str],
    char_weight: float = 0.7,
) -> np.ndarray:
    """Compute a combined similarity matrix between descriptions and transcripts.

    Combines character n-gram TF-IDF cosine similarity with rapidfuzz
    token set ratio. The ``char_weight`` parameter controls the balance
    between the two measures.

    Args:
        texts: List of preprocessed description strings.
        transcripts: List of preprocessed transcript strings.
        char_weight: Weight for the character n-gram measure (0-1).

    Returns:
        Float numpy array of shape (len(texts), len(transcripts)).
    """
    n = len(texts)
    m = len(transcripts)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=20000)
    vectorizer.fit(texts + transcripts)
    text_vec = vectorizer.transform(texts)
    trans_vec = vectorizer.transform(transcripts)
    char_sim = cosine_similarity(text_vec, trans_vec)
    lex_sim = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            lex_sim[i, j] = fuzz.token_set_ratio(texts[i], transcripts[j]) / 100.0
    return char_weight * char_sim + (1.0 - char_weight) * lex_sim


def detect_best_shift(
    sim: np.ndarray,
    max_shift: int = 2,
    mask: Optional[List[bool]] = None,
) -> Tuple[int, float]:
    """Find the constant transcript shift that maximises diagonal similarity.

    Args:
        sim: Similarity matrix of shape (n, n).
        max_shift: Maximum shift to test (inclusive).
        mask: Optional boolean mask; only masked-True rows contribute.

    Returns:
        Tuple of (best_shift, best_mean_similarity).
    """
    n = sim.shape[0]
    best_shift = 0
    best_mean = -math.inf
    masked_rows = None
    if mask is not None:
        masked_rows = [i for i, flag in enumerate(mask) if flag]
        if not masked_rows:
            return 0, -math.inf
    for shift in range(0, max_shift + 1):
        diag_scores: List[float] = []
        if mask is None:
            for i in range(n - shift):
                diag_scores.append(sim[i, i + shift])
        else:
            for i in masked_rows:
                j = i + shift
                if j < n:
                    diag_scores.append(sim[i, j])
        if not diag_scores:
            continue
        mean_sim = float(np.mean(diag_scores))
        if mean_sim > best_mean:
            best_mean = mean_sim
            best_shift = shift
    return best_shift, best_mean


def _to_bool(val: object) -> bool:
    """Convert a value to bool, treating NaN and missing as False."""
    if pd.isna(val):  # type: ignore[arg-type]
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "1", "yes", "y"):
            return True
        if v in ("false", "0", "no", "n"):
            return False
        try:
            return bool(int(v))
        except Exception:
            return False
    if isinstance(val, (int, float)):
        return bool(val)
    return False


def reorder_dataframe(
    df: pd.DataFrame,
    char_weight: float = 0.7,
    max_shift: int = 3,
    improvement_threshold: float = 1e-3,
) -> pd.DataFrame:
    """Reorder the ``transcript`` column to best match the ``text`` column.

    Only rows where ``trans`` is ``True`` (or all rows if ``trans`` is
    absent) participate in alignment. The function detects a constant
    offset shift and then solves a linear assignment problem to find the
    best one-to-one mapping. If the reordering does not improve the mean
    diagonal similarity by at least ``improvement_threshold``, the
    DataFrame is returned unchanged.

    Args:
        df: Input DataFrame with ``text`` and ``transcript`` columns.
        char_weight: Weight for the character n-gram similarity measure.
        max_shift: Maximum shift to test for constant offset detection.
        improvement_threshold: Minimum improvement to apply reordering.

    Returns:
        DataFrame with ``transcript`` column reordered (or unchanged).
    """
    if "text" not in df.columns or "transcript" not in df.columns:
        return df.copy()

    n = len(df)
    if "trans" in df.columns:
        mask_bool = [_to_bool(x) for x in df["trans"].tolist()]
    else:
        mask_bool = [True] * n

    true_indices = [i for i, flag in enumerate(mask_bool) if flag]
    false_indices = [i for i, flag in enumerate(mask_bool) if not flag]

    if not true_indices:
        return df.copy()

    k = len(true_indices)
    texts_true = [preprocess(df.iloc[i]["text"]) for i in true_indices]
    transcripts_prep = [preprocess(x) for x in df["transcript"].tolist()]
    transcripts_true_prep = [transcripts_prep[i] for i in true_indices]

    sim_true = build_similarity_matrix(texts_true, transcripts_true_prep, char_weight=char_weight)
    diag_initial = [sim_true[i, i] for i in range(k)]
    mean_before = float(np.mean(diag_initial)) if diag_initial else 0.0

    max_shift_eff = min(max_shift, k - 1)
    shift, _ = detect_best_shift(sim_true, max_shift=max_shift_eff)

    rotated_true_indices = true_indices[shift:] + true_indices[:shift]
    rotated_true_prep = [transcripts_prep[i] for i in rotated_true_indices]
    candidate_prep = rotated_true_prep + [transcripts_prep[i] for i in false_indices]
    candidate_indices = rotated_true_indices + false_indices

    sim2 = build_similarity_matrix(texts_true, candidate_prep, char_weight=char_weight)
    row_ind, col_ind = linear_sum_assignment(-sim2)
    assignment = {true_indices[row]: candidate_indices[col] for row, col in zip(row_ind, col_ind)}
    used_transcript_indices = set(candidate_indices[c] for c in col_ind)
    leftover_true_indices = [
        idx for idx in rotated_true_indices if idx not in used_transcript_indices
    ]
    leftover_true_transcripts = [df.iloc[idx]["transcript"] for idx in leftover_true_indices]

    final_transcripts_true_prep = []
    for row in true_indices:
        if row in assignment:
            source_idx = assignment[row]
            final_transcripts_true_prep.append(transcripts_prep[source_idx])
        else:
            final_transcripts_true_prep.append("")

    final_sim_true = build_similarity_matrix(
        texts_true, final_transcripts_true_prep, char_weight=char_weight
    )
    diag_final = [final_sim_true[i, i] for i in range(k)]
    mean_after = float(np.mean(diag_final)) if diag_final else 0.0
    improvement = mean_after - mean_before

    identity = shift == 0 and all(assignment.get(row) == row for row in true_indices)
    apply_reorder = not identity and (improvement >= improvement_threshold)

    df_out = df.copy()
    if "trans" in df_out.columns:
        df_out["trans"] = df_out["trans"].astype(object)

    if apply_reorder:
        for idx in false_indices:
            if idx in used_transcript_indices:
                df_out.at[idx, "transcript"] = np.nan
        for row in true_indices:
            if row in assignment:
                source_idx = assignment[row]
                df_out.at[row, "transcript"] = df.iloc[source_idx]["transcript"]
            else:
                df_out.at[row, "transcript"] = np.nan
        if leftover_true_transcripts:
            new_rows = []
            for t in leftover_true_transcripts:
                new_row = {col: np.nan for col in df_out.columns}
                new_row["transcript"] = t
                new_rows.append(new_row)
            df_out = pd.concat([df_out, pd.DataFrame(new_rows)], ignore_index=True)

    return df_out
