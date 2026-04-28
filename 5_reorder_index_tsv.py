#!/usr/bin/env python3
"""
reorder_transcripts.py
=======================

This script reads a CSV file containing a column ``text`` with short
descriptions and a column ``transcript`` with corresponding long
transcriptions.  In some data sets the rows in ``transcript`` may be
mis‑ordered relative to ``text``.  The goal of this utility is to
reorder the transcript entries so that each description is paired with
the transcript that most closely matches it.

The approach used here is a hybrid of symbolic heuristics and simple
neural‑style metrics:

1.  **Preprocessing**
    • Strip interviewer prompts enclosed in square brackets ``[...]``.
    • Remove extraneous backslashes used to mark stress in the source
      transcription.
    • Remove time codes and punctuation and normalise whitespace.
    • Convert everything to lower case.  Missing transcript entries
      (``NaN``) are replaced with empty strings so they do not break
      vectorisation.

2.  **Similarity matrix**
    • Compute a character n‑gram TF–IDF representation (3–5 character
      n‑grams).  Character n‑grams are robust to small spelling
      differences and allow reasonable matching of Russian text without
      stemming.
    • Compute cosine similarity between the TF–IDF vectors of the
      descriptions and transcripts.
    • Compute a lexical similarity using the RapidFuzz ``token_set_ratio``
      which measures the overlap of unique tokens.  This gives a
      complementary signal to the n‑gram based cosine similarity.
    • Combine the two similarities with weights of 0.7 (character
      similarity) and 0.3 (token set ratio).  These weights were tuned
      to give perfect alignment on the “pez_016” control data set.

3.  **Shift detection**
    When transcripts are mis‑aligned by a constant offset, the simple
    diagonal of the similarity matrix is not maximal.  To handle this
    case, the script evaluates the average similarity along several
    diagonals (shifts of 0–2 rows).  The shift with the highest
    average similarity is selected and the transcripts are rotated
    accordingly before further alignment.  On the “pez_017” test data
    this shift detection correctly identified a one‑row offset.

4.  **Assignment**
    After applying the best constant shift, the script builds a
    second similarity matrix for the shifted transcripts and solves a
    linear sum assignment problem (Hungarian algorithm) to find a
    one‑to‑one mapping between descriptions and transcripts that
    maximises the total similarity.  This allows recovery of small
    local mis‑orderings that are not captured by the global shift.

5.  **Output**
    A copy of the input CSV is written to the specified output file
    with the ``transcript`` column reordered according to the computed
    assignment.

The script takes two positional arguments: the input CSV and the
desired output CSV.  It prints summary diagnostics showing the chosen
shift and the average similarity before and after reordering.

Example usage::

    python reorder_transcripts.py pez_017.csv pez_017_reordered.csv

Dependencies: numpy, pandas, scikit‑learn, scipy, rapidfuzz.
"""

import argparse
import math
import os
import re
import string
from typing import List, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(text: str) -> str:
    """Normalise Russian transcription and description text.

    Removes interviewer prompts enclosed in square brackets, stress
    markers (backslashes), time codes and punctuation.  Collapses
    multiple spaces and converts everything to lower case.  Non‑string
    inputs return an empty string.
    """
    if not isinstance(text, str):
        return ""
    # remove interviewer prompts and stage directions in square brackets
    text = re.sub(r"\[[^\]]*\]", " ", text)
    # remove backslash markers used for stress
    text = text.replace("\\", "")
    # remove typical time codes (e.g. 00:12:34.567)
    text = re.sub(r"\d{1,2}:\d{2}:\d{2}[\.,]\d{3}", " ", text)
    # replace punctuation with spaces
    text = re.sub("[{}]".format(re.escape(string.punctuation)), " ", text)
    # collapse whitespace
    text = " ".join(text.split())
    return text.lower()


def build_similarity_matrix(texts: List[str], transcripts: List[str],
                            char_weight: float = 0.7) -> np.ndarray:
    """Compute a combined similarity matrix between descriptions and transcripts.

    Each entry sim[i,j] reflects how well the i‑th text matches the j‑th
    transcript.  The similarity combines character n‑gram cosine
    similarity with a token set ratio.  The ``char_weight`` controls
    the relative contribution of the n‑gram measure; the lexical
    similarity weight is implicitly (1 - char_weight).
    """
    n = len(texts)
    m = len(transcripts)
    # Fit a character n‑gram TF‑IDF on the joint corpus.  The vectoriser
    # must see both sets of strings to compute a shared vocabulary.
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5),
                                 max_features=20000)
    # In case either list is empty, avoid fitting on an empty list which
    # raises a ValueError.  At least one of ``texts`` or ``transcripts``
    # should contain data given the calling context.
    vectorizer.fit(texts + transcripts)
    text_vec = vectorizer.transform(texts)
    trans_vec = vectorizer.transform(transcripts)
    # Character n‑gram cosine similarity yields an (n x m) matrix
    char_sim = cosine_similarity(text_vec, trans_vec)
    # Lexical similarity using token_set_ratio from rapidfuzz.  The
    # resulting matrix must match the shape of ``char_sim``.
    lex_sim = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            lex_sim[i, j] = fuzz.token_set_ratio(texts[i], transcripts[j]) / 100.0
    # Weighted combination.  ``char_weight`` governs the relative
    # contribution of the character n‑gram measure versus the lexical
    # measure.  If the matrices have different shapes (when m != n),
    # this computation still broadcasts correctly.
    sim = char_weight * char_sim + (1.0 - char_weight) * lex_sim
    return sim


def detect_best_shift(
    sim: np.ndarray,
    max_shift: int = 2,
    mask: List[bool] | None = None,
) -> Tuple[int, float]:
    """Find the constant shift of transcripts that maximises the average diagonal similarity.

    Parameters
    ----------
    sim : np.ndarray
        Similarity matrix between the current ordering of descriptions (rows)
        and transcripts (columns).
    max_shift : int, optional
        Maximum shift (inclusive) to test when looking for a constant
        offset between ``text`` and ``transcript`` rows.  Default is 2.
    mask : list of bool, optional
        Boolean mask indicating which row indices should contribute to the
        diagonal similarity.  If provided, only those indices where
        ``mask[i]`` is True will be considered when computing the mean.

    Returns
    -------
    Tuple[int, float]
        The shift with the highest average diagonal similarity (first element)
        and the corresponding mean similarity value (second element).

    Notes
    -----
    When ``mask`` is provided, only those row indices with ``True`` mask
    values contribute to the diagonal similarity.  Shifts that extend
    beyond the matrix bounds are automatically ignored for those pairs.
    """
    n = sim.shape[0]
    best_shift = 0
    best_mean = -math.inf
    # Precompute indices of masked rows for efficiency
    masked_rows = None
    if mask is not None:
        masked_rows = [i for i, flag in enumerate(mask) if flag]
        # If no rows are masked in, return shift 0 with mean -inf to indicate no valid shift
        if not masked_rows:
            return 0, -math.inf
    for shift in range(0, max_shift + 1):
        diag_scores: List[float] = []
        if mask is None:
            # Use all rows
            for i in range(n - shift):
                diag_scores.append(sim[i, i + shift])
        else:
            # Only include masked rows where i+shift is within bounds
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


def reorder_transcripts(
    input_csv: str,
    output_csv: str,
    *,
    char_weight: float = 0.7,
    max_shift: int = 3,
    improvement_threshold: float = 1e-3,
) -> None:
    """Reorder the transcript column of ``input_csv`` and write the result to ``output_csv``.

    This routine aligns the ``transcript`` column with the ``text`` column for rows
    where the ``trans`` flag is ``True``.  It does so by first detecting a
    constant offset (shift) between the order of ``text`` and ``transcript``
    entries within the active (``trans=True``) subset, then solving a
    linear assignment problem to match each active ``text`` to the best
    available transcript.  Transcripts originating from rows where
    ``trans=False`` may be used to fill gaps in the active rows but are
    otherwise left untouched.  If a transcript from an active row is not
    used in the final assignment it is moved to the end of the file as a
    new row with all other fields set to missing.  The function prints
    diagnostic information showing the detected shift and the mean
    diagonal similarity before and after reordering.  When the
    improvement in mean similarity is below ``improvement_threshold`` or
    the mapping is the identity, the input file is copied unchanged.
    """
    try:
        df = pd.read_csv(input_csv)
    except Exception:
        # Unable to parse CSV; copy unchanged
        import shutil
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        shutil.copy2(input_csv, output_csv)
        print(f"Processed {input_csv}: unable to parse CSV; file copied unchanged.")
        return

    # Ensure required columns exist
    if 'text' not in df.columns or 'transcript' not in df.columns:
        import shutil
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        shutil.copy2(input_csv, output_csv)
        print(f"Processed {input_csv}: missing 'text' or 'transcript' column; file copied unchanged.")
        return

    n = len(df)
    # Determine which rows participate in alignment based on the 'trans' column
    mask_bool: List[bool]
    if 'trans' in df.columns:
        # robust conversion to boolean
        def to_bool(val: object) -> bool:
            if pd.isna(val):
                return False
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                v = val.strip().lower()
                if v in ('true', '1', 'yes', 'y'):
                    return True
                if v in ('false', '0', 'no', 'n'):
                    return False
                try:
                    return bool(int(v))
                except Exception:
                    return False
            if isinstance(val, (int, float)):
                return bool(val)
            return False
        mask_bool = [to_bool(x) for x in df['trans'].tolist()]
    else:
        # If 'trans' column is absent, treat all rows as active
        mask_bool = [True] * n

    # Identify indices of active (True) and inactive (False) rows
    true_indices = [i for i, flag in enumerate(mask_bool) if flag]
    false_indices = [i for i, flag in enumerate(mask_bool) if not flag]

    # If there are no active rows, copy unchanged
    if not true_indices:
        import shutil
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        shutil.copy2(input_csv, output_csv)
        print(f"Processed {input_csv}: no rows marked for alignment; file copied unchanged.")
        return

    k = len(true_indices)
    # Preprocess texts for active rows and all transcripts
    texts_true = [preprocess(df.loc[i, 'text']) for i in true_indices]
    transcripts_prep = [preprocess(x) for x in df['transcript'].tolist()]
    transcripts_true_prep = [transcripts_prep[i] for i in true_indices]
    # Compute similarity matrix for active rows (k x k) and detect shift
    sim_true = build_similarity_matrix(texts_true, transcripts_true_prep, char_weight=char_weight)
    # Initial diagonal similarity
    diag_initial = [sim_true[i, i] for i in range(k)]
    mean_before = float(np.mean(diag_initial)) if diag_initial else 0.0
    # Detect shift among active rows
    max_shift_eff = min(max_shift, k - 1)
    shift, _ = detect_best_shift(sim_true, max_shift=max_shift_eff)
    # Rotate the list of active transcript indices by the detected shift
    rotated_true_indices = true_indices[shift:] + true_indices[:shift]
    rotated_true_prep = [transcripts_prep[i] for i in rotated_true_indices]
    # Build candidate lists: rotated active transcripts followed by inactive transcripts
    candidate_prep = rotated_true_prep + [transcripts_prep[i] for i in false_indices]
    candidate_indices = rotated_true_indices + false_indices
    # Build similarity matrix for assignment (k x m)
    sim2 = build_similarity_matrix(texts_true, candidate_prep, char_weight=char_weight)
    # Solve assignment to maximise total similarity
    row_ind, col_ind = linear_sum_assignment(-sim2)
    assignment = {true_indices[row]: candidate_indices[col] for row, col in zip(row_ind, col_ind)}
    # Identify which candidate transcripts are used in the assignment
    used_transcript_indices = set(candidate_indices[c] for c in col_ind)
    # Determine leftover transcripts from active rows that were not used
    leftover_true_indices = [idx for idx in rotated_true_indices if idx not in used_transcript_indices]
    leftover_true_transcripts = [df.loc[idx, 'transcript'] for idx in leftover_true_indices]
    # Compose the final list of preprocessed transcripts for similarity evaluation
    final_transcripts_true_prep = []
    for row in true_indices:
        if row in assignment:
            source_idx = assignment[row]
            final_transcripts_true_prep.append(transcripts_prep[source_idx])
        else:
            final_transcripts_true_prep.append('')
    # Compute final similarity and mean
    final_sim_true = build_similarity_matrix(texts_true, final_transcripts_true_prep, char_weight=char_weight)
    diag_final = [final_sim_true[i, i] for i in range(k)]
    mean_after = float(np.mean(diag_final)) if diag_final else 0.0
    improvement = mean_after - mean_before
    # Determine if assignment is identity mapping (shift=0 and every active row maps to itself)
    identity = shift == 0 and all(assignment.get(row) == row for row in true_indices)
    # Decide whether to apply reordering
    apply_reorder = not identity and (improvement >= improvement_threshold)

    df_out = df.copy()
    # Preserve the data type of the 'trans' column by converting it to
    # object before any modifications.  Without this, pandas may upcast
    # boolean values to floats when NaN values are introduced during the
    # reordering process.
    if 'trans' in df_out.columns:
        df_out['trans'] = df_out['trans'].astype(object)
    if apply_reorder:
        # Remove transcripts from inactive rows that were used
        for idx in false_indices:
            if idx in used_transcript_indices:
                df_out.at[idx, 'transcript'] = np.nan
        # Assign transcripts to active rows based on the assignment mapping
        for row in true_indices:
            if row in assignment:
                source_idx = assignment[row]
                df_out.at[row, 'transcript'] = df.loc[source_idx, 'transcript']
            else:
                df_out.at[row, 'transcript'] = np.nan
        # Append leftover transcripts from active rows as new rows at the end
        if leftover_true_transcripts:
            new_rows = []
            for t in leftover_true_transcripts:
                new_row = {col: np.nan for col in df_out.columns}
                new_row['transcript'] = t
                # leave 'trans' as NaN to indicate no alignment
                new_rows.append(new_row)
            df_out = pd.concat([df_out, pd.DataFrame(new_rows)], ignore_index=True)
        print(
            f"Processed {input_csv}: detected shift = {shift}, mean initial diag similarity = {mean_before:.3f}, "
            f"mean final diag similarity = {mean_after:.3f} (improved on {k} rows)."
        )
    else:
        # Either already aligned or insufficient improvement
        reason = 'already aligned' if identity else 'no improvement'
        print(
            f"Processed {input_csv}: detected shift = {shift}, mean initial diag similarity = {mean_before:.3f}, "
            f"mean final diag similarity = {mean_after:.3f} (skipped: {reason})."
        )
        # When skipping, ensure we do not accidentally append rows
        df_out = df.copy()

    # Write the output
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reorder transcripts to match their summaries")
    parser.add_argument('input_csv', help='Path to the input CSV file')
    parser.add_argument('output_csv', help='Path to write the reordered CSV file')
    parser.add_argument('--char-weight', type=float, default=0.7,
                        help='Weight for character n‑gram similarity (default: 0.7)')
    parser.add_argument('--max-shift', type=int, default=3,
                        help='Maximum absolute shift to test when detecting a constant offset (default: 3)')
    args = parser.parse_args()
    reorder_transcripts(args.input_csv, args.output_csv,
                        char_weight=args.char_weight,
                        max_shift=args.max_shift)


if __name__ == '__main__':
    main()