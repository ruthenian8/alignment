from alignment.wer import compute_wer, format_wer_report, normalize_for_wer


def aligned_row(srt_text: str, transcript_text: str, *, matched: bool = True, score: str = "0.5"):
    return {
        "srt_text": srt_text,
        "transcript_text": transcript_text,
        "matched": matched,
        "score": score,
    }


def test_wer_skips_unmatched_and_zero_score_rows():
    rows = [
        aligned_row("[SPEAKER_00]: красный дом", "[АБ:] кра\\сный дом"),
        aligned_row("лишний текст", "ручной текст", score="0"),
        aligned_row("другой текст", "ручной текст", matched=False),
    ]
    stats, mismatches = compute_wer(rows)
    assert stats.rows == 1
    assert stats.reference_words == 2
    assert stats.wer == 0
    assert not mismatches


def test_wer_uses_normalized_text_without_bracketed_notes():
    rows = [
        aligned_row(
            "[SPEAKER_00]: костоправ его звали",
            "[ХВВ:] Костопра\\в его\\ зва\\ли [историю см. XXa-10].",
        ),
        aligned_row("зелёная луна", "[Соб.: Что?] зелёная руна"),
    ]
    stats, mismatches = compute_wer(rows)
    assert stats.rows == 2
    assert stats.reference_words == 5
    assert stats.substitutions == 1
    assert stats.deletions == 0
    assert stats.insertions == 0
    assert stats.wer == 0.2
    assert mismatches[("substitute", "руна", "луна")] == 1


def test_wer_report_prints_common_mismatches():
    stats, mismatches = compute_wer([aligned_row("красный кот", "кра\\сный дом")])
    report = format_wer_report(stats, mismatches, top=1)
    assert "wer\t0.5000" in report
    assert "1\tsubstitute\tдом\tкот" in report


def test_normalize_for_wer_handles_partial_brackets_from_alignment_spans():
    assert normalize_for_wer("[Что за дом?] дом") == "дом"
    assert normalize_for_wer("Что за дом?] дом") == "дом"
    assert normalize_for_wer("[Что за дом?") == ""
