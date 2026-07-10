from alignment.speakers import (
    bracket_speakers_from_text,
    fill_speaker_gaps,
    infer_table_speakers,
    leading_context_speaker,
    speaker_codes_from_stem,
    summarize_speaker_maps,
)


def aligned_row(index: int, normalized: str = "", speaker: str = "[SPEAKER_00]:") -> dict[str, str]:
    return {
        "srt_index": str(index),
        "speaker": speaker,
        "transcript_text": "",
        "normalized_text": normalized,
        "matched": "True" if normalized else "False",
        "score": "1.000" if normalized else "0.000",
    }


def test_leading_context_speaker_ignores_sentence_initial_one_letter_bracket():
    assert leading_context_speaker("[А почему так?] Ответ.") == ""
    assert leading_context_speaker("[Л:] Ответ.") == "Л"
    assert leading_context_speaker("[ГКЕ училась в интернате. Соб.: Вопрос?] Ответ.") == "ГКЕ"


def test_bridge_fill_between_identical_speaker_anchors():
    rows = [
        {"srt_index": 1, "tag": "АБ", "source": "preceding_marker", "matched": "True", "score": "1.000"},
        {"srt_index": 2, "tag": "", "source": "unmatched", "matched": "True", "score": "1.000"},
        {"srt_index": 3, "tag": "АБ", "source": "preceding_marker", "matched": "True", "score": "1.000"},
    ]
    filled = fill_speaker_gaps(rows, "")
    assert [row["tag"] for row in filled] == ["АБ", "АБ", "АБ"]
    assert filled[1]["source"] == "bridged_same_speaker"


def test_carry_forward_does_not_cross_unknown_collector_anchor():
    rows = [
        {"srt_index": 1, "tag": "АБ", "source": "preceding_marker", "matched": "True", "score": "1.000"},
        {"srt_index": 2, "tag": "UNK", "source": "collector_bracket", "matched": "True", "score": "1.000"},
        {"srt_index": 3, "tag": "", "source": "unmatched", "matched": "False", "score": "0.000"},
    ]
    filled = fill_speaker_gaps(rows, "")
    assert [row["tag"] for row in filled] == ["АБ", "UNK", ""]


def test_leading_context_can_resume_after_opening_collector_anchor():
    rows = [
        {"srt_index": 1, "tag": "UNK", "source": "collector_bracket", "matched": "True", "score": "1.000"},
        {"srt_index": 2, "tag": "", "source": "unmatched", "matched": "True", "score": "1.000"},
    ]
    filled = fill_speaker_gaps(rows, "ГКЕ")
    assert [row["tag"] for row in filled] == ["UNK", "ГКЕ"]
    assert filled[1]["source"] == "leading_context"


def test_infer_table_speakers_prefers_non_whisperx_table_speakers():
    inferred = infer_table_speakers(
        [aligned_row(1, "добрый день", speaker="[ДС]:")],
        "[А почему так?] до\\брый день",
        prefer_table_speakers=True,
    )
    assert inferred[1]["transcript_speaker"] == "[ДС]:"
    assert inferred[1]["speaker_source"] == "aligned_table"


def test_infer_table_speakers_uses_preceding_marker_for_following_matched_rows():
    inferred = infer_table_speakers(
        [
            aligned_row(1, "добрый день"),
            aligned_row(2, "красный дом"),
        ],
        "[АБ:] до\\брый день кра\\сный дом",
    )
    assert inferred[1]["transcript_speaker"] == "[АБ]:"
    assert inferred[2]["transcript_speaker"] == "[АБ]:"
    assert inferred[2]["speaker_source"] == "preceding_marker"


def test_infer_table_speakers_does_not_fill_unmatched_rows():
    inferred = infer_table_speakers(
        [
            aligned_row(1, "добрый день"),
            aligned_row(2),
        ],
        "[АБ:] до\\брый день",
    )
    assert inferred[1]["transcript_speaker"] == "[АБ]:"
    assert inferred[2]["transcript_speaker"] == ""
    assert inferred[2]["speaker_source"] == "unmatched"


def test_raw_transcript_inventory_helpers_extract_filename_and_bracket_speakers(tmp_path):
    path = tmp_path / "BVI&SIA&SNS_txt.txt"
    path.write_text("[БВИ:] текст [Что?] [???:] ответ", encoding="utf-8")
    assert speaker_codes_from_stem(path) == "BVI, SIA, SNS"
    assert bracket_speakers_from_text(path.read_text(encoding="utf-8")) == "БВИ, ???"


def test_summarize_speaker_maps_uses_requested_output_name(tmp_path):
    map_dir = tmp_path / "cut" / "pez_001" / "pez_001No0"
    map_dir.mkdir(parents=True)
    (map_dir / "custom_speakers.csv").write_text(
        "audio_file,text_file,text_original_file,srt_index,timestamp,whisperx_speaker,"
        "transcript_speaker,speaker_source,aligned_matched,alignment_score\n"
        "001.wav,001.txt,001_orig.txt,1,00-00-00-000,[SPEAKER_00]:,[ААК]:,speaker_hint,True,1.000\n"
        "002.wav,002.txt,002_orig.txt,2,00-00-01-000,[SPEAKER_00]:,,unmatched,True,0.800\n"
        "003.wav,003.txt,003_orig.txt,3,00-00-02-000,[SPEAKER_00]:,,unmatched,False,0.000\n",
        encoding="utf-8",
    )

    metrics = summarize_speaker_maps(tmp_path / "cut", tmp_path / "summary", "custom_speakers.csv")

    assert metrics["rows"] == 3
    assert metrics["inferred_rows"] == 1
    assert metrics["matched_blank_rows"] == 1
    assert metrics["unmatched_blank_rows"] == 1
