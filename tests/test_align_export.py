from pathlib import Path
from unittest.mock import patch

import pytest

from alignment.align import (
    AlignedSegment,
    align_segments,
    align_srt_file,
    aligned_to_speaker_map_rows,
    aligned_to_srt,
    apply_transcript_speakers,
    remove_alignment_notes,
)
from alignment.audio import build_cut_command
from alignment.cli import main
from alignment.export import (
    clean_speaker_code,
    export_aligned_srt_tree,
    export_segments,
    normalize_caption_text,
)
from alignment.srt import SrtSegment, parse_srt


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


def test_alignment_can_skip_leading_manual_context():
    srt = parse_srt(
        """
1
00:00:00,000 --> 00:00:01,000
[SPEAKER_00]: потому что не уронили

2
00:00:01,000 --> 00:00:02,000
[SPEAKER_00]: там неправильно
""".strip()
    )
    transcript = (
        "[Соб. длинное описание праздника. Соб.: А почему не пускали детей?] "
        "А потому\\ что не урони\\ли там, непра\\вильно."
    )

    aligned = align_segments(
        srt,
        transcript,
        max_span=5,
        similarity_threshold=0.2,
        allow_leading_transcript_skip=True,
    )

    assert [item.matched for item in aligned] == [True, True]
    assert "потому\\ что не урони\\ли" in aligned[0].transcript_text
    assert "там, непра\\вильно" in aligned[1].transcript_text


def test_leading_speaker_marker_carries_after_skipped_context():
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: ручной ответ\n")
    long_context = " ".join(["контекст"] * 80)
    transcript = f"[Л:] [{long_context}. Соб.: Вопрос?] Ручно\\й отве\\т."

    aligned = align_segments(
        srt,
        transcript,
        similarity_threshold=0.2,
        allow_leading_transcript_skip=True,
    )
    updated = apply_transcript_speakers(aligned, transcript, infer_missing=True)

    assert updated[0].matched is True
    assert updated[0].srt.speaker == "[Л]:"
    assert updated[0].speaker_source == "carried_forward_prev"


def test_alignment_can_require_starting_at_first_manual_token():
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: ответ\n")

    aligned = align_segments(
        srt,
        "лишний контекст ответ",
        similarity_threshold=0.8,
        allow_leading_transcript_skip=False,
    )

    assert aligned[0].matched is False


def test_alignment_does_not_force_unrelated_text_after_leading_skip():
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: unrelated\n")

    aligned = align_segments(srt, "лишний контекст ручной текст", similarity_threshold=0.9)

    assert aligned[0].matched is False


def test_transcript_speaker_tags_can_replace_srt_speakers():
    srt = parse_srt(
        """
1
00:00:00,000 --> 00:00:01,000
[SPEAKER_00]: добрый день

2
00:00:01,000 --> 00:00:02,000
[SPEAKER_00]: красный дом

3
00:00:02,000 --> 00:00:03,000
[SPEAKER_01]: новый день
""".strip()
    )
    transcript = "[ААК:] до\\брый день кра\\сный дом [РВВ:] но\\вый день"
    aligned = align_segments(srt, transcript, max_span=4, similarity_threshold=0.2)
    updated = apply_transcript_speakers(aligned, transcript, infer_missing=True)
    assert [item.srt.speaker for item in updated] == ["[ААК]:", "[ААК]:", "[РВВ]:"]
    assert "[ААК]: до\\брый день" in aligned_to_srt(updated)
    speaker_rows = aligned_to_speaker_map_rows(updated)
    assert speaker_rows[0]["whisperx_speaker"] == "[SPEAKER_00]:"
    assert speaker_rows[0]["transcript_speaker"] == "[ААК]:"
    assert speaker_rows[1]["speaker_source"] == "preceding_marker"


def test_unknown_bracket_questions_and_nonstandard_speakers_replace_srt_speakers():
    srt = parse_srt(
        """
1
00:00:00,000 --> 00:00:01,000
[SPEAKER_00]: как играть?

2
00:00:01,000 --> 00:00:02,000
[SPEAKER_01]: в ладушки

3
00:00:02,000 --> 00:00:03,000
[SPEAKER_02]: надо положить конфетки

4
00:00:03,000 --> 00:00:04,000
[SPEAKER_03]: непонятно как
""".strip()
    )
    transcript = (
        "[Как игра\\ть?] [М:] В ла\\душки. [ЛД:] На\\до положи\\ть конфе\\тки. "
        "[Мальчик рядом ???:] Непоня\\тно как."
    )
    aligned = align_segments(srt, transcript, max_span=5, similarity_threshold=0.2)
    updated = apply_transcript_speakers(aligned, transcript)
    assert [item.srt.speaker for item in updated] == ["[UNK]:", "[М]:", "[ЛД]:", "[???]:"]


def test_collector_bracket_replaces_srt_speaker_with_unknown():
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: ребята расскажите\n")
    transcript = "[Соб.: Ребя\\та, расскажи\\те?]"
    aligned = align_segments(srt, transcript, max_span=5, similarity_threshold=0.2)
    updated = apply_transcript_speakers(aligned, transcript)
    assert updated[0].srt.speaker == "[UNK]:"


def test_declarative_collector_bracket_replaces_srt_speaker_with_unknown():
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: да\n")
    transcript = "[Соб.: Да.]"
    aligned = align_segments(srt, transcript, max_span=5, similarity_threshold=0.2)
    updated = apply_transcript_speakers(aligned, transcript)
    assert updated[0].matched
    assert updated[0].srt.speaker == "[UNK]:"


def test_common_speaker_note_replaces_srt_speaker():
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: ручной ответ\n")
    transcript = "[ФМП спрашивает:] Ручно\\й отве\\т."
    aligned = align_segments(srt, transcript, max_span=5, similarity_threshold=0.2)
    updated = apply_transcript_speakers(aligned, transcript)
    assert updated[0].srt.speaker == "[ФМП]:"
    assert updated[0].speaker_source == "speaker_note"


def test_common_speaker_note_after_verb_replaces_srt_speaker():
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: ручной ответ\n")
    transcript = "[Сверху говорит громко Д:] Ручно\\й отве\\т."
    aligned = align_segments(srt, transcript, max_span=5, similarity_threshold=0.2)
    updated = apply_transcript_speakers(aligned, transcript)
    assert updated[0].srt.speaker == "[Д]:"
    assert updated[0].speaker_source == "speaker_note"


def test_long_speaker_commentary_does_not_assign_speaker():
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: ручной ответ\n")
    transcript = "[Собиратель спрашивает у ЛД имя перед началом записи.] Ручно\\й отве\\т."
    aligned = align_segments(srt, transcript, max_span=5, similarity_threshold=0.2)
    updated = apply_transcript_speakers(aligned, transcript)
    assert updated[0].srt.speaker == "[SPEAKER_00]:"
    assert updated[0].speaker_source == ""


def test_explicit_marker_precedes_common_speaker_note():
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: ручной ответ\n")
    transcript = "[АБ:] [ФМП спрашивает:] Ручно\\й отве\\т."
    aligned = align_segments(srt, transcript, max_span=5, similarity_threshold=0.2)
    updated = apply_transcript_speakers(aligned, transcript)
    assert updated[0].srt.speaker == "[АБ]:"
    assert updated[0].speaker_source == "preceding_marker"


def test_unknown_speaker_does_not_carry_forward_when_inferring_missing_speakers():
    srt = parse_srt(
        """
1
00:00:00,000 --> 00:00:01,000
[SPEAKER_00]: часовня какому празднику посвящена

2
00:00:01,000 --> 00:00:02,000
[SPEAKER_01]: часовня казанская
""".strip()
    )
    transcript = "[Часо\\вня како\\му пра\\зднику посвящена?] Часо\\вня Каза\\нская."
    aligned = align_segments(srt, transcript, max_span=8, similarity_threshold=0.2)
    updated = apply_transcript_speakers(aligned, transcript, infer_missing=True)
    assert [item.srt.speaker for item in updated] == ["[UNK]:", "[SPEAKER_01]:"]


def test_transcript_speaker_does_not_carry_to_unmatched_segments():
    srt = parse_srt(
        """
1
00:00:00,000 --> 00:00:01,000
[SPEAKER_00]: добрый день

2
00:00:01,000 --> 00:00:02,000
[SPEAKER_01]: unrelated
""".strip()
    )
    aligned = align_segments(srt, "[АБ:] до\\брый день", similarity_threshold=0.9)
    updated = apply_transcript_speakers(aligned, "[АБ:] до\\брый день", infer_missing=True)
    assert [item.matched for item in updated] == [True, False]
    assert [item.srt.speaker for item in updated] == ["[АБ]:", "[SPEAKER_01]:"]
    assert [item.transcript_speaker for item in updated] == ["[АБ]:", ""]


def test_mixed_collector_question_and_answer_keeps_srt_speaker():
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_01]: она была уже да\n")
    transcript = "[Она была уже?] Да,"
    aligned = align_segments(srt, transcript, max_span=6, similarity_threshold=0.2)
    updated = apply_transcript_speakers(aligned, transcript, infer_missing=True)
    assert updated[0].srt.speaker == "[SPEAKER_01]:"


def test_span_speaker_marker_labels_mixed_opening_context_and_answer():
    transcript = "[Когда выгоняли скот первый раз в году?] [ШВИ:] Мы выгоня\\ем."
    segment = SrtSegment(1, "00:00:00,000", "00:00:01,000", "[SPEAKER_02]:", "мы выгоняем")
    aligned = [
        AlignedSegment(
            segment,
            segment.speaker,
            transcript,
            "мы",
            True,
            0.609,
            0,
            len(transcript),
        )
    ]
    updated = apply_transcript_speakers(aligned, transcript, infer_missing=True)
    assert updated[0].srt.speaker == "[ШВИ]:"
    assert updated[0].speaker_source == "span_marker"


def test_collector_question_after_artificial_block_marker_gets_unknown():
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_01]: во что играете\n")
    transcript = "[МВ, ???:] [Соб.: Во что игра\\ете?]"
    aligned = align_segments(srt, transcript, max_span=8, similarity_threshold=0.2)
    updated = apply_transcript_speakers(aligned, transcript, infer_missing=True)
    assert updated[0].srt.speaker == "[UNK]:"


def test_alignment_skips_editorial_brackets_but_keeps_speaker_tags():
    srt = parse_srt(
        """
1
00:00:00,000 --> 00:00:01,000
[SPEAKER_00]: костоправ его звали

2
00:00:01,000 --> 00:00:02,000
[SPEAKER_01]: татьяна не знает
""".strip()
    )
    transcript = (
        "[ХВВ, ФМП:] [ХВВ спрашивает БВИ, помнит ли она знахаря. Отвечает ФМП:] "
        "Костопра\\в его\\ зва\\ли [историю про Костоправа см. XXa-10]. "
        "[ГЭС:] Татья\\на не зна\\ет?"
    )
    aligned = align_segments(srt, transcript, max_span=8, similarity_threshold=0.2)
    updated = apply_transcript_speakers(aligned, transcript)
    assert [item.matched for item in updated] == [True, True]
    assert [item.srt.speaker for item in updated] == ["[ХВВ, ФМП]:", "[ГЭС]:"]
    assert "историю про" not in updated[0].normalized_text
    assert "историю про" not in remove_alignment_notes(transcript)
    assert "[ГЭС:]" in remove_alignment_notes(transcript)


def test_alignment_skips_long_opening_editorial_brackets():
    srt = parse_srt(
        """
1
00:00:00,000 --> 00:00:01,000
[SPEAKER_00]: у меня было

2
00:00:01,000 --> 00:00:02,000
[SPEAKER_00]: в шортах крапива
""".strip()
    )
    long_note = " ".join(["дети стоят в кружке"] * 30)
    transcript = f"[ДС, А:] [{long_note}. ДС говорит:] У меня\\ бы\\ло, в шо\\ртах крапи\\ва."
    cleaned = remove_alignment_notes(transcript)
    aligned = align_segments(srt, cleaned, max_span=8, similarity_threshold=0.2)
    updated = apply_transcript_speakers(aligned, cleaned, infer_missing=True)
    assert long_note not in cleaned
    assert [item.matched for item in updated] == [True, True]
    assert [item.srt.speaker for item in updated] == ["[ДС]:", "[ДС]:"]


def test_transcript_block_footer_speaker_replaces_srt_speakers(tmp_path: Path):
    srt_path = tmp_path / "chunk.srt"
    transcript_path = tmp_path / "chunk.txt"
    output_path = tmp_path / "aligned.srt"
    srt_path.write_text(
        """
1
00:00:00,000 --> 00:00:01,000
[SPEAKER_00]: добрый день

2
00:00:01,000 --> 00:00:02,000
[SPEAKER_01]: красный дом
""".strip(),
        encoding="utf-8",
    )
    transcript_path.write_text(
        "\n".join(
            [
                "XXIIа-19",
                "Пежма-Берег",
                "АБМ, РВВ",
                "до\\брый день кра\\сный дом",
                "ААК",
            ]
        ),
        encoding="utf-8",
    )
    aligned = align_srt_file(
        srt_path,
        transcript_path.read_text(encoding="utf-8"),
        output_path,
        use_transcript_speakers=True,
    )
    assert [item.srt.speaker for item in aligned] == ["[ААК]:", "[ААК]:"]
    assert "[ААК]: до\\брый день" in output_path.read_text(encoding="utf-8")


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


def test_m4a_input_exports_wav_with_transcoding(tmp_path: Path):
    command = build_cut_command("input.m4a", tmp_path / "clip.wav", "00:00:00.000", "00:00:01.250")
    assert command[-3:] == ["-c:a", "pcm_s16le", str(tmp_path / "clip.wav")]


def test_export_rejects_non_wav_outputs(tmp_path: Path):
    try:
        build_cut_command("input.wav", tmp_path / "clip.m4a", "00:00:00.000", "00:00:01.250")
    except ValueError as error:
        assert ".wav" in str(error)
    else:
        raise AssertionError("Expected non-wav output to be rejected")


def test_normalize_caption_text_removes_stress_marks_but_keeps_readable_text():
    assert normalize_caption_text("Во­­­­­_т, ро\\дом  она\\.") == "Во­­­­­т, родом она."


def test_clean_speaker_code_is_filename_safe():
    assert clean_speaker_code("[ХВВ, БВИ, ГЭС]:") == "ХВВ-БВИ-ГЭС"
    assert clean_speaker_code("[???]:") == "UNK"
    assert clean_speaker_code("") == "UNKNOWN"


def test_export_aligned_srt_tree_matches_cut_samples_layout(tmp_path: Path):
    aligned_root = tmp_path / "aligned-root"
    audio_root = tmp_path / "audio"
    aligned_dir = aligned_root / "and_001" / "aligned"
    speaker_map_dir = aligned_root / "and_001" / "speaker_maps"
    audio_dir = audio_root / "and_001"
    aligned_dir.mkdir(parents=True)
    speaker_map_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    (audio_dir / "and_001No1.wav").write_bytes(b"not real wav")
    (speaker_map_dir / "and_001No1.speaker_map.csv").write_text(
        "srt_index,start,end,whisperx_speaker,transcript_speaker,speaker_source,matched,score\n"
        '1,"00:00:00,031","00:00:01,250",[SPEAKER_00]:,[АБ]:,preceding_marker,True,1.000\n'
        '2,"00:00:01,250","00:00:02,000",[SPEAKER_01]:,,unmatched,False,0.000\n',
        encoding="utf-8",
    )
    (aligned_dir / "and_001No1.aligned.srt").write_text(
        """
1
00:00:00,031 --> 00:00:01,250
[SPEAKER_00]: Во­­­­­_т, ро\\дом.

2
00:00:01,250 --> 00:00:02,000
[SPEAKER_01]: Да\\.
""".strip(),
        encoding="utf-8",
    )

    with patch("alignment.export.subprocess.run") as run:
        rows = export_aligned_srt_tree(
            aligned_root,
            audio_root,
            tmp_path / "cut_samples",
            tmp_path / "manifest.tsv",
            require_diarized_matches=True,
        )

    first = tmp_path / "cut_samples" / "and_001" / "and_001No1" / "001_АБ_00-00-00-031"
    second = tmp_path / "cut_samples" / "and_001" / "and_001No1" / "002_SPEAKER_01_00-00-01-250"
    assert len(rows) == 2
    assert rows[0]["speaker"] == "[АБ]:"
    assert first.with_suffix(".txt").read_text(encoding="utf-8") == "Во­­­­­т, родом."
    assert first.with_name(f"{first.name}_orig.txt").read_text(encoding="utf-8") == "Во­­­­­_т, ро\\дом."
    assert second.with_suffix(".txt").read_text(encoding="utf-8") == "Да."
    assert run.call_args_list[0].args[0] == build_cut_command(
        audio_dir / "and_001No1.wav",
        first.with_suffix(".wav"),
        "00:00:00.031",
        "00:00:01.250",
    )
    assert (first.parent / "speaker_map.csv").read_text(encoding="utf-8").startswith("srt_index,start")
    assert (tmp_path / "manifest.tsv").exists()


def test_export_aligned_srt_tree_rejects_matched_blank_speakers(tmp_path: Path):
    aligned_root = tmp_path / "aligned-root"
    audio_root = tmp_path / "audio"
    aligned_dir = aligned_root / "and_001" / "aligned"
    speaker_map_dir = aligned_root / "and_001" / "speaker_maps"
    audio_dir = audio_root / "and_001"
    aligned_dir.mkdir(parents=True)
    speaker_map_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    (audio_dir / "and_001No1.wav").write_bytes(b"not real wav")
    (speaker_map_dir / "and_001No1.speaker_map.csv").write_text(
        "srt_index,start,end,whisperx_speaker,transcript_speaker,speaker_source,matched,score\n"
        '1,"00:00:00,031","00:00:01,250",[SPEAKER_00]:,,unmatched,True,1.000\n',
        encoding="utf-8",
    )
    (aligned_dir / "and_001No1.aligned.srt").write_text(
        "1\n00:00:00,031 --> 00:00:01,250\n[SPEAKER_00]: текст\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Matched segments"):
        export_aligned_srt_tree(
            aligned_root,
            audio_root,
            tmp_path / "cut_samples",
            require_diarized_matches=True,
            run=False,
        )


def test_export_aligned_srt_tree_ignores_unmatched_speaker_map_rows(tmp_path: Path):
    aligned_root = tmp_path / "aligned-root"
    audio_root = tmp_path / "audio"
    aligned_dir = aligned_root / "and_001" / "aligned"
    speaker_map_dir = aligned_root / "and_001" / "speaker_maps"
    audio_dir = audio_root / "and_001"
    aligned_dir.mkdir(parents=True)
    speaker_map_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    (audio_dir / "and_001No1.wav").write_bytes(b"not real wav")
    (speaker_map_dir / "and_001No1.speaker_map.csv").write_text(
        "srt_index,start,end,whisperx_speaker,transcript_speaker,speaker_source,matched,score\n"
        '1,"00:00:00,031","00:00:01,250",[SPEAKER_00]:,[АБ]:,stale_carry,False,0.000\n',
        encoding="utf-8",
    )
    (aligned_dir / "and_001No1.aligned.srt").write_text(
        "1\n00:00:00,031 --> 00:00:01,250\n[SPEAKER_00]: текст\n",
        encoding="utf-8",
    )

    rows = export_aligned_srt_tree(aligned_root, audio_root, tmp_path / "cut_samples", run=False)

    assert rows[0]["clip_id"] == "001_SPEAKER_00_00-00-00-031"
    assert rows[0]["speaker"] == "[SPEAKER_00]:"


def test_export_aligned_srt_tree_can_skip_unmatched_rows(tmp_path: Path):
    aligned_root = tmp_path / "aligned-root"
    audio_root = tmp_path / "audio"
    aligned_dir = aligned_root / "and_001" / "aligned"
    speaker_map_dir = aligned_root / "and_001" / "speaker_maps"
    audio_dir = audio_root / "and_001"
    aligned_dir.mkdir(parents=True)
    speaker_map_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    (audio_dir / "and_001No1.wav").write_bytes(b"not real wav")
    (speaker_map_dir / "and_001No1.speaker_map.csv").write_text(
        "srt_index,start,end,whisperx_speaker,transcript_speaker,speaker_source,matched,score\n"
        '1,"00:00:00,031","00:00:01,250",[SPEAKER_00]:,[АБ]:,preceding_marker,True,1.000\n'
        '2,"00:00:01,250","00:00:02,000",[SPEAKER_01]:,,unmatched,False,0.000\n',
        encoding="utf-8",
    )
    (aligned_dir / "and_001No1.aligned.srt").write_text(
        """
1
00:00:00,031 --> 00:00:01,250
[SPEAKER_00]: ручной текст

2
00:00:01,250 --> 00:00:02,000
[SPEAKER_01]: asr fallback
""".strip(),
        encoding="utf-8",
    )

    rows = export_aligned_srt_tree(
        aligned_root,
        audio_root,
        tmp_path / "cut_samples",
        matched_only=True,
        run=False,
    )

    assert [row["clip_id"] for row in rows] == ["001_АБ_00-00-00-031"]
    assert not (
        tmp_path / "cut_samples" / "and_001" / "and_001No1" / "002_SPEAKER_01_00-00-01-250.txt"
    ).exists()


def test_export_aligned_srt_tree_can_reject_low_match_ratio(tmp_path: Path):
    aligned_root = tmp_path / "aligned-root"
    audio_root = tmp_path / "audio"
    aligned_dir = aligned_root / "and_001" / "aligned"
    speaker_map_dir = aligned_root / "and_001" / "speaker_maps"
    audio_dir = audio_root / "and_001"
    aligned_dir.mkdir(parents=True)
    speaker_map_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    (audio_dir / "and_001No1.wav").write_bytes(b"not real wav")
    (aligned_root / "and_001" / "summary.tsv").write_text(
        "name\tsrt\tmanual\taligned_srt\taligned_tsv\tspeaker_map\tsegments\t"
        "matched_segments\tmatch_ratio\tblank_speakers\tmatched_blank_speakers\tstatus\n"
        "and_001No1\t\t\t\t\t\t10\t1\t0.100\t9\t0\taligned\n",
        encoding="utf-8",
    )
    (speaker_map_dir / "and_001No1.speaker_map.csv").write_text(
        "srt_index,start,end,whisperx_speaker,transcript_speaker,speaker_source,matched,score\n"
        '1,"00:00:00,031","00:00:01,250",[SPEAKER_00]:,[АБ]:,preceding_marker,True,1.000\n',
        encoding="utf-8",
    )
    (aligned_dir / "and_001No1.aligned.srt").write_text(
        "1\n00:00:00,031 --> 00:00:01,250\n[SPEAKER_00]: ручной текст\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="below 0.200: and_001No1"):
        export_aligned_srt_tree(
            aligned_root,
            audio_root,
            tmp_path / "cut_samples",
            min_match_ratio=0.2,
            run=False,
        )


def test_export_aligned_srt_tree_requires_speaker_maps_when_guarded(tmp_path: Path):
    aligned_root = tmp_path / "aligned-root"
    audio_root = tmp_path / "audio"
    aligned_dir = aligned_root / "and_001" / "aligned"
    audio_dir = audio_root / "and_001"
    aligned_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    (audio_dir / "and_001No1.wav").write_bytes(b"not real wav")
    (aligned_dir / "and_001No1.aligned.srt").write_text(
        "1\n00:00:00,031 --> 00:00:01,250\n[SPEAKER_00]: текст\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing speaker map"):
        export_aligned_srt_tree(
            aligned_root,
            audio_root,
            tmp_path / "cut_samples",
            require_diarized_matches=True,
            run=False,
        )


def test_export_aligned_srt_tree_requires_complete_speaker_maps_when_guarded(tmp_path: Path):
    aligned_root = tmp_path / "aligned-root"
    audio_root = tmp_path / "audio"
    aligned_dir = aligned_root / "and_001" / "aligned"
    speaker_map_dir = aligned_root / "and_001" / "speaker_maps"
    audio_dir = audio_root / "and_001"
    aligned_dir.mkdir(parents=True)
    speaker_map_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    (audio_dir / "and_001No1.wav").write_bytes(b"not real wav")
    (speaker_map_dir / "and_001No1.speaker_map.csv").write_text(
        "srt_index,start,end,whisperx_speaker,transcript_speaker,speaker_source,matched,score\n"
        '1,"00:00:00,031","00:00:01,250",[SPEAKER_00]:,[АБ]:,preceding_marker,True,1.000\n',
        encoding="utf-8",
    )
    (aligned_dir / "and_001No1.aligned.srt").write_text(
        """
1
00:00:00,031 --> 00:00:01,250
[SPEAKER_00]: первый

2
00:00:01,250 --> 00:00:02,000
[SPEAKER_01]: второй
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lacks rows for SRT indices 2"):
        export_aligned_srt_tree(
            aligned_root,
            audio_root,
            tmp_path / "cut_samples",
            require_diarized_matches=True,
            run=False,
        )


def test_export_aligned_map_cli_accepts_diarized_matches_guard(tmp_path: Path):
    aligned_root = tmp_path / "aligned-root"
    audio_root = tmp_path / "audio"
    output_root = tmp_path / "cut_samples"
    aligned_dir = aligned_root / "and_001" / "aligned"
    speaker_map_dir = aligned_root / "and_001" / "speaker_maps"
    audio_dir = audio_root / "and_001"
    aligned_dir.mkdir(parents=True)
    speaker_map_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    (audio_dir / "and_001No1.wav").write_bytes(b"not real wav")
    (speaker_map_dir / "and_001No1.speaker_map.csv").write_text(
        "srt_index,start,end,whisperx_speaker,transcript_speaker,speaker_source,matched,score\n"
        '1,"00:00:00,031","00:00:01,250",[SPEAKER_00]:,[АБ]:,preceding_marker,True,1.000\n',
        encoding="utf-8",
    )
    (aligned_dir / "and_001No1.aligned.srt").write_text(
        "1\n00:00:00,031 --> 00:00:01,250\n[SPEAKER_00]: текст\n",
        encoding="utf-8",
    )

    with patch("alignment.export.subprocess.run"):
        main(
            [
                "export-aligned-map",
                str(aligned_root),
                str(audio_root),
                str(output_root),
                "--require-diarized-matches",
                "--matched-only",
            ]
        )

    assert (output_root / "and_001" / "and_001No1" / "001_АБ_00-00-00-031.wav").exists() is False
    assert (output_root / "and_001" / "and_001No1" / "001_АБ_00-00-00-031.txt").exists()
