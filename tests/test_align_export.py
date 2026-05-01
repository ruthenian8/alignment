from pathlib import Path
from unittest.mock import patch

from alignment.align import align_segments, align_srt_file, aligned_to_srt, apply_transcript_speakers
from alignment.audio import build_cut_command
from alignment.export import export_segments
from alignment.srt import parse_srt


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


def test_mixed_collector_question_and_answer_keeps_srt_speaker():
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_01]: она была уже да\n")
    transcript = "[Она была уже?] Да,"
    aligned = align_segments(srt, transcript, max_span=6, similarity_threshold=0.2)
    updated = apply_transcript_speakers(aligned, transcript, infer_missing=True)
    assert updated[0].srt.speaker == "[SPEAKER_01]:"


def test_collector_question_after_artificial_block_marker_gets_unknown():
    srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_01]: во что играете\n")
    transcript = "[МВ, ???:] [Соб.: Во что игра\\ете?]"
    aligned = align_segments(srt, transcript, max_span=8, similarity_threshold=0.2)
    updated = apply_transcript_speakers(aligned, transcript, infer_missing=True)
    assert updated[0].srt.speaker == "[UNK]:"


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
