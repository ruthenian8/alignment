"""Tests for alignment.srt module."""

from alignment.srt import format_srt, parse_srt

SAMPLE_SRT_COMMA = """\
1
00:00:00,031 --> 00:00:06,358
[SPEAKER_00]: Вот эта часовня ваша, она кому, какому празднику посвящена?

2
00:00:06,418 --> 00:00:10,843
[SPEAKER_01]: Так, часовня эта казанская, образ иконы казанской.

3
00:00:10,903 --> 00:00:18,132
[SPEAKER_01]: Так, дату, когда она построена, я сказать не могу.
"""


def test_parse_srt_comma_separator():
    srt = "1\n00:00:00,031 --> 00:00:06,358\nПривет.\n"
    segs = parse_srt(srt)
    assert len(segs) == 1
    assert segs[0].index == 1
    assert segs[0].start == "00:00:00,031"
    assert segs[0].end == "00:00:06,358"
    assert segs[0].text == "Привет."
    assert segs[0].speaker == ""


def test_parse_srt_dot_separator():
    srt = "1\n00:00:00.031 --> 00:00:06.358\nПривет.\n"
    segs = parse_srt(srt)
    assert len(segs) == 1
    assert segs[0].start == "00:00:00,031"
    assert segs[0].end == "00:00:06,358"


def test_parse_srt_with_speaker():
    srt = "1\n00:00:00,000 --> 00:00:03,000\n[SPEAKER_01]: Привет мир.\n"
    segs = parse_srt(srt)
    assert len(segs) == 1
    assert segs[0].speaker == "[SPEAKER_01]:"
    assert segs[0].text == "Привет мир."


def test_format_srt_roundtrip():
    srt = "1\n00:00:00,031 --> 00:00:06,358\nПривет.\n\n2\n00:00:07,000 --> 00:00:10,000\nМир.\n"
    segs = parse_srt(srt)
    output = format_srt(segs)
    segs2 = parse_srt(output)
    assert len(segs2) == 2
    assert segs2[0].start == segs[0].start
    assert segs2[0].end == segs[0].end
    assert segs2[0].text == segs[0].text
    assert segs2[1].text == segs[1].text


def test_parse_srt_multiline_text():
    srt = "1\n00:00:00,031 --> 00:00:06,358\n[SPEAKER_00]: Первая строка\nтекст продолжается.\n"
    segs = parse_srt(srt)
    assert len(segs) == 1
    assert "Первая строка" in segs[0].text
    assert "текст продолжается" in segs[0].text


def test_parse_srt_multiple_blocks():
    segs = parse_srt(SAMPLE_SRT_COMMA)
    assert len(segs) == 3
    assert segs[0].speaker == "[SPEAKER_00]:"
    assert segs[1].speaker == "[SPEAKER_01]:"
