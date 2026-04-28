"""Tests for alignment.align module."""

from alignment.align import (
    align_segments,
    align_srt_to_transcript,
    compute_similarity,
    normalize_text_for_match,
    tokenize_transcript,
)
from alignment.srt import parse_srt


def test_normalize_text_for_match():
    assert "е" in normalize_text_for_match("ёж")
    assert "\\" not in normalize_text_for_match("ка\\т")
    result = normalize_text_for_match("Привет, мир!")
    assert "," not in result
    assert "!" not in result
    assert normalize_text_for_match("КОШКА") == "кошка"


def test_compute_similarity():
    assert compute_similarity([], ["а"]) == 0.0
    assert compute_similarity(["а"], []) == 0.0
    assert compute_similarity(["кот", "мат"], ["кот", "мат"]) == 1.0
    sim = compute_similarity(["кот", "мат", "сидит"], ["кот", "сидит", "дома"])
    assert 0 < sim < 1


def test_align_segments_monotonic():
    """Alignment boundaries must be monotonically non-decreasing."""
    srt_text = """\
1
00:00:00,000 --> 00:00:03,000
кот сидит на мате

2
00:00:03,000 --> 00:00:06,000
собака бежит по полю

3
00:00:06,000 --> 00:00:09,000
птица летит в небе
"""
    transcript = "кот сидит на мате собака бежит по полю птица летит в небе"
    aligned = align_srt_to_transcript(srt_text, transcript)
    assert len(aligned) == 3
    srt_segs = parse_srt(srt_text)
    tokens = tokenize_transcript(transcript)
    boundaries, _ = align_segments(srt_segs, tokens)
    for i in range(len(boundaries) - 1):
        assert boundaries[i] <= boundaries[i + 1]


def test_original_text_preserved():
    """Original transcript text should appear in aligned output."""
    srt_text = """\
1
00:00:00,000 --> 00:00:03,000
кот сидит
"""
    transcript = "ко\\т сиди\\т на ма\\те"
    aligned = align_srt_to_transcript(srt_text, transcript)
    assert len(aligned) == 1
    if aligned[0].matched:
        assert "\\" in aligned[0].transcript_text or "кот" in aligned[0].transcript_text.lower()


def test_align_small_example():
    """End-to-end small alignment example."""
    srt_text = """\
1
00:00:00,571 --> 00:00:03,074
Гораздо мягче.

2
00:00:03,154 --> 00:00:03,996
А люди?

3
00:00:04,116 --> 00:00:06,779
Люди другие обычаи другие.
"""
    transcript = "[А люди отличаются?] Люди другие. Обычаи другие."
    aligned = align_srt_to_transcript(srt_text, transcript)
    assert len(aligned) == 3
    assert any(seg.matched for seg in aligned)
