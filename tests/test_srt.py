from alignment.srt import SrtSegment, format_srt, normalize_timestamp, parse_srt, timestamp_to_ms


def test_timestamp_parses_comma_and_dot_milliseconds():
    assert timestamp_to_ms("00:01:02,003") == 62003
    assert timestamp_to_ms("00:01:02.003") == 62003
    assert normalize_timestamp("00:01:02,003", decimal=".") == "00:01:02.003"


def test_srt_round_trip_preserves_speaker_and_multiline_text():
    text = "1\n00:00:00.001 --> 00:00:02,003\n[SPEAKER_01]: line one\nline two\n"
    segments = parse_srt(text)
    assert segments == [SrtSegment(1, "00:00:00,001", "00:00:02,003", "[SPEAKER_01]:", "line one\nline two")]
    assert parse_srt(format_srt(segments)) == segments
