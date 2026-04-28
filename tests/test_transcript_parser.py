"""Tests for alignment.transcript_parser module."""

import tempfile
from pathlib import Path

from alignment.transcript_parser import parse_transcript_file

SAMPLE = """\
XXIIа-19
Пежма-Берег'2018
АБМ, РВВ
[Часовня в вашей деревне?] Так, часовня казанская.
ААК

XXIа-доп.
Пежма-Берег'2018
АБМ, РВВ
Глубоко верующей была моя бабушка.
Ещё одна строка текста.
ААК
"""


def _write_tmp(content: str) -> Path:
    tf = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False)
    tf.write(content)
    tf.flush()
    return Path(tf.name)


def test_parse_transcript_basic():
    path = _write_tmp(SAMPLE)
    records = parse_transcript_file(path)
    assert len(records) == 2
    assert records[0].archive_id == "XXIIа-19"
    assert records[1].archive_id == "XXIа-доп."


def test_transcript_interviewers_count():
    path = _write_tmp(SAMPLE)
    records = parse_transcript_file(path)
    assert records[0].interviewers == ["АБМ", "РВВ"]
    assert records[0].interviewees == ["ААК"]


def test_transcript_text_joins():
    path = _write_tmp(SAMPLE)
    records = parse_transcript_file(path)
    assert "Глубоко верующей" in records[1].text
    assert "Ещё одна строка" in records[1].text


def test_bracketed_prompts_preserved():
    path = _write_tmp(SAMPLE)
    records = parse_transcript_file(path)
    assert "[Часовня в вашей деревне?]" in records[0].text
