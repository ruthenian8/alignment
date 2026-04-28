"""Tests for alignment.index_parser module."""

from alignment.index_parser import parse_index_plaintext

SIMPLE_INDEX = """\
pez_001.wma
Место'2018
Информант
Соб.: Собиратель

00:00:00,000 – Описание первой записи.
00:05:00,000 – Вторая запись. НЕ РАСПИСАНО
00:10:00,000 – Третья запись.
"""


def test_parse_plaintext_index_basic():
    rows = parse_index_plaintext(SIMPLE_INDEX, "pez_001")
    assert len(rows) == 3
    assert rows[0].start == "00:00:00.000"
    assert rows[1].start == "00:05:00.000"
    assert rows[2].start == "00:10:00.000"


def test_parse_ne_raspisano():
    rows = parse_index_plaintext(SIMPLE_INDEX, "pez_001")
    assert rows[0].trans is True
    assert rows[1].trans is False
    assert rows[2].trans is True


def test_continuation_linking():
    index_text = """\
00:00:00,000 – Первая запись. Продолжается см. 00:05:00,000 продолж
00:05:00,000 – Продолжение первой записи.
"""
    rows = parse_index_plaintext(index_text, "pez_test")
    assert rows[0].cont == "00:05:00.000"
    assert rows[1].prev == "0"


def test_names_assigned():
    rows = parse_index_plaintext(SIMPLE_INDEX, "pez_001", suffix=".wav")
    assert rows[0].name == "pez_001No0.wav"
    assert rows[1].name == "pez_001No1.wav"
    assert rows[2].name == "pez_001No2.wav"
