from pathlib import Path
from zipfile import ZipFile

from alignment.index_parser import parse_index_file, parse_index_text
from alignment.transcript_parser import parse_transcript_text


def test_plaintext_index_parses_untranscribed_and_continuation_linking():
    text = """
00:00:00,000 - Начало. Окончание см. 00:00:10,000
00:00:05.000 – НЕ РАСПИСАНО
00:00:07,500Без пробела и тире.
00:00:10,000 Окончание.
""".strip()
    rows = parse_index_text(text, audio_stem="sample")
    assert rows[0]["start"] == "00:00:00.000"
    assert rows[0]["cont"] == "00:00:10.000"
    assert rows[3]["prev"] == "0"
    assert rows[1]["trans"] == "False"
    assert rows[2]["text"] == "00:00:07,500 - Без пробела и тире."
    assert rows[3]["text"] == "00:00:10,000 - Окончание."
    assert rows[0]["name"] == "sampleNo0.wav"


def test_real_pez_index_fixture_parses_rows():
    rows = parse_index_text(
        Path("data/index_plaintext/pez_001.txt").read_text(encoding="utf-8-sig"), audio_stem="pez_001"
    )
    assert len(rows) == 38
    assert rows[0]["text"].startswith("00:00:00,000 - Часовня")
    assert rows[2]["trans"] == "False"


def test_docx_index_parser_reads_word_document_xml(tmp_path: Path):
    docx = tmp_path / "index.docx"
    document_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>00:00:00,000 - Из DOCX.</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    with ZipFile(docx, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
    rows = parse_index_file(docx, audio_stem="docx")
    assert rows[0]["text"] == "00:00:00,000 - Из DOCX."
    assert rows[0]["name"] == "docxNo0.wav"


def test_transcript_parser_keeps_bracketed_prompts_and_stress_marks():
    text = """
XXIIа-19
Пежма-Берег
АБМ, РВВ
[Вопрос?] Так, часо\\вня.
ААК
""".strip()
    rows = parse_transcript_text(text)
    assert rows[0]["transcript"] == "[ААК:] [Вопрос?] Так, часо\\вня."
    assert rows[0]["max_speakers"] == 3
    assert rows[0]["min_speakers"] == 1
