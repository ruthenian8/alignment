from alignment.join import join_rows
from alignment.reorder import reorder_rows


def test_join_uses_tsv_style_rows_and_leaves_inactive_transcript_empty():
    index = [
        {"start": "00:00:00.000", "trans": "True", "cont": "", "prev": "", "text": "alpha", "name": "a.wav"},
        {"start": "00:00:01.000", "trans": "False", "cont": "", "prev": "", "text": "skip", "name": "b.wav"},
        {"start": "00:00:02.000", "trans": "True", "cont": "", "prev": "", "text": "beta", "name": "c.wav"},
    ]
    transcripts = [
        {"transcript": "first", "max_speakers": "2", "min_speakers": "1"},
        {"transcript": "second", "max_speakers": "1", "min_speakers": "1"},
    ]
    rows = join_rows(index, transcripts)
    assert rows[0]["transcript"] == "first"
    assert rows[1]["transcript"] == ""
    assert rows[2]["transcript"] == "second"


def test_reorder_identity_and_one_row_shift():
    identity = [
        {"trans": "True", "text": "яблоко красное", "transcript": "яблоко красное"},
        {"trans": "True", "text": "груша зеленая", "transcript": "груша зеленая"},
    ]
    assert reorder_rows(identity) == identity

    shifted = [
        {"trans": "True", "text": "яблоко красное", "transcript": "груша зеленая"},
        {"trans": "True", "text": "груша зеленая", "transcript": "яблоко красное"},
    ]
    reordered = reorder_rows(shifted)
    assert reordered[0]["transcript"] == "яблоко красное"
    assert reordered[1]["transcript"] == "груша зеленая"


def test_reorder_ignores_inactive_rows_and_missing_transcripts():
    rows = [
        {"trans": "True", "text": "первый рассказ", "transcript": "второй рассказ"},
        {"trans": "False", "text": "не расписано", "transcript": ""},
        {"trans": "True", "text": "второй рассказ", "transcript": "первый рассказ"},
    ]
    reordered = reorder_rows(rows)
    assert reordered[1]["transcript"] == ""
    assert reordered[0]["transcript"] == "первый рассказ"
