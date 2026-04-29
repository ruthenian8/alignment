from pathlib import Path

from alignment.cli import main


def test_small_end_to_end_parse_join_reorder_smoke(tmp_path: Path):
    index = tmp_path / "index.txt"
    transcript = tmp_path / "transcript.txt"
    index.write_text("00:00:00,000 - Первый.\n00:00:01,000 - Второй. НЕ РАСПИСАНО\n", encoding="utf-8")
    transcript.write_text("id\nplace\nA\nручно\\й текст\nB\n", encoding="utf-8")
    index_tsv = tmp_path / "index.tsv"
    transcript_tsv = tmp_path / "transcript.tsv"
    joined_tsv = tmp_path / "joined.tsv"
    reordered_tsv = tmp_path / "reordered.tsv"
    main(["parse-index", index.as_posix(), index_tsv.as_posix(), "--audio-stem", "tiny"])
    main(["parse-transcript", transcript.as_posix(), transcript_tsv.as_posix()])
    main(["join", index_tsv.as_posix(), transcript_tsv.as_posix(), joined_tsv.as_posix()])
    main(["reorder", joined_tsv.as_posix(), reordered_tsv.as_posix()])
    output = reordered_tsv.read_text(encoding="utf-8")
    assert "ручно\\й текст" in output
    assert "tinyNo0.wav" in output
