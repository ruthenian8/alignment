"""Tests for corpus speaker-map helper script logic."""

from tools.align_corpus_speaker_maps import best_block_speaker, text_blocks_with_speakers


def test_best_block_speaker_tolerates_small_transcript_differences():
    raw = """
VIII-15 в
Пежма
АБМ, БИС
[Как вели купленную корову домой?» Ой, там то\\жо при\\говор есть, а во\\т и забы\\ла при\\говор-то.
КЛП
""".strip()
    transcript = "[Как вели купленную корову домой?] Ой, там то\\жо при\\говор есть, а во\\т забы\\ла."

    assert best_block_speaker(transcript, text_blocks_with_speakers(raw)) == "КЛП"
