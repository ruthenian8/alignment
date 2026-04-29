import pytest

from alignment.embeddings import (
    align_pairs_with_embeddings,
    clean_text_for_embedding,
    cosine_similarity,
    extract_dialect_text,
    parse_manual_annotation,
    prepare_text_for_embedding,
    segment_text_with_pauses,
)


class FakeModel:
    def encode(self, sentences: list[str], **kwargs: object) -> list[list[float]]:
        vectors = []
        for sentence in sentences:
            if "добрый" in sentence:
                vectors.append([1.0, 0.0])
            elif "дом" in sentence:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.2, 0.2])
        return vectors


def test_parse_manual_annotation_marks_bracketed_prompts():
    segments = parse_manual_annotation("[Как дела?] До\\брый день. [Соб.: шум] Кра\\сный дом.")
    assert [(segment.text, segment.is_dialect) for segment in segments] == [
        ("Как дела?", False),
        ("До\\брый день.", True),
        ("Кра\\сный дом.", True),
    ]


def test_extract_dialect_text_removes_matching_interviewer_srt_segment():
    srt = """
1
00:00:00,000 --> 00:00:01,000
[SPEAKER_00]: Как дела?

2
00:00:01,000 --> 00:00:02,000
[SPEAKER_01]: Добрый день.
""".strip()
    extraction = extract_dialect_text(srt, "[Как дела?] До\\брый день.", threshold=0.8)
    assert extraction.whisper_text == "Добрый день."
    assert extraction.manual_text == "До\\брый день."
    assert extraction.stats["whisper_standard"] == 1


def test_embedding_text_segmentation_and_cleaning_preserve_originals():
    segments = segment_text_with_pauses("А, ну э\\то са\\мое… Тут вот. Потом длиннее фраза.")
    pairs = prepare_text_for_embedding(segments)
    assert pairs[0][0].startswith("А, ну э\\то")
    assert clean_text_for_embedding("кра\\сный_ дом") == "красный дом"


def test_align_pairs_with_embeddings_uses_optional_model_and_preserves_text():
    pairs = align_pairs_with_embeddings(
        "добрый день один два три четыре. красный дом один два три четыре.",
        "до\\брый день один два три четыре. кра\\сный дом один два три четыре.",
        model=FakeModel(),
        threshold=0.9,
        join_short=False,
    )
    assert [pair.manual_text for pair in pairs] == [
        "до\\брый день один два три четыре",
        "кра\\сный дом один два три четыре",
    ]
    assert all(pair.score == pytest.approx(1.0) for pair in pairs)


def test_cosine_similarity_handles_zero_vectors():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
