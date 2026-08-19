import pytest

from alignment.annotations import BracketKind, classify_bracket_text, is_note_only_transcript_row

REAL_CLASSIFIER_EXAMPLES = {
    BracketKind.SPEAKER_TAG: [
        "АБН:",
        "ГИН:",
        "ФМП:",
        "БСА:",
        "БЕВ:",
        "ФМП",
        "ХВВ",
        "Д:",
        "МВ:",
        "???:",
    ],
    BracketKind.COLLECTOR_UTTERANCE: [
        "Часовня в вашей деревне какому празднику посвящена?",
        "Но она была уже?",
        "Косково?",
        "Кулакова Нина Андреевна?",
        "А она сюда замуж вышла?",
        "Что значит «десятым женихом»?",
        "Соб.: Да.",
        "Соб.: Это уже недавно.",
        "Соб.: Такая как миска",
        "Соб.: Интересно.",
    ],
    BracketKind.SPOKEN_UTTERANCE: [
        "А монеты в могилу не кидают в могилу\\",
        "Хором:",
        "Говорят одновременно:",
        "вставая на цыпочки и подтягива\\ясь руками за столешницу",
        "ГЕА говорит: «Труди\\ться да моли\\ться, говоря\\т, никогда\\ не грех».",
        "Поёт:",
        "Федько\\во",
        "Поет:",
        "поет:",
        "Смеё\\тся.",
    ],
    BracketKind.SPEAKER_NOTE: [
        "БЕВ смеется:",
        "ГОА одновременно:",
        "БЮА одновременно:",
        "ЧВП одновременно:",
        "ДЛА одновременно:",
        "ДАД усмехается:",
        "ДЛА согласно мычит:",
        "ДАД удивляется:",
        "Собиратель спрашивает у ЛД имя.",
        "Сверху с лестницы говорит громко Д:",
    ],
    BracketKind.EDITORIAL_NOTE: [
        "Смеётся.",
        "показывает размер «в обхват»",
        "рисует рукой в воздухе спираль",
        "указывает на некрашеные доски, которыми замощена дорожка на участке",
        "показывает, как шанежки ставят на сковороде в печь",
        "смеется",
        "Показывает, как резали хлеб, прижимая к груди.",
        "показывает рукой ровное место",
        "нрзб. 00:56:02,504",
        "молчит",
    ],
    BracketKind.UNKNOWN: [
        "Нина Михайловна Третьякова, которую собиратели хотели опросить в Берегу, но не опросили пока",
        "… - рассказывает про книгу Кладовикова.",
        "вздыхает",
        "в Мезени",
        "Далее отсылет к книге «О рыбаках и поморах Севера».",
        "информант КАМ",
        "ни",
        "… отвлекается на телефонный звонок",
        "Отрицательно мычит.",
        "Задумалась.",
    ],
}


@pytest.mark.parametrize(
    ("expected", "span"),
    [(kind, span) for kind, spans in REAL_CLASSIFIER_EXAMPLES.items() for span in spans],
)
def test_real_bracket_examples_classify_by_rule(expected: BracketKind, span: str) -> None:
    assert classify_bracket_text(span) == expected


def test_classifies_speaker_and_collector_brackets() -> None:
    assert classify_bracket_text("Л:") == BracketKind.SPEAKER_TAG
    assert classify_bracket_text("АИ-1:") == BracketKind.SPEAKER_TAG
    assert classify_bracket_text("МВ, ???:") == BracketKind.SPEAKER_TAG
    assert classify_bracket_text("Соб.: Да.") == BracketKind.COLLECTOR_UTTERANCE
    assert classify_bracket_text("А где это было?") == BracketKind.COLLECTOR_UTTERANCE


def test_classifies_spoken_and_editorial_brackets() -> None:
    assert classify_bracket_text("Поет: Вот настало утро.") == BracketKind.SPOKEN_UTTERANCE
    assert classify_bracket_text("ДС говорит:") == BracketKind.SPEAKER_NOTE
    assert classify_bracket_text("См. XIII-9 а.") == BracketKind.EDITORIAL_NOTE
    assert classify_bracket_text("Непонятная фраза без явного маркера") == BracketKind.UNKNOWN


def test_note_only_rows_are_dropped_only_for_clear_editorial_notes() -> None:
    assert is_note_only_transcript_row("[См. XIII-9 а.]")
    assert not is_note_only_transcript_row("[Соб.: Да.]")
    assert not is_note_only_transcript_row("[Поет: Вот настало утро.]")
    assert not is_note_only_transcript_row("[Непонятная фраза без явного маркера]")
