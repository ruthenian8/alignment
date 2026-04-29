import os
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import hf_hub_download
from thefuzz import fuzz # нужeн pip install
from rusenttokenize import ru_sent_tokenize # нужeн pip install


def normalize_whisper(whisper):
    # разбивает текст Whisper на предложения
    sentencized_whisper = ru_sent_tokenize(whisper)

    # словарь: нормализованный текст -> оригинальный сегмент Whisper
    # можно в принципе перейти на индексы, если хочется оптимизировать
    mapping = {}

    for sent in sentencized_whisper:
        # паттерн для удаления номера сегмента, таймкодов и SPEAKER-тега
        pattern = re.compile(
            r'\d+\n\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+\n\[SPEAKER_\d+\]:'
        )

        # удаляем служебную разметку, оставляя только текст
        corrected = re.sub(pattern, '', sent)

        # сохраняем соответствие "очищенный текст -> оригинальный сегмент"
        mapping[corrected] = sent

    return mapping


def normalize_trans(transcript):
    # удаляем все конструкции в квадратных скобках
    transcript = re.sub(r'\[[^\[\]]+\]', '', transcript)

    # разбиваем транскрипт на предложения
    sentencized_trans = ru_sent_tokenize(transcript)

    # словарь: нормализованный текст -> оригинальное предложение
    mapping = {}

    for sent in sentencized_trans:
        # паттерн для очистки многоточий, экранированных символов
        # и остаточных скобочных конструкций
        pattern = re.compile(r'(?:[…\\]|\[[^\[\]]+\])+')

        # финальная нормализация предложения
        corrected = re.sub(pattern, '', sent)

        mapping[corrected] = sent

    return mapping


def right_condition(sent, current_transcript, alpha=90, beta=90):
    # проверка совпадения по token_set_ratio
    if fuzz.token_set_ratio(sent, current_transcript) > alpha:
        return True

    # проверка совпадения по partial_ratio
    if fuzz.partial_ratio(sent, current_transcript) > beta:
        return True

    # если ни одно условие не выполнено — совпадение недостаточное
    return False


def check_next(idx, sent, trans_norm, num, aligned, retry=5):
    # ограничиваем число попыток, чтобы не выйти за границы списка
    if num + retry > len(trans_norm):
        retry = len(trans_norm) - num - 2

    # сохраняем исходную позицию в транскрипте
    true_number = num

    # пробуем сдвигаться вперёд по транскрипту
    for i in range(retry):

        if num + 1 == len(trans_norm):
            break

        num += 1
        current_transcript = trans_norm[num]

        entered = False

        # пока текущее предложение транскрипта подходит —
        # добавляем его к текущему сегменту Whisper
        while num < len(trans_norm) and right_condition(sent, current_transcript):
            entered = True
            aligned[idx].append(current_transcript)

            if num >= len(trans_norm) - 1:
                break

            num += 1
            current_transcript = trans_norm[num]

        # если совпадение было найдено — возвращаем новую позицию
        if entered:
            return num, aligned

    # если совпадение так и не найдено — возвращаем исходную позицию
    return true_number, aligned


def align_fc(whisper_mapping, trans_mapping, alpha=75, beta=80):
    # словарь: индекс предложения Whisper -> список нормализованных предложений транскрипта
    aligned = {}

    # указатель текущей позиции в транскрипте
    num = 0

    # нормализованные предложения транскрипта и Whisper
    trans_norm = list(trans_mapping.keys())
    whisper_norm = list(whisper_mapping.keys())

    # проходим по предложениям Whisper по порядку
    for idx, sent in enumerate(whisper_norm):
        if idx not in aligned:
            aligned[idx] = []

        # если транскрипт закончился — прекращаем выравнивание
        if num < len(trans_norm):
            current_transcript = trans_norm[num]
        else:
            break

        entered = False

        # основной цикл: пока предложения совпадают — считаем их выровненными
        while num < len(trans_norm) and right_condition(sent, current_transcript, alpha, beta):
            entered = True
            aligned[idx].append(current_transcript)

            if num >= len(trans_norm) - 1:
                break

            num += 1
            current_transcript = trans_norm[num]

        # если прямого совпадения не было — пробуем найти его дальше
        else:
            num, aligned = check_next(idx, sent, trans_norm, num, aligned)
            current_transcript = trans_norm[num]

    return aligned


def final_process(aligned, whisper_mapping, transcript_mapping):
    # итоговый словарь с восстановленным оригинальным текстом
    final_aligned = {}

    # список пар (нормализованный текст, оригинальный сегмент Whisper)
    whisper_norm = list(whisper_mapping.items())

    for i, value in aligned.items():
        # оригинальный сегмент Whisper
        prefix = whisper_norm[i][1]
        final_aligned[i] = []

        # если соответствий не найдено — оставляем Whisper как есть
        if not value:
            final_aligned[i].append(whisper_norm[i][1])
        else:
            # паттерн для извлечения заголовка .srt (номер + таймкод + SPEAKER)
            pattern = re.compile(
                r'\d+\n\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+\n\[SPEAKER_\d+\]:'
            )

            # добавляем заголовок Whisper, если он есть
            prefix = re.match(pattern, prefix)
            if prefix:
                final_aligned[i].append(prefix.group())

            # добавляем оригинальные предложения транскрипта
            for val in value:
                final_aligned[i].append(transcript_mapping[val])

    return final_aligned


REPO_ID = "hse-prs-folklore/corpus"


def analyze_folder(basedir, filename):
    # загружаем CSV-файл с метаданными из HuggingFace Hub
    # вам не нужно – используйте предоставленные данные
    csv_file_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        token=os.environ['HF_TOKEN']
    )

    orig_srt_file = []
    orig_whisper = []
    orig_transcript = []
    aligned = []

    df = pd.read_csv(csv_file_path)

    # проходим по записям датасета
    for name, transcript, is_trans in tqdm(zip(df['name'], df['transcript'], df['trans'])):
        # пропускаем записи без транскрипта
        if not is_trans:
            continue

        # получаем путь к соответствующему .srt файлу
        srt_file = Path(name).with_suffix('.srt')
        srt_file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f'{basedir}/{srt_file}',
            repo_type="dataset",
            token=os.environ['HF_TOKEN']
        )

        # читаем Whisper-разметку
        with open(srt_file_path) as file:
            whisper = file.read()

        # нормализация Whisper и транскрипта
        whisper_norm = normalize_whisper(whisper)
        trans_norm = normalize_trans(transcript)

        # выравнивание
        alignment = align_fc(whisper_norm, trans_norm)

        # восстановление оригинального текста
        new_aligned = final_process(alignment, whisper_norm, trans_norm)

        # склеиваем результат в единый текст
        final_alignment = '\n\n'.join([' '.join(x) for x in new_aligned.values()])

        orig_whisper.append(whisper)
        orig_srt_file.append(srt_file)
        orig_transcript.append(transcript)
        aligned.append(final_alignment)

    new_df = pd.DataFrame(
        {
            'srt_file_names': orig_srt_file,
            'orig_whisper': orig_whisper,
            'orig_transcript': orig_transcript,
            'alignment': aligned
        }
    )

    return new_df
