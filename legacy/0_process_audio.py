import os
import csv
import subprocess
import math
import re

# НАЗВАНИЕ ПО СХЕМЕ ABC_001
def get_count(counter: int):
    s = str(counter + 1)
    return "0" * (3 - len(s)) + s

def normalize_name(prefix: str, counter: int):
    return f"{prefix}_{get_count(counter)}"

# Переименование файла
def rename_audio(basedir, old_name, new_name, ext):
    old_file = os.path.join(basedir, f"{old_name}.{ext}")
    new_file = os.path.join(basedir, f"{new_name}.{ext}")
    if os.path.exists(old_file):
        os.rename(old_file, new_file)
    else:
        raise FileNotFoundError(old_file)

# Конвертация в .wav
def convert_to_wav(path_in, path_out):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", path_in,
        "-ar", "16000",
        "-ac", "1",
        path_out
    ]
    subprocess.run(cmd, check=True, capture_output=True)

# Длительность
def get_duration(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(res.stdout.strip())
    except:
        return None

# Средняя громкость
VOL_PATTERN = re.compile(r"mean_volume:\s*(-?\d+\.?\d*)")
def get_mean_volume(path):
    cmd = [
        "ffmpeg",
        "-i", path,
        "-af", "volumedetect",
        "-f", "null",
        "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, stderr = proc.communicate()
    m = VOL_PATTERN.search(stderr)
    if m:
        return float(m.group(1))
    return None

def run(prefix, basedir, output_csv="result.csv"):
    """Для каждого файла:
        - новое имя (prefix_XXX)
        - конвертация в WAV
        - вычисление duration, mean volume
        - запись в CSV
    """
    results = []
    for idx, filename in enumerate(sorted(os.listdir(basedir))):
        orig_name, orig_ext = filename.rsplit(".", 1)
        full_in = os.path.join(basedir, filename)

        new_name = normalize_name(prefix, idx)
        rename_audio(basedir, orig_name, new_name, orig_ext)
        renamed = os.path.join(basedir, f"{new_name}.{orig_ext}")

        wav_path = os.path.join(basedir, f"{new_name}.wav")
        print(wav_path)
        res = convert_to_wav(renamed, wav_path)

        duration = get_duration(wav_path)
        mean_vol = get_mean_volume(wav_path)

        results.append([
            new_name,          # индекс (без расширения)
            orig_name,         # оригинальное имя (без расширения)
            orig_ext,          # оригинальное расширение
            duration,          # продолжительность файла
            mean_vol          # средняя скорость
        ])

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "original_name",
            "original_extension",
            "duration_sec",
            "mean_volume_dB"
        ])
        writer.writerows(results)

    print(f"CSV сохранён: {output_csv}")

if __name__ == "__main__":
    run(
        prefix="pez",
        basedir="/home/daniiligantev/Downloads/audio/files/audio/Pezhma",
        output_csv="result.csv"
    )
