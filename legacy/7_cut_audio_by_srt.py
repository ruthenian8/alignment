import os
import re
import subprocess
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
from dataclasses import dataclass


REPO_ID = "hse-prs-folklore/corpus"
HF_TOKEN = ''

@dataclass
class Segment:
    index: int
    start: str
    end: str
    speaker: str
    text: str


def parse_srt(text: str) -> list[Segment]:
    if isinstance(text, float):
        print(text)
    blocks = re.split(r"\n\s*\n", text.strip())
    segments = []

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue

        index = int(lines[0].strip())

        start, end = re.findall(
            r"(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1]
        )

        start = start.replace(",", ".")
        end = end.replace(",", ".")

        match = re.match(r"\[(.+?)\]:\s*(.*)", lines[2])
        speaker = match.group(1) if match else "UNKNOWN"
        text_line = match.group(2) if match else lines[2]

        segments.append(
            Segment(
                index=index,
                start=start,
                end=end,
                speaker=speaker,
                text=text_line.strip()
            )
        )

    return segments

def safe_time(t: str) -> str:
    return t.replace(":", "-").replace(".", "-")


def make_basename(seg: Segment) -> str:
    return f"{seg.index:03}_{seg.speaker}_{safe_time(seg.start)}"


def cut_audio_with_text(input_audio, segments, segments_clean, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seg, seg_clean in zip(segments, segments_clean):
        base = make_basename(seg)

        audio_path = output_dir / f"{base}.wav"
        text_path = output_dir / f"{base}_orig.txt"
        text_clean_path = output_dir / f"{base}.txt"

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-ss", seg.start,
            "-to", seg.end,
            "-i", input_audio,
            "-map", "0:a",
            "-c:a", "copy",
            audio_path.as_posix()
        ]

        subprocess.run(ffmpeg_cmd, check=True)

        text_path.write_text(
            seg.text,
            encoding="utf-8"
        )

        text_clean_path.write_text(
            seg_clean.text,
            encoding="utf-8"
        )


def download_file(filename, local_dir):
    hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset",
                token=HF_TOKEN, local_dir=local_dir)


if __name__ == "__main__":
    local_dir = ''
    path = '/pez' 
    for file in sorted(os.listdir(path)):
        full_path = os.path.join(path, file)
        folder_name = os.path.splitext(file)[0]
        alignment = pd.read_csv(full_path)
        srt_names = alignment['srt_file_names']
        srt_alignment_clean = alignment['alignment_clean']
        srt_alignment = alignment['alignment']
        path_in_corpus = f'cut_audio/{folder_name}'

        result_folder = f'/result/{folder_name}'
        os.makedirs(result_folder, exist_ok=True)
        for name, align, align_clean in zip(srt_names, srt_alignment, srt_alignment_clean):
            if isinstance(align, float):
                continue
            print('NAME', name)
            base = os.path.splitext(name)[0]
            file_path = base + '.wav'
            filename = f'{path_in_corpus}/{file_path}'
            download_file(filename, local_dir)
            input_audio = f'/cut_audio/{folder_name}/{file_path}'
            segments = parse_srt(align)
            segments_clean = parse_srt(align_clean)
            output_dir = f'{result_folder}/{base}'
            cut_audio_with_text(input_audio, segments, segments_clean, output_dir)