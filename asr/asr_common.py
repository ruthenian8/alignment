from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Sampler

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".webm"}
TARGET_SR = 16_000


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


@dataclass
class AudioItem:
    path: Path
    duration_s: float


class AudioDataset(Dataset):
    def __init__(self, items: Sequence[AudioItem], target_sr: int = TARGET_SR):
        self.items = list(items)
        self.target_sr = target_sr
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _get_resampler(self, orig_sr: int) -> Optional[torchaudio.transforms.Resample]:
        if orig_sr == self.target_sr:
            return None
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.target_sr)
        return self._resamplers[orig_sr]

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        wav, sr = torchaudio.load(str(item.path))
        if wav.numel() == 0:
            raise RuntimeError(f"Empty audio file: {item.path}")
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        resampler = self._get_resampler(sr)
        if resampler is not None:
            wav = resampler(wav)
        wav = wav.squeeze(0).contiguous()
        return {
            "path": str(item.path),
            "audio": wav,
            "num_samples": int(wav.numel()),
            "duration_s": wav.numel() / self.target_sr,
        }


class DurationBucketBatchSampler(Sampler[List[int]]):
    """
    Groups sorted-by-duration files into batches constrained by both item count
    and total audio seconds. This keeps short clips highly batched while avoiding
    pathological padding for the longest clips.
    """

    def __init__(
        self,
        items: Sequence[AudioItem],
        max_batch_size: int,
        max_batch_audio_s: float,
        shuffle: bool = False,
    ) -> None:
        self.items = list(items)
        self.max_batch_size = max_batch_size
        self.max_batch_audio_s = max_batch_audio_s
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[List[int]]:
        import random

        pairs = list(enumerate(self.items))
        pairs.sort(key=lambda x: x[1].duration_s)

        batches: List[List[int]] = []
        current: List[int] = []
        current_audio_s = 0.0
        for idx, item in pairs:
            if current and (
                len(current) >= self.max_batch_size
                or current_audio_s + item.duration_s > self.max_batch_audio_s
            ):
                batches.append(current)
                current = []
                current_audio_s = 0.0
            current.append(idx)
            current_audio_s += item.duration_s
        if current:
            batches.append(current)

        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        # Approximation good enough for progress bars.
        total_audio = sum(x.duration_s for x in self.items)
        by_audio = math.ceil(total_audio / max(self.max_batch_audio_s, 1e-6))
        by_count = math.ceil(len(self.items) / max(self.max_batch_size, 1))
        return max(by_audio, by_count)


def collate_audio(batch: Sequence[dict]) -> dict:
    audios = [x["audio"] for x in batch]
    lengths = torch.tensor([x["num_samples"] for x in batch], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    attention_mask = torch.arange(padded.shape[1])[None, :] < lengths[:, None]
    return {
        "paths": [x["path"] for x in batch],
        "audio": padded,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "durations_s": [x["duration_s"] for x in batch],
    }


def discover_audio_files(input_dir: Optional[str], manifest: Optional[str], glob_pattern: str) -> List[Path]:
    paths: List[Path] = []
    if manifest:
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Supports plain path per line or JSONL with a "path" field.
                if line.startswith("{"):
                    obj = json.loads(line)
                    p = obj.get("path")
                    if p:
                        paths.append(Path(p))
                else:
                    paths.append(Path(line))
    if input_dir:
        root = Path(input_dir)
        for p in root.rglob(glob_pattern):
            if p.suffix.lower() in AUDIO_EXTENSIONS:
                paths.append(p)
    deduped = []
    seen = set()
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            deduped.append(Path(rp))
    return deduped


def build_items(paths: Sequence[Path]) -> List[AudioItem]:
    items: List[AudioItem] = []

    for p in paths:
        try:
            waveform, sample_rate = torchaudio.load(str(p))
            duration_s = waveform.shape[-1] / float(sample_rate)
        except Exception as e:
            logging.warning("Skipping %s: failed to read audio (%s)", p, e)
            continue

        items.append(AudioItem(path=p, duration_s=duration_s))

    return items

def create_dataloader(
    items: Sequence[AudioItem],
    max_batch_size: int,
    max_batch_audio_s: float,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = AudioDataset(items)
    batch_sampler = DurationBucketBatchSampler(
        items=items,
        max_batch_size=max_batch_size,
        max_batch_audio_s=max_batch_audio_s,
        shuffle=False,
    )
    kwargs = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_audio,
        pin_memory=pin_memory,
        **kwargs,
    )


def write_results(path: str, rows: Sequence[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    if out_path.suffix.lower() == ".csv":
        if not rows:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                pass
            return
        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return
    raise ValueError("Output file must end in .jsonl or .csv")


def add_shared_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--input-dir", type=str, default=None, help="Directory scanned recursively for audio files.")
    parser.add_argument("--manifest", type=str, default=None, help="Text file with one path per line or JSONL with a path field.")
    parser.add_argument("--glob", type=str, default="*", help="Recursive glob applied under --input-dir.")
    parser.add_argument("--output", type=str, required=True, help="Output .jsonl or .csv file.")
    parser.add_argument("--model-id", type=str, required=True, help="HF model id or local path.")
    parser.add_argument("--language", type=str, default=None, help="Language or dialect tag if supported by the model.")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="Used mainly by Whisper.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda, cuda:0, cpu, etc.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Model compute dtype.")
    parser.add_argument("--max-batch-size", type=int, default=32, help="Maximum utterances per batch.")
    parser.add_argument("--max-batch-audio-s", type=float, default=180.0, help="Total audio seconds per batch cap.")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1), help="CPU workers for audio decoding.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor per worker.")
    parser.add_argument("--pin-memory", action="store_true", help="Pin host memory before GPU transfer.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of audio files.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir and not args.manifest:
        raise ValueError("Provide at least one of --input-dir or --manifest.")
    if args.max_batch_size < 1:
        raise ValueError("--max-batch-size must be >= 1")
    if args.max_batch_audio_s <= 0:
        raise ValueError("--max-batch-audio-s must be > 0")


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_items_from_args(args: argparse.Namespace) -> List[AudioItem]:
    validate_args(args)
    setup_logging(args.verbose)
    paths = discover_audio_files(args.input_dir, args.manifest, args.glob)
    if args.limit is not None:
        paths = paths[: args.limit]
    logging.info("Discovered %d candidate audio files", len(paths))
    items = build_items(paths)
    logging.info("Prepared %d readable audio files", len(items))
    if not items:
        raise RuntimeError("No readable audio files found")
    return items


def finalize_and_write(output_path: str, rows: Sequence[dict]) -> None:
    write_results(output_path, rows)
    logging.info("Wrote %d rows to %s", len(rows), output_path)
