"""Shared helpers for optional batched ASR scripts."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - depends on optional ASR environment.
    torch = None

try:
    import torchaudio
except ImportError:  # pragma: no cover - depends on optional ASR environment.
    torchaudio = None

if torch is not None:
    from torch.utils.data import DataLoader, Dataset, Sampler
else:
    DataLoader = None

    class Dataset:
        """Placeholder base class used when torch is not installed."""

    class Sampler:
        """Placeholder base class used when torch is not installed."""

        def __class_getitem__(cls, _item):
            return cls


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".webm"}
TARGET_SR = 16_000


def setup_logging(verbose: bool = False) -> None:
    """Configure process-wide logging for ASR scripts."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


@dataclass
class AudioItem:
    """Audio path plus duration metadata used for batching."""

    path: Path
    duration_s: float


class AudioDataset(Dataset):
    """Load, mono-mix, and resample audio files lazily for inference."""

    def __init__(self, items: Sequence[AudioItem], target_sr: int = TARGET_SR):
        self._torchaudio = require_torchaudio()
        self.items = list(items)
        self.target_sr = target_sr
        self._resamplers: dict[int, object] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _get_resampler(self, orig_sr: int) -> object | None:
        if orig_sr == self.target_sr:
            return None
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = self._torchaudio.transforms.Resample(orig_sr, self.target_sr)
        return self._resamplers[orig_sr]

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        wav, sr = self._torchaudio.load(str(item.path))
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


class DurationBucketBatchSampler(Sampler[list[int]]):
    """Group duration-sorted files into batches constrained by count and seconds."""

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

    def __iter__(self) -> Iterator[list[int]]:
        import random

        pairs = list(enumerate(self.items))
        pairs.sort(key=lambda x: x[1].duration_s)

        batches: list[list[int]] = []
        current: list[int] = []
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
    """Pad variable-length audio tensors and return batch metadata."""
    torch_lib = require_torch()
    audios = [x["audio"] for x in batch]
    lengths = torch_lib.tensor([x["num_samples"] for x in batch], dtype=torch_lib.long)
    padded = torch_lib.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    attention_mask = torch_lib.arange(padded.shape[1])[None, :] < lengths[:, None]
    return {
        "paths": [x["path"] for x in batch],
        "audio": padded,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "durations_s": [x["duration_s"] for x in batch],
    }


def discover_audio_files(input_dir: str | None, manifest: str | None, glob_pattern: str) -> list[Path]:
    """Collect audio paths from an input directory and/or path manifest."""
    paths: list[Path] = []
    if manifest:
        manifest_path = Path(manifest)
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Supports plain path per line or JSONL with a "path" field.
                if line.startswith("{"):
                    obj = json.loads(line)
                    p = obj.get("path")
                    if p:
                        paths.append(_resolve_manifest_path(manifest_path, p))
                else:
                    paths.append(_resolve_manifest_path(manifest_path, line))
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


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    """Resolve relative manifest entries against the manifest file directory."""
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def require_torch():
    """Return torch or raise a clear runtime error for optional ASR commands."""
    if torch is None:
        raise RuntimeError("ASR inference requires torch. Install the versions pinned in freeze.txt.")
    return torch


def require_torchaudio():
    """Return torchaudio or raise a clear runtime error for optional ASR commands."""
    if torchaudio is None:
        raise RuntimeError("ASR inference requires torchaudio. Install the versions pinned in freeze.txt.")
    return torchaudio


def audio_duration_s(path: Path) -> float:
    """Return audio duration using metadata when available, falling back to decoding.

    Some torchaudio builds expose ``torchaudio.info``, while others only expose
    ``torchaudio.load``. The fallback keeps duration discovery compatible with
    the frozen ASR environment without changing the supported input formats.
    """
    ta = require_torchaudio()
    info = getattr(ta, "info", None)
    if callable(info):
        try:
            metadata = info(str(path))
            if metadata.num_frames > 0 and metadata.sample_rate > 0:
                return metadata.num_frames / float(metadata.sample_rate)
        except Exception as e:
            logging.debug("Falling back to audio decode for %s after metadata read failed: %s", path, e)

    waveform, sample_rate = ta.load(str(path))
    if sample_rate <= 0:
        raise RuntimeError(f"Invalid sample rate for {path}: {sample_rate}")
    return waveform.shape[-1] / float(sample_rate)


def build_items(paths: Sequence[Path]) -> list[AudioItem]:
    """Build duration metadata for readable audio files."""
    items: list[AudioItem] = []

    for p in paths:
        try:
            duration_s = audio_duration_s(p)
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
    """Create the duration-bucketed audio dataloader."""
    if DataLoader is None:
        raise RuntimeError("ASR inference requires torch. Install the versions pinned in freeze.txt.")
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
    """Write ASR result rows as JSONL or CSV."""
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


def add_shared_args(
    parser: argparse.ArgumentParser,
    *,
    model_id_required: bool = True,
    default_model_id: str | None = None,
) -> argparse.ArgumentParser:
    """Add common ASR command-line arguments to a parser."""
    parser.add_argument(
        "--input-dir", type=str, default=None, help="Directory scanned recursively for audio files."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Text file with one path per line or JSONL with a path field.",
    )
    parser.add_argument("--glob", type=str, default="*", help="Recursive glob applied under --input-dir.")
    parser.add_argument("--output", type=str, required=True, help="Output .jsonl or .csv file.")
    parser.add_argument(
        "--model-id",
        type=str,
        required=model_id_required,
        default=default_model_id,
        help="Model id, local path, or backend-specific model name.",
    )
    parser.add_argument(
        "--language", type=str, default=None, help="Language or dialect tag if supported by the model."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Used mainly by Whisper.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda, cuda:0, cpu, etc.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model compute dtype.",
    )
    parser.add_argument("--max-batch-size", type=int, default=32, help="Maximum utterances per batch.")
    parser.add_argument(
        "--max-batch-audio-s", type=float, default=180.0, help="Total audio seconds per batch cap."
    )
    parser.add_argument(
        "--num-workers", type=int, default=min(8, os.cpu_count() or 1), help="CPU workers for audio decoding."
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor per worker."
    )
    parser.add_argument("--pin-memory", action="store_true", help="Pin host memory before GPU transfer.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of audio files.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate common ASR command-line arguments."""
    if not args.input_dir and not args.manifest:
        raise ValueError("Provide at least one of --input-dir or --manifest.")
    if args.max_batch_size < 1:
        raise ValueError("--max-batch-size must be >= 1")
    if args.max_batch_audio_s <= 0:
        raise ValueError("--max-batch-audio-s must be > 0")


def torch_dtype_from_name(name: str):
    """Map a CLI dtype name to a torch dtype."""
    torch_lib = require_torch()
    mapping = {
        "float16": torch_lib.float16,
        "bfloat16": torch_lib.bfloat16,
        "float32": torch_lib.float32,
    }
    return mapping[name]


def resolve_device_and_dtype(device_name: str, dtype_name: str) -> tuple:
    """Return a usable torch device and dtype, failing early for invalid CUDA use."""
    torch_lib = require_torch()
    device = torch_lib.device(device_name)
    if device.type == "cuda" and not torch_lib.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Use --device cpu or install CUDA support."
        )
    dtype = torch_dtype_from_name(dtype_name)
    if device.type == "cpu" and dtype != torch_lib.float32:
        logging.warning("Using float32 on CPU instead of %s", dtype_name)
        dtype = torch_lib.float32
    return device, dtype


def load_items_from_args(args: argparse.Namespace) -> list[AudioItem]:
    """Load duration metadata for audio files described by CLI arguments."""
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
    """Write output rows and log a short completion message."""
    write_results(output_path, rows)
    logging.info("Wrote %d rows to %s", len(rows), output_path)
