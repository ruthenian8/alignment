"""Run GigaAM ASR over local utterance-level audio files."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

try:
    from .asr_common import (
        AudioItem,
        add_shared_args,
        finalize_and_write,
        load_items_from_args,
        require_torch,
        resolve_device_and_dtype,
    )
except ImportError:
    from asr_common import (
        AudioItem,
        add_shared_args,
        finalize_and_write,
        load_items_from_args,
        require_torch,
        resolve_device_and_dtype,
    )
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - depends on optional ASR environment.

    def tqdm(iterable, **_kwargs):
        """Fallback iterator when tqdm is not installed."""
        return iterable


DEFAULT_GIGAAM_MODEL = "v3_e2e_rnnt"


def parse_args() -> argparse.Namespace:
    """Parse GigaAM ASR command-line arguments."""
    parser = argparse.ArgumentParser(description="GigaAM ASR inference over short local audio files.")
    add_shared_args(
        parser,
        model_id_required=False,
        default_model_id=DEFAULT_GIGAAM_MODEL,
    )
    parser.add_argument(
        "--download-root",
        type=str,
        default=None,
        help="Optional GigaAM checkpoint cache directory.",
    )
    parser.add_argument(
        "--use-flash",
        action="store_true",
        help="Enable GigaAM flash attention if the installed environment supports it.",
    )
    return parser.parse_args()


def load_gigaam_model(args: argparse.Namespace):
    """Load a GigaAM ASR model on the requested device."""
    torch_lib = require_torch()
    device, dtype = resolve_device_and_dtype(args.device, args.dtype)
    try:
        import gigaam
    except ImportError as e:
        raise RuntimeError(
            "GigaAM ASR requires the gigaam package. Install it from "
            "https://github.com/salute-developers/GigaAM with the torch extra."
        ) from e

    fp16_encoder = device.type != "cpu" and dtype == torch_lib.float16
    logging.info("Loading GigaAM model: %s", args.model_id)
    return gigaam.load_model(
        args.model_id,
        fp16_encoder=fp16_encoder,
        use_flash=args.use_flash,
        device=device,
        download_root=args.download_root,
    )


def result_text(result) -> str:
    """Extract text from current and older GigaAM transcription return values."""
    return getattr(result, "text", result).strip()


def transcribe_items(model, items: Sequence[AudioItem], model_id: str) -> list[dict]:
    """Transcribe audio items and return common ASR result rows."""
    rows: list[dict] = []
    for item in tqdm(items, desc="gigaam"):
        text = result_text(model.transcribe(str(item.path)))
        rows.append(
            {
                "path": str(item.path),
                "text": text,
                "duration_s": round(float(item.duration_s), 3),
                "model_id": model_id,
                "model_type": "gigaam",
            }
        )
    return rows


def main() -> None:
    """Run GigaAM ASR inference."""
    args = parse_args()
    items = load_items_from_args(args)
    too_long = [item.path for item in items if item.duration_s > 25.0]
    if too_long:
        examples = ", ".join(str(p) for p in too_long[:3])
        raise RuntimeError(
            "GigaAM .transcribe supports clips up to 25 seconds. "
            f"Cut long audio before ASR; examples over the limit: {examples}"
        )
    model = load_gigaam_model(args)
    finalize_and_write(args.output, transcribe_items(model, items, args.model_id))


if __name__ == "__main__":
    main()
