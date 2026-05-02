"""Run batched XLS-R or Wav2Vec2 CTC ASR over local audio files."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

try:
    from .asr_common import (
        add_shared_args,
        create_dataloader,
        finalize_and_write,
        load_items_from_args,
        require_torch,
        resolve_device_and_dtype,
    )
except ImportError:
    from asr_common import (
        add_shared_args,
        create_dataloader,
        finalize_and_write,
        load_items_from_args,
        require_torch,
        resolve_device_and_dtype,
    )
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress display.

    def tqdm(iterable, **_kwargs):
        return iterable


try:
    from transformers import (
        AutoProcessor,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
    )
except ImportError:  # pragma: no cover - depends on optional ASR environment.
    AutoProcessor = None
    Wav2Vec2CTCTokenizer = None
    Wav2Vec2FeatureExtractor = None
    Wav2Vec2ForCTC = None
    Wav2Vec2Processor = None


def parse_args() -> argparse.Namespace:
    """Parse XLS-R ASR command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batched XLS-R / Wav2Vec2-CTC ASR inference over many short audio files."
    )
    add_shared_args(parser)
    parser.add_argument(
        "--processor-id",
        type=str,
        default=None,
        help="Optional separate processor/tokenizer id if different from --model-id.",
    )
    parser.add_argument(
        "--vocab-json",
        type=Path,
        default=None,
        help="Optional local CTC vocab.json used with pretrained model weights.",
    )
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        default=None,
        help="Optional local tokenizer.json paired with --vocab-json.",
    )
    return parser.parse_args()


def require_transformers() -> None:
    """Fail with a clear message when the optional Transformers stack is absent."""
    if AutoProcessor is None or Wav2Vec2ForCTC is None:
        raise RuntimeError(
            "XLS-R inference requires transformers. Install the versions pinned in freeze.txt."
        )


def local_tokenizer_special_tokens(vocab_json: Path) -> dict[str, str]:
    """Infer common Wav2Vec2 CTC special tokens from a local vocab file."""
    with vocab_json.open(encoding="utf-8") as f:
        vocab = json.load(f)
    if not isinstance(vocab, dict):
        raise ValueError(f"Expected a JSON object in {vocab_json}")

    def pick(candidates: tuple[str, ...], fallback: str) -> str:
        return next((token for token in candidates if token in vocab), fallback)

    return {
        "unk_token": pick(("[UNK]", "<unk>", "<UNK>"), "[UNK]"),
        "pad_token": pick(("[PAD]", "<pad>", "<PAD>"), "[PAD]"),
        "word_delimiter_token": pick(("|", " "), "|"),
    }


def load_processor(processor_id: str, vocab_json: Path | None, tokenizer_json: Path | None):
    """Load the XLS-R processor, optionally replacing only the CTC tokenizer."""
    require_transformers()
    if vocab_json is None and tokenizer_json is None:
        return AutoProcessor.from_pretrained(processor_id)
    if vocab_json is None:
        raise ValueError("--tokenizer-json requires --vocab-json")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(processor_id)
    tokenizer_kwargs = {
        "vocab_file": str(vocab_json),
        **local_tokenizer_special_tokens(vocab_json),
    }
    if tokenizer_json is not None:
        tokenizer_kwargs["tokenizer_file"] = str(tokenizer_json)
    tokenizer = Wav2Vec2CTCTokenizer(**tokenizer_kwargs)
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def local_vocab_model_kwargs(processor) -> dict:
    """Return model load kwargs that align the CTC head with a local tokenizer."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Local vocab model initialization requires a processor tokenizer")

    kwargs = {
        "vocab_size": len(tokenizer),
        "ignore_mismatched_sizes": True,
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


def main() -> None:
    """Run XLS-R ASR inference."""
    args = parse_args()
    items = load_items_from_args(args)
    device, dtype = resolve_device_and_dtype(args.device, args.dtype)
    torch_lib = require_torch()

    processor_id = args.processor_id or args.model_id
    logging.info("Loading XLS-R processor: %s", processor_id)
    processor = load_processor(processor_id, args.vocab_json, args.tokenizer_json)
    logging.info("Loading XLS-R model: %s", args.model_id)
    require_transformers()
    model_kwargs = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if args.vocab_json is not None:
        model_kwargs.update(local_vocab_model_kwargs(processor))
    model = Wav2Vec2ForCTC.from_pretrained(args.model_id, **model_kwargs)
    model.to(device)
    model.eval()

    loader = create_dataloader(
        items,
        max_batch_size=args.max_batch_size,
        max_batch_audio_s=args.max_batch_audio_s,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
    )

    results: list[dict] = []
    autocast_enabled = device.type == "cuda" and dtype in (torch_lib.float16, torch_lib.bfloat16)

    for batch in tqdm(loader, desc="xlsr"):
        inputs = processor(
            batch["audio"].numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device=device, non_blocking=True)
        attention_mask = None
        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
            attention_mask = inputs.attention_mask.to(device=device, non_blocking=True)

        with (
            torch_lib.inference_mode(),
            torch_lib.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled),
        ):
            logits = model(input_values, attention_mask=attention_mask).logits
        pred_ids = torch_lib.argmax(logits, dim=-1)
        texts = processor.batch_decode(pred_ids)

        for path, text, dur in zip(batch["paths"], texts, batch["durations_s"], strict=True):
            results.append(
                {
                    "path": path,
                    "text": text.strip(),
                    "duration_s": round(float(dur), 3),
                    "model_id": args.model_id,
                    "model_type": "xlsr",
                }
            )

    finalize_and_write(args.output, results)


if __name__ == "__main__":
    main()
