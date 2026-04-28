from __future__ import annotations

import argparse
import logging
from typing import List

import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, AutoProcessor

from asr_common import add_shared_args, create_dataloader, finalize_and_write, load_items_from_args, torch_dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched MMS ASR inference over many short audio files.")
    add_shared_args(parser)
    parser.add_argument("--language-code", type=str, default=None, help="MMS adapter language code such as rus, deu, etc.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_items_from_args(args)
    dtype = torch_dtype_from_name(args.dtype)
    device = torch.device(args.device)

    logging.info("Loading MMS model: %s", args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if args.language_code:
        if hasattr(processor.tokenizer, "set_target_lang"):
            processor.tokenizer.set_target_lang(args.language_code)
        if hasattr(model, "load_adapter"):
            model.load_adapter(args.language_code)
        logging.info("Activated MMS adapter language: %s", args.language_code)
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

    results: List[dict] = []
    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)

    for batch in tqdm(loader, desc="mms"):
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

        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            logits = model(input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)
        texts = processor.batch_decode(pred_ids)

        for path, text, dur in zip(batch["paths"], texts, batch["durations_s"]):
            results.append({
                "path": path,
                "text": text.strip(),
                "duration_s": round(float(dur), 3),
                "model_id": args.model_id,
                "model_type": "mms",
                "language_code": args.language_code,
            })

    finalize_and_write(args.output, results)


if __name__ == "__main__":
    main()
