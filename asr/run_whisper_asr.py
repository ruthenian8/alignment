from __future__ import annotations

import argparse
import logging
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from asr_common import add_shared_args, create_dataloader, finalize_and_write, load_items_from_args, torch_dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched Whisper ASR inference over many short audio files.")
    add_shared_args(parser)
    parser.add_argument("--attn-implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"], help="Transformer attention backend if supported.")
    parser.add_argument("--chunk-length-s", type=float, default=0.0, help="Optional long-audio chunking. Keep 0 for 2-10 second clips.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_items_from_args(args)
    dtype = torch_dtype_from_name(args.dtype)
    device = torch.device(args.device)

    logging.info("Loading Whisper model: %s", args.model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation=args.attn_implementation,
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id)

    loader = create_dataloader(
        items,
        max_batch_size=args.max_batch_size,
        max_batch_audio_s=args.max_batch_audio_s,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
    )

    forced_decoder_ids = None
    if hasattr(processor, "get_decoder_prompt_ids"):
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)

    results: List[dict] = []
    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)

    for batch in tqdm(loader, desc="whisper"):
        inputs = processor(
            batch["audio"].numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device=device, dtype=dtype, non_blocking=True)

        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            generated = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=args.max_new_tokens,
            )

        texts = processor.batch_decode(generated, skip_special_tokens=True)
        for path, text, dur in zip(batch["paths"], texts, batch["durations_s"]):
            results.append({
                "path": path,
                "text": text.strip(),
                "duration_s": round(float(dur), 3),
                "model_id": args.model_id,
                "model_type": "whisper",
            })

    finalize_and_write(args.output, results)


if __name__ == "__main__":
    main()
