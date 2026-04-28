# Uniform ASR inference scripts

Files:
- `run_whisper_asr.py`
- `run_mms_asr.py`
- `run_xlsr_asr.py`
- `asr_common.py`

All three scripts share the same core CLI:

```bash
python SCRIPT.py \
  --input-dir /data/audio \
  --glob '*.wav' \
  --output /data/preds.jsonl \
  --model-id MODEL_OR_PATH \
  --device cuda:0 \
  --dtype float16 \
  --max-batch-size 48 \
  --max-batch-audio-s 240 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --pin-memory
```

Notes:
- Use either `--input-dir` or `--manifest` or both.
- `--manifest` may be plain text with one path per line, or JSONL with a `path` field.
- Output can be `.jsonl` or `.csv`.
- Files are duration-bucketed before batching, which greatly reduces padding waste for 2-10 second clips.
- GPU efficiency comes mainly from one large batched inference stream on the 40 GB card, while CPU workers decode and resample in parallel.

## Recommended settings for a 40 GB GPU

### Whisper
```bash
python run_whisper_asr.py \
  --input-dir /data/audio \
  --glob '*.wav' \
  --output /data/whisper_preds.jsonl \
  --model-id openai/whisper-large-v3 \
  --language russian \
  --task transcribe \
  --device cuda:0 \
  --dtype float16 \
  --max-batch-size 64 \
  --max-batch-audio-s 320 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --pin-memory
```

### MMS
```bash
python run_mms_asr.py \
  --input-dir /data/audio \
  --glob '*.wav' \
  --output /data/mms_preds.jsonl \
  --model-id facebook/mms-1b-all \
  --language-code rus \
  --device cuda:0 \
  --dtype float16 \
  --max-batch-size 48 \
  --max-batch-audio-s 240 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --pin-memory
```

### XLS-R
```bash
python run_xlsr_asr.py \
  --input-dir /data/audio \
  --glob '*.wav' \
  --output /data/xlsr_preds.jsonl \
  --model-id /models/my-xlsr-russian-dialect-ctc \
  --processor-id /models/my-xlsr-russian-dialect-ctc \
  --device cuda:0 \
  --dtype float16 \
  --max-batch-size 64 \
  --max-batch-audio-s 320 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --pin-memory
```

## Dependency sketch

```bash
pip install torch torchaudio transformers accelerate sentencepiece tqdm
```

Some Whisper checkpoints may also benefit from:

```bash
pip install safetensors
```
