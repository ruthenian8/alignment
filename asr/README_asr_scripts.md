# Uniform ASR inference scripts

Files:
- `run_whisper_asr.py`
- `run_mms_asr.py`
- `run_xlsr_asr.py`
- `run_gigaam_asr.py`
- `asr_common.py`
- `evaluate_corpus.py`

All model runners share the same core CLI:

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
- GigaAM uses the same discovery and output format, but calls the upstream short-clip `transcribe()` API
  rather than the batched Hugging Face processor path.

## Corpus WER for cut samples

`evaluate_corpus.py` expects utterance-level files laid out like `cut_samples/pez_001/pez_001No1/`:

```text
001_SPEAKER_00_00-00-00-031.wav
001_SPEAKER_00_00-00-00-031.txt
001_SPEAKER_00_00-00-00-031_orig.txt
```

The sibling `.txt` file is the normalized reference. `_orig.txt` is preserved by the corpus builder but is
not used for ASR WER.

Evaluate an existing prediction file:

```bash
python asr/evaluate_corpus.py \
  hf-repo/cut_samples \
  build/asr-eval \
  --predictions build/asr-eval/predictions.jsonl
```

Run inference first, then evaluate:

```bash
python asr/evaluate_corpus.py \
  hf-repo/cut_samples \
  build/asr-eval-whisper \
  --asr-command "python asr/run_whisper_asr.py --manifest {manifest} --output {predictions} --model-id openai/whisper-large-v3 --language russian --task transcribe --device cuda:0 --dtype float16"
```

When `--asr-command` is used, the evaluator checks that every manifest path has a prediction.
If a batch or worker drops rows, it writes `missing_predictions_retry_1.txt` and reruns the same
command on only the missing files once by default. Use `--retry-missing N` to change the retry
count, or `--allow-missing-predictions` to keep partial results after retries.

GigaAM works the same way for utterance-level clips:

```bash
python asr/evaluate_corpus.py \
  hf-repo/cut_samples \
  build/asr-eval-gigaam \
  --asr-command "python asr/run_gigaam_asr.py --manifest {manifest} --output {predictions} --model-id v3_e2e_rnnt --device cuda:0 --dtype float16"
```

Outputs:
- `audio_manifest.txt`: audio paths passed to the ASR runner.
- `predictions.jsonl`: default inference output path when `--asr-command` is used.
- `missing_predictions.txt`: final missing paths when prediction coverage is incomplete.
- `per_utterance.csv`: per-file reference, prediction, edit counts, and WER.
- `mismatches.csv`: all substitutions, deletions, and insertions sorted by frequency.
- `wer_report.txt`: global WER plus the most common error types.

Existing prediction files may include a `reference_path` field. When present, `evaluate_corpus.py`
uses that text file as the reference for the row; otherwise it uses the default sibling `.txt`
next to the audio path. Use `--prediction-reference-path-field` if a copied prediction file uses a
different field name.

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

To use open pretrained XLS-R/Wav2Vec2 CTC weights with a local tokenizer, keep `--model-id` pointed at
the pretrained weights and pass the local vocabulary files separately:

```bash
python run_xlsr_asr.py \
  --input-dir /data/audio \
  --glob '*.wav' \
  --output /data/xlsr_preds.jsonl \
  --model-id facebook/wav2vec2-xls-r-1b \
  --processor-id facebook/wav2vec2-xls-r-1b \
  --vocab-json /models/local-xlsr-tokenizer/vocab.json \
  --tokenizer-json /models/local-xlsr-tokenizer/tokenizer.json \
  --device cuda:0 \
  --dtype float16
```

`--tokenizer-json` is optional, but when it is supplied `--vocab-json` is required. The feature
extractor is still loaded from `--processor-id`; the local tokenizer supplies `vocab_size` and
`pad_token_id` for the CTC head while the open pretrained weights are loaded from `--model-id`.

### GigaAM

Use `v3_e2e_rnnt` by default on a 40 GB GPU. It is the largest current GigaAM end-to-end ASR
variant and keeps punctuation/text normalization, while still being small enough for fp16 inference on
that card. GigaAM's regular `transcribe()` API is intended for clips up to 25 seconds, which matches
the utterance-level `cut_samples/...` layout used by the corpus evaluator.

```bash
python run_gigaam_asr.py \
  --input-dir /data/audio \
  --glob '*.wav' \
  --output /data/gigaam_preds.jsonl \
  --model-id v3_e2e_rnnt \
  --device cuda:0 \
  --dtype float16
```

`--model-id` may be omitted for GigaAM; it defaults to `v3_e2e_rnnt`. Add `--download-root` to control
the GigaAM checkpoint cache, or `--use-flash` only when flash attention is installed and tested in the
runtime environment.

## Dependency sketch

```bash
pip install torch torchaudio transformers accelerate sentencepiece tqdm
```

For GigaAM:

```bash
git clone https://github.com/salute-developers/GigaAM.git
python -m pip install -e GigaAM[torch]
```

Some Whisper checkpoints may also benefit from:

```bash
pip install safetensors
```
