# alignment

Build an aligned speech corpus from long-form audio, coarse timestamp indexes, manual transcripts, and WhisperX SRT files.

The supported code path is the `alignment/` package and the `alignment` CLI. Historical numbered scripts are preserved in `legacy/`; tutorial notebooks are in `notebooks/`. The optional `asr/` utilities are separate from the parser/alignment pipeline and are not required for tests.

## Workflow

```text
source audio + coarse index + manual transcript
  -> index TSV
  -> transcript TSV
  -> joined TSV
  -> reordered TSV
  -> WhisperX SRT per chunk
  -> aligned SRT/manual transcript pairs
  -> final corpus clips + text + manifest
```

Example commands:

```bash
alignment parse-index data/index_plaintext/pez_001.txt build/pez_001/index.tsv --audio-stem pez_001
alignment parse-transcript data/transcript_plaintext/pez_001.txt build/pez_001/transcript.tsv
alignment join build/pez_001/index.tsv build/pez_001/transcript.tsv build/pez_001/joined.tsv
alignment reorder build/pez_001/joined.tsv build/pez_001/reordered.tsv
alignment align-srt data/whisper_srt/pez_001/pez_001No0.srt transcript.txt outputs/pez_001No0.srt --output-tsv outputs/pez_001No0.tsv
alignment align-srt data/whisper_srt/pez_001/pez_001No0.srt transcript.txt outputs/pez_001No0.srt --use-transcript-speakers --infer-missing-speakers --output-speaker-map outputs/pez_001No0.speaker_map.csv
alignment align-embeddings data/whisper_srt/pez_001/pez_001No0.srt transcript.txt outputs/pez_001No0_embs.tsv
alignment wer outputs/pez_001No0.tsv --top 20
alignment align-map mapping.csv srt_dir build/aligned --use-transcript-speakers --infer-missing-speakers --require-diarized-matches --min-match-ratio 0.2
alignment export-corpus chunk.wav outputs/pez_001No0.srt outputs/pez_001No0.srt outputs/clips outputs/manifest.tsv
alignment export-aligned-map build/align-map-wx-transcripts-srt-speakers hf-repo/cut_audio build/cut_samples --manifest build/cut_samples/manifest.tsv --require-diarized-matches --matched-only --min-match-ratio 0.2
python tools/verify_export_manifest.py build/cut_samples/manifest.tsv --summary-root build/align-map-wx-transcripts-srt-speakers
```

Write derived files under `build/` or `outputs/`. Keep files in `data/` as source fixtures.

## Schemas

Human-facing intermediate files are UTF-8 TSV.

`index.tsv` columns:

- `start`: chunk start timestamp, normalized as `HH:MM:SS.mmm`.
- `trans`: `True` when the row should receive a manual transcript; `False` for `НЕ РАСПИСАНО`.
- `cont`: start timestamp of the row continued by this row, when recoverable.
- `prev`: row index of the continuation source, when recoverable.
- `text`: original coarse index text.
- `name`: deterministic chunk audio filename.

`transcript.tsv` columns:

- `id`: 1-based transcript block number.
- `transcript`: manual transcript text, including stress marks and bracketed prompts. When a block has a valid final-line speaker footer, `parse-transcript` prefixes it as a `[TAG:]` marker so `align-srt --use-transcript-speakers` can replace WhisperX speaker codes; the marker is removed from final aligned text.
- `max_speakers`: interviewer plus interviewee count inferred from block headers.
- `min_speakers`: interviewee count inferred from block footer.

`joined.tsv` adds `transcript`, `max_speakers`, and `min_speakers` to every index row. Inactive rows keep those fields empty.

`aligned.tsv` columns:

- `index_name`: chunk or index identifier.
- `srt_index`, `start`, `end`, `speaker`: timing scaffold from WhisperX SRT.
- `srt_text`: original SRT text.
- `transcript_text`: matched original manual transcript span.
- `normalized_text`: normalized alignment text used for matching/debugging.
- `matched`: explicit `True`/`False` alignment status.
- `score`: token-overlap alignment score.

`speaker_map.csv` columns:

- `srt_index`, `start`, `end`: aligned SRT segment location.
- `whisperx_speaker`: original WhisperX speaker code, preserved for debugging only.
- `transcript_speaker`: actual speaker inferred from the manual transcript, or empty when the row is not safely attributable.
- `speaker_source`: provenance such as `preceding_marker`, `speaker_note`, `block_footer`, `speaker_hint`, `collector_bracket`, or `carried_forward_prev`.
- `matched`, `score`: alignment status used to distinguish genuinely blank matched rows from skipped or misaligned utterances.

Speaker recovery uses manual transcript evidence only. Explicit speaker markers and collector brackets have priority; short common speaker notes such as `[ФМП спрашивает:]` can provide a weak local speaker anchor. WhisperX speaker-code consistency is not used as evidence.

`align-map` also writes `summary.tsv`. Its `matched_blank_speakers` column is the primary audit field for diarization coverage: it should be zero when every successfully matched caption has a transcript-derived speaker. `match_ratio` records the share of SRT segments that matched manual transcript text for each chunk, so very low-coverage chunks can be treated as likely misaligned instead of trusted.
Use `--require-diarized-matches` to make diarization coverage a CLI quality gate and to fail when any mapping row cannot be aligned to an SRT. Add `--min-match-ratio` when final outputs should also reject low-coverage chunk alignments.
By default, alignment removes bracketed editorial/reference notes from recovered manual spans while preserving bracketed utterances and speaker evidence. Use `--keep-alignment-notes` on `align-srt` or `align-map` when auditing classifier behavior and original note placement.

`manifest.tsv` columns:

- `clip_id`, `audio_path`, `text_path`, `text_original_path`: deterministic exported file references.
- `start`, `end`, `speaker`: clip metadata.
- `text`: clean output text.
- `text_original`: original aligned text.

`export-aligned-map` consumes `align-map` output laid out as `pez_001/aligned/pez_001No1.aligned.srt`
and matching chunk audio laid out as `pez_001/pez_001No1.wav`. It writes the same directory shape as
`cut_samples`: `pez_001/pez_001No1/001_SPEAKER_00_00-00-00-031.wav`, plus a normalized `.txt`
caption and the original stressed/manual caption as `_orig.txt`.
When a matching `speaker_maps/{chunk}.speaker_map.csv` exists, transcript-derived speakers from that
map are used in exported clip names and manifest rows.
Use `--require-diarized-matches` to fail export if any matched segment lacks a transcript-derived speaker
or if a required speaker map is missing. By default, export preserves all aligned SRT rows, including
unmatched rows that fall back to the original SRT text. Add `--matched-only` to export only rows that
were matched to manual transcript text. Add `--min-match-ratio` to make export re-check the
`align-map` summary and fail low-coverage chunks even if the alignment stage was run without a gate.
Add `--exclude-quality-failures build/aligned-with-speaker-maps/quality_failures.tsv` to skip chunks
already marked as missing, undiarized, or low-coverage by a speaker-map audit.
Use repeated `--corpus` options when an aligned root contains multiple corpus directories but the
audio root or export job should cover only a subset.

`tools/verify_export_manifest.py` verifies the final exported manifest against `summary.tsv` and
an optional `quality_failures.tsv`. It checks the manifest row count, rejects blank speakers,
rejects leftover `[SPEAKER_*]` labels, and confirms that chunks excluded by quality audit are not
present in final outputs. Use the same repeated `--corpus` filters as the export command when
verifying a subset export. Add `--check-files` after real exports to verify that referenced audio
and caption files exist and that caption files match the manifest text fields.

`align-embeddings` is an optional side path for the old embedding experiment. It removes bracketed interviewer prompts, segments dialect text around pauses, and aligns segment pairs with a lazily loaded `sentence-transformers` model. It is not the default aligner, and normal parser/alignment tests do not require external models.

`wer` computes global word error rate from an `aligned.tsv`, using only rows with `matched=True` and `score > 0`. It normalizes both SRT and manual text with the same text-normalization path used by alignment, drops square-bracketed notes and tags, and prints the most common substitutions, deletions, and insertions.

## Development

Install the package in editable mode if you want the `alignment` command on your PATH:

```bash
python -m pip install -e .
```

Quality gates:

```bash
pytest
ruff check .
ruff format --check .
```

The test suite uses small fixtures and does not require WhisperX, GPU dependencies, or network access.
