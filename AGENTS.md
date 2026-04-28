# AGENTS.md

This file gives coding-agent instructions for maintaining and refactoring this repository.

## Mission

Turn this repository into one coherent, tested, documented codebase for producing an aligned speech corpus from:

- long-form source audio;
- coarse plaintext or DOCX index files with approximate timestamps;
- manual plaintext transcripts;
- intermediary WhisperX subtitle files (`.srt`).

The supported pipeline should emit short audio clips plus aligned text and metadata, and it should be runnable both as importable Python code and from a CLI.

## What is already true in this repository

Treat these as grounded findings from the current repo state:

- Root-level numbered scripts are the current pipeline prototypes.
- `0_process_audio.py` is an ingestion helper with hardcoded local paths.
- `1_index_to_tsv_cut_audio_v1.py` and `1_index_to_tsv_cut_audio_v2.py` are two generations of the same stage.
- `3_transcripts_to_tsv.py` captures a real transcript input format and should be preserved in cleaned form.
- `4_add_transcripts_to_index_tsv.py` is conceptually necessary but currently mixes CSV and TSV assumptions.
- `5_reorder_index_tsv.py` contains useful reorder logic and should survive the refactor.
- `6_align_srt_w_index_tsv_v1.py`, `v2.py`, `v3.py`, and `embs.py` are competing alignment experiments.
- `6_align_srt_w_index_tsv_v3.py` is the best starting point for the canonical aligner.
- `6_align_srt_w_index_tsv_embs.py` is notebook-style experimental code and should not remain a production entry point.
- `7_cut_audio_by_srt.py` contains useful export logic but is tightly coupled to Hugging Face paths and tokens.
- `asr/` is a cleaner utility area and should remain optional and clearly separated from the core alignment pipeline.
- `data/` already contains small sample fixtures, including `pez_001`, which should be preserved for regression tests and examples.
- The README is currently very sparse, so agents should not assume the documented workflow is complete or current.

## Core principles

1. **Bias toward simplification.** Keep one supported path per stage.
2. **Keep the repository human-readable.** Prefer straightforward modules and functions over elaborate abstractions.
3. **Archive before deleting.** If an old script still contains useful behavior or algorithm history, move it to `legacy/` or `notebooks/` after its value is preserved.
4. **Do not preserve duplicate entry points.** Multiple versions may exist during migration, but the supported path must end up singular and obvious.
5. **Keep raw inputs immutable.** Write derived files under `build/`, `outputs/`, or another explicit output directory.
6. **Use UTF-8 everywhere.** Russian text, stress marks, bracketed markup, and SRT timestamps must round-trip safely.
7. **Preserve original text separately from normalized text.** Manual transcripts are the source of truth for final text output; normalized text exists only to support matching and alignment.
8. **Remove environment-specific assumptions.** No hardcoded user paths, Colab-only flows, Hugging Face tokens, or machine-specific cache locations.
9. **Prefer explicit CLI arguments over hidden constants.** Add config files only when they clearly reduce repetition in a multi-step workflow.
10. **Clarity beats cleverness.** Avoid over-optimized or opaque logic, especially in alignment code.

## Canonical supported pipeline

The repository should converge on this single workflow:

```text
source audio + coarse index + manual transcript
  -> normalized audio metadata
  -> index TSV
  -> transcript TSV
  -> joined working TSV
  -> reordered working TSV
  -> WhisperX SRT per chunk
  -> aligned SRT/manual transcript pairs
  -> final corpus clips + text + manifest
```

Use TSV for human-facing intermediate tables unless there is a strong reason to change the whole pipeline consistently.

## Recommended package shape

Keep the implementation compact. A simple package like this is enough:

```text
alignment/
  __init__.py
  cli.py
  audio.py
  io.py
  index_parser.py
  transcript_parser.py
  reorder.py
  srt.py
  align.py
  export.py
asr/
  ...
```

Notes:

- Only add `models.py`, `config.py`, or other framework-like modules if they clearly earn their keep.
- Small dataclasses are welcome, but do not introduce boilerplate-heavy schema layers without a concrete need.
- Keep pure parsing and alignment logic inside the package; keep filesystem and subprocess work at the edges.

## Canonical sources to promote

When consolidating the current scripts, use this direction:

- **Index parsing / audio cutting:** start from `1_index_to_tsv_cut_audio_v2.py`; borrow any missing useful behavior from `v1.py` only if tests justify it.
- **Transcript parsing:** preserve the real input assumptions from `3_transcripts_to_tsv.py`, but refactor into a clean parser module.
- **Join step:** keep the purpose of `4_add_transcripts_to_index_tsv.py`, but fix the CSV/TSV mismatch and make the schema explicit.
- **Reordering:** migrate the useful logic from `5_reorder_index_tsv.py` into a supported module.
- **SRT parsing/writing:** consolidate into one canonical `srt.py`; borrow robust bits from `6_align_srt_w_index_tsv_v2.py` if they improve round-trip safety.
- **Alignment:** use `6_align_srt_w_index_tsv_v3.py` as the canonical starting point.
- **Export:** keep the useful cutting/text-export behavior from `7_cut_audio_by_srt.py`, but remove all Hugging Face coupling from the core path.
- **ASR tools:** preserve `asr/` as optional utilities, not as a hard dependency of parser/alignment tests.

After migration, archive or remove root-level legacy variants from the supported path.

## Immediate cleanup rules

### Archive legacy and experimental material

Move legacy or exploratory files into a clearly named area such as `legacy/` or `notebooks/`.

At minimum, treat these as archive candidates once their useful behavior is preserved:

- `1_index_to_tsv_cut_audio_v1.py`
- `6_align_srt_w_index_tsv_v1.py`
- `6_align_srt_w_index_tsv_embs.py`
- tutorial notebooks that are not part of the supported execution path

### Keep one active implementation per responsibility

There should be exactly one supported module for each of these jobs:

- audio ingestion and normalization;
- coarse index parsing;
- transcript parsing;
- joining transcripts to index rows;
- transcript reordering;
- SRT parsing and formatting;
- transcript/SRT alignment;
- corpus clip export.

## Data contract rules

Standardize the intermediate schemas and document them explicitly.

At minimum, define and document the intended columns for:

- coarse index rows;
- transcript rows;
- joined index-plus-transcript rows;
- aligned subtitle rows;
- final corpus manifest rows.

Rules:

- Do not mix CSV and TSV assumptions between stages.
- Preserve the original transcript text in its own field.
- Put cleaned or normalized alignment text in separate fields with explicit names.
- Keep speaker labels when recoverable.
- Represent skipped or unmatched alignments explicitly rather than hiding them.

## Refactoring rules

- Replace repeated timestamp regexes and parser snippets with shared helpers.
- Replace repeated SRT parsing and writing code with one canonical implementation.
- Replace repeated text normalization code with clearly named functions such as `normalize_for_match()` and `clean_for_output()`.
- Replace `os.system(...)` with `subprocess.run([...], check=True)`.
- Replace hardcoded paths such as local download folders, `/pez`, `/result`, and user home directories with CLI arguments.
- Remove hardcoded tokens and repository IDs from production code.
- Use `pathlib.Path` consistently.
- Remove notebook-export debris such as `# In[...]`, shell magics, and inline experimentation from production modules.
- Separate algorithmic logic from I/O so the hard parts stay testable.

## Alignment-specific guidance

The canonical aligner should be based on the dynamic-programming approach in `6_align_srt_w_index_tsv_v3.py`.

Expected behavior:

- preserve original manual transcript text in outputs;
- use WhisperX timing as the temporal scaffold;
- enforce monotonic alignment;
- support skipped segments explicitly;
- keep speaker information when available;
- return structured Python objects before serializing to SRT or TSV.

Do not keep multiple aligners as equal-status production paths unless benchmarks and tests show a real need.
A fallback heuristic is acceptable as an internal helper, not as a second top-level workflow.

## Optimization guidance

Optimize for clarity first and runtime second.

Good optimizations:

- precompile regexes used in tight loops;
- cache normalized transcript tokens where it reduces repeated work;
- use `ffprobe` for duration instead of loading audio just to inspect metadata;
- batch ASR by duration, as the `asr/` utilities already do;
- mock ffmpeg-heavy subprocess calls in tests;
- add structured logging instead of scattered `print()` statements.

Avoid these failure modes:

- clever vectorization that makes the code hard to reason about;
- giant config layers before the pipeline is stable;
- forcing GPU or large-model dependencies into the core package;
- treating experimental embedding code as part of the default path.

## Documentation rules

Use one docstring style consistently. Prefer concise Google-style docstrings.

Required documentation standard:

- every production module starts with a short module docstring;
- every public function and class has a docstring;
- every CLI command has useful `--help` text;
- README examples match the real commands and file layout;
- schema columns are documented in plain language.

Do not add verbose comments that only restate the code.
Use comments to explain domain assumptions, data quirks, or non-obvious tradeoffs.

## Testing requirements

Create and maintain a real `pytest` suite.

Cover at least:

- timestamp parsing for both comma and dot millisecond formats;
- SRT parse/write round trips, including speaker labels and multiline text;
- plaintext and DOCX index parsing;
- `НЕ РАСПИСАНО` and continuation linking via `cont` and `prev`;
- transcript parsing with bracketed prompts, interviewer text, and stress marks;
- the current CSV/TSV mismatch as a regression test;
- reorder behavior for identity, one-row shift, inactive rows, and missing transcript rows;
- alignment monotonicity on small handcrafted examples;
- preservation of original transcript text in aligned output;
- deterministic export naming and ffmpeg command construction using mocks;
- at least one small end-to-end smoke test using tiny fixtures.

Testing strategy:

- use small fixture strings whenever possible;
- use `pez_001` or trimmed copies of it for regression coverage;
- do not make network calls from tests;
- do not require WhisperX or GPU dependencies for parser/alignment tests.

## CLI guidance

Replace numbered scripts with one CLI entry point and clear subcommands, for example:

```text
alignment parse-index
alignment parse-transcript
alignment join
alignment reorder
alignment align-srt
alignment export-corpus
alignment run-all
```

Each command should:

- accept explicit input and output paths;
- write stable, documented outputs;
- fail with clear, actionable error messages;
- avoid hidden directory conventions unless they are clearly documented.

## Quality gates

Before calling a refactor complete, run the checks that are actually configured:

```bash
pytest
ruff check .
ruff format --check .
```

Add `mypy` only when the public API has stabilized enough for it to provide signal without driving noisy boilerplate.
Type hints are encouraged for public functions, but do not contort the design just to satisfy static typing.

## Definition of done

A refactor is complete when:

- there is one documented path for each pipeline stage;
- redundant root-level variants are archived or removed from the supported path;
- no supported script depends on hardcoded local paths, tokens, or Colab-only commands;
- intermediate schemas are consistent and documented;
- deterministic parser and alignment tests pass on the included sample fixtures;
- the README matches the real code layout and commands;
- a new contributor can understand the supported workflow without reading legacy files first.

## Final instruction

Make the repository smaller, clearer, and easier to trust.

Preserve useful history in `legacy/` when needed, but keep the supported path singular, simple, and readable.
