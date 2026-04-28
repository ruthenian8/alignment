# alignment

A Python package for producing aligned speech corpora from long-form audio,
coarse index files, manual transcripts, and WhisperX subtitles.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9 and `ffmpeg`/`ffprobe` on `PATH`.

## Pipeline overview

```
source audio + coarse index + manual transcript
  -> index TSV           (alignment parse-index)
  -> transcript TSV      (alignment parse-transcript)
  -> joined TSV          (alignment join)
  -> reordered TSV       (alignment reorder)
  -> WhisperX SRT        (external: WhisperX)
  -> aligned TSV         (alignment align-srt)
  -> corpus clips + manifest  (alignment export-corpus)
```

## CLI usage

```bash
# Parse a plaintext index file
alignment parse-index data/index_plaintext/pez_001.txt --output build/index.tsv

# Parse a plaintext transcript file
alignment parse-transcript data/transcript_plaintext/pez_001.txt --output build/transcript.tsv

# Join index and transcript
alignment join build/index.tsv build/transcript.tsv build/joined.tsv

# Reorder transcript column to align with index descriptions
alignment reorder build/joined.tsv --output build/joined_reordered.tsv

# Align a WhisperX SRT to a transcript (after running WhisperX separately)
alignment align-srt chunk.srt transcript.txt build/aligned.tsv

# Export corpus clips
alignment export-corpus audio.wav build/clips/ --aligned-tsv build/aligned.tsv

# Or run the first three stages at once
alignment run-all data/index_plaintext/pez_001.txt data/transcript_plaintext/pez_001.txt \
    --work-dir build/ --base-name pez_001
```

## Package structure

```
alignment/          # Main package
  __init__.py
  cli.py            # CLI entry point (subcommands)
  srt.py            # SRT parsing and formatting
  index_parser.py   # Plaintext and DOCX index parsing
  transcript_parser.py  # Plaintext transcript parsing
  io.py             # TSV I/O and schema constants
  join.py           # Join index + transcript DataFrames
  reorder.py        # Transcript reordering (DP + TF-IDF)
  align.py          # DP alignment of SRT to transcript
  audio.py          # ffmpeg audio cutting utilities
  export.py         # Corpus clip export and manifest
asr/                # Optional ASR utilities (WhisperX helpers)
data/               # Sample fixtures (pez_001)
legacy/             # Archived prototype scripts
notebooks/          # Exploratory notebooks
tests/              # pytest suite
```

## Intermediate schemas

### Index TSV columns
| Column | Type | Description |
|--------|------|-------------|
| start | str | Start timestamp (HH:MM:SS.mmm) |
| trans | bool | Whether segment has been transcribed |
| cont | str | Continuation timestamp or empty |
| prev | str | Index of predecessor segment or empty |
| text | str | Raw description text |
| name | str | Output audio filename |

### Transcript TSV columns
| Column | Type | Description |
|--------|------|-------------|
| id | int | 1-based record index |
| transcript | str | Full transcript text |
| max_speakers | int | Total speaker count |
| min_speakers | int | Interviewee count |

### Aligned TSV columns
| Column | Type | Description |
|--------|------|-------------|
| index | int | SRT segment index |
| start | str | Start timestamp |
| end | str | End timestamp |
| speaker | str | Speaker tag or empty |
| original_text | str | WhisperX ASR text |
| transcript_text | str | Aligned manual transcript span |
| matched | bool | Whether alignment succeeded |

## Development

```bash
pip install -e .
pytest
ruff check alignment/ tests/
ruff format alignment/ tests/
```
