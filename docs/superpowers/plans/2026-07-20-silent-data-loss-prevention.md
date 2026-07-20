# Silent Data Loss Prevention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make corpus export reject index/path collisions before writing and make manifest verification prove per-chunk completeness instead of relying on aggregate row counts.

**Architecture:** Refactor `alignment.export` into pure planning, whole-operation validation, and side-effect execution while keeping its public export functions stable. Extend the standalone verifier with uniqueness checks and explicit speaker-map index-set reconciliation, using `--matched-only` to mirror export semantics.

**Tech Stack:** Python 3.11+, standard-library `dataclasses`, `csv`, `pathlib`, `subprocess`, pytest, Ruff.

## Global Constraints

- Keep raw inputs immutable; write only derived export files.
- Use UTF-8 for every caption, manifest, and speaker-map read or write.
- Do not change alignment scoring, matching decisions, or transcript normalization.
- Preflight every selected chunk before the first ffmpeg invocation, caption write, speaker-map copy, or manifest write.
- Permit deterministic reruns only when existing normalized and original captions exactly match the planned text.
- Do not delete stale files outside the newly planned manifest.
- Audio is regenerated after successful preflight; execution is not transactional against ffmpeg or disk failures.

---

### Task 1: Enforce paired-SRT index integrity

**Files:**
- Modify: `alignment/export.py:145-225`
- Test: `tests/test_align_export.py`

**Interfaces:**
- Consumes: `parse_srt(text: str) -> list[SrtSegment]`.
- Produces: `_segments_by_unique_index(segments: list[SrtSegment], label: str) -> dict[int, SrtSegment]`; `export_segments(...)` retains its existing public signature.

- [ ] **Step 1: Write failing tests for duplicate and mismatched indices**

Append tests that prove paired SRT conversion cannot collapse rows or omit cleaned text:

```python
@pytest.mark.parametrize("side", ["original", "clean"])
def test_export_segments_rejects_duplicate_srt_indices(tmp_path: Path, side: str):
    duplicate = (
        "1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: first\n\n"
        "1\n00:00:01,000 --> 00:00:02,000\n[SPEAKER_00]: second\n"
    )
    single = "1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: only\n"
    original, clean = (duplicate, single) if side == "original" else (single, duplicate)

    with patch("alignment.export.subprocess.run") as run:
        with pytest.raises(ValueError, match=rf"duplicate {side} SRT indices: 1"):
            export_segments("input.wav", original, clean, tmp_path)

    run.assert_not_called()
    assert not tmp_path.exists()


def test_export_segments_requires_identical_srt_index_sets(tmp_path: Path):
    original = (
        "1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: first\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n[SPEAKER_00]: second\n"
    )
    clean = (
        "1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: clean first\n\n"
        "3\n00:00:01,000 --> 00:00:02,000\n[SPEAKER_00]: clean extra\n"
    )

    with pytest.raises(
        ValueError,
        match=r"paired SRT index mismatch; missing clean indices: 2; unexpected clean indices: 3",
    ):
        export_segments("input.wav", original, clean, tmp_path, run=False)

    assert not tmp_path.exists()
```

- [ ] **Step 2: Run the new tests and confirm RED**

Run: `pytest tests/test_align_export.py -k 'duplicate_srt_indices or identical_srt_index_sets' -v`

Expected: both behaviors fail because duplicate indices are collapsed and unequal sets are accepted.

- [ ] **Step 3: Add unique-index validation before output creation**

Add this helper near `_sample` and use it at the start of `export_segments`:

```python
def _segments_by_unique_index(
    segments: list[SrtSegment], label: str
) -> dict[int, SrtSegment]:
    """Index segments while rejecting duplicate SRT indices."""
    indexed: dict[int, SrtSegment] = {}
    duplicates: set[int] = set()
    for segment in segments:
        if segment.index in indexed:
            duplicates.add(segment.index)
        indexed[segment.index] = segment
    if duplicates:
        values = [str(index) for index in sorted(duplicates)]
        raise ValueError(f"duplicate {label} SRT indices: {_sample(values)}")
    return indexed
```

Replace the current dictionary comprehension in `export_segments` with:

```python
original_segments = parse_srt(original_srt)
clean_segments = parse_srt(clean_srt)
original_by_index = _segments_by_unique_index(original_segments, "original")
clean_by_index = _segments_by_unique_index(clean_segments, "clean")
missing = sorted(original_by_index.keys() - clean_by_index.keys())
unexpected = sorted(clean_by_index.keys() - original_by_index.keys())
if missing or unexpected:
    parts = ["paired SRT index mismatch"]
    if missing:
        parts.append(f"missing clean indices: {_sample([str(index) for index in missing])}")
    if unexpected:
        parts.append(
            f"unexpected clean indices: {_sample([str(index) for index in unexpected])}"
        )
    raise ValueError("; ".join(parts))
clean_text_by_index = {index: segment.text for index, segment in clean_by_index.items()}
```

- [ ] **Step 4: Run focused and existing export tests**

Run: `pytest tests/test_align_export.py -k 'export_segments or export_builds' -v`

Expected: PASS, including the new validation tests and deterministic export test.

- [ ] **Step 5: Commit Task 1**

```bash
git add alignment/export.py tests/test_align_export.py
git commit -m "Reject inconsistent paired SRT indices"
```

---

### Task 2: Preflight clip plans and existing captions

**Files:**
- Modify: `alignment/export.py:1-340`
- Test: `tests/test_align_export.py`

**Interfaces:**
- Consumes: `_segments_by_unique_index(...)`; `build_cut_command(...)`.
- Produces: frozen internal `ExportPlan`; `_plan_srt_segments(...) -> list[ExportPlan]`; `_validate_export_plans(plans: list[ExportPlan]) -> None`; `_execute_export_plans(plans: list[ExportPlan], run: bool) -> list[dict[str, str]]`.

- [ ] **Step 1: Write failing tests for collisions, conflicting reruns, and safe reruns**

Add `from dataclasses import replace` to the test imports, then append:

```python
def test_export_rejects_clip_collision_before_any_write(tmp_path: Path):
    original = (
        "1\n00:00:00,000 --> 00:00:01,000\n[A/B]: first\n\n"
        "1\n00:00:00,000 --> 00:00:01,000\n[AB]: second\n"
    )

    with patch("alignment.export.subprocess.run") as run:
        with pytest.raises(ValueError, match="duplicate original SRT indices"):
            export_segments("input.wav", original, original, tmp_path)

    run.assert_not_called()
    assert not tmp_path.exists()


def test_export_rejects_conflicting_existing_caption_before_ffmpeg(tmp_path: Path):
    base = "001_SPEAKER_00_00-00-00-000"
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / f"{base}.txt").write_text("older text", encoding="utf-8")
    original = "1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: original\n"
    clean = "1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: replacement\n"

    with patch("alignment.export.subprocess.run") as run:
        with pytest.raises(ValueError, match="existing caption conflicts"):
            export_segments("input.wav", original, clean, tmp_path)

    run.assert_not_called()
    assert (tmp_path / f"{base}.txt").read_text(encoding="utf-8") == "older text"


def test_export_allows_identical_existing_captions_on_rerun(tmp_path: Path):
    original = "1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: original\n"
    clean = "1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: clean\n"
    export_segments("input.wav", original, clean, tmp_path, run=False)

    with patch("alignment.export.subprocess.run") as run:
        rows = export_segments("input.wav", original, clean, tmp_path)

    assert len(rows) == 1
    run.assert_called_once()
```

Also add a direct plan-validation test for the sanitizer collision that distinct SRT indices normally prevent from sharing a clip ID:

```python
def test_plan_validation_rejects_duplicate_output_paths(tmp_path: Path):
    from alignment.export import ExportPlan, _validate_export_plans

    segment = SrtSegment(1, "00:00:00,000", "00:00:01,000", "[АБ]:", "one")
    plan = ExportPlan(
        segment=segment,
        clip_id="001_АБ_00-00-00-000",
        audio_path=tmp_path / "same.wav",
        text_path=tmp_path / "same.txt",
        original_text_path=tmp_path / "same_orig.txt",
        text="one",
        command=["ffmpeg"],
    )

    with pytest.raises(ValueError, match="duplicate clip IDs"):
        _validate_export_plans([plan, replace(plan, segment=replace(segment, index=2))])
```

- [ ] **Step 2: Run collision/rerun tests and confirm RED**

Run: `pytest tests/test_align_export.py -k 'collision or conflicting_existing or identical_existing or duplicate_output_paths' -v`

Expected: failures because `ExportPlan` and preflight validation do not exist and captions are overwritten today.

- [ ] **Step 3: Introduce pure plans and validation**

Import `dataclass` and define:

```python
@dataclass(frozen=True)
class ExportPlan:
    """One validated clip export operation."""

    segment: SrtSegment
    clip_id: str
    audio_path: Path
    text_path: Path
    original_text_path: Path
    text: str
    command: list[str]

    def manifest_row(self) -> dict[str, str]:
        """Return the manifest row represented by this plan."""
        return {
            "clip_id": self.clip_id,
            "audio_path": str(self.audio_path),
            "text_path": str(self.text_path),
            "text_original_path": str(self.original_text_path),
            "start": normalize_timestamp(self.segment.start, decimal="."),
            "end": normalize_timestamp(self.segment.end, decimal="."),
            "speaker": self.segment.speaker,
            "text": self.text,
            "text_original": self.segment.text,
        }
```

Implement `_plan_srt_segments` by moving path, text, and command construction out of `_export_srt_segments` without creating directories. Implement validation with a `Counter` for `clip_id`, `audio_path`, `text_path`, and `original_text_path`; report each non-unique category. Validate each existing caption path by exact UTF-8 content comparison. Implement execution as the only function that creates parents, invokes `subprocess.run`, and writes captions. Keep `_export_srt_segments` as a wrapper that plans, validates, and executes so callers remain compatible.

Core validation shape:

```python
def _validate_export_plans(plans: list[ExportPlan]) -> None:
    """Reject collisions and conflicting deterministic reruns before writes."""
    fields = {
        "clip IDs": [plan.clip_id for plan in plans],
        "audio paths": [str(plan.audio_path) for plan in plans],
        "text paths": [str(plan.text_path) for plan in plans],
        "original-text paths": [str(plan.original_text_path) for plan in plans],
    }
    failures = []
    for label, values in fields.items():
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            failures.append(f"duplicate {label}: {_sample(duplicates)}")
    for plan in plans:
        for path, expected in (
            (plan.text_path, plan.text),
            (plan.original_text_path, plan.segment.text),
        ):
            if path.exists() and path.read_text(encoding="utf-8") != expected:
                failures.append(f"existing caption conflicts with planned text: {path}")
    if failures:
        raise ValueError("; ".join(failures))
```

- [ ] **Step 4: Run the focused tests and full export test file**

Run: `pytest tests/test_align_export.py -k 'collision or conflicting_existing or identical_existing or duplicate_output_paths' -v`

Expected: PASS.

Run: `pytest tests/test_align_export.py -v`

Expected: PASS with no changed public export behavior.

- [ ] **Step 5: Commit Task 2**

```bash
git add alignment/export.py tests/test_align_export.py
git commit -m "Preflight corpus clip exports"
```

---

### Task 3: Preflight the entire aligned tree

**Files:**
- Modify: `alignment/export.py:225-340`
- Test: `tests/test_align_export.py`

**Interfaces:**
- Consumes: `_plan_srt_segments(...)`, `_validate_export_plans(...)`, `_execute_export_plans(...)` from Task 2.
- Produces: `export_aligned_srt_tree(...)` with its existing signature and whole-tree validation semantics.

- [ ] **Step 1: Write a failing no-partial-output tree test**

```python
def test_export_tree_preflights_later_chunks_before_writing(tmp_path: Path):
    aligned_root = tmp_path / "aligned"
    audio_root = tmp_path / "audio"
    output_root = tmp_path / "output"
    for chunk, text in (("and_001No1", "first"), ("and_001No2", "second")):
        aligned_dir = aligned_root / "and_001" / "aligned"
        audio_dir = audio_root / "and_001"
        aligned_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / f"{chunk}.wav").write_bytes(b"wav")
        (aligned_dir / f"{chunk}.aligned.srt").write_text(
            f"1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: {text}\n",
            encoding="utf-8",
        )
    conflict = output_root / "and_001" / "and_001No2" / "001_SPEAKER_00_00-00-00-000.txt"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("conflicting old caption", encoding="utf-8")

    with patch("alignment.export.subprocess.run") as run:
        with pytest.raises(ValueError, match="existing caption conflicts"):
            export_aligned_srt_tree(aligned_root, audio_root, output_root)

    run.assert_not_called()
    first = output_root / "and_001" / "and_001No1" / "001_SPEAKER_00_00-00-00-000.txt"
    assert not first.exists()
```

- [ ] **Step 2: Run the new tree test and confirm RED**

Run: `pytest tests/test_align_export.py::test_export_tree_preflights_later_chunks_before_writing -v`

Expected: FAIL because the first chunk caption is written before the second chunk conflict is discovered.

- [ ] **Step 3: Separate tree discovery/planning from execution**

In `export_aligned_srt_tree`, keep the current corpus, exclusion, ratio, audio, diarization, and speaker-map checks, but replace the per-chunk call to `export_aligned_srt` with plan accumulation. Store `(speaker_map, target_dir)` copy operations separately. After every selected chunk has been parsed and planned:

```python
_validate_export_plans(plans)
rows = _execute_export_plans(plans, run=run)
for speaker_map, target_dir in speaker_map_copies:
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(speaker_map, target_dir / "speaker_map.csv")
if manifest_path is not None:
    write_tsv(manifest_path, rows, MANIFEST_COLUMNS)
return rows
```

Do not call the public `export_aligned_srt` during tree planning because it executes immediately. Apply speaker maps and `matched_only` filtering before `_plan_srt_segments`, preserving their present order and behavior.

- [ ] **Step 4: Verify whole-tree preflight and regressions**

Run: `pytest tests/test_align_export.py::test_export_tree_preflights_later_chunks_before_writing -v`

Expected: PASS.

Run: `pytest tests/test_align_export.py -v`

Expected: PASS, including guarded, filtered, and matched-only tree exports.

- [ ] **Step 5: Commit Task 3**

```bash
git add alignment/export.py tests/test_align_export.py
git commit -m "Preflight complete aligned export trees"
```

---

### Task 4: Detect duplicate manifest identities and paths

**Files:**
- Modify: `tools/verify_export_manifest.py:45-100`
- Test: `tests/test_verify_export_manifest.py`

**Interfaces:**
- Consumes: parsed manifest dictionaries.
- Produces: `duplicate_value_failures(rows: list[dict[str, str]]) -> list[str]`, called unconditionally by `manifest_failures`.

- [ ] **Step 1: Write a parameterized failing uniqueness test**

```python
import pytest


@pytest.mark.parametrize(
    ("field", "label"),
    [
        ("clip_id", "clip IDs"),
        ("audio_path", "audio paths"),
        ("text_path", "text paths"),
        ("text_original_path", "original-text paths"),
    ],
)
def test_verify_manifest_reports_duplicate_identity_fields(
    tmp_path: Path, field: str, label: str
) -> None:
    values = {
        "clip_id": "001_АБ_00-00-00-000",
        "audio_path": "/out/chunk/001.wav",
        "text_path": "/out/chunk/001.txt",
        "text_original_path": "/out/chunk/001_orig.txt",
        "speaker": "[АБ]:",
    }
    second = {key: f"{value}.other" for key, value in values.items()}
    second["speaker"] = "[АБ]:"
    second[field] = values[field]
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "\t".join(values) + "\n"
        + "\t".join(values.values()) + "\n"
        + "\t".join(second.values()) + "\n",
        encoding="utf-8",
    )

    _, failures = verify_manifest(manifest)

    assert failures == [f"1 duplicate manifest {label}: {values[field]}"]
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `pytest tests/test_verify_export_manifest.py::test_verify_manifest_reports_duplicate_identity_fields -v`

Expected: four failures because duplicate values are currently accepted.

- [ ] **Step 3: Implement non-empty duplicate detection**

Import `Counter` and add:

```python
def duplicate_value_failures(rows: list[dict[str, str]]) -> list[str]:
    """Return failures for duplicate non-empty manifest identities and paths."""
    fields = {
        "clip_id": "clip IDs",
        "audio_path": "audio paths",
        "text_path": "text paths",
        "text_original_path": "original-text paths",
    }
    failures = []
    for field, label in fields.items():
        counts = Counter(row.get(field, "").strip() for row in rows)
        duplicates = sorted(value for value, count in counts.items() if value and count > 1)
        if duplicates:
            failures.append(
                f"{len(duplicates)} duplicate manifest {label}: {sample(duplicates)}"
            )
    return failures
```

Call `failures.extend(duplicate_value_failures(manifest_rows))` immediately after the aggregate count check in `manifest_failures`.

- [ ] **Step 4: Run verifier tests**

Run: `pytest tests/test_verify_export_manifest.py -v`

Expected: PASS; older sparse manifests remain valid because blank values are ignored by uniqueness checks.

- [ ] **Step 5: Commit Task 4**

```bash
git add tools/verify_export_manifest.py tests/test_verify_export_manifest.py
git commit -m "Detect duplicate exported manifest paths"
```

---

### Task 5: Reconcile manifest and speaker-map indices per chunk

**Files:**
- Modify: `tools/verify_export_manifest.py:135-285`
- Modify: `README.md:32-34,119-127`
- Test: `tests/test_verify_export_manifest.py`

**Interfaces:**
- Consumes: `parse_bool(value: str) -> bool`; manifest `clip_id` and `audio_path`; colocated exported `speaker_map.csv`.
- Produces: `speaker_map_failures(manifest_rows: list[dict[str, str]], *, matched_only: bool = False) -> list[str]`; `verify_manifest(..., matched_only: bool = False)`; CLI `--matched-only`.

- [ ] **Step 1: Write failing balanced-count and matched-only tests**

Add these two-row speaker-map fixtures:

```python
def test_verify_manifest_reconciles_indices_when_total_count_is_balanced(tmp_path: Path) -> None:
    chunk = tmp_path / "and_001" / "and_001No1"
    chunk.mkdir(parents=True)
    (chunk / "speaker_map.csv").write_text(
        "srt_index,start,end,whisperx_speaker,transcript_speaker,speaker_source,matched,score\n"
        "1,00:00:00,00:00:01,[SPEAKER_00]:,[АБ]:,marker,True,1.000\n"
        "2,00:00:01,00:00:02,[SPEAKER_00]:,[АБ]:,marker,True,1.000\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "clip_id\taudio_path\tspeaker\n"
        f"001_АБ_00-00-00-000\t{chunk / '001.wav'}\t[АБ]:\n"
        f"003_АБ_00-00-02-000\t{chunk / '003.wav'}\t[АБ]:\n",
        encoding="utf-8",
    )

    _, failures = verify_manifest(manifest, check_speaker_maps=True)

    assert "1 expected speaker-map indices are missing from manifest: and_001No1:2" in failures
    assert "1 manifest indices are absent from speaker maps: and_001No1:3" in failures


def test_verify_manifest_matched_only_reconciles_only_matched_rows(tmp_path: Path) -> None:
    chunk = tmp_path / "and_001" / "and_001No1"
    chunk.mkdir(parents=True)
    (chunk / "speaker_map.csv").write_text(
        "srt_index,start,end,whisperx_speaker,transcript_speaker,speaker_source,matched,score\n"
        "1,00:00:00,00:00:01,[SPEAKER_00]:,[АБ]:,marker,True,1.000\n"
        "2,00:00:01,00:00:02,[SPEAKER_00]:,,unmatched,False,0.000\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "clip_id\taudio_path\tspeaker\n"
        f"001_АБ_00-00-00-000\t{chunk / '001.wav'}\t[АБ]:\n",
        encoding="utf-8",
    )

    _, failures = verify_manifest(
        manifest, check_speaker_maps=True, matched_only=True
    )

    assert failures == []
```

Add one test with duplicate `srt_index` rows in `speaker_map.csv` and one with two distinct manifest clip IDs beginning with the same numeric index; assert each emits an explicit duplicate-index failure.

- [ ] **Step 2: Run reconciliation tests and confirm RED**

Run: `pytest tests/test_verify_export_manifest.py -k 'reconciles or duplicate_index' -v`

Expected: failures because current dictionary construction collapses speaker-map duplicates and checks only manifest rows individually.

- [ ] **Step 3: Preserve speaker-map row lists and compare per-directory index sets**

Replace the map cache value with `list[dict[str, str]] | None`. For each chunk directory:

1. parse every non-empty speaker-map index without collapsing duplicates;
2. choose all rows or only `matched=True` rows according to `matched_only`;
3. parse manifest indices from the numeric `clip_id` prefix;
4. report duplicate eligible speaker-map indices and duplicate manifest indices;
5. compare the two sets and report missing and unexpected `chunk:index` labels;
6. retain the existing matched-state and transcript-speaker provenance checks for each manifest row.

Thread `matched_only` through `verify_manifest`, `main`, and this new parser option:

```python
parser.add_argument(
    "--matched-only",
    action="store_true",
    help="Expect only speaker-map rows marked matched, mirroring export-aligned-map.",
)
```

The complete-exchange error strings used by tests are:

```python
f"{len(missing)} expected speaker-map indices are missing from manifest: {sample(missing)}"
f"{len(unexpected)} manifest indices are absent from speaker maps: {sample(unexpected)}"
```

- [ ] **Step 4: Update command documentation**

Add `--matched-only` to the verifier example on README line 34. Extend the verifier paragraph to state that it always rejects duplicate IDs/paths, that speaker-map checking reconciles indices per chunk, and that verifier `--matched-only` must mirror the export flag.

- [ ] **Step 5: Run focused verifier and CLI tests**

Run: `pytest tests/test_verify_export_manifest.py tests/test_cli_smoke.py -v`

Expected: PASS, including complete and matched-only reconciliation.

- [ ] **Step 6: Commit Task 5**

```bash
git add tools/verify_export_manifest.py tests/test_verify_export_manifest.py README.md
git commit -m "Reconcile exported clips with speaker maps"
```

---

### Task 6: Run completion gates and audit the safety contract

**Files:**
- Modify only if a gate exposes a defect in files already listed above.

**Interfaces:**
- Consumes: all export and verifier behavior from Tasks 1-5.
- Produces: evidence that the approved design and repository quality gates are satisfied.

- [ ] **Step 1: Run focused safety tests**

Run: `pytest tests/test_align_export.py tests/test_verify_export_manifest.py -v`

Expected: PASS; confirm the output includes tests for duplicate SRT indices, paired index mismatch, plan collisions, conflicting/identical reruns, whole-tree preflight, duplicate manifest fields, balanced missing/extra rows, duplicate per-chunk indices, and matched-only reconciliation.

- [ ] **Step 2: Run the full test suite**

Run: `pytest`

Expected: PASS with no network, WhisperX, or GPU requirement. Environment-dependent tests may be skipped only by their existing declared markers.

- [ ] **Step 3: Run Ruff gates**

Run: `ruff check .`

Expected: `All checks passed!`

Run: `ruff format --check .`

Expected: all files already formatted.

- [ ] **Step 4: Audit each design invariant against evidence**

Inspect `git diff --check`, the focused test names/output, and the final implementations. Confirm explicitly that validation happens before `mkdir`, `subprocess.run`, `Path.write_text`, `shutil.copyfile`, and `write_tsv`; paired SRT sets are identical; existing conflicting captions abort; manifest identity fields are unique; and speaker-map completeness is checked per chunk for both full and matched-only modes.

- [ ] **Step 5: Commit any gate-only corrections**

If the gates required changes within the approved scope:

```bash
git add alignment/export.py tools/verify_export_manifest.py tests/test_align_export.py tests/test_verify_export_manifest.py README.md
git commit -m "Polish silent data loss safeguards"
```

If no corrections were required, record the successful commands in the handoff without creating an empty commit.
