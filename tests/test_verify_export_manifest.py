"""Tests for exported manifest verification helper."""

from pathlib import Path

import pytest

from tools.verify_export_manifest import verify_manifest


def write_summary(root: Path) -> None:
    summary_dir = root / "and_001"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.tsv").write_text(
        "name\tsegments\tmatched_segments\tmatch_ratio\tmatched_blank_speakers\tstatus\n"
        "and_001No1\t2\t2\t1.000\t0\taligned\n"
        "and_001No2\t2\t1\t0.100\t0\taligned\n",
        encoding="utf-8",
    )
    other_dir = root / "pom_001"
    other_dir.mkdir(parents=True)
    (other_dir / "summary.tsv").write_text(
        "name\tsegments\tmatched_segments\tmatch_ratio\tmatched_blank_speakers\tstatus\n"
        "pom_001No1\t10\t10\t1.000\t0\taligned\n",
        encoding="utf-8",
    )


def test_verify_manifest_accepts_diarized_manifest_without_failed_chunks(tmp_path: Path) -> None:
    summary_root = tmp_path / "aligned"
    write_summary(summary_root)
    failures = tmp_path / "quality_failures.tsv"
    failures.write_text(
        "job\tname\treason\tvalue\tthreshold\tstatus\n"
        "and_001\tand_001No2\tlow_match_ratio\t0.100\t0.200\taligned\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "audio_path\tspeaker\n"
        "/tmp/out/and_001/and_001No1/001_АБ.wav\t[АБ]:\n"
        "/tmp/out/and_001/and_001No1/002_АБ.wav\t[АБ]:\n",
        encoding="utf-8",
    )

    metrics, failures_found = verify_manifest(
        manifest,
        summary_root=summary_root,
        quality_failures=failures,
        corpora={"and_001"},
    )

    assert failures_found == []
    assert metrics["manifest_rows"] == 2
    assert metrics["expected_rows"] == 2
    assert metrics["excluded_chunks"] == 1


def test_verify_manifest_reports_counts_speakers_and_excluded_chunks(tmp_path: Path) -> None:
    summary_root = tmp_path / "aligned"
    write_summary(summary_root)
    failures = tmp_path / "quality_failures.tsv"
    failures.write_text(
        "job\tname\treason\tvalue\tthreshold\tstatus\n"
        "and_001\tand_001No2\tlow_match_ratio\t0.100\t0.200\taligned\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "audio_path\tspeaker\n"
        "/tmp/out/and_001/and_001No1/001_АБ.wav\t\n"
        "/tmp/out/and_001/and_001No2/001_SPEAKER_00.wav\t[SPEAKER_00]:\n",
        encoding="utf-8",
    )

    _, failures_found = verify_manifest(
        manifest,
        summary_root=summary_root,
        quality_failures=failures,
        corpora={"and_001"},
    )

    assert failures_found == [
        "1 manifest rows have blank speakers",
        "1 manifest rows keep WhisperX speaker codes",
        "1 manifest rows come from excluded quality-failure chunks",
    ]


def test_verify_manifest_can_check_referenced_files(tmp_path: Path) -> None:
    audio = tmp_path / "001_АБ.wav"
    text = tmp_path / "001_АБ.txt"
    original = tmp_path / "001_АБ_orig.txt"
    audio.write_bytes(b"RIFF")
    text.write_text("нормальный текст", encoding="utf-8")
    original.write_text("норма\\льный текст", encoding="utf-8")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "audio_path\ttext_path\ttext_original_path\tspeaker\ttext\ttext_original\n"
        f"{audio}\t{text}\t{original}\t[АБ]:\tнормальный текст\tнорма\\льный текст\n",
        encoding="utf-8",
    )

    _, failures_found = verify_manifest(manifest, check_files=True)

    assert failures_found == []


def test_verify_manifest_reports_missing_and_stale_referenced_files(tmp_path: Path) -> None:
    text = tmp_path / "001_АБ.txt"
    original = tmp_path / "001_АБ_orig.txt"
    text.write_text("старый текст", encoding="utf-8")
    original.write_text("старый оригинал", encoding="utf-8")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "audio_path\ttext_path\ttext_original_path\tspeaker\ttext\ttext_original\n"
        f"{tmp_path / 'missing.wav'}\t{text}\t{original}\t[АБ]:\tновый текст\tновый оригинал\n"
        "\t\t\t[АБ]:\tеще текст\tеще оригинал\n",
        encoding="utf-8",
    )

    _, failures_found = verify_manifest(manifest, check_files=True)

    assert failures_found == [
        "2 manifest audio files are missing",
        "1 manifest text files are missing",
        "1 manifest text files differ from manifest text",
        "1 manifest original-text files are missing",
        "1 manifest original-text files differ from manifest text_original",
    ]


def test_verify_manifest_reports_kept_summary_quality_failures(tmp_path: Path) -> None:
    summary_root = tmp_path / "aligned"
    summary_dir = summary_root / "and_001"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.tsv").write_text(
        "name\tsegments\tmatched_segments\tmatch_ratio\tmatched_blank_speakers\tstatus\n"
        "and_001No1\t2\t2\t0.100\t1\taligned\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "audio_path\tspeaker\n"
        "/tmp/out/and_001/and_001No1/001_АБ.wav\t[АБ]:\n"
        "/tmp/out/and_001/and_001No1/002_АБ.wav\t[АБ]:\n",
        encoding="utf-8",
    )

    _, failures_found = verify_manifest(
        manifest,
        summary_root=summary_root,
        corpora={"and_001"},
        min_match_ratio=0.2,
    )

    assert failures_found == [
        "1 kept matched summary rows have blank transcript speakers",
        "1 kept chunks are below minimum match ratio 0.200: and_001No1=0.100",
    ]


def test_verify_manifest_can_check_speaker_map_provenance(tmp_path: Path) -> None:
    chunk = tmp_path / "and_001" / "and_001No1"
    chunk.mkdir(parents=True)
    (chunk / "speaker_map.csv").write_text(
        "srt_index,start,end,whisperx_speaker,transcript_speaker,speaker_source,matched,score\n"
        "1,00:00:00,00:00:01,[SPEAKER_00]:,[АБ]:,preceding_marker,True,1.000\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        f"clip_id\taudio_path\tspeaker\n001_АБ_00-00-00-000\t{chunk / '001_АБ_00-00-00-000.wav'}\t[АБ]:\n",
        encoding="utf-8",
    )

    _, failures_found = verify_manifest(manifest, check_speaker_maps=True)

    assert failures_found == []


def test_verify_manifest_reports_speaker_map_provenance_failures(tmp_path: Path) -> None:
    chunk = tmp_path / "and_001" / "and_001No1"
    chunk.mkdir(parents=True)
    (chunk / "speaker_map.csv").write_text(
        "srt_index,start,end,whisperx_speaker,transcript_speaker,speaker_source,matched,score\n"
        "1,00:00:00,00:00:01,[SPEAKER_00]:,[ВГ]:,preceding_marker,False,1.000\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "clip_id\taudio_path\tspeaker\n"
        f"001_АБ_00-00-00-000\t{chunk / '001_АБ_00-00-00-000.wav'}\t[АБ]:\n"
        f"002_АБ_00-00-01-000\t{chunk / '002_АБ_00-00-01-000.wav'}\t[АБ]:\n"
        f"003_АБ_00-00-02-000\t{tmp_path / 'missing' / '003_АБ.wav'}\t[АБ]:\n",
        encoding="utf-8",
    )

    _, failures_found = verify_manifest(manifest, check_speaker_maps=True)

    assert failures_found == [
        "1 manifest indices are absent from speaker maps: and_001No1:2",
        "1 manifest rows have no speaker-map provenance file",
        "1 manifest rows are absent from speaker-map provenance",
        "1 manifest rows point to unmatched speaker-map rows",
        "1 manifest speakers differ from speaker-map provenance",
    ]


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
        + "\t".join(values.values())
        + "\n"
        + "\t".join(second.values())
        + "\n",
        encoding="utf-8",
    )

    _, failures = verify_manifest(manifest)

    assert failures == [f"1 duplicate manifest {label}: {values[field]}"]


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

    _, failures = verify_manifest(manifest, check_speaker_maps=True, matched_only=True)

    assert failures == []


def test_verify_manifest_reports_duplicate_speaker_map_indices(tmp_path: Path) -> None:
    chunk = tmp_path / "and_001" / "and_001No1"
    chunk.mkdir(parents=True)
    (chunk / "speaker_map.csv").write_text(
        "srt_index,start,end,whisperx_speaker,transcript_speaker,speaker_source,matched,score\n"
        "1,00:00:00,00:00:01,[SPEAKER_00]:,[АБ]:,marker,True,1.000\n"
        "1,00:00:01,00:00:02,[SPEAKER_00]:,[АБ]:,marker,True,1.000\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "clip_id\taudio_path\tspeaker\n"
        f"001_АБ_00-00-00-000\t{chunk / '001.wav'}\t[АБ]:\n",
        encoding="utf-8",
    )

    _, failures = verify_manifest(manifest, check_speaker_maps=True)

    assert "1 duplicate eligible speaker-map indices: and_001No1:1" in failures


def test_verify_manifest_reports_duplicate_per_chunk_manifest_indices(tmp_path: Path) -> None:
    chunk = tmp_path / "and_001" / "and_001No1"
    chunk.mkdir(parents=True)
    (chunk / "speaker_map.csv").write_text(
        "srt_index,start,end,whisperx_speaker,transcript_speaker,speaker_source,matched,score\n"
        "1,00:00:00,00:00:01,[SPEAKER_00]:,[АБ]:,marker,True,1.000\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "clip_id\taudio_path\tspeaker\n"
        f"001_АБ_00-00-00-000\t{chunk / '001-a.wav'}\t[АБ]:\n"
        f"001_АБ_00-00-00-001\t{chunk / '001-b.wav'}\t[АБ]:\n",
        encoding="utf-8",
    )

    _, failures = verify_manifest(manifest, check_speaker_maps=True)

    assert "1 duplicate per-chunk manifest indices: and_001No1:1" in failures
