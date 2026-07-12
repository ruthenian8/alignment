"""Tests for exported manifest verification helper."""

from pathlib import Path

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
