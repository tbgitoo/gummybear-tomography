"""M6.3 sequence-directory, frame-order, and filename validation tests."""

from __future__ import annotations

from gummybear_validation.milestone_06.validate_sequence import (
    validate_generated_sequence,
)


def test_valid_minimal_sequence_passes_with_structured_summary(
    m6_sequence_factory,
):
    sequence_dir, _manifest, _write = m6_sequence_factory()

    result = validate_generated_sequence(sequence_dir)

    assert result.ok
    assert result.frame_count == 2
    assert result.role_names == ("clean", "particle", "observed")
    assert result.post_jpeg_identity_checked is True
    assert result.pre_jpeg_identity_checked is False
    assert len(result.checked_files) == 13
    assert result.summary()["ok"] is True
    assert any("Pre-JPEG" in warning for warning in result.warnings)


def test_missing_manifest_fails(tmp_path):
    sequence_dir = tmp_path / "missing_manifest"
    sequence_dir.mkdir()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert "manifest.json is missing" in result.errors[0]


def test_invalid_json_manifest_fails(m6_sequence_factory):
    sequence_dir, _manifest, _write = m6_sequence_factory()
    (sequence_dir / "manifest.json").write_text("{broken", encoding="utf-8")

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("not valid readable JSON" in error for error in result.errors)


def test_non_sequential_frame_indices_fail(m6_sequence_factory):
    sequence_dir, manifest, write_manifest = m6_sequence_factory()
    manifest["frames"][1]["frame_index"] = 3
    write_manifest()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("sequential frame_index" in error for error in result.errors)


def test_filename_frame_index_mismatch_and_order_fail(m6_sequence_factory):
    sequence_dir, manifest, write_manifest = m6_sequence_factory()
    first = manifest["frames"][0]["filenames"]["clean"]
    second = manifest["frames"][1]["filenames"]["clean"]
    manifest["frames"][0]["filenames"]["clean"] = second
    manifest["frames"][1]["filenames"]["clean"] = first
    write_manifest()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("filename encodes frame index" in error for error in result.errors)
    assert any("do not sort in acquisition order" in error for error in result.errors)


def test_role_filename_angle_tokens_must_align(m6_sequence_factory):
    sequence_dir, manifest, write_manifest = m6_sequence_factory()
    original = manifest["frames"][0]["filenames"]["observed"]
    renamed = original.replace("angle_+0000.00", "angle_+0001.00")
    (sequence_dir / original).rename(sequence_dir / renamed)
    manifest["frames"][0]["filenames"]["observed"] = renamed
    write_manifest()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("different angles" in error for error in result.errors)
    assert any("does not match angle_deg" in error for error in result.errors)
