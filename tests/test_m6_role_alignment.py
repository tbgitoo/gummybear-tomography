"""M6.3 role alignment and compression-aware composition tests."""

from __future__ import annotations

import numpy as np
from PIL import Image

from gummybear_validation.milestone_06.validate_sequence import (
    validate_generated_sequence,
)


def test_missing_required_role_file_fails(m6_sequence_factory):
    sequence_dir, manifest, _write = m6_sequence_factory()
    missing = sequence_dir / manifest["frames"][0]["filenames"]["clean"]
    missing.unlink()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("file does not exist" in error for error in result.errors)


def test_mismatched_role_dimensions_fail(m6_sequence_factory):
    sequence_dir, manifest, _write = m6_sequence_factory()
    observed_path = sequence_dir / manifest["frames"][0]["filenames"]["observed"]
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(
        observed_path,
        format="JPEG",
        quality=95,
    )

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("image size" in error for error in result.errors)
    assert any("dimensions do not align" in error for error in result.errors)


def test_identical_no_corruption_roles_pass_approximate_post_jpeg_check(
    m6_sequence_factory,
):
    sequence_dir, _manifest, _write = m6_sequence_factory()

    result = validate_generated_sequence(sequence_dir)

    assert result.ok
    assert result.post_jpeg_identity_checked is True
    assert result.pre_jpeg_identity_checked is False
    assert any("approximately" in warning for warning in result.warnings)


def test_large_observed_particle_difference_fails(m6_sequence_factory):
    sequence_dir, manifest, _write = m6_sequence_factory()
    observed_path = sequence_dir / manifest["frames"][0]["filenames"]["observed"]
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8), mode="L").save(
        observed_path,
        format="JPEG",
        quality=95,
    )

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("compression-aware tolerance" in error for error in result.errors)


def test_missing_optional_anomaly_preview_passes(m6_sequence_factory):
    sequence_dir, manifest, _write = m6_sequence_factory(anomaly=False)

    result = validate_generated_sequence(sequence_dir)

    assert "anomaly_preview" not in manifest["roles"]
    assert result.ok
    assert "anomaly" not in result.role_names


def test_listed_anomaly_preview_missing_fails(m6_sequence_factory):
    sequence_dir, manifest, _write = m6_sequence_factory(anomaly=True)
    anomaly_path = sequence_dir / manifest["frames"][0]["filenames"]["anomaly"]
    anomaly_path.unlink()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any(
        "role 'anomaly' file does not exist" in error for error in result.errors
    )


def test_anomaly_preview_is_non_authoritative(m6_sequence_factory):
    sequence_dir, manifest, write_manifest = m6_sequence_factory(anomaly=True)
    manifest["representation"]["anomaly_preview"]["authoritative"] = True
    write_manifest()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("authoritative must be false" in error for error in result.errors)
