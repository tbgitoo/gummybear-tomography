"""M6.3 manifest, provenance, portability, and cache-policy tests."""

from __future__ import annotations

from gummybear_validation.milestone_06.validate_sequence import (
    validate_generated_sequence,
)


def test_missing_required_top_level_field_fails(m6_sequence_factory):
    sequence_dir, manifest, write_manifest = m6_sequence_factory()
    del manifest["forward_model_tier"]
    write_manifest()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("forward_model_tier" in error for error in result.errors)


def test_missing_legacy_split_and_seed_is_ok(m6_sequence_factory):
    sequence_dir, manifest, write_manifest = m6_sequence_factory()
    manifest.pop("split", None)
    manifest.pop("seed", None)
    write_manifest()

    result = validate_generated_sequence(sequence_dir)

    assert result.ok


def test_wrong_schema_field_spelling_fails(m6_sequence_factory):
    sequence_dir, manifest, write_manifest = m6_sequence_factory()
    manifest["schemaVersion"] = manifest.pop("schema_version")
    write_manifest()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("Forbidden schema-version field" in error for error in result.errors)
    assert any("schema_version" in error for error in result.errors)


def test_absolute_workbook_and_stl_paths_fail(m6_sequence_factory):
    sequence_dir, manifest, write_manifest = m6_sequence_factory()
    manifest["workbook"]["workbook_path"] = "/Users/example/private/plan.xlsx"
    manifest["phantom"]["stl_path"] = "/Users/example/private/bear.stl"
    write_manifest()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("workbook path" in error for error in result.errors)
    assert any("phantom.stl_path" in error for error in result.errors)


def test_windows_absolute_workbook_path_fails(m6_sequence_factory):
    sequence_dir, manifest, write_manifest = m6_sequence_factory()
    manifest["workbook"]["workbook_path"] = (
        r"C:\Users\example\private\m6_generation_plan.xlsx"
    )
    write_manifest()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("workbook path" in error for error in result.errors)


def test_missing_stl_sha256_fails(m6_sequence_factory):
    sequence_dir, manifest, write_manifest = m6_sequence_factory()
    del manifest["phantom"]["stl_sha256"]
    write_manifest()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("phantom.stl_sha256" in error for error in result.errors)


def test_setup_workbook_coordinates_are_required(m6_sequence_factory):
    sequence_dir, manifest, write_manifest = m6_sequence_factory()
    del manifest["setups"]["particle"]["workbook_name"]
    del manifest["setups"]["camera"]["workbook_sheet"]
    del manifest["setups"]["camera"]["source_sheet"]
    write_manifest()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("particle.workbook_name" in error for error in result.errors)
    assert any("camera requires workbook_sheet" in error for error in result.errors)


def test_null_diffusion_operator_cache_and_disabled_persistence_pass(
    m6_sequence_factory,
):
    sequence_dir, manifest, _write = m6_sequence_factory()

    result = validate_generated_sequence(sequence_dir)

    assert manifest["caches"]["persistent_cache_used"] is False
    assert manifest["caches"]["diffusion_operator_cache"] is None
    assert result.ok


def test_non_null_diffusion_operator_cache_fails(m6_sequence_factory):
    sequence_dir, manifest, write_manifest = m6_sequence_factory()
    manifest["caches"]["diffusion_operator_cache"] = "serialized-fem-operator"
    write_manifest()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("diffusion_operator_cache must be null" in error for error in result.errors)


def test_forbidden_serialized_runtime_reference_fails(m6_sequence_factory):
    sequence_dir, manifest, write_manifest = m6_sequence_factory()
    manifest["generation"]["solver_handle"] = "pickled-handle"
    write_manifest()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("forbidden serialized FEM/runtime state" in error for error in result.errors)


def test_pixel_orientation_contract_is_required(m6_sequence_factory):
    sequence_dir, manifest, write_manifest = m6_sequence_factory()
    manifest["representation"]["pixel_orientation"] = {
        "camera_up": "image_bottom",
        "transform_from_camera_sample_grid": "none",
    }
    write_manifest()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("pixel_orientation" in error for error in result.errors)


def test_negative_stage_timing_and_wrong_assignment_fail(m6_sequence_factory):
    sequence_dir, manifest, write_manifest = m6_sequence_factory()
    manifest["generation"]["stage_seconds"]["camera_capture"] = -1.0
    manifest["generation"]["diagnostics"]["source_assignment"] = "midpoint"
    write_manifest()

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("camera_capture" in error for error in result.errors)
    assert any("source_assignment" in error for error in result.errors)
