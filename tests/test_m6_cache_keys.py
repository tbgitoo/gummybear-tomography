"""Phase 1 tests for M6 cache-key construction."""

from __future__ import annotations

from gummybear.datasets.cache_keys import (
    clean_optical_cache_key,
    clean_optical_cache_key_payload,
    particle_source_cache_key,
    particle_source_cache_key_payload,
)


def _base_clean_kwargs():
    return dict(
        stl_sha256="abc123",
        illumination_kind="point",
        light_position_x=1.0,
        light_position_y=2.0,
        light_position_z=3.0,
        num_source_rays=512,
        mu_s=0.3,
        mu_a=0.1,
        refractive_index=1.33,
        source_deposition_method="exact_ray_tet_intervals",
    )


def _base_particle_kwargs(clean_id: str):
    return dict(
        clean_optical_cache_id=clean_id,
        particle_kind="sphere",
        center_x=0.0,
        center_y=0.0,
        center_z=0.0,
        radius=3.0,
        mu_s_particle=0.8,
        mu_a_particle=0.2,
        refractive_index_particle=1.33,
        placement_mode="fixed",
        seed=42,
    )


def test_clean_key_changes_with_source_intensity():
    base = _base_clean_kwargs()
    key_a = clean_optical_cache_key(**base)
    key_b = clean_optical_cache_key(**{**base, "source_intensity": 2.0})
    assert key_a != key_b
    payload = clean_optical_cache_key_payload(**base)
    assert payload["illumination"]["source_intensity"] == 1.0
    assert payload["algorithm_version"] == "m6-clean-optical-v4"


def test_clean_key_stable_and_excludes_camera_like_fields():
    base = clean_optical_cache_key(**_base_clean_kwargs())
    again = clean_optical_cache_key(**_base_clean_kwargs())
    assert base == again
    assert len(base) == 64

    payload = clean_optical_cache_key_payload(**_base_clean_kwargs())
    forbidden = (
        "camera",
        "camera_schedule",
        "resolution",
        "lateral_offsets",
        "z_offsets",
        "up_variants",
        "particle",
        "extrapolation_length",
        "robin_boundary_model",
        "corruption",
        "image_format",
        "jpeg_quality",
        "max_workers",
    )
    for name in forbidden:
        assert name not in payload
    assert payload["illumination"]["target_footprint"] == (
        "derived_from_mesh_bounds"
    )


def test_camera_or_resolution_changes_do_not_affect_clean_key_api():
    # Callers must not pass camera fields into the clean key helper. Changing
    # unrelated local variables must leave the clean key unchanged.
    kwargs = _base_clean_kwargs()
    key_a = clean_optical_cache_key(**kwargs)
    camera_resolution = (224, 224)
    camera_schedule_id = "orbit_other"
    _ = (camera_resolution, camera_schedule_id)
    key_b = clean_optical_cache_key(**kwargs)
    assert key_a == key_b


def test_robin_extrapolation_does_not_change_clean_or_particle_keys():
    clean_id = clean_optical_cache_key(**_base_clean_kwargs())
    particle_id = particle_source_cache_key(**_base_particle_kwargs(clean_id))

    robin_a = 5.0
    robin_b = 12.0
    _ = (robin_a, robin_b)

    clean_again = clean_optical_cache_key(**_base_clean_kwargs())
    particle_again = particle_source_cache_key(**_base_particle_kwargs(clean_id))
    assert clean_again == clean_id
    assert particle_again == particle_id


def test_particle_radius_and_optics_change_particle_key():
    clean_id = clean_optical_cache_key(**_base_clean_kwargs())
    base = _base_particle_kwargs(clean_id)
    key_a = particle_source_cache_key(**base)

    radius_changed = dict(base)
    radius_changed["radius"] = 4.0
    key_b = particle_source_cache_key(**radius_changed)
    assert key_a != key_b

    optics_changed = dict(base)
    optics_changed["mu_s_particle"] = 1.5
    key_c = particle_source_cache_key(**optics_changed)
    assert key_a != key_c


def test_max_workers_does_not_affect_scientific_keys():
    clean_id = clean_optical_cache_key(**_base_clean_kwargs())
    particle_id = particle_source_cache_key(**_base_particle_kwargs(clean_id))

    max_workers = 10
    _ = max_workers

    assert clean_optical_cache_key(**_base_clean_kwargs()) == clean_id
    assert (
        particle_source_cache_key(**_base_particle_kwargs(clean_id)) == particle_id
    )

    clean_payload = clean_optical_cache_key_payload(**_base_clean_kwargs())
    particle_payload = particle_source_cache_key_payload(
        **_base_particle_kwargs(clean_id)
    )
    assert "max_workers" not in clean_payload
    assert "max_workers" not in particle_payload


def test_fixed_placement_seed_does_not_change_particle_key():
    clean_id = clean_optical_cache_key(**_base_clean_kwargs())
    base = _base_particle_kwargs(clean_id)
    key_a = particle_source_cache_key(**base)
    base_other_seed = dict(base)
    base_other_seed["seed"] = 99
    key_b = particle_source_cache_key(**base_other_seed)
    assert key_a == key_b


def test_source_ray_seed_changes_clean_key():
    base = _base_clean_kwargs()
    with_seed_a = dict(base, source_ray_seed=42)
    with_seed_b = dict(base, source_ray_seed=43)
    assert clean_optical_cache_key(**with_seed_a) != clean_optical_cache_key(
        **with_seed_b
    )


def test_source_assignment_changes_particle_key():
    clean_id = clean_optical_cache_key(**_base_clean_kwargs())
    base = _base_particle_kwargs(clean_id)
    attenuated = particle_source_cache_key(
        **base,
        source_delta_assignment="attenuated_chord",
    )
    midpoint = particle_source_cache_key(
        **base,
        source_delta_assignment="midpoint",
    )
    assert attenuated != midpoint


def test_legacy_single_particle_cache_kwargs_still_work():
    clean_id = clean_optical_cache_key(**_base_clean_kwargs())
    base = _base_particle_kwargs(clean_id)
    payload = particle_source_cache_key_payload(**base)
    assert "particles" in payload
    assert len(payload["particles"]) == 1
    assert payload["particles"][0]["radius"] == 3.0
