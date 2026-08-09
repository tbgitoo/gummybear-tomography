"""Installed pytest checks for particle transport pair and source-deposition contracts."""

import numpy as np

from gummybear.particles.geometry import ParticleSet, ParticleSphere, intersect_segments_with_particles
from gummybear.particles.perturbation import build_affected_transport_pairs
 
from ..text_output import array_mini_summary
from ..helpers import event_segment_index, pair_path_id

import pytest




from gummybear.optics.diffusion_mesh import DiffusionMesh, DiffusionMeshMetadata
from gummybear.optics.source_deposition import (
    RaySegmentBundle,
    deposit_ray_source,
)
from gummybear.particles import (
    ParticleSphere,
    assert_downstream_background_shadow,
    build_affected_transport_pairs,
    compute_transport_source_correction,
    deposit_particle_scatter_sources,
    nearest_tet_centroid,
)






@pytest.mark.milestone("Milestone 5B")
@pytest.mark.proves(
    "Clean/particle-altered transport pairs are constructed exactly for rays " 
    "with particle-hit segments, as verified by explicit particle-intersection " 
    "events."
)

def test_affected_pair_indices_match_intersection_events():
    """Check affected transport pairs align with particle intersection events.

    Builds a tiny two-path segment bundle with one analytic sphere hit, then
    verifies ``build_affected_transport_pairs`` marks exactly the intersected
    segment and parent path. **Pass:** affected segment indices and path ids
    match the intersection event set; the lone pair exposes clean/dirty
    intervals for path 42.

    Notebook / protocol: M5B
    """
    segments = RaySegmentBundle(
        starts=np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        ends=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        intensities=np.ones(4, dtype=float),
        ray_ids=np.array([42, 42, 99, 99], dtype=int),
        path_order=np.array([0, 1, 0, 1], dtype=int),
    )

    particles = ParticleSet.from_particles(
        [
            ParticleSphere(
                center=(1.0, 0.0, 0.0),
                radius=0.25,
                mu_abs=1.0,
            )
        ]
    )

    events = intersect_segments_with_particles(
        segments.starts,
        segments.ends,
        particles,
    )

    result = build_affected_transport_pairs(
        segments,
        particles,
        events=events,
    )

    event_segment_indices = np.asarray(
        sorted(set(event_segment_index(event) for event in events)),
        dtype=int,
    )

    event_path_ids = np.asarray(
        sorted(
            set(
                int(segments.ray_ids[segment_index])
                for segment_index in event_segment_indices
            )
        ),
        dtype=int,
    )

    assert result.affected_segment_indices.tolist() == event_segment_indices.tolist()
    assert tuple(result.affected_path_ids) == tuple(event_path_ids)

    assert result.affected_path_ids == (42,)
    assert result.affected_segment_indices.tolist() == [1]
    assert len(result.pairs) == 1

    pair = result.pairs[0]

    assert pair_path_id(pair) == 42
    assert len(pair.clean_intervals) == 2
    assert len(pair.dirty_intervals) > 0




@pytest.mark.milestone("Milestone 5C")
@pytest.mark.proves(
    "For each particle-hit transport path, downstream background scattering "
    "behind the particle is lower along the particle-altered path than along "
    "the corresponding clean path."
)
def test_particle_creates_downstream_scattering_shadow():
    """Verify particle-hit paths cast a downstream background scattering shadow.

    Uses a synthetic three-segment path and single downstream tet. **Pass:**
    ``assert_downstream_background_shadow`` reports zero violations and dirty
    background deposition is strictly less than clean on downstream elements.

    Notebook / protocol: M5C
    """
    segments = RaySegmentBundle(
        starts=np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        ends=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        intensities=np.ones(3, dtype=float),
        ray_ids=np.array([17, 17, 17], dtype=int),
        path_order=np.array([0, 1, 2], dtype=int),
    )

    particles = [
        ParticleSphere(
            center=(0.5, 0.0, 0.0),
            radius=0.25,
            mu_abs=3.0,
            mu_scat=0.0,
        )
    ]

    result = build_affected_transport_pairs(
        segments,
        particles,
        mu_s=1.0,
        mu_a=0.0,
    )

    assert result.affected_path_ids == (17,)
    assert len(result.pairs) == 1

    pair = result.pairs[0]

    # One tetrahedron around the downstream [1, 2] path interval.  The
    # package assertion performs exact ray-tet deposition for the clean and
    # dirty background intervals; TransportInterval itself intentionally does
    # not store element-deposited E_scat.
    nodes = np.array(
        [
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    downstream_mesh = DiffusionMesh(
        nodes=nodes,
        tets=np.array([[0, 1, 2, 3]], dtype=int),
        centroids=nodes.mean(axis=0, keepdims=True),
        volumes=np.array([2.0 / 3.0], dtype=float),
        metadata=DiffusionMeshMetadata(
            stl_hash="test",
            geometry_id="downstream_tet",
            meshing_method="synthetic",
            num_elements=1,
            num_nodes=4,
        ),
        netgen_mesh=None,
    )

    summary = assert_downstream_background_shadow(
        downstream_mesh,
        pair,
        mu_s=1.0,
        mu_a=0.0,
        density=False,
    )
    deposition = summary["deposition"]

    assert summary["n_downstream_elements"] == 1
    assert summary["n_violations"] == 0
    assert deposition.E_clean_elem.sum() > 0.0
    assert deposition.E_dirty_background_elem.sum() > 0.0
    assert (
        deposition.E_dirty_background_elem.sum()
        < deposition.E_clean_elem.sum()
    )





def _single_tet_mesh() -> DiffusionMesh:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return DiffusionMesh(
        nodes=nodes,
        tets=np.array([[0, 1, 2, 3]], dtype=int),
        centroids=nodes.mean(axis=0, keepdims=True),
        volumes=np.array([1.0 / 6.0], dtype=float),
        metadata=DiffusionMeshMetadata(
            stl_hash="test",
            geometry_id="single_tet",
            meshing_method="synthetic",
            num_elements=1,
            num_nodes=4,
        ),
        netgen_mesh=None,
    )




@pytest.mark.milestone("Milestone 5C")
@pytest.mark.proves(
    "The particle-induced source delta equals the sum of primary-ray "
    "attenuation and energy directly scattered by the particle."
)

def test_source_correction_identity_and_inspectable_split():
    """Check transport source-delta splits and reconstructs particle deposition.

    **Pass:** ``delta_E_transport == delta_background + delta_scatter``;
    ``E_particle == E_clean + delta_transport``; ``S_particle == E_particle/volume``;
    metadata marks ``source_model == affected_transport_pair_delta``.

    Notebook / protocol: M5C
    """
    mesh = _single_tet_mesh()
    c = mesh.centroids[0]

    segments = RaySegmentBundle(
        starts=np.array([[-0.5, c[1], c[2]]]),
        ends=np.array([[1.5, c[1], c[2]]]),
        intensities=np.array([1.0]),
        ray_ids=np.array([11]),
    )

    particles = [
        ParticleSphere(
            center=c,
            radius=0.1,
            mu_abs=1.0,
            mu_scat=2.0,
        )
    ]

    pairs = build_affected_transport_pairs(segments, particles, mu_s=1.0, mu_a=0.0)

    clean = deposit_ray_source(
        mesh,
        segments,
        mu_s=1.0,
        mu_a=0.0,
    )

    source = compute_transport_source_correction(
        mesh,
        pairs,
        clean.E_scat_elem,
        mu_s=1.0,
        mu_a=0.0,
    )

    assert np.allclose(
        source.delta_E_transport_elem,
        source.delta_E_background_elem + source.delta_E_particle_scat_elem,
    )

    assert np.allclose(
        source.E_particle_elem,
        source.E_clean_elem + source.delta_E_transport_elem,
    )

    assert np.allclose(
        source.S_particle,
        source.E_particle_elem / mesh.volumes,
    )

    assert source.metadata["source_model"] == "affected_transport_pair_delta"

import numpy as np
import pytest

from importlib import resources

from gummybear.geometry import load_stl

from gummybear.optics import (
    OpticalMaterialConfig,
    PointLightConfig,
    SourceSamplingParams,
    generate_diffusion_mesh,
    in_object_segments_from_rays,
    make_source_ray_bundle,
    refract_ray_bundle,
)

from gummybear.particles import (
    ParticleSet,
    ParticleSphere,
    build_affected_transport_pairs,
    deposit_transport_pair_sources,
    intersect_segments_with_particles,
)

from gummybear.particles.access_helpers import pair_path_id


def _fixture_stl_context():
    stl_resource = resources.files("gummybear_validation.test_data").joinpath(
        "gummybear_fixture.stl"
    )

    return resources.as_file(stl_resource)


def _path_events_for_id(events, segments, path_id):
    ray_ids = np.asarray(segments.ray_ids, dtype=int)

    return [
        event
        for event in events
        if int(ray_ids[int(event.segment_index)]) == int(path_id)
    ]


def _interval_point(interval, which):
    if which == "start":
        names = ("start", "point0", "start_point", "p0")
    else:
        names = ("end", "point1", "end_point", "p1")

    if isinstance(interval, dict):
        for name in names:
            if name in interval and interval[name] is not None:
                return np.asarray(interval[name], dtype=float)
    else:
        for name in names:
            if hasattr(interval, name):
                value = getattr(interval, name)
                if value is not None:
                    return np.asarray(value, dtype=float)

    raise AttributeError("Could not read interval " + which + " point.")


def _path_origin_direction_from_pair(pair):
    clean_intervals = tuple(pair.clean_intervals)

    p0 = _interval_point(clean_intervals[0], "start")
    p1 = _interval_point(clean_intervals[-1], "end")

    direction = p1 - p0
    direction = direction / np.linalg.norm(direction)

    return p0, direction


def _build_m5c_fixture_scene():
    with _fixture_stl_context() as stl_path:
        surface_mesh = load_stl(stl_path)
        diff_mesh = generate_diffusion_mesh(stl_path, target_elements=1000)

    assert diff_mesh.netgen_mesh is not None, (
        "DiffusionMesh.netgen_mesh is missing for the M5C fixture scene."
    )

    hi = diff_mesh.bounds[1]
    c = diff_mesh.bounds.mean(axis=0)

    mat = OpticalMaterialConfig(
        mu_scatter=0.3,
        mu_absorption=0.1,
        n_refractive=1.33,
    )

    light = PointLightConfig(
        position=(c[0] + 15.0, c[1] + 15.0, hi[2] + 15.0),
        intensity=1.0,
    )

    source_rays = make_source_ray_bundle(
        light,
        surface_mesh.bounds,
        SourceSamplingParams(n_rays=512, seed=0),
    )

    entry = refract_ray_bundle(
        surface_mesh,
        source_rays,
        n_from=1.0,
        n_to=mat.n_refractive,
    )

    segments = in_object_segments_from_rays(surface_mesh, entry.rays)

    particles = ParticleSet.from_particles(
        [
            ParticleSphere(
                center=c,
                radius=3.0,
                mu_abs=0.2,
                mu_scat=0.8,
                particle_id="p000_mixed_phoenix",
            )
        ]
    )

    ray_ids = np.asarray(segments.ray_ids, dtype=int)

    events = intersect_segments_with_particles(
        segments.starts,
        segments.ends,
        particles,
    )

    events = sorted(
        list(events),
        key=lambda ev: (
            int(ev.segment_index),
            float(ev.entry_t),
            int(ev.particle_index),
        ),
    )

    assert events, "No particle intersection events were found."

    event_segment_indices = np.asarray(
        sorted(set(int(ev.segment_index) for ev in events)),
        dtype=int,
    )

    event_path_ids = np.asarray(
        sorted(set(int(ray_ids[s]) for s in event_segment_indices)),
        dtype=int,
    )

    pair_result = build_affected_transport_pairs(
        segments,
        particles,
        events=events,
        mu_s=mat.mu_scatter,
        mu_a=mat.mu_absorption,
    )

    pairs = tuple(pair_result.pairs)

    assert len(pairs) > 0, "No affected transport pairs were constructed."

    path_ids_with_pairs = set(int(pair_path_id(pair)) for pair in pairs)

    assert 19 in path_ids_with_pairs, (
        "Expected path_id 19 in the fixed M5C fixture scene. "
        "Available path ids are: " + repr(sorted(path_ids_with_pairs))
    )

    assert 19 in set(int(x) for x in event_path_ids), (
        "Expected path_id 19 among particle-hit transport rays."
    )

    return {
        "path_id": 19,
        "pair_result": pair_result,
        "segments": segments,
        "events": events,
        "diff_mesh": diff_mesh,
        "mu_s": mat.mu_scatter,
        "mu_a": mat.mu_absorption,
    }


def _local_source_profile_for_path(
    path_id,
    pair_result,
    segments,
    events,
    diff_mesh,
    mu_s,
    mu_a,
    density=True,
    assignment="attenuated_chord",
    window_extra=8,
):
    pair = next(
        pair
        for pair in pair_result.pairs
        if int(pair_path_id(pair)) == int(path_id)
    )

    path_events = _path_events_for_id(events, segments, path_id)

    if not path_events:
        raise ValueError("No particle-intersection event found for path.")

    origin, direction = _path_origin_direction_from_pair(pair)

    entry_s = min(
        float(np.asarray(event.entry_point, dtype=float) @ direction - origin @ direction)
        for event in path_events
    )

    exit_s = max(
        float(np.asarray(event.exit_point, dtype=float) @ direction - origin @ direction)
        for event in path_events
    )

    deposition = deposit_transport_pair_sources(
        diff_mesh,
        pair,
        mu_s=mu_s,
        mu_a=mu_a,
        assignment=assignment,
    )

    volumes = np.asarray(diff_mesh.volumes, dtype=float)

    clean = np.asarray(deposition.E_clean_elem, dtype=float)
    dirty_bg = np.asarray(deposition.E_dirty_background_elem, dtype=float)
    particle_scat = np.asarray(deposition.E_particle_scat_elem, dtype=float)
    dirty_total = np.asarray(deposition.E_dirty_total_elem, dtype=float)

    if density:
        clean = clean / volumes
        dirty_bg = dirty_bg / volumes
        particle_scat = particle_scat / volumes
        dirty_total = dirty_total / volumes

    delta_total = dirty_total - clean

    centroids = np.asarray(diff_mesh.centroids, dtype=float)
    elem_s = centroids @ direction - origin @ direction

    active = (
        (np.abs(clean) > 0.0)
        | (np.abs(dirty_bg) > 0.0)
        | (np.abs(particle_scat) > 0.0)
        | (np.abs(dirty_total) > 0.0)
    )

    active_indices = np.nonzero(active)[0]
    active_indices = active_indices[np.argsort(elem_s[active_indices])]

    near_particle = (
        (elem_s[active_indices] >= entry_s - window_extra)
        & (elem_s[active_indices] <= exit_s + window_extra)
    )

    shown = active_indices[near_particle]

    region = []

    for i in shown:
        if elem_s[i] < entry_s:
            region.append("before")
        elif elem_s[i] <= exit_s:
            region.append("inside")
        else:
            region.append("after")

    clean_shown = clean[shown]
    dirty_total_shown = dirty_total[shown]

    ratio = np.full(clean_shown.shape, np.nan, dtype=float)
    valid = clean_shown > 0.0
    ratio[valid] = dirty_total_shown[valid] / clean_shown[valid]

    return {
        "elem": shown,
        "s": elem_s[shown],
        "region": np.asarray(region, dtype=object),
        "clean": clean_shown,
        "dirty_bg": dirty_bg[shown],
        "particle_scat": particle_scat[shown],
        "dirty_total": dirty_total_shown,
        "delta_total": delta_total[shown],
        "dirty_total_over_clean": ratio,
        "entry_s": entry_s,
        "exit_s": exit_s,
    }


def _assert_entry_enhancement_then_shadow(profile):
    region = profile["region"]
    clean = profile["clean"]
    delta_total = profile["delta_total"]
    ratio = profile["dirty_total_over_clean"]

    before = region == "before"
    inside = region == "inside"
    after = region == "after"

    clean_scale = max(float(np.nanmax(clean)), 1.0)

    if np.any(before):
        assert np.all(np.abs(delta_total[before]) <= 1e-10 * clean_scale), (
            "Source deposition changed before particle entry."
        )

    inside_indices = np.nonzero(inside)[0]
    after_indices = np.nonzero(after)[0]

    assert len(inside_indices) >= 3, (
        "Need at least three inside-particle elements to test entry and attenuation behavior."
    )

    assert len(after_indices) >= 1, (
        "Need at least one downstream element to test the shadow."
    )

    inside_valid = inside_indices[np.isfinite(ratio[inside_indices])]
    after_valid = after_indices[np.isfinite(ratio[after_indices])]

    assert len(inside_valid) >= 3, (
        "Need at least three valid inside-particle clean/dirty ratios."
    )

    assert len(after_valid) >= 1, (
        "Need at least one valid downstream clean/dirty ratio."
    )

    first_inside = inside_valid[0]
    later_inside = inside_valid[inside_valid > first_inside]

    assert len(later_inside) >= 1, (
        "Need at least one later inside-particle element after the entry element."
    )

    assert ratio[first_inside] > 1.0, (
        "The first inside-particle source element is not enhanced. "
        "Particle scattering should be deposited locally at particle entry."
    )

    assert np.any(ratio[later_inside] < 1.0), (
        "No later inside-particle suppression was found. "
        "Combined attenuation should eventually dominate after entry enhancement."
    )

    assert np.any(ratio[after_valid] < 1.0), (
        "No downstream shadow was found after particle exit."
    )


@pytest.mark.milestone("Milestone 5C")
@pytest.mark.proves(
    "Strong particle scattering raises source deposition at particle entry "
    "before combined attenuation produces a downstream shadow."
)
def test_particle_source_deposition_rises_at_entry_then_falls_downstream():
    """Check entry enhancement then attenuation/shadow on a fixture STL triangle mesh scene.

    Builds the shared M5C fixture, profiles source density along path 19, and
    asserts: no change before entry; ratio > 1 at first inside element; later
    inside and downstream ratios fall below 1.

    Notebook / protocol: M5C
    """
    scene = _build_m5c_fixture_scene()

    profile = _local_source_profile_for_path(
        scene["path_id"],
        scene["pair_result"],
        scene["segments"],
        scene["events"],
        scene["diff_mesh"],
        scene["mu_s"],
        scene["mu_a"],
        density=True,
        assignment="attenuated_chord",
        window_extra=8,
    )

    _assert_entry_enhancement_then_shadow(profile)
