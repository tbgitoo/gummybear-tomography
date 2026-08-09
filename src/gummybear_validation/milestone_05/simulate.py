"""Prepare inspectable particle-transport simulation state for validation notebooks.

Builds scene geometry from an STL triangle mesh file, source deposition, finite-element method (FEM)
diffusion solves, camera samples, and scalar diagnostics via :func:`run_m5d_simulation`.
Does not plot figures or redefine particle-source deposition contracts.

Main entry point:

    run_m5d_simulation(config)

Notebook / protocol: M5D
"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from gummybear.geometry import load_stl
from gummybear.optics import (
    OpticalMaterialConfig,
    PointLightConfig,
    SourceSamplingParams,
    compose_hybrid_image,
    compute_refractive_direct_image,
    deposit_ray_source,
    generate_diffusion_mesh,
    in_object_segments_from_rays,
    make_source_ray_bundle,
    refract_ray_bundle,
    sample_diffuse_image,
    solve_diffusion,
)
from gummybear.particles import (
    ParticleSet,
    build_affected_transport_pairs,
    compute_transport_source_correction,
)
from gummybear.rays import (
    PinholeCameraConfig,
    first_visible_hits_with_points,
    make_camera_rays,
)


@dataclass
class M5DSimulationConfig:
    """Inputs for one full clean vs particle-altered forward-model validation run.

    Provides an STL triangle mesh file path or pre-built meshes, optical material, illumination, camera,
    and particle specification. Set ``compute_direct=True`` to also form hybrid
    direct+diffuse camera images.

    Attributes:
        stl_path: Path to phantom STL triangle mesh file when meshes are not pre-supplied.
        material: Bulk optical coefficients and refractive index.
        light: Point-light position and intensity for source rays.
        camera: Pinhole camera intrinsics/extrinsics for image sampling.
        particles: ``ParticleSet``, sphere list, or callable ``(diff_mesh) → …``.
        target_elements: Coarse diffusion mesh element budget.
        n_source_rays: Number of exterior source rays.
        ray_seed: RNG seed for source-ray sampling.
        source_sampling_mode: Source-ray sampling mode string.
        extrapolation_length: Robin boundary condition (extrapolated-flux leakage) extrapolation length for diffusion.
        exitance_scale: Diffuse camera exitance multiplier.
        direct_scale: Direct-image scale when ``compute_direct`` is enabled.
        alpha: Hybrid blend weight between direct and diffuse images.
        n_from: Exterior refractive index for refraction at entry.
        source_delta_assignment: Particle-scatter deposition assignment mode.
        compute_direct: Whether to compute direct and hybrid camera channels.
        surface_mesh: Optional pre-loaded surface mesh (skips STL load).
        diff_mesh: Optional pre-built diffusion mesh (skips mesh generation).
    """

    stl_path: Any
    material: OpticalMaterialConfig
    light: PointLightConfig
    camera: PinholeCameraConfig
    particles: Any

    target_elements: int = 1000
    n_source_rays: int = 1000
    ray_seed: int = 0
    source_sampling_mode: str = "point_uniform"

    extrapolation_length: float = 5.0
    exitance_scale: float = 10.0
    direct_scale: float = 1.0
    alpha: float = 0.0

    n_from: float = 1.0
    source_delta_assignment: str = "attenuated_chord"

    compute_direct: bool = False

    surface_mesh: Any = None
    diff_mesh: Any = None


@dataclass
class M5DScene:
    """Geometry, transport segments, and camera hit maps for one simulation scene.

    Attributes:
        surface_mesh: Phantom surface mesh used for refraction and camera rays.
        diff_mesh: Coarse diffusion mesh (must retain ``netgen_mesh`` for finite-element method (FEM) solve).
        material, light, camera, particles: Copied from config / resolution.
        source_rays: Exterior source-ray bundle before refraction.
        refracted: Refracted entry bundle inside the object.
        segments: In-object transport segments with path ids.
        camera_rays: Pinhole rays for the image grid.
        H, W: Image height and width.
        cam_valid, cam_depth, cam_faces, cam_points: Per-pixel camera hits.
        camera_mask: ``(H, W)`` boolean visibility mask.
        hit_faces_img: ``(H, W)`` first-hit face indices.
        view_directions: ``(H, W, 3)`` unit view vectors per pixel.
    """

    surface_mesh: Any
    diff_mesh: Any
    material: OpticalMaterialConfig
    light: PointLightConfig
    camera: PinholeCameraConfig
    particles: ParticleSet

    source_rays: Any
    refracted: Any
    segments: Any

    camera_rays: Any
    H: int
    W: int
    cam_valid: np.ndarray
    cam_depth: np.ndarray
    cam_faces: np.ndarray
    cam_points: np.ndarray
    camera_mask: np.ndarray
    hit_faces_img: np.ndarray
    view_directions: np.ndarray


@dataclass
class M5DSourceState:
    """Clean and particle-altered volumetric source fields plus pair bookkeeping.

    Attributes:
        clean_deposition: Ray-source deposition without particle perturbation.
        pair_result: Affected transport pairs and metadata.
        source_delta: Transport source correction (background + scatter split).
        S_clean, S_particle, S_delta: Element source densities ``E / volume``.
        active_source_mask: Boolean mask of elements with any source activity.
    """

    clean_deposition: Any
    pair_result: Any
    source_delta: Any

    S_clean: np.ndarray
    S_particle: np.ndarray
    S_delta: np.ndarray
    active_source_mask: np.ndarray


@dataclass
class M5DDiffusionState:
    """Diffusion solves for clean, particle, and delta sources.

    Attributes:
        clean_solve, particle_solve, delta_solve: Finite-element method (FEM) solve results.
        Phi_delta_nodes, Phi_delta_tets: Particle-minus-clean fluence field ``Phi`` values.
        linearity_error: Relative ``‖Phi_particle−Phi_clean − Phi_delta‖`` error.
        solve_seconds_clean, solve_seconds_particle, solve_seconds_delta: Timings.
    """

    clean_solve: Any
    particle_solve: Any
    delta_solve: Any

    Phi_delta_nodes: np.ndarray
    Phi_delta_tets: np.ndarray
    linearity_error: float

    solve_seconds_clean: float
    solve_seconds_particle: float
    solve_seconds_delta: float


@dataclass
class M5DCameraState:
    """Sampled diffuse (and optional direct/hybrid) camera images.

    Attributes:
        clean_diffuse, particle_diffuse: Diffuse-only camera samples.
        Delta_I_diffuse: Pixelwise particle-minus-clean diffuse image.
        clean_direct, particle_direct, Delta_I_direct: Present when direct path
            is computed.
        clean_hybrid, particle_hybrid, Delta_I_total: Hybrid compose results.
        particle_ray_weights, affected_ray_ids: Direct-path attenuation factors.
    """

    clean_diffuse: Any
    particle_diffuse: Any
    Delta_I_diffuse: np.ndarray

    clean_direct: Any = None
    particle_direct: Any = None
    Delta_I_direct: Any = None

    clean_hybrid: Any = None
    particle_hybrid: Any = None
    Delta_I_total: Any = None

    particle_ray_weights: Any = None
    affected_ray_ids: Any = None


@dataclass
class M5DSimulationDiagnostics:
    """Scalar identity checks and run statistics for notebook display.

    Identity errors are relative Euclidean (L2) norms. **Pass** thresholds are enforced
    inside the builder (for example linearity_error < 1e-8).

    Attributes:
        source_delta_identity_error: ‖Δ_transport − (Δ_bg + Δ_scat)‖ / ref.
        source_reconstruction_error: ‖E_particle − (E_clean + Δ)‖ / ref.
        S_reconstruction_error: ‖S_particle − (S_clean + S_delta)‖ / ref.
        linearity_error: Diffusion superposition check on nodal fluence field ``Phi``.
        n_source_rays, n_refracted, n_segments, n_transport_paths: Counts.
        n_affected_paths, affected_fraction: Particle-hit path statistics.
        source_assignment, source_distribution: Deposition metadata.
        operator_key_hash: Cached diffusion operator identifier.
        residual_clean, residual_particle, residual_delta: FEM residual norms.
    """

    source_delta_identity_error: float
    source_reconstruction_error: float
    S_reconstruction_error: float
    linearity_error: float

    n_source_rays: int
    n_refracted: int
    n_segments: int
    n_transport_paths: int
    n_affected_paths: int
    affected_fraction: float

    source_assignment: str
    source_distribution: Any

    operator_key_hash: Any
    residual_clean: float
    residual_particle: float
    residual_delta: float


@dataclass
class M5DSimulationResult:
    """Full bundle returned by :func:`run_m5d_simulation`.

    Attributes:
        config: Input configuration snapshot.
        scene: Meshes, segments, and camera maps.
        source: Deposition and source-density fields.
        diffusion: Fluence solves and linearity diagnostic.
        camera: Sampled images (diffuse-only or hybrid).
        diagnostics: Scalar checks and run statistics.
    """

    config: M5DSimulationConfig
    scene: M5DScene
    source: M5DSourceState
    diffusion: M5DDiffusionState
    camera: M5DCameraState
    diagnostics: M5DSimulationDiagnostics


def source_active_element_mask(
    E_clean_elem,
    E_particle_elem,
    delta_E_background_elem,
    delta_E_particle_scat_elem,
    delta_E_transport_elem,
):
    """Return elements with nonzero clean, particle, or delta source energy.

    See also the plotting helper of the same name under
    ``gummybear_validation.plotting``.

    Returns:
        numpy.ndarray: Boolean mask, length ``num_elements``.
    """
    E_clean_elem = np.asarray(E_clean_elem, dtype=float)
    E_particle_elem = np.asarray(E_particle_elem, dtype=float)
    delta_E_background_elem = np.asarray(delta_E_background_elem, dtype=float)
    delta_E_particle_scat_elem = np.asarray(
        delta_E_particle_scat_elem,
        dtype=float,
    )
    delta_E_transport_elem = np.asarray(delta_E_transport_elem, dtype=float)

    return (
        (np.abs(E_clean_elem) > 0.0)
        | (np.abs(E_particle_elem) > 0.0)
        | (np.abs(delta_E_background_elem) > 0.0)
        | (np.abs(delta_E_particle_scat_elem) > 0.0)
        | (np.abs(delta_E_transport_elem) > 0.0)
    )


def relative_norm(a, b):
    """Relative Euclidean (L2) error ``‖a − b‖ / max(‖b‖, ε)`` for identity checks.

    Args:
        a: Observed array-like vector.
        b: Reference array-like vector.

    Returns:
        float: Relative error suitable for diagnostics (not an assert).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denominator = max(
        float(np.linalg.norm(b)),
        np.finfo(float).eps,
    )

    return float(np.linalg.norm(a - b) / denominator)


def particle_mu_total(particle):
    """Return total extinction coefficient (absorption plus scatter) for a sphere.

    Uses ``particle.mu_total`` when present; otherwise sums ``mu_abs`` and
    ``mu_scat``.
    """
    if hasattr(particle, "mu_total"):
        return float(particle.mu_total)

    return float(particle.mu_abs) + float(particle.mu_scat)


def load_or_build_meshes(config):
    """Load or reuse surface and diffusion meshes from a simulation config.

    Requires ``netgen_mesh`` on the diffusion mesh for downstream finite-element method (FEM) solves.

    Args:
        config: :class:`M5DSimulationConfig` with ``stl_path`` and/or meshes.

    Returns:
        tuple: ``(surface_mesh, diff_mesh)``.

    Raises:
        ValueError: When required paths/meshes are missing or finite-element method (FEM) handle absent.
    """
    stl_path = None

    if config.stl_path is not None:
        stl_path = Path(config.stl_path)

    if config.surface_mesh is None:
        if stl_path is None:
            raise ValueError("Provide stl_path or surface_mesh.")

        surface_mesh = load_stl(stl_path)
    else:
        surface_mesh = config.surface_mesh

    if config.diff_mesh is None:
        if stl_path is None:
            raise ValueError("Provide stl_path or diff_mesh.")

        diff_mesh = generate_diffusion_mesh(
            stl_path,
            target_elements=config.target_elements,
        )
    else:
        diff_mesh = config.diff_mesh

    if diff_mesh.netgen_mesh is None:
        raise ValueError(
            "M5D diffusion solve requires a live netgen_mesh handle."
        )

    return surface_mesh, diff_mesh


def resolve_particles(particles, diff_mesh):
    """Normalize particles to a validated :class:`ParticleSet`.

    Accepts a ``ParticleSet``, a sequence of spheres, or a callable
    ``(diff_mesh) → …`` used by notebooks for placement callbacks.

    Args:
        particles: Particle specification from config.
        diff_mesh: Diffusion mesh passed to callables.

    Returns:
        ParticleSet: Overlap-validated particle group.
    """
    if callable(particles):
        resolved = particles(diff_mesh)
    else:
        resolved = particles

    if isinstance(resolved, ParticleSet):
        return resolved.require_valid()
    if isinstance(resolved, (list, tuple)):
        return ParticleSet.from_particles(resolved)
    return resolved


def build_m5d_scene(config):
    """Construct transport segments and camera hit maps for one simulation config.

    Runs source-ray sampling, refraction, in-object segmentation, and camera
    first-surface visibility. **Pass:** at least one in-object segment exists
    and segment ``ray_ids`` map to source-ray indices.

    Args:
        config: Simulation configuration.

    Returns:
        M5DScene: Scene bundle for source/deposition and camera sampling.

    Raises:
        AssertionError: When transport or ray-id invariants fail.
        ValueError: When meshes cannot be built.
    """
    surface_mesh, diff_mesh = load_or_build_meshes(config)
    particles = resolve_particles(config.particles, diff_mesh)

    source_params = SourceSamplingParams(
        mode=config.source_sampling_mode,
        n_rays=config.n_source_rays,
        seed=config.ray_seed,
    )

    source_rays = make_source_ray_bundle(
        config.light,
        surface_mesh.bounds,
        source_params,
    )

    refracted = refract_ray_bundle(
        surface_mesh,
        source_rays,
        n_from=config.n_from,
        n_to=config.material.n_refractive,
    )

    segments = in_object_segments_from_rays(
        surface_mesh,
        refracted.rays,
        parent_ray_ids=refracted.parent_indices,
    )

    camera_rays = make_camera_rays(config.camera)
    H, W = camera_rays.sample_shape

    cam_valid, cam_depth, cam_faces, cam_points = first_visible_hits_with_points(
        surface_mesh,
        camera_rays,
    )

    camera_mask = cam_valid.reshape(H, W)
    hit_faces_img = cam_faces.reshape(H, W)
    view_directions = -camera_rays.directions.reshape(H, W, 3)

    if segments.n_segments <= 0:
        raise AssertionError("No in-object transport segments were created.")

    expected_ray_ids = np.arange(source_rays.n_rays)

    if not np.all(np.isin(segments.ray_ids, expected_ray_ids)):
        raise AssertionError(
            "Segment ray_ids do not map to source-ray ids."
        )

    return M5DScene(
        surface_mesh=surface_mesh,
        diff_mesh=diff_mesh,
        material=config.material,
        light=config.light,
        camera=config.camera,
        particles=particles,
        source_rays=source_rays,
        refracted=refracted,
        segments=segments,
        camera_rays=camera_rays,
        H=H,
        W=W,
        cam_valid=cam_valid,
        cam_depth=cam_depth,
        cam_faces=cam_faces,
        cam_points=cam_points,
        camera_mask=camera_mask,
        hit_faces_img=hit_faces_img,
        view_directions=view_directions,
    )


def build_m5d_source_state(config, scene):
    """Deposit clean and particle-altered sources and verify split identities.

    **Pass:** at least one affected pair exists; transport delta equals
    background plus scatter; ``S_particle == S_clean + S_delta``; assignment
    mode matches config; ``S_particle`` is nonnegative within tolerance.

    Args:
        config: Simulation configuration (assignment mode).
        scene: Built scene with segments and particles.

    Returns:
        M5DSourceState: Deposition products and active-element mask.

    Raises:
        AssertionError: When bookkeeping identities or assignment checks fail.
    """
    clean_deposition = deposit_ray_source(
        scene.diff_mesh,
        scene.segments,
        material=scene.material,
    )

    pair_result = build_affected_transport_pairs(
        scene.segments,
        scene.particles,
        material=scene.material,
    )

    source_delta = compute_transport_source_correction(
        scene.diff_mesh,
        pair_result,
        clean_deposition.E_scat_elem,
        material=scene.material,
        assignment=config.source_delta_assignment,
    )

    S_clean = np.asarray(clean_deposition.S_clean, dtype=float)
    S_particle = np.asarray(source_delta.S_particle, dtype=float)

    volumes = np.asarray(scene.diff_mesh.volumes, dtype=float)
    S_delta = np.asarray(source_delta.delta_E_transport_elem, dtype=float)
    S_delta = S_delta / volumes

    if not pair_result.pairs:
        raise AssertionError(
            "Challenge particle did not intersect any transport paths."
        )

    expected_delta = (
        source_delta.delta_E_background_elem
        + source_delta.delta_E_particle_scat_elem
    )

    if not np.allclose(source_delta.delta_E_transport_elem, expected_delta):
        raise AssertionError("Source-delta split identity failed.")

    expected_E_particle = (
        source_delta.E_clean_elem
        + source_delta.delta_E_transport_elem
    )

    if not np.allclose(source_delta.E_particle_elem, expected_E_particle):
        raise AssertionError(
            "Particle source reconstruction identity failed."
        )

    if not np.allclose(S_particle, S_clean + S_delta):
        raise AssertionError("S_particle is not S_clean plus S_delta.")

    assignment_mode = source_delta.particle_scatter_deposition.assignment_mode

    if assignment_mode != config.source_delta_assignment:
        raise AssertionError("Unexpected particle-scatter assignment mode.")

    if np.min(S_particle) < -1e-12:
        raise AssertionError(
            "S_particle contains negative values below tolerance."
        )

    active_source_mask = source_active_element_mask(
        source_delta.E_clean_elem,
        source_delta.E_particle_elem,
        source_delta.delta_E_background_elem,
        source_delta.delta_E_particle_scat_elem,
        source_delta.delta_E_transport_elem,
    )

    return M5DSourceState(
        clean_deposition=clean_deposition,
        pair_result=pair_result,
        source_delta=source_delta,
        S_clean=S_clean,
        S_particle=S_particle,
        S_delta=S_delta,
        active_source_mask=active_source_mask,
    )


def timed_diffusion_solve(scene, config, source):
    """Run one diffusion solve and return result plus wall-clock seconds."""
    started = perf_counter()

    result = solve_diffusion(
        scene.diff_mesh,
        source,
        D=scene.material.diffusion_coefficient,
        mu_a=scene.material.mu_absorption,
        extrapolation_length=config.extrapolation_length,
    )

    seconds = perf_counter() - started

    return result, seconds


def build_m5d_diffusion_state(config, scene, source):
    """Solve diffusion for clean, particle, and delta sources.

    **Pass:** all three solves share one operator key; superposition
    ``Φ_particle − Φ_clean ≈ Φ_delta`` with relative error below ``1e-8``.

    Args:
        config: Simulation configuration (extrapolation length).
        scene: Scene with mesh and material.
        source: Built source state with ``S_clean``, ``S_particle``, ``S_delta``.

    Returns:
        M5DDiffusionState: Fluence fields, linearity error, and timings.

    Raises:
        AssertionError: When operator keys differ or linearity check fails.
    """
    clean_solve, clean_seconds = timed_diffusion_solve(
        scene,
        config,
        source.S_clean,
    )

    particle_solve, particle_seconds = timed_diffusion_solve(
        scene,
        config,
        source.S_particle,
    )

    delta_solve, delta_seconds = timed_diffusion_solve(
        scene,
        config,
        source.S_delta,
    )

    Phi_delta_nodes = particle_solve.Phi_nodes - clean_solve.Phi_nodes
    Phi_delta_tets = particle_solve.Phi_tets - clean_solve.Phi_tets

    denominator = max(
        float(np.linalg.norm(delta_solve.Phi_nodes)),
        np.finfo(float).eps,
    )

    linearity_error = float(
        np.linalg.norm(Phi_delta_nodes - delta_solve.Phi_nodes)
        / denominator
    )

    same_operator = (
        clean_solve.operator_cache_key
        == particle_solve.operator_cache_key
        == delta_solve.operator_cache_key
    )

    if not same_operator:
        raise AssertionError(
            "Clean, particle, and delta solves used different operators."
        )

    if linearity_error >= 1e-8:
        raise AssertionError("Diffusion linearity check failed.")

    return M5DDiffusionState(
        clean_solve=clean_solve,
        particle_solve=particle_solve,
        delta_solve=delta_solve,
        Phi_delta_nodes=Phi_delta_nodes,
        Phi_delta_tets=Phi_delta_tets,
        linearity_error=linearity_error,
        solve_seconds_clean=float(clean_seconds),
        solve_seconds_particle=float(particle_seconds),
        solve_seconds_delta=float(delta_seconds),
    )


def attenuated_direct_ray_weights(scene, source):
    """Apply Beer–Lambert attenuation along particle chords to source-ray weights.

    Multiplies each affected source-ray weight by ``exp(−μ_total * chord_length)``
    per particle event on that path.

    Args:
        scene: Scene with ``source_rays.weights`` and ``particles``.
        source: Source state exposing ``pair_result.pairs``.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Attenuated weights and sorted
        affected ray ids.
    """
    weights = np.asarray(scene.source_rays.weights, dtype=float).copy()
    affected_ray_ids = []

    for pair in source.pair_result.pairs:
        path_id = int(pair.path_id)

        if path_id < 0 or path_id >= len(weights):
            raise ValueError(
                "Affected pair path_id does not index source ray weights: "
                + str(path_id)
            )

        particle_factor = 1.0

        for event in pair.particle_events:
            particle = scene.particles[event.particle_index]
            mu_total = particle_mu_total(particle)
            length = float(event.path_length_inside_particle)
            particle_factor = particle_factor * np.exp(-mu_total * length)

        weights[path_id] = weights[path_id] * particle_factor
        affected_ray_ids.append(path_id)

    affected_ray_ids = np.asarray(
        sorted(set(affected_ray_ids)),
        dtype=int,
    )

    return weights, affected_ray_ids


def build_m5d_camera_state(config, scene, source, diffusion):
    """Sample diffuse camera images and optionally direct/hybrid channels.

    Diffuse delta is zero outside ``camera_mask``. When ``compute_direct`` is
    enabled, hybrid deltas must satisfy
    ``Δ_total ≈ alpha * Δ_direct + Δ_diffuse``.

    Args:
        config: Simulation configuration.
        scene: Scene with camera hits and masks.
        source: Source state (pair list for direct attenuation).
        diffusion: Fluence solves for clean and particle sources.

    Returns:
        M5DCameraState: Sampled images and optional direct/hybrid products.

    Raises:
        AssertionError: When mask or hybrid identity checks fail.
    """
    clean_diffuse = sample_diffuse_image(
        scene.diff_mesh,
        diffusion.clean_solve.Phi_nodes,
        scene.cam_points,
        scene.cam_valid,
        (scene.H, scene.W),
        exitance_scale=config.exitance_scale,
    )

    particle_diffuse = sample_diffuse_image(
        scene.diff_mesh,
        diffusion.particle_solve.Phi_nodes,
        scene.cam_points,
        scene.cam_valid,
        (scene.H, scene.W),
        exitance_scale=config.exitance_scale,
    )

    Delta_I_diffuse = (
        particle_diffuse.I_diffuse
        - clean_diffuse.I_diffuse
    )

    if not np.all(Delta_I_diffuse[~scene.camera_mask] == 0.0):
        raise AssertionError(
            "Diffuse camera delta is nonzero outside camera mask."
        )

    if not config.compute_direct:
        return M5DCameraState(
            clean_diffuse=clean_diffuse,
            particle_diffuse=particle_diffuse,
            Delta_I_diffuse=Delta_I_diffuse,
        )

    particle_ray_weights, affected_ray_ids = attenuated_direct_ray_weights(
        scene,
        source,
    )

    clean_direct = compute_refractive_direct_image(
        scene.surface_mesh,
        scene.source_rays,
        scene.material,
        scene.hit_faces_img,
        scene.view_directions,
        direct_scale=config.direct_scale,
        apply_attenuation=True,
        camera_mask=scene.camera_mask,
    )

    particle_direct = compute_refractive_direct_image(
        scene.surface_mesh,
        scene.source_rays,
        scene.material,
        scene.hit_faces_img,
        scene.view_directions,
        ray_weights=particle_ray_weights,
        direct_scale=config.direct_scale,
        apply_attenuation=True,
        camera_mask=scene.camera_mask,
    )

    clean_hybrid = compose_hybrid_image(
        clean_direct.I_direct,
        clean_diffuse.I_diffuse,
        alpha=config.alpha,
        camera_mask=scene.camera_mask,
    )

    particle_hybrid = compose_hybrid_image(
        particle_direct.I_direct,
        particle_diffuse.I_diffuse,
        alpha=config.alpha,
        camera_mask=scene.camera_mask,
    )

    Delta_I_direct = (
        particle_direct.I_direct
        - clean_direct.I_direct
    )

    Delta_I_total = (
        particle_hybrid.I_total
        - clean_hybrid.I_total
    )

    expected_total = (
        config.alpha * Delta_I_direct
        + Delta_I_diffuse
    )

    if not np.allclose(Delta_I_total, expected_total):
        raise AssertionError("Hybrid camera delta identity failed.")

    return M5DCameraState(
        clean_diffuse=clean_diffuse,
        particle_diffuse=particle_diffuse,
        Delta_I_diffuse=Delta_I_diffuse,
        clean_direct=clean_direct,
        particle_direct=particle_direct,
        Delta_I_direct=Delta_I_direct,
        clean_hybrid=clean_hybrid,
        particle_hybrid=particle_hybrid,
        Delta_I_total=Delta_I_total,
        particle_ray_weights=particle_ray_weights,
        affected_ray_ids=affected_ray_ids,
    )


def build_m5d_diagnostics(config, scene, source, diffusion):
    """Summarize identity errors, transport counts, and finite-element method (FEM) residuals.

    Does not raise; notebooks compare returned scalars to expected tolerances.

    Args:
        config: Simulation configuration.
        scene: Built scene.
        source: Source state.
        diffusion: Diffusion state with solve diagnostics.

    Returns:
        M5DSimulationDiagnostics: Scalar summary for logging and plots.
    """
    source_delta = source.source_delta

    expected_delta = (
        source_delta.delta_E_background_elem
        + source_delta.delta_E_particle_scat_elem
    )

    source_delta_identity_error = relative_norm(
        source_delta.delta_E_transport_elem,
        expected_delta,
    )

    expected_E_particle = (
        source_delta.E_clean_elem
        + source_delta.delta_E_transport_elem
    )

    source_reconstruction_error = relative_norm(
        source_delta.E_particle_elem,
        expected_E_particle,
    )

    S_reconstruction_error = relative_norm(
        source.S_particle,
        source.S_clean + source.S_delta,
    )

    n_paths = max(scene.segments.n_transport_paths, 1)
    n_affected = len(source.pair_result.pairs)

    particle_dep = source_delta.particle_scatter_deposition

    return M5DSimulationDiagnostics(
        source_delta_identity_error=source_delta_identity_error,
        source_reconstruction_error=source_reconstruction_error,
        S_reconstruction_error=S_reconstruction_error,
        linearity_error=diffusion.linearity_error,
        n_source_rays=int(scene.source_rays.n_rays),
        n_refracted=int(scene.refracted.n_refracted),
        n_segments=int(scene.segments.n_segments),
        n_transport_paths=int(scene.segments.n_transport_paths),
        n_affected_paths=int(n_affected),
        affected_fraction=float(n_affected / n_paths),
        source_assignment=particle_dep.assignment_mode,
        source_distribution=particle_dep.metadata.get("distribution"),
        operator_key_hash=diffusion.clean_solve.diagnostics.get(
            "operator_key_hash"
        ),
        residual_clean=float(diffusion.clean_solve.residual_norm),
        residual_particle=float(diffusion.particle_solve.residual_norm),
        residual_delta=float(diffusion.delta_solve.residual_norm),
    )


def run_m5d_simulation(config):
    """Run the full inspectable particle-transport pipeline for one configuration.

    Orchestrates scene build → source deposition → three diffusion solves →
    camera sampling → diagnostics. Does not plot. Raises when internal **pass**
    checks in the builders fail.

    Args:
        config: :class:`M5DSimulationConfig` describing phantom, optics, and
            particles.

    Returns:
        M5DSimulationResult: Scene, fields, images, and scalar diagnostics.

    Notebook / protocol: M5D

    See also:
        :func:`~gummybear.optics.source_deposition.deposit_ray_source` — clean volumetric source ``S``.
        :func:`~gummybear.optics.diffusion_solve.solve_diffusion` — fluence ``Phi`` solve.
        :func:`~gummybear.particles.perturbation.build_affected_transport_pairs` — particle transport pairs.
        :func:`~gummybear.optics.hybrid_compose.compose_hybrid_image` — direct + diffuse camera image.
    """
    scene = build_m5d_scene(config)
    source = build_m5d_source_state(config, scene)
    diffusion = build_m5d_diffusion_state(config, scene, source)
    camera = build_m5d_camera_state(config, scene, source, diffusion)
    diagnostics = build_m5d_diagnostics(config, scene, source, diffusion)

    return M5DSimulationResult(
        config=config,
        scene=scene,
        source=source,
        diffusion=diffusion,
        camera=camera,
        diagnostics=diagnostics,
    )