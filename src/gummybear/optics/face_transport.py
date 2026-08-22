"""Face-level source coverage, refractive transport, and direct camera sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh

from ..rays.source import SourceRayBundle
from ..rays.visibility import first_visible_hits, first_visible_hits_with_points
from .face_illumination import sample_face_values_to_image
from .material import OpticalMaterialConfig
from .refraction import refract_direction


@dataclass(frozen=True)
class FaceOpticalState:
    """Per-face transported energy and mean outgoing direction after source rays.

    ``face_energy[f]`` accumulates weighted ray energy deposited on face ``f``.
    ``b_out[f]`` is the energy-weighted mean outgoing (or incident baseline)
    unit direction. ``hit_count`` and ``valid`` record coverage statistics.

    Attributes
    ----------
    face_energy:
        Accumulated energy per face, shape ``[n_faces]``.
    b_out:
        Mean outgoing direction per face, shape ``[n_faces, 3]``.
    hit_count:
        Number of contributing rays per face, shape ``[n_faces]``.
    valid:
        ``True`` where ``hit_count > 0``, shape ``[n_faces]``.
    """

    face_energy: np.ndarray  # [n_faces]
    b_out: np.ndarray  # [n_faces, 3]
    hit_count: np.ndarray  # [n_faces]
    valid: np.ndarray  # [n_faces], bool

    @property
    def n_faces(self) -> int:
        """Number of mesh faces represented in this state."""
        return len(self.face_energy)

    def energy_density(self, face_areas: np.ndarray) -> np.ndarray:
        """Area-normalized energy ``face_energy / face_areas`` (diagnostic only).

        Faces with zero area return zero density.
        """
        face_areas = np.asarray(face_areas, dtype=float)
        return np.divide(
            self.face_energy,
            face_areas,
            out=np.zeros_like(self.face_energy),
            where=face_areas > 0,
        )


def _normalize_weighted_directions(
    direction_sum: np.ndarray,
    hit_count: np.ndarray,
) -> np.ndarray:
    """Reduce per-face accumulated ray directions to one unit direction.

    Multiple rays contributing to the same surface face may not have identical
    directions. This helper reconciles those contributions into one
    representative per-face direction for the face-level transport model.

    ``direction_sum`` contains accumulated, typically energy-weighted, direction
    contributions per face. ``hit_count`` is used to leave faces without contributing rays at zero. 
    The final
    normalization discards magnitude; only the representative direction is
    retained.
    """
    b_out = np.zeros_like(direction_sum, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        b_out = np.divide(
            direction_sum,
            hit_count[:, None],
            out=b_out,
            where=hit_count[:, None] > 0,
        )
        norms = np.linalg.norm(b_out, axis=-1, keepdims=True)
        b_out = np.divide(b_out, norms, out=b_out, where=norms > 0)
    return b_out


def accumulate_source_coverage(
    mesh: trimesh.Trimesh,
    source_rays: SourceRayBundle,
) -> FaceOpticalState:
    """Accumulate non-refracted source-ray hits into per-face energy and directions.

    Uses first-surface entry hits only. ``b_out`` is the energy-weighted mean
    *incident* source direction (baseline without Snell refraction). Ray
    weights come from ``source_rays.weights``.

    Parameters
    ----------
    mesh:
        Surface mesh for visibility tests.
    source_rays:
        Weighted source rays with ``origins``, ``directions``, ``weights``.

    Returns
    -------
    FaceOpticalState
        Entry-face coverage; ``valid`` marks faces with at least one hit.

    Notebook / protocol:
        M3 Stage 1 — non-refracted source coverage baseline.
    """
    n_faces = len(mesh.faces)
    source_valid, _source_depth, source_entry_faces = first_visible_hits(mesh, source_rays)

    face_energy = np.zeros(n_faces, dtype=float)
    hit_count = np.zeros(n_faces, dtype=np.int64)
    b_in_sum = np.zeros((n_faces, 3), dtype=float)

    active = source_valid & (source_entry_faces >= 0)
    faces = source_entry_faces[active]
    weights = source_rays.weights[active]
    dirs = source_rays.directions[active]

    np.add.at(face_energy, faces, weights)
    np.add.at(hit_count, faces, 1)
    np.add.at(b_in_sum, faces, dirs * weights[:, None])

    b_out = _normalize_weighted_directions(b_in_sum, hit_count)
    valid = hit_count > 0
    return FaceOpticalState(
        face_energy=face_energy,
        b_out=b_out,
        hit_count=hit_count,
        valid=valid,
    )


def sample_face_state_to_camera(
    state: FaceOpticalState,
    hit_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample exit-face energy and ``b_out`` onto camera ``hit_faces``.

    Requires camera first-surface ``hit_faces`` from a visibility pass (same
    shape as the target image). Background pixels receive zero energy and
    zero vectors; ``valid_image`` is false on misses and faces with no transport.

    Parameters
    ----------
    state:
        Per-face transport result (typically exit faces after refraction).
    hit_faces:
        Camera hit face index per pixel, shape ``[H, W]`` or flat ``[N]``.

    Returns
    -------
    energy_image:
        Sampled face energy, same shape as ``hit_faces``, background ``0.0``.
    b_out_image:
        Sampled outgoing directions, shape ``hit_faces.shape + (3,)``.
    valid_image:
        Boolean mask: hit face exists and ``state.valid[face]``.
    """
    hit_faces = np.asarray(hit_faces, dtype=int)
    energy_image = sample_face_values_to_image(
        hit_faces=hit_faces,
        face_values=state.face_energy,
        background_value=0.0,
    )
    b_out_image = np.zeros(hit_faces.shape + (3,), dtype=float)
    for c in range(3):
        b_out_image[..., c] = sample_face_values_to_image(
            hit_faces=hit_faces,
            face_values=state.b_out[:, c],
            background_value=0.0,
        )
    valid_face = sample_face_values_to_image(
        hit_faces=hit_faces,
        face_values=state.valid.astype(float),
        background_value=0.0,
    )
    valid_image = (hit_faces >= 0) & (valid_face > 0.5)
    return energy_image, b_out_image, valid_image


def refractive_exit_view_coupling(
    b_out_image: np.ndarray,
    view_directions: np.ndarray,
) -> np.ndarray:
    """Heuristic positive directional coupling between exit direction and camera view.

    Unit-normalizes ``b_out_image``  per pixel, then
    returns ``0.5 * (cos(b · v) + 1)``. This maps opposite directions to 0,
    perpendicular directions to 0.5, and aligned directions to 1.

    Assumes ``view_directions`` are already unit-normalized.    

    The factor is intentionally non-negative: an exit face should contribute
    less when its transported direction is poorly aligned with the camera, but
    it should not create negative intensity or act as an absorber in the direct
    channel. This is a bounded forward-contribution heuristic, not a
    full BRDF, Fresnel model, or radiometric emission model.

    Parameters
    ----------
    b_out_image:
        Outgoing directions, shape ``[..., 3]``.
    view_directions:
        Camera view directions, same leading shape, trailing dim ``3``.

    Returns
    -------
    np.ndarray
        Coupling in ``[0, 1]``, shape ``b_out_image.shape[:-1]``.
    """
    b_out = np.asarray(b_out_image, dtype=float)
    view = np.asarray(view_directions, dtype=float)
    if b_out.shape[:-1] != view.shape[:-1] or b_out.shape[-1] != 3 or view.shape[-1] != 3:
        raise ValueError(
            "b_out_image and view_directions must share leading shape and trailing dim 3"
        )
    b_norm = np.linalg.norm(b_out, axis=-1, keepdims=True)
    b_unit = b_out / np.maximum(b_norm, 1e-12)
    cos_bv = np.clip(np.sum(b_unit * view, axis=-1), -1.0, 1.0)
    return 0.5 * (cos_bv + 1.0)


@dataclass(frozen=True)
class RefractiveDirectImageResult:
    """Refractive direct-channel image and intermediate face/coupling fields.

    Attributes
    ----------
    I_direct:
        Direct camera intensity after coupling and scaling, shape ``[H, W]``.
    face_state:
        Exit-face :class:`FaceOpticalState` from transport.
    energy_image, coupling, face_valid:
        Sampled energy, view coupling, and validity on the camera grid.
    metadata:
        Pipeline bookkeeping (scale, attenuation flags, etc.).
    """

    I_direct: np.ndarray
    face_state: FaceOpticalState
    energy_image: np.ndarray
    coupling: np.ndarray
    face_valid: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "I_direct", np.asarray(self.I_direct, dtype=float))
        object.__setattr__(self, "energy_image", np.asarray(self.energy_image, dtype=float))
        object.__setattr__(self, "coupling", np.asarray(self.coupling, dtype=float))
        object.__setattr__(self, "face_valid", np.asarray(self.face_valid, dtype=bool))
        object.__setattr__(self, "metadata", dict(self.metadata))


def compute_refractive_direct_image(
    mesh: trimesh.Trimesh,
    source_rays: SourceRayBundle,
    material: OpticalMaterialConfig,
    hit_faces: np.ndarray,
    view_directions: np.ndarray,
    *,
    ray_weights: np.ndarray | None = None,
    direct_scale: float = 1.0,
    n_environment: float = 1.0,
    apply_attenuation: bool = True,
    camera_mask: np.ndarray | None = None,
) -> RefractiveDirectImageResult:
    """Form the refractive direct camera channel from source rays and hit faces.

    Pipeline:

    1. :func:`propagate_entry_exit_transport` — entry Snell refraction, internal
       traverse to exit face, exit refraction, optional Beer–Lambert on internal
       path length (uses ``ray_weights`` override when supplied).
    2. :func:`sample_face_state_to_camera` — sample exit energy and ``b_out``
       through ``hit_faces``.
    3. :func:`refractive_exit_view_coupling` and ``direct_scale``.

    ``hit_faces`` and ``view_directions`` must come from a camera visibility
    pass. For particle perturbations pass attenuated ``ray_weights`` here; do
    not rescale an already-rendered ``I_direct_clean`` in image space.

    Parameters
    ----------
    mesh:
        Surface mesh.
    source_rays:
        Weighted source rays.
    material:
        Refractive index and attenuation coefficients.
    hit_faces:
        Camera first-surface face indices, shape ``[H, W]``.
    view_directions:
        Mesh-to-camera unit directions, shape ``[H, W, 3]``.
    ray_weights:
        Optional per-ray weight override, shape ``[n_rays]``.
    direct_scale:
        Explicit scale on the direct channel (record in metadata).
    n_environment:
        Exterior refractive index (typically ``1.0`` for air).
    apply_attenuation:
        Apply ``exp(-mu_total * exit_depth)`` along internal chords.
    camera_mask:
        Optional boolean mask zeroing pixels outside the valid region.

    Returns
    -------
    RefractiveDirectImageResult

    Notebook / protocol:
        M3 Stage 3 direct channel; pairs with M4 diffuse sampling for hybrid compose.
    """
    face_state = propagate_entry_exit_transport(
        mesh,
        source_rays,
        material,
        n_environment=n_environment,
        apply_attenuation=apply_attenuation,
        ray_weights=ray_weights,
    )
    energy_img, b_out_img, face_valid_img = sample_face_state_to_camera(
        face_state, hit_faces
    )
    coupling = refractive_exit_view_coupling(b_out_img, view_directions)
    I_direct = float(direct_scale) * energy_img * coupling
    if camera_mask is not None:
        I_direct = np.where(camera_mask, I_direct, 0.0)
    I_direct = np.where(face_valid_img, I_direct, 0.0)

    return RefractiveDirectImageResult(
        I_direct=I_direct,
        face_state=face_state,
        energy_image=energy_img,
        coupling=coupling,
        face_valid=face_valid_img,
        metadata={
            "pipeline": "m3_entry_exit_refraction_sample_to_camera",
            "direct_scale": float(direct_scale),
            "apply_attenuation": bool(apply_attenuation),
            "used_ray_weight_override": ray_weights is not None,
        },
    )


def propagate_entry_exit_transport(
    mesh: trimesh.Trimesh,
    source_rays: SourceRayBundle,
    material: OpticalMaterialConfig,
    *,
    n_environment: float = 1.0,
    eps: float = 1e-6,
    apply_attenuation: bool = False,
    ray_weights: np.ndarray | None = None,
) -> FaceOpticalState:
    """Transport source rays through entry refraction, interior traverse, and exit refraction.

    For each source ray that hits an entry face:

    1. Refract at entry (``n_environment → material.n_refractive``).
    2. Cast an internal ray from ``entry_point + eps * internal_dir`` to the
       first exit hit.
    3. Refract at exit (``material.n_refractive → n_environment``).
    4. Accumulate exit-face energy and energy-weighted outgoing direction.

    This is a single-entry/single-exit direct-transport model.
    Each source ray contributes through one internal path between entry and exit.
    The model does not spawn secondary rays, for example from reflection,
    repeated refraction, or scattering events. Wave-optic effects such as
    Fresnel amplitude coefficients, phase, and interference are also outside
    this direct-channel approximation. Diffuse transport is modeled separately
    by the FEM diffusion solve.

    When ``apply_attenuation=True``, Beer–Lambert uses ``material.mu_total *
    exit_depth`` on the internal segment only. ``ray_weights`` optionally
    replaces ``source_rays.weights`` for this step (e.g. particle-attenuated
    weights); shape must be ``[n_rays]``.

    Parameters
    ----------
    mesh:
        Surface mesh.
    source_rays:
        Weighted source rays.
    material:
        Optical material for refraction and attenuation.
    n_environment:
        Exterior refractive index.
    eps:
        Internal origin offset to avoid self-hits (mesh units).
    apply_attenuation:
        Apply exponential attenuation along internal path length.
    ray_weights:
        Optional per-ray weight override.

    Returns
    -------
    FaceOpticalState
        Exit-face energy and ``b_out`` after refraction.

    Notebook / protocol:
        M3 Stage 3 entry/exit transport; alias :data:`compute_refracted_face_field`.
    """
    n_faces = len(mesh.faces)
    source_valid, _source_depth, source_entry_faces, entry_points = (
        first_visible_hits_with_points(mesh, source_rays)
    )

    weights_in = np.asarray(source_rays.weights, dtype=float)
    if ray_weights is not None:
        ray_weights = np.asarray(ray_weights, dtype=float)
        if ray_weights.shape != weights_in.shape:
            raise ValueError(
                f"ray_weights must be shape [{weights_in.shape[0]}], got {ray_weights.shape}"
            )
        weights_in = ray_weights

    internal_dirs = np.zeros_like(source_rays.directions)
    internal_valid = np.zeros(source_rays.n_rays, dtype=bool)

    for ray_idx in range(source_rays.n_rays):
        entry_face = int(source_entry_faces[ray_idx])
        if entry_face < 0 or not source_valid[ray_idx]:
            continue
        refracted, ok = refract_direction(
            direction=source_rays.directions[ray_idx],
            normal=mesh.face_normals[entry_face],
            n_from=n_environment,
            n_to=material.n_refractive,
        )
        if ok:
            internal_dirs[ray_idx] = refracted
            internal_valid[ray_idx] = True

    active = np.where(internal_valid)[0]
    if active.size == 0:
        return FaceOpticalState(
            face_energy=np.zeros(n_faces, dtype=float),
            b_out=np.zeros((n_faces, 3), dtype=float),
            hit_count=np.zeros(n_faces, dtype=np.int64),
            valid=np.zeros(n_faces, dtype=bool),
        )

    # Offset the internal ray origin slightly into the object to avoid
    # immediately re-hitting the entry face due to numerical precision.
    internal_origins = entry_points[active] + eps * internal_dirs[active]
    internal_bundle = SourceRayBundle(
        origins=internal_origins,
        directions=internal_dirs[active],
        weights=weights_in[active],
        sample_shape=None,
    )

    exit_valid, exit_depth, exit_faces = first_visible_hits(mesh, internal_bundle)

    exit_face_energy = np.zeros(n_faces, dtype=float)
    exit_hit_count = np.zeros(n_faces, dtype=np.int64)
    b_out_sum_exit = np.zeros((n_faces, 3), dtype=float)

    for local_idx in range(internal_bundle.n_rays):
        if not exit_valid[local_idx]:
            continue
        exit_f = int(exit_faces[local_idx])
        if exit_f < 0:
            continue

        outgoing, ok = refract_direction(
            direction=internal_bundle.directions[local_idx],
            normal=mesh.face_normals[exit_f],
            n_from=material.n_refractive,
            n_to=n_environment,
        )
        if not ok:
            continue

        # Optional Beer-Lambert attenuation along the single internal chord.
        w = float(internal_bundle.weights[local_idx])
        if apply_attenuation:
            path_length = float(exit_depth[local_idx])
            if np.isfinite(path_length) and path_length > 0.0:
                w = w * float(np.exp(-material.mu_total * path_length))

        exit_face_energy[exit_f] += w
        exit_hit_count[exit_f] += 1
        b_out_sum_exit[exit_f] += outgoing * w

    exit_b_out = _normalize_weighted_directions(b_out_sum_exit, exit_hit_count)
    return FaceOpticalState(
        face_energy=exit_face_energy,
        b_out=exit_b_out,
        hit_count=exit_hit_count,
        valid=exit_hit_count > 0,
    )


def propagate_source_rays(
    source_rays: SourceRayBundle,
    mesh: trimesh.Trimesh,
    material: OpticalMaterialConfig | None = None,
    *,
    refract: bool = True,
    apply_attenuation: bool = False,
    n_environment: float = 1.0,
) -> FaceOpticalState:
    """Dispatch source-side transport to coverage-only or refractive pipelines.

    Parameters
    ----------
    source_rays:
        Weighted source rays.
    mesh:
        Surface mesh.
    material:
        Required when ``refract=True``; ignored for coverage-only mode.
    refract:
        If ``False``, run non-refracted entry coverage only.
        If ``True``, run entry/exit refraction transport.
    apply_attenuation:
        Passed to :func:`propagate_entry_exit_transport` when refracting.
    n_environment:
        Exterior refractive index for refraction.

    Returns
    -------
    FaceOpticalState

    Raises
    ------
    ValueError
        If ``refract=True`` but ``material`` is ``None``.
    """
    if not refract:
        return accumulate_source_coverage(mesh, source_rays)

    if material is None:
        raise ValueError("material is required when refract=True")

    return propagate_entry_exit_transport(
        mesh=mesh,
        source_rays=source_rays,
        material=material,
        n_environment=n_environment,
        apply_attenuation=apply_attenuation,
    )


# Alias used in the M3 plan.
compute_refracted_face_field = propagate_entry_exit_transport
