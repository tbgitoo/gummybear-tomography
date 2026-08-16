"""Export an M8 physical-setup POV-Ray scene from sample metadata."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from gummybear.paths import display_text_paths

from .anomaly_zscore import DEFAULT_ZSCORE_CLIP, write_anomaly_zscore_plates
from .caption_overlay import overlay_workflow_captions
from .load_sample import PhysicalSetup, load_m8_physical_setup
from .paths import default_output_pov, repo_root
from .pov_scene import IllustrationCameraParams, build_pov_scene
from .stl_to_mesh2 import write_stl_mesh2_inc


def povray_command(
    exe: str,
    pov_path: Path,
    png_path: Path,
    *,
    width: int,
    height: int,
) -> list[str]:
    """Unix POV-Ray 3.7 argv (Homebrew/macOS). Do not pass Windows ``/EXIT``."""
    return [
        exe,
        f"+I{pov_path}",
        f"+O{png_path.name}",
        f"+L{pov_path.parent}",
        f"+W{int(width)}",
        f"+H{int(height)}",
        "+FN",
        "-D",
    ]


def render_pov_file(
    pov_path: str | Path,
    png_path: str | Path | None = None,
    *,
    povray_bin: str = "povray",
    width: int = 1280,
    height: int = 960,
) -> Path | None:
    """Render ``.pov`` to PNG when POV-Ray is on PATH. Return None if missing.

    Export does not require POV-Ray. This helper is optional.

    Homebrew POV-Ray 3.7 I/O restrictions allow writing the *current*
    directory (and ``/tmp``), not arbitrary siblings. The process therefore
    ``cwd``s to the PNG directory and adds ``+L`` so the ``.inc`` mesh include
    still resolves. ``/EXIT`` is a Windows option and is not passed.
    """
    exe = shutil.which(povray_bin)
    if exe is None:
        return None
    pov_path = Path(pov_path)
    if png_path is None:
        png_path = pov_path.with_suffix(".png")
        if pov_path.parent.name == "pov":
            png_path = pov_path.parent.parent / "renders" / (pov_path.stem + ".png")
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = povray_command(exe, pov_path, png_path, width=width, height=height)
    proc = subprocess.run(
        cmd,
        cwd=str(png_path.parent),
        capture_output=True,
        text=True,
    )
    combined = f"{proc.stderr or ''}{proc.stdout or ''}"
    filtered = display_text_paths(combined)
    if filtered:
        sys.stderr.write(filtered if filtered.endswith("\n") else filtered + "\n")
    if proc.returncode:
        raise subprocess.CalledProcessError(
            proc.returncode,
            [display_text_paths(part) for part in cmd],
            output=display_text_paths(proc.stdout or ""),
            stderr=display_text_paths(proc.stderr or ""),
        )
    return png_path


def export_m8_physical_scene(
    sample_index: int = 0,
    camera_angle_deg: float = 180,
    output_pov: str | Path | None = "outputs/pov/m8_physical_scene.pov",
    *,
    manifest_path: str | Path | None = None,
    data_root: str | Path | None = None,
    repo_root_path: str | Path | None = None,
    render: bool = False,
    setup: PhysicalSetup | None = None,
    illustration_yaw_deg: float | None = None,
    illustration_fov_deg: float | None = None,
    illustration_distance_scale: float | None = None,
    illustration_n_rays: int | None = None,
    illustration_particle_radius_mm: float | None = None,
    illustration_light_cone_length_frac: float | None = None,
    illustration_camera_distance: float | None = None,
    illustration_light_distance: float | None = None,
    illustration_mesh_edge_radius_mm: float | None = None,
    illustration_orbit_cameras: int | None = None,
    illustration_orbit_step_deg: float | None = None,
    illustration_camera_rays: str | None = None,
    illustration_particle_light: float | None = None,
    illustration_anomaly_plates: bool | None = None,
    illustration_zscore_clip: float | None = None,
    illustration_orbit_fade: bool | None = None,
    illustration_inset_plate: bool | None = None,
    illustration_inset_stack: int | None = None,
    illustration_caption_views_xy: tuple[float, float] | None = None,
    illustration_caption_deep_learning_xy: tuple[float, float] | None = None,
    illustration_caption_localization_xy: tuple[float, float] | None = None,
) -> dict[str, Path]:
    """Write a POV-Ray 3.7 scene for one M8 sample's physical setup.

    Args:
        sample_index: Which discovered M8 manifest to use when
            ``manifest_path`` is omitted (0 = first).
        camera_angle_deg: Acquisition view to draw as the camera object
            (M8 single-view studies use 180).
        output_pov: Destination ``.pov`` path. Relative paths are resolved
            against the repository root.
        manifest_path: Optional explicit ``manifest.json``.
        data_root: Directory of sequence folders (default M8 generated root).
        repo_root_path: Override repository root (tests).
        render: If True, attempt ``povray`` after writing the scene.
        setup: Optional preloaded :class:`PhysicalSetup` (tests).
        illustration_yaw_deg: POV viewpoint yaw from behind the pinhole
            toward the side overview (degrees). ``None`` = package default.
        illustration_fov_deg: POV-Ray camera ``angle`` (degrees).
        illustration_distance_scale: Stand-off behind the pinhole, as a
            multiple of the pinhole–bear distance (larger = further back).
        illustration_n_rays: Illustration source-ray count (``0`` = no ray
            cylinders, only the catalog light marker). ``None`` = default.
        illustration_particle_radius_mm: Drawn particle-marker radius in mm.
            ``None`` = automatic from mesh span.
        illustration_light_cone_length_frac: Catalog-light cone length as a
            fraction of light-to-AABB-centre distance. Intensity fades to 0
            at that length. ``None`` = 0.5.
        illustration_camera_distance: Scale the drawn optical camera along
            its look-at ray (1 = catalog distance, same angle). Illustration
            only; does not change generation geometry.
        illustration_light_distance: Scale the drawn catalog light along the
            AABB-centre ray (1 = catalog distance). Illustration only.
        illustration_mesh_edge_radius_mm: Cylinder radius for unique STL
            triangle edges (mm). ``None`` = automatic from mesh span.
            ``0`` omits the wire overlay.
        illustration_orbit_cameras: Ghost optical cameras on each side of the
            illustrated pinhole, on a world-z orbit. ``None`` = 4. ``0`` = none.
        illustration_orbit_step_deg: Angular spacing of those ghosts (degrees).
            ``None`` = 20. Same z-axis radius as the illustrated camera.
        illustration_camera_rays: Frustum-edge cylinders. ``"all"`` (default)
            draws four rays on the main camera and every orbit ghost;
            ``"single"`` only the main camera; ``"none"`` draws no camera rays.
        illustration_particle_light: Green point-light intensity at the
            particle marker (``1`` = default local fill of the bear; ``0`` =
            glowing sphere only, no ``light_source``).
        illustration_anomaly_plates: Glue per-view z-score anomaly images on
            the back of each optical camera (opposite the lens). ``None``/True
            writes PNGs from ``anomaly_raw`` when present.
        illustration_zscore_clip: Symmetric clip of plate intensities in
            z-score units (``None`` = 2). Smaller values raise bear contrast.
        illustration_orbit_fade: If True (default), orbit cameras grow more
            transparent with angular step. If False, they stay opaque.
        illustration_inset_plate: Stack of z-score plates near the centre of
            the illustration view. Front plate is the optical camera whose
            back-plate appears largest; further plates are consecutive orbit
            views on thin camera-body blocks. ``None``/True when plates exist.
        illustration_inset_stack: Number of blocks in that stack (``None`` = 3).
        illustration_caption_views_xy: PNG overlay ``(x, y)`` as fractions of
            width/height (y from the top) for "single-view or multi-view input".
            ``None`` = package default.
        illustration_caption_deep_learning_xy: Same for two-line "Deep Learning".
        illustration_caption_localization_xy: Same for "3D localization".

    Returns:
        Dict with ``pov``, ``inc``, and optionally ``png`` / ``png_plain`` paths.
    """
    repo = repo_root(repo_root_path) if repo_root_path is not None else repo_root()
    if setup is None:
        setup = load_m8_physical_setup(
            sample_index=sample_index,
            camera_angle_deg=float(camera_angle_deg),
            manifest_path=manifest_path,
            data_root=data_root,
            repo_root_path=repo,
            n_illustration_rays=illustration_n_rays,
        )
    if output_pov is None:
        pov_path = default_output_pov(repo)
    else:
        pov_path = Path(output_pov)
        if not pov_path.is_absolute():
            pov_path = repo / pov_path
    pov_path.parent.mkdir(parents=True, exist_ok=True)
    inc_path = pov_path.with_name(pov_path.stem + "_bear.inc")
    write_stl_mesh2_inc(setup.stl_path, inc_path)
    cam_kwargs: dict[str, float] = {}
    if illustration_yaw_deg is not None:
        cam_kwargs["yaw_deg"] = float(illustration_yaw_deg)
    if illustration_fov_deg is not None:
        cam_kwargs["fov_deg"] = float(illustration_fov_deg)
    if illustration_distance_scale is not None:
        cam_kwargs["distance_scale"] = float(illustration_distance_scale)
    view = IllustrationCameraParams(**cam_kwargs) if cam_kwargs else None
    plates: dict[float, str] = {}
    if illustration_anomaly_plates is not False:
        plates = write_anomaly_zscore_plates(
            setup,
            pov_path.parent,
            stem=pov_path.stem,
            clip=(
                DEFAULT_ZSCORE_CLIP
                if illustration_zscore_clip is None
                else float(illustration_zscore_clip)
            ),
        )
    scene = build_pov_scene(
        setup,
        bear_inc_name=inc_path.name,
        illustration_camera_params=view,
        particle_radius_mm=illustration_particle_radius_mm,
        light_cone_length_frac=illustration_light_cone_length_frac,
        optical_camera_distance_scale=illustration_camera_distance,
        catalog_light_distance_scale=illustration_light_distance,
        mesh_edge_radius_mm=illustration_mesh_edge_radius_mm,
        orbit_cameras=illustration_orbit_cameras,
        orbit_step_deg=illustration_orbit_step_deg,
        camera_rays=illustration_camera_rays,
        particle_light=illustration_particle_light,
        anomaly_plates=plates,
        orbit_fade=illustration_orbit_fade,
        inset_plate=illustration_inset_plate,
        inset_stack=illustration_inset_stack,
    )
    pov_path.write_text(scene, encoding="utf-8")
    out: dict[str, Path] = {"pov": pov_path, "inc": inc_path}
    if render:
        png = render_pov_file(pov_path)
        if png is not None:
            if illustration_inset_plate is not False and plates:
                plain = png.with_name(f"{png.stem}_plain{png.suffix}")
                shutil.copy2(png, plain)
                overlay_workflow_captions(
                    png,
                    source=plain,
                    views_xy=illustration_caption_views_xy,
                    deep_learning_xy=illustration_caption_deep_learning_xy,
                    localization_xy=illustration_caption_localization_xy,
                )
                out["png_plain"] = plain
            out["png"] = png
    return out
