"""Export the M8 network-inference POV-Ray schematic."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import trimesh

from .anomaly_zscore import DEFAULT_ZSCORE_CLIP
from .export_m8_physical_scene import render_pov_file
from .load_sample import PhysicalSetup, load_m8_physical_setup
from .network_activations import NetworkActivationBundle, collect_m8_network_activations
from .network_captions import overlay_network_captions
from .network_pov_scene import build_network_pov_scene
from .network_texture_export import write_network_texture_pngs
from .paths import default_cad_dir, repo_root
from .pov_scene import bear_triangle_edge_cylinders
from .stl_to_mesh2 import write_stl_mesh2_inc


def default_network_pov(root: Path | None = None) -> Path:
    base = repo_root() if root is None else Path(root)
    return base / "outputs" / "pov" / "m8_network_scene.pov"


def _placeholder_cube_mesh() -> trimesh.Trimesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [3, 2, 6],
            [3, 6, 7],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype=int,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _edge_cylinders_for_mesh(mesh: trimesh.Trimesh) -> str:
    extents = np.asarray(mesh.extents, dtype=float).reshape(-1)
    span = float(np.max(extents)) if extents.size else 1.0
    return bear_triangle_edge_cylinders(
        mesh, radius=max(span * 0.0035, 0.05), transmit=0.88
    )


def _write_placeholder_bear_inc(inc_path: Path) -> tuple[np.ndarray, float, float, str]:
    """Unit cube mesh2 so schematic export still works without an STL."""
    mesh = _placeholder_cube_mesh()
    inc_path.write_text(
        "\n".join(
            [
                "#declare BearMesh = mesh2 {",
                "  vertex_vectors { 8,",
                "    <0,0,0>, <1,0,0>, <1,1,0>, <0,1,0>,",
                "    <0,0,1>, <1,0,1>, <1,1,1>, <0,1,1>",
                "  }",
                "  face_indices { 12,",
                "    <0,1,2>, <0,2,3>, <4,6,5>, <4,7,6>,",
                "    <0,4,5>, <0,5,1>, <3,2,6>, <3,6,7>,",
                "    <0,3,7>, <0,7,4>, <1,5,6>, <1,6,2>",
                "  }",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return np.array([0.5, 0.5, 0.5], dtype=float), 1.0, 0.0, _edge_cylinders_for_mesh(mesh)


def _resolve_network_bear(
    *,
    repo: Path,
    pov_path: Path,
    setup: PhysicalSetup | None,
) -> tuple[str, np.ndarray, float, float, str]:
    inc_path = pov_path.with_name(pov_path.stem + "_bear.inc")
    stl = None
    if setup is not None and Path(setup.stl_path).is_file():
        stl = Path(setup.stl_path)
    else:
        candidate = default_cad_dir(repo) / "proto_bear.stl"
        if candidate.is_file():
            stl = candidate
    if stl is None:
        centroid, extent, zmin, edges = _write_placeholder_bear_inc(inc_path)
        return inc_path.name, centroid, extent, zmin, edges
    mesh = trimesh.load(stl, force="mesh")
    write_stl_mesh2_inc(stl, inc_path)
    extents = np.asarray(mesh.extents, dtype=float).reshape(-1)
    return (
        inc_path.name,
        np.asarray(mesh.centroid, dtype=float).reshape(3),
        float(np.max(extents)),
        float(np.asarray(mesh.bounds, dtype=float)[0, 2]),
        _edge_cylinders_for_mesh(mesh),
    )


def export_m8_network_scene(
    sample_index: int = 0,
    camera_angle_deg: float = 180,
    output_pov: str | Path | None = "outputs/pov/m8_network_scene.pov",
    *,
    manifest_path: str | Path | None = None,
    data_root: str | Path | None = None,
    repo_root_path: str | Path | None = None,
    render: bool = False,
    setup: PhysicalSetup | None = None,
    activations: NetworkActivationBundle | None = None,
    checkpoint_path: str | Path | None = None,
    illustration_yaw_deg: float | None = None,
    illustration_fov_deg: float | None = None,
    illustration_distance_scale: float | None = None,
    illustration_zscore_clip: float | None = None,
    caption_positions: dict[str, tuple[float, float]] | None = None,
    caption_colors: dict[str, tuple[int, int, int]] | None = None,
    input_stack_center: tuple[float, float, float] | None = None,
    input_stack_scale: float | None = None,
    fourier_pane_yaw_deg: float | None = None,
    fourier_embed_offset: tuple[float, float, float] | None = None,
    fourier_embed_scale: float | None = None,
    fourier_group_offset: tuple[float, float, float] | None = None,
    gap_embed_yaw_deg: float | None = None,
    gap_embed_scale: float | None = None,
    gap_embed_shift: float | None = None,
    cnn_fourier_front: float | None = None,
    slice_colormap: str | None = None,
    slice_colormap_clip: float | None = None,
) -> dict[str, Path]:
    """Write the network-inference schematic. Optional POV-Ray + PNG captions."""
    repo = repo_root(repo_root_path) if repo_root_path is not None else repo_root()
    if output_pov is None:
        pov_path = default_network_pov(repo)
    else:
        pov_path = Path(output_pov)
        if not pov_path.is_absolute():
            pov_path = repo / pov_path
    pov_path.parent.mkdir(parents=True, exist_ok=True)
    clip = DEFAULT_ZSCORE_CLIP if illustration_zscore_clip is None else float(
        illustration_zscore_clip
    )
    loaded: PhysicalSetup | None = setup
    bundle = activations
    if bundle is None:
        if loaded is None:
            loaded = load_m8_physical_setup(
                sample_index=sample_index,
                camera_angle_deg=float(camera_angle_deg),
                manifest_path=manifest_path,
                data_root=data_root,
                repo_root_path=repo,
            )
        bundle = collect_m8_network_activations(
            loaded,
            repo_root_path=repo,
            checkpoint_path=checkpoint_path,
            angle_deg=float(camera_angle_deg),
        )
    textures = write_network_texture_pngs(
        bundle,
        pov_path.parent,
        stem=pov_path.stem,
        zscore_clip=clip,
        slice_colormap=(
            "minmax" if slice_colormap is None else str(slice_colormap)
        ),
        slice_colormap_clip=(
            2.0 if slice_colormap_clip is None else float(slice_colormap_clip)
        ),
    )
    bear_inc, bear_centroid, bear_extent, bear_zmin, bear_edges = _resolve_network_bear(
        repo=repo, pov_path=pov_path, setup=loaded
    )
    scene = build_network_pov_scene(
        bundle,
        textures,
        yaw_deg=80.0 if illustration_yaw_deg is None else float(illustration_yaw_deg),
        fov_deg=60.0 if illustration_fov_deg is None else float(illustration_fov_deg),
        distance_scale=(
            1.05
            if illustration_distance_scale is None
            else float(illustration_distance_scale)
        ),
        input_stack_center=input_stack_center,
        input_stack_scale=(
            1.0 if input_stack_scale is None else float(input_stack_scale)
        ),
        fourier_pane_yaw_deg=(
            0.0 if fourier_pane_yaw_deg is None else float(fourier_pane_yaw_deg)
        ),
        fourier_embed_offset=fourier_embed_offset,
        fourier_embed_scale=(
            1.0 if fourier_embed_scale is None else float(fourier_embed_scale)
        ),
        fourier_group_offset=fourier_group_offset,
        gap_embed_yaw_deg=(
            0.0 if gap_embed_yaw_deg is None else float(gap_embed_yaw_deg)
        ),
        gap_embed_scale=(
            1.0 if gap_embed_scale is None else float(gap_embed_scale)
        ),
        gap_embed_shift=(
            0.0 if gap_embed_shift is None else float(gap_embed_shift)
        ),
        cnn_fourier_front=(
            0.0 if cnn_fourier_front is None else float(cnn_fourier_front)
        ),
        bear_inc_name=bear_inc,
        bear_mesh_centroid=bear_centroid,
        bear_mesh_extent=bear_extent,
        particle_catalog_radius=(
            0.5 if loaded is None else float(loaded.particle_radius)
        ),
        bear_edge_solids=bear_edges,
        bear_mesh_zmin=bear_zmin,
    )
    pov_path.write_text(scene, encoding="utf-8")
    out: dict[str, Path] = {"pov": pov_path}
    if render:
        png = render_pov_file(pov_path)
        if png is not None:
            plain = png.with_name(f"{png.stem}_plain{png.suffix}")
            shutil.copy2(png, plain)
            overlay_network_captions(
                png,
                source=plain,
                positions=caption_positions,
                colors=caption_colors,
            )
            out["png_plain"] = plain
            out["png"] = png
    return out
