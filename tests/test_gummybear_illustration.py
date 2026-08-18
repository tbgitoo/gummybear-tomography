"""POV-Ray illustration exporter: data-driven M8 physical setup scenes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gummybear_illustration import export_m8_physical_scene, load_m8_physical_setup
from gummybear_illustration.export_m8_physical_scene import povray_command, render_pov_file
from gummybear_illustration.load_sample import (
    fallback_illumination_rays,
    illustration_optical_ray_segments,
)


def _write_ascii_tet_stl(path: Path) -> None:
    """Minimal tetrahedron in simulation millimetres."""
    verts = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (0.0, 10.0, 0.0),
        (0.0, 0.0, 10.0),
    ]
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    lines = ["solid tet"]
    for i, j, k in faces:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        for idx in (i, j, k):
            x, y, z = verts[idx]
            lines.append(f"      vertex {x} {y} {z}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid tet")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest(
    *,
    sequence_id: str,
    stl_rel: str,
    include_camera_position: bool = True,
    angle_deg: float = 180.0,
) -> dict:
    frame = {
        "frame_index": 0,
        "angle_deg": angle_deg,
        "axis": [0.0, 0.0, 1.0],
        "camera_kind": "orbit",
        "distance": 80.0,
        "elevation_deg": 0.0,
        "look_at": [3.0, 3.0, 3.0],
        "up": [0.0, 0.0, 1.0],
        "resolution": [8, 8],
        "fov_deg": 35.0,
        "filenames": {},
    }
    if include_camera_position:
        frame["camera_position"] = [3.0, -77.0, 3.0]
    return {
        "schema_version": "1.6-m6-draft",
        "sequence_id": sequence_id,
        "phantom": {"phantom_id": "test_tet", "stl_path": stl_rel},
        "setups": {
            "optical": {
                "optical_setup_id": "opt_test",
                "light_position_x": 15.0,
                "light_position_y": 15.0,
                "light_position_z": 40.0,
                "refractive_index": 1.5,
            },
            "particles": {
                "particle_group_id": "g1",
                "count": 1,
                "items": [
                    {
                        "center_x": 2.0,
                        "center_y": 3.0,
                        "center_z": 4.0,
                        "radius": 0.8,
                    }
                ],
            },
        },
        "frames": [frame],
    }


def _write_sample(tmp: Path, repo: Path, *, include_camera_position: bool = True) -> Path:
    cad = repo / "cad"
    cad.mkdir(exist_ok=True)
    stl = cad / "test_illustration_tet.stl"
    _write_ascii_tet_stl(stl)
    seq = tmp / "seq_m8_000"
    seq.mkdir()
    man = _manifest(
        sequence_id="seq_m8_000",
        stl_rel="cad/test_illustration_tet.stl",
        include_camera_position=include_camera_position,
    )
    path = seq / "manifest.json"
    path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return path


def test_export_contains_metadata_markers(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    man = _write_sample(tmp_path, repo)
    out = repo / "outputs" / "pov" / "scene.pov"
    with pytest.warns(UserWarning, match="FALLBACK"):
        result = export_m8_physical_scene(
            sample_index=0,
            camera_angle_deg=180,
            output_pov=out,
            manifest_path=man,
            repo_root_path=repo,
        )
    text = result["pov"].read_text(encoding="utf-8")
    assert "#version 3.7;" in text
    assert "mesh2" in result["inc"].read_text(encoding="utf-8")
    assert "<2, 3, 4>" in text.replace(".00000", "") or "<2, 3, 4>" in text
    assert "15" in text and "40" in text
    assert "FALLBACK" in text
    assert "cylinder {" in text
    assert "plane { z," in text
    assert "rgbt <0.68, 0.69, 0.71, 0.72>" in text
    assert "Acquisition camera" in text
    assert "location <3, -77, 3>" not in text
    assert "Behind the M8" in text
    assert "angle 46" in text
    assert "ambient 0.95" in text
    assert "sky_sphere" in text
    assert "<0.08, 0.18, 0.42>" in text
    assert "ambient_light rgb <0.28, 0.28, 0.30>" in text
    assert "Scene illumination (illustration only)" in text
    assert "light_source { <15, 15, 40>" not in text
    assert "cone {" in text
    assert "Catalog illumination cone" in text
    assert "Bear triangle edges" in text
    assert "rgb <0.10, 0.10, 0.11>" in text
    assert text.count("rgb <0.10, 0.10, 0.11>") == 6
    assert "Tomography orbit cameras" in text


def test_illustration_camera_params_reach_pov(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    man = _write_sample(tmp_path, repo)
    out = repo / "outputs" / "pov" / "scene.pov"
    with pytest.warns(UserWarning, match="FALLBACK"):
        result = export_m8_physical_scene(
            sample_index=0,
            camera_angle_deg=180,
            output_pov=out,
            manifest_path=man,
            repo_root_path=repo,
            illustration_yaw_deg=25.0,
            illustration_fov_deg=51.0,
            illustration_distance_scale=2.0,
            illustration_camera_distance=2.5,
            illustration_light_distance=1.4,
        )
    text = result["pov"].read_text(encoding="utf-8")
    assert "angle 51" in text
    assert "yaw_deg=25" in text
    assert "distance_scale=2" in text
    assert "optical_camera_distance_scale=2.5" in text
    assert "catalog_light_distance_scale=1.4" in text
    with pytest.warns(UserWarning, match="FALLBACK"):
        result2 = export_m8_physical_scene(
            sample_index=0,
            camera_angle_deg=180,
            output_pov=out,
            manifest_path=man,
            repo_root_path=repo,
            illustration_light_cone_length_frac=0.25,
        )
    text2 = result2["pov"].read_text(encoding="utf-8")
    assert "length 0.25 of light-to-centre" in text2


def test_zero_illustration_rays_skips_fallback(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    man = _write_sample(tmp_path, repo)
    setup = load_m8_physical_setup(
        manifest_path=man,
        repo_root_path=repo,
        camera_angle_deg=180,
        n_illustration_rays=0,
    )
    assert setup.illumination_rays == ()
    assert setup.refracted_rays == ()
    assert setup.illumination_rays_are_fallback is False


def test_particle_marker_radius_written(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    man = _write_sample(tmp_path, repo)
    out = repo / "outputs" / "pov" / "scene.pov"
    with pytest.warns(UserWarning, match="FALLBACK"):
        result = export_m8_physical_scene(
            sample_index=0,
            camera_angle_deg=180,
            output_pov=out,
            manifest_path=man,
            repo_root_path=repo,
            illustration_particle_radius_mm=1.25,
        )
    text = result["pov"].read_text(encoding="utf-8")
    assert "sphere { <2, 3, 4>, 1.25" in text
    assert "Particle illumination light_source" in text
    assert "light_source {\n  <2, 3, 4>" in text
    with pytest.warns(UserWarning, match="FALLBACK"):
        off = export_m8_physical_scene(
            sample_index=0,
            camera_angle_deg=180,
            output_pov=out,
            manifest_path=man,
            repo_root_path=repo,
            illustration_particle_light=0.0,
        )
    off_text = off["pov"].read_text(encoding="utf-8")
    assert "Particle light_source omitted" in off_text
    assert "Particle illumination light_source" not in off_text


def test_mesh_edge_cylinder_radius(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    man = _write_sample(tmp_path, repo)
    out = repo / "outputs" / "pov" / "scene.pov"
    with pytest.warns(UserWarning, match="FALLBACK"):
        result = export_m8_physical_scene(
            sample_index=0,
            camera_angle_deg=180,
            output_pov=out,
            manifest_path=man,
            repo_root_path=repo,
            illustration_mesh_edge_radius_mm=0.17,
        )
    text = result["pov"].read_text(encoding="utf-8")
    assert ", 0.17" in text
    assert "Bear triangle edges" in text
    with pytest.warns(UserWarning, match="FALLBACK"):
        result0 = export_m8_physical_scene(
            sample_index=0,
            camera_angle_deg=180,
            output_pov=out,
            manifest_path=man,
            repo_root_path=repo,
            illustration_mesh_edge_radius_mm=0.0,
        )
    text0 = result0["pov"].read_text(encoding="utf-8")
    assert "omitted (radius 0)" in text0
    assert "rgb <0.10, 0.10, 0.11>" not in text0


def test_tomography_orbit_cameras(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    man = _write_sample(tmp_path, repo)
    out = repo / "outputs" / "pov" / "scene.pov"
    with pytest.warns(UserWarning, match="FALLBACK"):
        result = export_m8_physical_scene(
            sample_index=0,
            camera_angle_deg=180,
            output_pov=out,
            manifest_path=man,
            repo_root_path=repo,
            illustration_orbit_cameras=2,
            illustration_orbit_step_deg=30,
        )
    text = result["pov"].read_text(encoding="utf-8")
    assert "orbit_cameras=2" in text
    assert "orbit_step_deg=30" in text
    assert "orbit angle_deg=30" in text
    assert "orbit angle_deg=-30" in text
    assert "orbit angle_deg=60" in text
    assert "orbit angle_deg=-60" in text
    assert "z-axis radius preserved" in text
    assert text.count("orbit angle_deg=") == 4
    with pytest.warns(UserWarning, match="FALLBACK"):
        result0 = export_m8_physical_scene(
            sample_index=0,
            camera_angle_deg=180,
            output_pov=out,
            manifest_path=man,
            repo_root_path=repo,
            illustration_orbit_cameras=0,
        )
    assert "omitted (count 0)" in result0["pov"].read_text(encoding="utf-8")


def test_anomaly_zscore_plates_on_camera_back(tmp_path: Path):
    from gummybear.datasets.role_images import write_float_raw_tif

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    man = _write_sample(tmp_path, repo)
    seq = man.parent
    anomaly_dir = seq / "anomaly"
    anomaly_dir.mkdir()
    blob = np.zeros((8, 8), dtype=np.float32)
    blob[3:5, 3:5] = 4.0
    write_float_raw_tif(anomaly_dir / "a180.raw.tif", blob)
    blob2 = np.zeros((8, 8), dtype=np.float32)
    blob2[1:3, 1:3] = -3.0
    write_float_raw_tif(anomaly_dir / "a210.raw.tif", blob2)
    blob3 = np.zeros((8, 8), dtype=np.float32)
    blob3[5:7, 2:4] = 5.0
    write_float_raw_tif(anomaly_dir / "a150.raw.tif", blob3)
    payload = json.loads(man.read_text(encoding="utf-8"))
    payload["frames"][0]["filenames"] = {
        "anomaly_raw": "anomaly/a180.raw.tif",
    }
    frame210 = dict(payload["frames"][0])
    frame210["frame_index"] = 1
    frame210["angle_deg"] = 210.0
    frame210["filenames"] = {"anomaly_raw": "anomaly/a210.raw.tif"}
    payload["frames"].append(frame210)
    frame150 = dict(payload["frames"][0])
    frame150["frame_index"] = 2
    frame150["angle_deg"] = 150.0
    frame150["filenames"] = {"anomaly_raw": "anomaly/a150.raw.tif"}
    payload["frames"].append(frame150)
    man.write_text(json.dumps(payload), encoding="utf-8")
    out = repo / "outputs" / "pov" / "scene.pov"
    with pytest.warns(UserWarning, match="FALLBACK"):
        result = export_m8_physical_scene(
            sample_index=0,
            camera_angle_deg=180,
            output_pov=out,
            manifest_path=man,
            repo_root_path=repo,
            illustration_orbit_cameras=1,
            illustration_orbit_step_deg=30,
        )
    text = result["pov"].read_text(encoding="utf-8")
    assert 'png "scene_zscore_180.00deg.png"' in text
    assert 'png "scene_zscore_210.00deg.png"' in text
    assert 'png "scene_zscore_150.00deg.png"' in text
    assert "3 blocks" in text
    assert "world-upright" in text
    assert "pointing right in the POV view" in text
    assert "localization task" in text
    assert "Localization xy plate" in text
    assert "shadowless" in text
    assert "single-view or multi-view input" not in text
    assert "single or multiple views" not in text
    assert "Deep Learning" not in text
    assert "3D localization" not in text
    assert not (out.parent / "scene_caption_deep_learning.png").is_file()
    assert not (out.parent / "scene_caption_views.png").is_file()
    assert not (out.parent / "scene_caption_localization.png").is_file()
    assert (out.parent / "scene_zscore_180.00deg.png").is_file()
    assert (out.parent / "scene_zscore_210.00deg.png").is_file()
    assert "uv_vectors { 4, <1,0>, <0,0>, <0,1>, <1,1> }" in text
    with pytest.warns(UserWarning, match="FALLBACK"):
        off = export_m8_physical_scene(
            sample_index=0,
            camera_angle_deg=180,
            output_pov=out,
            manifest_path=man,
            repo_root_path=repo,
            illustration_anomaly_plates=False,
        )
    assert "image_map" not in off["pov"].read_text(encoding="utf-8")
    assert "Inset z-score plate omitted" in off["pov"].read_text(encoding="utf-8")
    with pytest.warns(UserWarning, match="FALLBACK"):
        faded = export_m8_physical_scene(
            sample_index=0,
            camera_angle_deg=180,
            output_pov=out,
            manifest_path=man,
            repo_root_path=repo,
            illustration_orbit_cameras=1,
            illustration_orbit_step_deg=30,
            illustration_orbit_fade=False,
        )
    faded_text = faded["pov"].read_text(encoding="utf-8")
    assert "Orbit fade off" in faded_text
    assert "transmit=0" in faded_text
    assert "Inset z-score plate stack (illustration)" in text
    with pytest.warns(UserWarning, match="FALLBACK"):
        no_inset = export_m8_physical_scene(
            sample_index=0,
            camera_angle_deg=180,
            output_pov=out,
            manifest_path=man,
            repo_root_path=repo,
            illustration_inset_plate=False,
        )
    assert "Inset z-score plate stack (illustration)" not in no_inset["pov"].read_text(
        encoding="utf-8"
    )


def test_zscore_gray_clip_saturates():
    from gummybear_illustration.anomaly_zscore import zscore_to_gray_uint8

    z = np.array([[-10.0, 0.0, 10.0]], dtype=float)
    rgb = zscore_to_gray_uint8(z, clip=2.0)
    assert rgb[0, 0, 0] == 0
    assert rgb[0, 1, 0] in {127, 128}
    assert rgb[0, 2, 0] == 255


def test_camera_rays_modes(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    man = _write_sample(tmp_path, repo)
    out = repo / "outputs" / "pov" / "scene.pov"
    kwargs = dict(
        sample_index=0,
        camera_angle_deg=180,
        output_pov=out,
        manifest_path=man,
        repo_root_path=repo,
        illustration_orbit_cameras=1,
        illustration_orbit_step_deg=30,
    )
    with pytest.warns(UserWarning, match="FALLBACK"):
        all_text = export_m8_physical_scene(
            **kwargs, illustration_camera_rays="all"
        )["pov"].read_text(encoding="utf-8")
    assert "camera_rays=all" in all_text
    assert "Camera frustum edges from illustrated pinhole" in all_text
    assert "Orbit camera frustum edges (camera_rays=all)" in all_text
    with pytest.warns(UserWarning, match="FALLBACK"):
        single_text = export_m8_physical_scene(
            **kwargs, illustration_camera_rays="single"
        )["pov"].read_text(encoding="utf-8")
    assert "camera_rays=single" in single_text
    assert "Camera frustum edges from illustrated pinhole" in single_text
    assert "Orbit camera frustum edges" not in single_text
    with pytest.warns(UserWarning, match="FALLBACK"):
        none_text = export_m8_physical_scene(
            **kwargs, illustration_camera_rays="none"
        )["pov"].read_text(encoding="utf-8")
    assert "camera_rays=none" in none_text
    assert "Camera frustum omitted (camera_rays=none)" in none_text
    assert "Camera frustum edges from illustrated pinhole" not in none_text
    assert "Orbit camera frustum edges" not in none_text


def test_particle_and_light_from_manifest(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    man = _write_sample(tmp_path, repo)
    with pytest.warns(UserWarning, match="FALLBACK"):
        setup = load_m8_physical_setup(
            manifest_path=man,
            repo_root_path=repo,
            camera_angle_deg=180,
        )
    assert setup.particle_center.tolist() == [2.0, 3.0, 4.0]
    assert setup.particle_radius == pytest.approx(0.8)
    assert setup.light_position.tolist() == [15.0, 15.0, 40.0]
    assert setup.camera_position.tolist() == [3.0, -77.0, 3.0]
    assert setup.illumination_rays_are_fallback
    assert len(setup.illumination_rays) >= 1
    assert setup.illumination_rays[0][0].tolist() == [15.0, 15.0, 40.0]
    assert len(setup.refracted_rays) >= 1


def test_real_ray_sidecar_disables_fallback(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    man = _write_sample(tmp_path, repo)
    sidecar = man.parent / "illustration_source_rays.json"
    sidecar.write_text(
        json.dumps({"segments": [[[15.0, 15.0, 40.0], [2.0, 3.0, 4.0]]]}),
        encoding="utf-8",
    )
    setup = load_m8_physical_setup(
        manifest_path=man,
        repo_root_path=repo,
        camera_angle_deg=180,
    )
    assert setup.illumination_rays_are_fallback is False
    assert len(setup.illumination_rays) == 1


def test_reconstruct_camera_when_position_missing(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    man = _write_sample(tmp_path, repo, include_camera_position=False)
    with pytest.warns(UserWarning):
        setup = load_m8_physical_setup(
            manifest_path=man,
            repo_root_path=repo,
            camera_angle_deg=180,
        )
    assert setup.camera_position.shape == (3,)
    assert abs(float(setup.camera_position[1])) > 10.0


def test_missing_particle_raises(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    man_path = _write_sample(tmp_path, repo)
    payload = json.loads(man_path.read_text(encoding="utf-8"))
    del payload["setups"]["particles"]
    man_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="particle"):
        load_m8_physical_setup(manifest_path=man_path, repo_root_path=repo)


def test_fallback_rays_are_deterministic(tmp_path: Path):
    import numpy as np
    from gummybear.geometry.io import load_stl
    from gummybear.optics.light_source import PointLightConfig

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    man = _write_sample(tmp_path, repo)
    mesh = load_stl(repo / "cad" / "test_illustration_tet.stl")
    light = PointLightConfig(position=(15.0, 15.0, 40.0))
    a = fallback_illumination_rays(light=light, mesh=mesh, n_to=1.5)
    b = fallback_illumination_rays(light=light, mesh=mesh, n_to=1.5)
    assert len(a) >= 1
    assert a[0][0].tolist() == [15.0, 15.0, 40.0]
    assert a[0][1].tolist() == b[0][1].tolist()
    ends = np.stack([seg[1] for seg in a], axis=0)
    assert float(np.linalg.norm(ends.max(axis=0) - ends.min(axis=0))) > 0.5
    _illum, refracted = illustration_optical_ray_segments(
        light=light, mesh=mesh, n_to=1.5
    )
    assert len(refracted) >= 1


def test_warning_location_uses_display_path(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    man = _write_sample(tmp_path, repo)
    with pytest.warns(UserWarning) as recorded:
        load_m8_physical_setup(
            manifest_path=man,
            repo_root_path=repo,
            camera_angle_deg=180,
        )
    filename = str(recorded[0].filename)
    assert not filename.startswith("/Users/")
    from gummybear.paths import display_path

    assert filename == display_path(filename) or not Path(filename).is_absolute()


def test_render_helper_returns_none_without_povray(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    assert render_pov_file(tmp_path / "x.pov") is None


def test_povray_command_is_unix_style(tmp_path: Path):
    pov = tmp_path / "scene.pov"
    png = tmp_path / "renders" / "scene.png"
    cmd = povray_command("/opt/homebrew/bin/povray", pov, png, width=128, height=96)
    assert "/EXIT" not in cmd
    assert f"+O{png.name}" in cmd
    assert f"+L{pov.parent}" in cmd
    assert f"+I{pov}" in cmd


def test_povray_command_work_threads(tmp_path: Path):
    pov = tmp_path / "scene.pov"
    png = tmp_path / "renders" / "scene.png"
    cmd = povray_command(
        "/opt/homebrew/bin/povray",
        pov,
        png,
        width=128,
        height=96,
        work_threads=1,
    )
    assert "+WT1" in cmd


def test_discover_sample_index(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    data = tmp_path / "data"
    man = _write_sample(tmp_path, repo)
    # Move sequence under data_root
    dest = data / "seq_m8_000"
    dest.mkdir(parents=True)
    (dest / "manifest.json").write_text(man.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.warns(UserWarning, match="FALLBACK"):
        setup = load_m8_physical_setup(
            sample_index=0,
            data_root=data,
            repo_root_path=repo,
            camera_angle_deg=180,
        )
    assert setup.sequence_id == "seq_m8_000"


def test_overlay_workflow_captions(tmp_path: Path):
    from PIL import Image

    from gummybear_illustration.caption_overlay import (
        DEEP_LEARNING_XY,
        LOCALIZATION_XY,
        VIEWS_XY,
        overlay_workflow_captions,
    )

    path = tmp_path / "scene.png"
    Image.new("RGB", (1280, 960), (255, 255, 255)).save(path)
    overlay_workflow_captions(path)
    pixels = np.asarray(Image.open(path))
    dark = pixels.min(axis=2) < 40
    assert dark.any()
    h, w = pixels.shape[:2]

    def _window(frac, box=40):
        x = int(frac[0] * w)
        y = int(frac[1] * h)
        return dark[max(0, y - box) : y + box, max(0, x - box) : x + box].any()

    assert _window(VIEWS_XY)
    assert _window(LOCALIZATION_XY)
    assert _window(DEEP_LEARNING_XY)
    custom = tmp_path / "custom.png"
    Image.new("RGB", (1280, 960), (255, 255, 255)).save(custom)
    overlay_workflow_captions(
        custom,
        views_xy=(0.12, 0.20),
        deep_learning_xy=(0.50, 0.30),
        localization_xy=(0.88, 0.20),
    )
    pixels = np.asarray(Image.open(custom))
    dark = pixels.min(axis=2) < 40
    h, w = pixels.shape[:2]
    assert dark[int(0.20 * h) - 40 : int(0.20 * h) + 40, int(0.12 * w) - 40 : int(0.12 * w) + 40].any()
