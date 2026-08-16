# Gummybear illustration — POV-Ray physical setup exporter

**Audience:** Contributors adding scientific setup figures  
**Scope:** A *separate* Python package `gummybear_illustration` that writes POV-Ray 3.7 scenes from M8 sample metadata. It must not change the optical simulation pipeline (`gummybear.datasets.sequence_generation`, FEM, ray transport).  
**Companion notebook:** `figures/notebooks/render_m8_physical_scene.ipynb`

---

## 1. Purpose

Produce a **reproducible physical-setup illustration** of the first M8 sample: the real STL phantom, the labelled particle sphere, the point-light location, the 180° acquisition camera as a scene object, and a subtle laboratory coordinate grid.

This is **not** a generic decorative diagram. Every graphic element maps to project data or to an **explicitly labelled fallback**.

---

## 2. What is in-scope / out-of-scope

| In scope | Out of scope |
|----------|----------------|
| Read existing `manifest.json` + STL | Running generation, FEM, or NGSolve |
| Write `.pov` / `.inc` under `outputs/` | Committing large renders or converted meshes |
| Optional `povray` subprocess if installed | Requiring POV-Ray for `pytest` or export |
| Tiny fixture manifests in `tests/` | Editing `src/gummybear/` physics |

---

## 3. Data sources (authoritative)

M8 sequences are generated artifacts. Identity lives in each sequence `manifest.json` (schema currently `1.6-m6-draft`, older `1.*-m6-draft` drafts still appear in tests).

| Element | Manifest / disk field | Notes |
|---------|----------------------|--------|
| Bear mesh | `phantom.stl_path` (+ repo `stl_root`) | Same coordinates as `gummybear.geometry.io.load_stl` (no rescale). Typical file: `cad/proto_bear_head.stl`. |
| Particle | `setups.particles.items[0]` (`center_*`, `radius`) | Single-particle M8. Fallback: `setups.particle` if `items` missing. |
| Light | `setups.optical.light_position_{x,y,z}` | Point illumination. |
| Camera (180°) | `frames[]` with `angle_deg ≈ 180` | Production writer stores `camera_position`, `look_at`, `up`, `fov_deg`. |
| Look-at policy | Mesh AABB centroid | Matches `CAMERA_LOOK_AT_POLICY = "mesh_bounds_centroid"` in `gummybear.datasets.cache_keys` and `sequence_generation.capture_frame`. Used only if a frame omits `look_at`. |
| Illumination rays | **Not** stored in the manifest | Counts exist under `generation.diagnostics` (`n_source_rays`, …). Full segments live in optical caches, which this package **must not** load. |

**First M8 sample:** `sample_index=0` among sequence directories under the M8 data root that contain `manifest.json`, in catalog/workbook order when a workbook is supplied, otherwise lexicographic `sequence_id`. Default data root: `data/generated/m8_1/single_particle`. Tests pass an explicit `manifest_path`.

---

## 4. Fallback policy (must warn)

| Situation | Behaviour |
|-----------|-----------|
| Real source-ray polylines absent | Rebuild a downsampled optical path: `make_source_ray_bundle` (`point_uniform`, `n_rays=96`, `seed=0`) → `refract_ray_bundle` (`n_from` from generation runtime defaults, `n_to` from manifest IOR) → `in_object_segments_from_rays`. Draw exterior (light→entry) and interior chords with the same cylinder style. Warn FALLBACK. |
| Frame missing `camera_position` | Reconstruct orbit pose from `angle_deg`, `distance`, `elevation_deg`, `axis` plus look-at = mesh bounds centroid — same formula as `_orbit_camera`. Warn that this is reconstruction, not a new invented pose. |
| Frustum rays | If no stored camera ray bundle, draw **4–8** frustum edges from pinhole + `fov_deg` + `look_at`. Document as visualization of intrinsics, not sampled pixels. |

Never silently invent a particle, light, or STL.

---

## 5. Package layout

Import path: `gummybear_illustration` under `src/` (setuptools `include` alongside `gummybear*`).

```text
src/gummybear_illustration/
    __init__.py                 # export_m8_physical_scene, version
    paths.py                    # repo root, default M8 data/cad paths
    load_sample.py              # PhysicalSetup dataclass + manifest load
    pov_primitives.py           # POV 3.7 snippets (sphere, cylinder, box, plane, mesh2)
    pov_scene.py                # assemble scene string + illustration camera
    export_m8_physical_scene.py # public exporter + optional povray helper
    stl_to_mesh2.py             # STL → mesh2 .inc (trimesh, already a dist dependency)

figures/notebooks/render_m8_physical_scene.ipynb
outputs/pov/                    # generated .pov/.inc (gitignored)
outputs/renders/                # optional PNG (gitignored)
tests/test_gummybear_illustration.py
plans/01_gummybear_illustration.md  # this file
```

Public API:

```python
export_m8_physical_scene(
    sample_index=0,
    camera_angle_deg=180,
    output_pov="outputs/pov/m8_physical_scene.pov",
    *,
    manifest_path=None,
    data_root=None,
    repo_root=None,
    render=False,
)
```

---

## 6. POV-Ray scene (scientific style)

- `#version 3.7;` `assumed_gamma 1.0`
- **World axes = simulation axes** (mm, **z-up**). Illustration `camera { sky <0,0,1> }` so POV does not remap to y-up.
- Translucent amber/orange `mesh2` for the bear (`transmit`, mild `finish`, `interior { ior … }` from optical `refractive_index` when present).
- Opaque/semi-opaque contrasting sphere at the particle centre and radius.
- Small emissive sphere at the light; a `light_source` at the same point.
- Neutral gray box + short cylinder for the **acquisition** camera (scene object). The POV **render** camera is a separate 3/4 overview of the AABB — labelled as illustration viewpoint, not an M8 pose.
- Thin colored cylinders for illumination rays and camera frustum.
- Reflective plane slightly below `z_min` with a faint x–y grid and short x/y axis hints; faint z arrow only.

---

## 7. Tests (no POV-Ray, no FEM)

- Tiny ASCII STL (few triangles) + synthetic manifest with known xyz.
- Export writes valid tokens: `#version 3.7`, `mesh2`, particle/light coordinates, `cylinder`, `plane`, acquisition-camera comment.
- Fallback-ray path emits `UserWarning` and POV comment `FALLBACK`.
- Missing particle/light/STL raises.
- `render=True` with no `povray` on `PATH` skips without failing export.

---

## 8. Notebook

Thin: set `sample_index`, `camera_angle_deg`, output paths, call `export_m8_physical_scene`, print repo-relative paths and warnings. Optional render checkbox. No `pip install -e`. Quiet install note only if needed: `pip install "." -c requirements.txt` from repo root.

---

## 9. Non-goals (do not regress)

- Do not import or call `DefaultSmokePhysicsBackend`, NGSolve, or source caches to fetch rays.
- Do not rewrite manifests or bump `schema_version`.
- Do not put absolute user paths in committed notebook outputs.
