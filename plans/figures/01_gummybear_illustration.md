# Gummybear illustration — POV-Ray physical setup exporter

**Audience:** Contributors adding scientific setup figures  
**Scope:** A *separate* Python package `gummybear_illustration` that writes POV-Ray 3.7 scenes from M8 sample metadata, optionally renders a PNG, then draws workflow captions on that PNG. It must not change the optical simulation pipeline (`gummybear.datasets.sequence_generation`, FEM, ray transport).  
**Companion notebook:** `figures/notebooks/render_m8_physical_scene.ipynb`

This document describes the **as-built** pipeline, not the original sketch.

---

## 1. Purpose

Produce a **reproducible physical-setup illustration** of the first M8 sample: the real STL phantom, the labelled particle, the catalog point-light location, the 180° acquisition camera as a scene object, an orbit of ghost cameras with z-score back-plates, and a **workflow inset** (plate stack → arrow → mini coordinate system).

Captions are **not** 3D POV text. They are drawn in 2D after the PNG exists so size and alignment can be iterated without re-tracing.

This is **not** a generic decorative diagram. Every graphic element maps to project data or to an **explicitly labelled fallback**.

---

## 2. What is in-scope / out-of-scope

| In scope | Out of scope |
|----------|----------------|
| Read existing `manifest.json` + STL + float `.raw.tif` anomaly frames | Running generation, FEM, or NGSolve |
| Write `.pov` / `.inc` under `outputs/` | Committing large renders, converted meshes, or z-score plate PNGs |
| Optional `povray` subprocess if installed | Requiring POV-Ray for `pytest` or `.pov` export |
| PIL caption overlay on the rendered PNG | 3D `text` / `image_map` labels in the POV scene |
| Tiny fixture manifests in `tests/` | Editing `src/gummybear/` physics |

---

## 3. Data sources (authoritative)

M8 sequences are generated artifacts. Identity lives in each sequence `manifest.json` (schema currently `1.6-m6-draft`, older `1.*-m6-draft` drafts still appear in tests).

| Element | Manifest / disk field | Notes |
|---------|----------------------|--------|
| Bear mesh | `phantom.stl_path` (+ repo `stl_root`) | Same coordinates as `gummybear.geometry.io.load_stl` (no rescale). Typical file: `cad/proto_bear_head.stl`. |
| Particle | `setups.particles.items[0]` (`center_*`, `radius`) | Single-particle M8. Fallback: `setups.particle` if `items` missing. Drawn radius may be overridden for illustration. |
| Light | `setups.optical.light_position_{x,y,z}` | Catalog point illumination. Drawn as a **yellow marker sphere**, not a POV `light_source`. |
| Camera (180°) | `frames[]` with `angle_deg ≈ 180` | Production writer stores `camera_position`, `look_at`, `up`, `fov_deg`. |
| Look-at policy | Mesh AABB centroid | Matches `CAMERA_LOOK_AT_POLICY = "mesh_bounds_centroid"` in `gummybear.datasets.cache_keys` and `sequence_generation.capture_frame`. Used only if a frame omits `look_at`. |
| Anomaly plates | `frames[].filenames.anomaly_raw` | Float `.raw.tif` is authoritative. Per-view z-score PNGs are written next to the `.pov` for `image_map` back-plates. |
| Illumination rays | **Not** stored in the manifest | Counts exist under `generation.diagnostics`. Full segments live in optical caches, which this package **must not** load. |

**First M8 sample:** `sample_index=0` among sequence directories under the M8 data root that contain `manifest.json`, in catalog/workbook order when a workbook is supplied, otherwise lexicographic `sequence_id`. Default data root: `data/generated/m8_1/single_particle`. Tests pass an explicit `manifest_path`.

**Illustration-only distance scales** (`illustration_camera_distance`, `illustration_light_distance`) move the drawn camera and catalog light **along their real rays**. Angles and look-at stay catalog; this does not change generation geometry.

---

## 4. Fallback policy (must warn)

| Situation | Behaviour |
|-----------|-----------|
| Real source-ray polylines absent | Rebuild a downsampled optical path: `make_source_ray_bundle` (`point_uniform`, default `n_rays=96`, `seed=0`) → `refract_ray_bundle` → `in_object_segments_from_rays`. Draw exterior (light→entry) and interior chords. Warn FALLBACK. Notebook often sets `n_rays=0` (marker only, no cylinders). |
| Frame missing `camera_position` | Reconstruct orbit pose from `angle_deg`, `distance`, `elevation_deg`, `axis` plus look-at = mesh bounds centroid — same formula as `_orbit_camera`. Warn that this is reconstruction, not a new invented pose. |
| Frustum rays | If no stored camera ray bundle, draw **four** frustum edges from pinhole + `fov_deg` + `look_at`. Modes: `"all"` (main + orbit ghosts), `"single"` (main only), `"none"`. Document as visualization of intrinsics, not sampled pixels. |

Never silently invent a particle, light, or STL.

Warnings and POV-Ray logs must not leak absolute user paths: `gummybear.paths.display_path` / `display_text_paths` rewrite them (`~`, repo-relative, `<temp>/`). `render_pov_file` filters stdout/stderr and `CalledProcessError` payloads the same way.

---

## 5. Package layout

Import path: `gummybear_illustration` under `src/` (setuptools `include` alongside `gummybear*`).

```text
src/gummybear_illustration/
    __init__.py                 # export_m8_physical_scene, render_pov_file, version
    paths.py                    # repo root, default M8 data/cad paths
    load_sample.py              # PhysicalSetup dataclass + manifest load
    pov_primitives.py           # POV 3.7 snippets (sphere, cylinder, box, plane, mesh2)
    pov_scene.py                # assemble scene + illustration camera + inset
    anomaly_zscore.py           # .raw.tif → greyscale z-score PNGs for plates
    caption_overlay.py          # 2D labels on the rendered PNG
    export_m8_physical_scene.py # public exporter + optional povray helper
    stl_to_mesh2.py             # STL → mesh2 .inc (trimesh)

figures/notebooks/render_m8_physical_scene.ipynb
outputs/pov/                    # generated .pov/.inc/z-score PNGs (gitignored)
outputs/renders/                # captioned PNG + `_plain.png` (gitignored)
tests/test_gummybear_illustration.py
plans/01_gummybear_illustration.md  # this file
```

Public API (keyword illustration knobs omitted; see `export_m8_physical_scene`):

```python
export_m8_physical_scene(
    sample_index=0,
    camera_angle_deg=180,
    output_pov="outputs/pov/m8_physical_scene.pov",
    *,
    manifest_path=None,
    data_root=None,
    repo_root_path=None,
    render=False,
)
```

After `src/` changes: `pip install "." -c requirements.txt` from the repo root (**no** editable install), then restart the notebook kernel.

---

## 6. Pipeline (POV-Ray, then PNG captions)

```text
manifest + STL + anomaly_raw
        │
        ▼
  build_pov_scene  →  .pov + bear .inc + z-score plate PNGs
        │
        ▼  (optional, render=True and povray on PATH)
  render_pov_file  →  outputs/renders/<stem>_plain.png
        │                 (copy of the ray-traced image)
        ▼
  overlay_workflow_captions  →  outputs/renders/<stem>.png
```

1. **Export** writes the scene. Algorithms stay in `src/`; the notebook only sets knobs.
2. **POV-Ray 3.7** traces geometry only. Homebrew I/O allows writing the *current* directory: `render_pov_file` `cwd`s to the PNG directory, passes `+O<name>` and `+L` for the `.inc`. Do not pass Windows `/EXIT`. Default size is 1280×960 PNG (`+FN`, `-D`).
3. **Captions** are PIL text on a copy of that PNG. The uncaptioned file is kept as `<stem>_plain.png` so caption positions can be iterated **without** another trace.

If `povray` is missing, export still succeeds (`png` omitted). Overlay runs only when a PNG was produced and the inset stack is on (anomaly plates exist).

---

## 7. POV-Ray scene (as implemented)

- `#version 3.7;` `assumed_gamma 1.0`
- **World axes = simulation axes** (mm, **z-up**). Illustration `camera { sky <0,0,1> }` so POV does not remap to y-up.
- **Illustration viewpoint** (`IllustrationCameraParams`): yaw from behind the pinhole, FOV, stand-off as a multiple of pinhole–bear distance. Package defaults are 60° / 46° / 1.58; the notebook currently uses 80° / 60° / 2.1.
- **Bear:** grey `rgbt <0.68, 0.69, 0.71, 0.72>`, no gloss, `interior { ior 1.02 }`. Unique STL edges as thin opaque cylinders (notebook often `0.01` mm).
- **Particle:** green sphere at the catalog centre. Optional local `light_source` (notebook `0` = sphere only). Drawn radius is an illustration knob (not forced to catalog `radius`).
- **Catalog light:** yellow emissive **sphere** at the (optionally scaled) catalog position. **Not** a POV `light_source`. Optional transmissive cone toward the AABB centre (length as a fraction of that distance; intensity fades to 0).
- **Scene lighting:** separate key/fill/sky (`scene_illumination_lights`, `sky_and_horizon`) so the catalog marker does not light the mesh.
- **Acquisition camera:** gray box + cylinder at the illustrated pinhole. Z-score PNG on the **back** (opposite the lens).
- **Orbit ghosts:** `n` cameras on each side of the illustrated pinhole, world-z orbit, same |xy| radius. Optional transmit fade with angular step (`orbit_fade=False` in the current notebook).
- **Inset (lower-left):** stack of thin **world-upright** camera-body blocks. Front PNG = optical camera whose back-plate appears largest; later plates are consecutive orbit views, fanned toward the bear and screen up/right (`screen_right = -view_right` because of POV sky-up).
- **Arrow:** cylinder + cone to the right of the stack, pointing screen-right.
- **Mini triad (lower-right):** world x red, y green-gray, z blue; smaller/dimmer green particle; reflective xy plate so the ball casts a soft shadow (shadowless fill light).
- Coordinate plane slightly below `z_min` with a faint x–y grid.

**Do not** put caption strings in the `.pov` file (no `text`, no caption `image_map` quads).

---

## 8. Caption overlay (2D, after render)

Module: `caption_overlay.overlay_workflow_captions`. Font: DejaVu Sans via matplotlib `findfont`. Black text, one size (`FONT_HEIGHT_FRAC` of image height).

| Label | Default `(x, y)` | Anchor |
|-------|------------------|--------|
| `single-view or multi-view input` | `(0.40, 0.94)` | `ms` (shared baseline with localization unless the notebook overrides) |
| `Deep` / `Learning` (two lines) | `(0.60, 0.76)` | `mm` |
| `3D localization` | `(0.78, 0.88)` | `ms` |

Coordinates are **fractions of PNG width/height**; PIL **y is from the top**. Notebook knobs override the package defaults without a reinstall **after** the overlay kwargs exist in the installed wheel:

```python
CAPTION_VIEWS_XY = (0.3, 0.96)
CAPTION_DEEP_LEARNING_XY = (0.53, 0.66)
CAPTION_LOCALIZATION_XY = (0.78, 0.88)
```

Export kwargs: `illustration_caption_views_xy`, `illustration_caption_deep_learning_xy`, `illustration_caption_localization_xy`. The notebook’s last cell redraws from `_plain.png` only.

---

## 9. Tests (no POV-Ray, no FEM)

- Tiny ASCII STL + synthetic manifest with known xyz.
- Export writes valid tokens: `#version 3.7`, `mesh2`, particle/light coordinates, inset comments, **no** caption strings in the `.pov`.
- Fallback-ray path emits `UserWarning` and POV comment `FALLBACK`.
- Missing particle/light/STL raises.
- `render=True` with no `povray` on `PATH` skips without failing export.
- Overlay unit test: dummy RGB PNG, dark pixels near the requested fractions; custom `(x, y)` is honoured.

---

## 10. Notebook

Thin: knobs in one cell, `export_m8_physical_scene(...)` in the next, caption-only overlay in the last. Print repo-relative paths (`gummybear.paths.display_path` / `repo_relative_path`). Optional `RENDER=True`. No `pip install -e`. Install note: `pip install "." -c requirements.txt` from repo root.

Caption iteration: change the three `CAPTION_*_XY` tuples, re-run the config cell and the last cell. Re-run the POV export cell only when geometry knobs change.

---

## 11. Non-goals (do not regress)

- Do not import or call `DefaultSmokePhysicsBackend`, NGSolve, or source caches to fetch rays.
- Do not rewrite manifests or bump `schema_version`.
- Do not put absolute user paths in committed notebook outputs or POV-Ray log echoes.
- Do not move workflow captions back into the POV scene.
