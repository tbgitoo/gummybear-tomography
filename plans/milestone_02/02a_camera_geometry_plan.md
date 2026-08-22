# Milestone 2A — First-Surface Camera Geometry Rendering Plan

**Source:** `plans/00_architecture.md` §5 / Milestone 2A (and appearance roadmap)  
**Status:** Package implemented; evidence notebook is `02a_camera_view` under `notebooks/milestone_02/`  
**Scope:** Prove first-surface camera visibility with orthographic (debug) + perspective (default) cameras.  
**Evidence notebook:** `notebooks/milestone_02/02a_camera_view.ipynb` 
**Package modules:** `src/gummybear/rays/camera.py`, `src/gummybear/rays/visibility.py`

### As implemented in this repository

| Item | Status here |
|------|-------------|
| Orthographic + pinhole configs / ray builders | `gummybear.rays.camera` |
| First-hit visibility, `hits_to_image`, intensity proxy | `gummybear.rays.visibility` |
| Re-exports | `gummybear.rays` |
| Evidence notebook | `notebooks/milestone_02/02a_camera_view.ipynb` perspective smoke path (install + `display_path`, inspect, pinhole rays, first-hit → intensity plot, optional bounds / `mesh.show()`). Ortho / side-by-side / normal RGB panels omitted. |
| Dedicated M2A pytest module | **No**; rays exercised via later tests (e.g. M3A face-transport) + smoke import |
| `docs/physics_model.md` M2A notes | Present — camera vs face-centered illumination invariants |

API shapes below match the code in `src/gummybear/rays/`.

---

## 1. Purpose

> Can I generate camera-view geometry images of the bear?

Invariant (both camera modes):

```text
pixel → camera ray → first visible surface hit → pixel property
```

```text
M2A.1 Orthographic first-surface visibility   (debug / validation)
M2A.2 Perspective first-surface visibility    (default for M2B+)
```

Pipeline:

```text
validated STL mesh → camera rays
                   → first visible surface hit
                   → mask / silhouette
                   → first-surface depth image
                   → optional normal map
                   → simple intensity proxy (constant or b·v)
```

This is **not** a shadow projection, attenuation image, or path-length integral. Depth = camera-to-first-surface distance only.

---

## 2. Philosophy / non-goals

Prefer package APIs + a thin notebook; modest resolution (default 256²); no `Renderer` / `CameraBackend` / `Scene` frameworks. Notebook visual output is milestone evidence.

**Out of scope for M2A:** Beer–Lambert / thickness (M2B); entry/exit pairing; Snell (M3); source-ray tracing / caustics / GPU; datasets / ML; Lambertian `n·b` / `n·v` as default intensity.

---

## 3. Inputs

| Item | Source |
|------|--------|
| STL | `cad/proto_bear.stl` (M1) |
| Load / inspect | `inspect_stl(path)` → `{mesh, summary, validation}` |
| Units | millimetres |
| Call form | `inspect_stl(ROOT / "cad" / "proto_bear.stl")` — not `inspect_stl(mesh, path)` |

STL coordinates are authoritative. Default camera `up` = `+Z` = `(0, 0, 1)`.

---

## 4. Targets

**M2A.1 Orthographic (debug):** `OrthographicCameraConfig` + `make_orthographic_rays` → first-hit → mask / depth / optional normals / intensity. Parallel rays; origins on a plane through `camera_position`.

**M2A.2 Perspective (default):** `PinholeCameraConfig` + `make_pinhole_rays` → same first-hit pipeline; shared camera center; per-pixel directions. Notebook evidence should exercise this path on the bear.

---

## 5. Camera geometry 

### OrthographicCameraConfig

```python
@dataclass(frozen=True)
class OrthographicCameraConfig:
    camera_position: tuple[float, float, float] = (0.0, -20.0, 0.0)
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: float = 20.0          # full side length in mm
    resolution: int = 256
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    margin: float = 0.15        # reserved; unused by make_orthographic_rays today
```

`make_orthographic_rays(cam)`: basis `forward / right / up`; origins on a grid in the plane through `camera_position` spanning `[-size/2, size/2]`; all directions = `forward`; return `CameraRayBundle(..., sample_shape=(H,W))`. Caller places the camera outside the mesh (no auto bbox framing).

### PinholeCameraConfig

```python
@dataclass(frozen=True)
class PinholeCameraConfig:
    camera_position: tuple[float, float, float] = (0.0, -30.0, 0.0)
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_deg: float = 60.0
    resolution: int = 256
```

`make_pinhole_rays(cam)`: same basis; image plane at fixed `image_dist = 1.0` along `forward` (**not** halfway to `look_at`); `half_size = tan(fov/2) * image_dist`; all origins = `camera_position`; directions = normalize(image_point − camera).

`make_camera_rays(camera)` dispatches on config type. Use `sample_shape` to reshape flat `[N, …]` arrays to images.

### First-hit (shared)

`first_visible_hits(mesh, rays: CameraRayBundle)` uses `mesh.ray.intersects_location(..., multiple_hits=False)`. Directions renormalized so `depth = dot(location - origin, direction_unit)` (Euclidean range). Misses → `valid_mask=False`, `hit_depth=NaN`, `hit_face=-1`. Depth is **not** internal thickness.

### Intensity proxy

```python
simple_camera_intensity(mask, beam_direction, view_direction, mode="bdotv")
# "constant" → mask.astype(float)
# "bdotv"    → ((dot(b̂,v̂)+1)/2) * mask     # global factor, not Lambertian
```

---

## 6. API surface

```text
camera.py       OrthographicCameraConfig, PinholeCameraConfig, CameraRayBundle,
                make_orthographic_rays, make_pinhole_rays, make_camera_rays
visibility.py   first_visible_hits, hits_to_image, simple_camera_intensity
__init__.py     re-exports
```

```python
@dataclass(frozen=True)
class CameraRayBundle:
    origins: np.ndarray            # [N, 3]
    directions: np.ndarray         # [N, 3]
    sample_shape: tuple[int, int]  # (H, W)

def first_visible_hits(mesh, rays) -> tuple[valid_mask, hit_depth, hit_face]: ...

def hits_to_image(mesh, valid_mask, hit_depth, hit_faces, resolution
) -> tuple[mask_image, depth_image, normal_image]: ...  # miss normals = NaN
```

Plain NumPy / dataclasses only.

---

## 7. Implementation phases (intent vs evidence)

| Phase | Intent | Evidence |
|-------|--------|----------|
| Ortho primitive / bear | Debug camera | API present; not required in the notebook |
| Perspective bear | Default camera | API + notebook evidence |
| Ortho vs perspective | Side-by-side | Optional; omitted from notebook |
| Normal-map diagnostics | RGB normals from `hits_to_image` | Available from API; omitted from  notebook |
| Intensity proxy | constant / b·v | Implemented; notebook uses `mode="bdotv"` |
| Promote out of notebook | Package APIs | **Done** |
| Invariant pytest | Ray shape / unit-dir / non-empty mask | **Not** as a dedicated M2A module |

Optional follow-ups: ortho smoke cell/test; document or remove unused `margin`.

---

## 8. Notebook outline — `notebooks/milestone_02/02a_camera_view.ipynb`

Perspective smoke path:

1. Resolve `ROOT` via `pyproject.toml`; constrained non-editable install; print via `display_path`.
2. Imports: `inspect_stl`, `PinholeCameraConfig` / `make_camera_rays`, `first_visible_hits`, `hits_to_image`, `simple_camera_intensity`.
3. `inspect_stl(cad/proto_bear.stl)` → mesh (+ summary).
4. Default `PinholeCameraConfig` + `make_camera_rays`; quick ray sanity (direction norms / unique origins).
5. `first_visible_hits` → `hits_to_image` → `simple_camera_intensity(..., mode="bdotv")`.
6. Plot intensity proxy; optional bounds / `mesh.show()`.

Omitted vs a full tutorial: orthographic path, primitive checks, side-by-side ortho/perspective, normal-map RGB panel, multi-angle sweep.

---

## 9. Tests

**Recommended invariants:** ortho directions equal/unit; pinhole shared origin + diverse dirs; centered primitive → non-empty mask; optional box front-face depth ≈ expected range.

**Do not add:** pixel-perfect bear goldens; Beer–Lambert checks (M2B); dataclass-default-only tests.

---

## 10. Success criteria

| Criterion | Package here |
|-----------|--------------|
| Perspective first-hit API | **Yes** |
| Orthographic API | **Yes** |
| Depth = first-surface range | **Yes** |
| No Beer–Lambert / path-length in M2A | **Yes** |
| No Lambertian default; no renderer framework | **Yes** |
| Perspective default for M2B sampling | **Yes** |
| Evidence notebook (perspective path) | **Yes** — `notebooks/milestone_02/02a_camera_view.ipynb` |
| Dedicated M2A pytest module | **No** (optional follow-up) |

---

## 11. Handoff to Milestone 2B

**M2B may assume:** camera pass works — pinhole (or `make_camera_rays`) → `CameraRayBundle` → `first_visible_hits` → mask, first-surface depth, normals, and especially **`hit_faces`** as pixel→face map (`rays.sample_shape` to reshape). `simple_camera_intensity` is a non-physical M2A proxy only. Ortho remains for diagnostics.

**M2B must still implement** a **face-centered illumination pass**, then sample into the camera:

```text
illumination sampling space = mesh faces
camera sampling space       = camera pixels
```

1. Per face `f`: centroid `P_f`; `L_proxy[f]` = mesh thickness along illumination direction `b` through `P_f` (not along the camera ray); `T_face[f] = exp(-μ · L_proxy[f])`.
2. Sample: `I_proxy[pixel] ≈ I_bg · T_face[hit_faces[pixel]] · g(b·v)`.
3. Keep `g(b·v)` global; spatial variation from `T_face`.
4. Compare M2A vs M2B at ≥2 angles; document limitations (no refraction; faceted OK).

M2B reuses the M2A camera pass; it does not replace it. Detail: `plans/milestone_02/02b_…` when imported; normative invariants in `plans/00_architecture.md`.

---

## Appendix — Signature drift (draft → as-built)

1. Ortho uses `camera_position` / `look_at` / `size` (not bare `direction` + auto origin plane).
2. Default `up` = `(0,0,1)`. Otherwise bear is not upright
3. Pinhole defaults `(0,-30,0)`, `fov_deg=60`; image plane at `image_dist=1.0`.
4. Intersection via `intersects_location(..., multiple_hits=False)`.
5. `hits_to_image(mesh, …)` requires mesh for normals.
6. Intensity default `"bdotv"`.
7. `inspect_stl(path)` returns a dict.
8. Rays return `CameraRayBundle` (not bare tuples); `first_visible_hits(mesh, rays)` takes the bundle.
9. `OrthographicCameraConfig.margin` present but unused by the ray builder.

---

*When contracts change, bump `schema_version` / `preprocess_contract_version` rather than silently editing semantics.*
