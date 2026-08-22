# Milestone 2A — First-Surface Camera Geometry

**Source:** `plans/00_architecture.md` §5  
**Scope:** First-surface camera visibility — orthographic (debug) + perspective (default for M2B+).  
**Core:** `gummybear.rays` (`camera.py`, `visibility.py`)  
**Evidence:** `notebooks/milestone_02/02a_camera_view.ipynb`

---

## Status (this repo)

Implementation summary: The milestone originally planned parallel orthographic and perspective tutorial paths. During implementation, perspective rendering became the primary architecture because it corresponds to the real camera-view geometry used by all subsequent milestones. Orthographic rendering remains available as a diagnostic and validation tool, but is not developed further beyond its core API.

A key architectural outcome of M2A was the formalization of ray bundles as a reusable computational abstraction. `CameraRayBundle` established a common representation consisting of origins, directions, and sampling topology, providing the conceptual foundation for later energy-deposition ray bundles. Camera and illumination passes remain physically independent and operate on separate bundle instances; the shared element is the ray-bundle formalism and generic operations such as visibility and intersection (via `first_visible_hits`), and sampling. Future milestones may further generalize this contract through a `RayBundleProtocol`.


**Implemented** — perspective path + notebook + validation helpers. Ortho notebook panels and full pytest suite remain optional.

| Item | Status |
|------|--------|
| Ortho + pinhole ray builders, first-hit, `hits_to_image`, intensity proxy | **Done** — `gummybear.rays` |
| Auto-framing, basis, trimesh pose | **Done** — `gummybear_validation.helpers.camera_helpers` |
| Aligned wireframe plot | **Done** — `gummybear_validation.plotting.pinhole_camera` |
| Evidence notebook | **Done** — head mesh, auto-framed pinhole, mask / b·v / wireframe |
| Tests | **Partial** — `tests/test_camera_helpers.py` (basis, framing, trimesh pose) |
| Ortho / side-by-side / normal RGB in notebook | **Omitted** (APIs exist) |
| `docs/physics_model.md` M2A notes | **Done** |

**Notes:** Notebook uses `cad/proto_bear_head.stl` (tall Z) for framing stress-test; canonical phantom remains `cad/proto_bear.stl`. `gummybear.rays` does not auto-frame — notebook calls `pinhole_camera_framing_mesh` (−Y standoff, vertex-derived FOV). Wireframe matches raster; trimesh Jupyter three.js viewer does not (use wireframe panel).

---

## Purpose & invariant

> Camera-view geometry images of the bear.

```text
pixel → camera ray → first visible surface hit → pixel property
```

Pipeline: STL → rays → first hit → mask, first-surface depth, optional normals, intensity proxy (`constant` or global `b·v`). **Not** shadow projection, Beer–Lambert, or path-length depth.

**Non-goals:** M2B thickness; M3 refraction; renderer frameworks; Lambertian `n·b` / `n·v`; datasets / ML.

---

## Inputs

- **STL:** mm coordinates; `inspect_stl(path)` → `{mesh, summary, validation}`.
- **Up hint:** `(0, 0, 1)`.
- **Notebook mesh:** `proto_bear_head.stl`; plans elsewhere use `proto_bear.stl`.

---

## Camera API (planned contract = as-built in `gummybear.rays`)

**Orthographic** — `OrthographicCameraConfig(camera_position, look_at, size, resolution, up)` + `make_orthographic_rays`: parallel rays, origins on a `size×size` mm grid at `camera_position`. Caller places camera outside mesh. `margin` field unused.

**Pinhole** — `PinholeCameraConfig(camera_position, look_at, up, fov_deg, resolution)` + `make_pinhole_rays`: shared origin; directions through image plane at `image_dist=1.0` (`half_size = tan(fov/2)`). Defaults `(0,-30,0)` → origin look-at clip the offset bear unless reframed.

**Dispatch:** `make_camera_rays(config)` → `CameraRayBundle(origins, directions, sample_shape)`.

**First hit:** `first_visible_hits(mesh, rays)` via `intersects_location(..., multiple_hits=False)`; depth = Euclidean range along ray; misses → NaN / invalid.

**Images:** `hits_to_image(mesh, …)` → mask, depth, normals (NaN on miss).

**Intensity:** `simple_camera_intensity(mask, beam_direction, view_direction, mode="bdotv")`.

**Validation helpers** (not in `gummybear.rays`): `pinhole_camera_framing_mesh`, `pinhole_camera_basis`, `trimesh_camera_transform`, `plot_pinhole_wireframe`.

---

## Notebook (as-built)

`notebooks/milestone_02/02a_camera_view.ipynb`: install + imports → `inspect_stl(head)` → `pinhole_camera_framing_mesh` → `first_visible_hits` → mask + intensity + aligned wireframe.

**Optional / not in notebook:** default pinhole on full bear, ortho path, side-by-side, normal RGB, multi-angle, pyglet viewer.

---

## Tests & success

| | |
|--|--|
| **Done** | Perspective + ortho APIs; first-surface depth; notebook evidence; helpers in package |
| **Partial** | `test_camera_helpers.py` only |
| **Optional** | Ortho/primitive invariant tests; bear goldens |
| **Skip** | Beer–Lambert tests; default-dataclass-only tests |

---

## M2B handoff

M2B **reuses** M2A camera pass: `make_camera_rays` → `first_visible_hits` → **`hit_faces`** (pixel→face map). `simple_camera_intensity` is a non-physical placeholder.

M2B **adds** a **transient debug** face→pixel wiring check (see [`02b_translucent_camera_proxy_plan.md`](02b_translucent_camera_proxy_plan.md)):

- **Durable:** illumination and camera stay separate passes; sample face fields via `hit_faces`.
- **Transient (not repo forward model):** `L_proxy` / `T_face` / Beer–Lambert compose — superseded by M3+ refraction and M4+ volume physics; **not** used in sequence generation.

Do not read M2B as classical tomography (line integrals along camera rays / sinograms).

Illumination and camera stay separate passes. Detail: `02b` plan / `plans/00_architecture.md`.

---

## Appendix — drift from early drafts

Ortho/pinhole use `camera_position` + `look_at` + `size`/`fov_deg`; bundle type `CameraRayBundle`; `first_visible_hits(mesh, rays)`; `inspect_stl` returns dict; auto-framing in `gummybear_validation`, not core rays.

---

*Bump `schema_version` / `preprocess_contract_version` on semantic change — do not silently rewrite meaning.*
