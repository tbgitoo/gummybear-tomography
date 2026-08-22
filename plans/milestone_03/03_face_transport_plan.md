# Milestone 3 — Face Energy and Mean Beam Direction

**Source:** `plans/00_architecture.md` §5.1  
**Upstream:** `GummyBearTomography/plans/milestone_03/implementation_plan.md` (shortened)  
**Role:** First **durable** face-level transport — source rays, Snell refraction, exit-face deposition. Replaces M2B Beer–Lambert proxies; not diffusion (M4) or full sequence compose (M5+).  
**Core:** `gummybear.rays.source`, `gummybear.optics.face_transport`, `refraction`, `source_sampling`, `material`  
**Evidence:** `notebooks/milestone_03/` — **Done** (`03_0` Stage 1 → `03_1` Stage 2 → `03_2` Stage 3)

---

## Read this first

**Per-face outputs only:**

```text
face_energy[f]   hit_count[f]   b_out[f]   valid[f]
```

**Not in scope:** wavefronts, packet histories, Monte Carlo rendering, source-to-camera path solving, scattering, ray-framework abstractions.

```text
LightConfig → SourceRayBundle (geometry only)
OpticalMaterialConfig + mesh → FaceOpticalState (refraction + exit deposit)
CameraRayBundle → first_visible_hits → hit_faces → sample state to image
```

Camera and source rays share structure (`origins`, `directions`, `sample_shape`) but **different semantics**. `first_visible_hits` is reused **unchanged** (3-tuple return). Join passes only via `hit_faces` + `sample_face_values_to_image`. M2B `L_proxy` / `I_proxy` are debug-only; **`FaceOpticalState`** is what M4+ builds on.

---

## Architecture

```text
LightConfig ──► make_source_ray_bundle(bbox, sampling)     [NO material]
                      │
                      ▼
              SourceRayBundle ──► first_visible_hits ──► entry faces
                      │
OpticalMaterialConfig ┤
                      ▼
         propagate_source_rays / compute_refracted_face_field
                      │
                      ▼
              FaceOpticalState

CameraConfig ──► CameraRayBundle ──► first_visible_hits ──► hit_faces
                                              │
                                              ▼
                              sample_face_state_to_camera (diagnostics)
```

| Phase | Owns |
|-------|------|
| Ray generation | `LightConfig`, bbox, `SourceSamplingParams` → parallel rays + weights |
| Transport | `OpticalMaterialConfig` → entry Snell, internal trace, exit deposit; optional Beer–Lambert on **internal** path only |
| Camera | M2A/M2B visibility — **no changes** |

---

## Core types

**`FaceOpticalState`** — `face_energy [n_faces]` (raw weight, not area-normalized by default), `b_out [n_faces,3]` (unit where valid), `hit_count`, `valid`. Zero hits ⇒ zero energy, zero `b_out`, `valid=False`. Optional: `energy_density(face_areas)`.

**`OpticalMaterialConfig`** — `n_refractive` (default 1.33) drives Snell refraction in M3; `mu_total = mu_absorption + mu_scatter` for attenuation **after** geometry is stable.

**`SourceRayBundle`** — frozen: `origins`, `directions` (unit), `weights ≥ 0`, optional `sample_shape`. **Do not merge with `CameraRayBundle`.** Satisfies `RayBundleProtocol`; `first_visible_hits_with_points` when positions needed — do not extend default 3-tuple return.

**Production (full M3 stack — Stage 3):**

```python
source_rays = make_source_ray_bundle(light, mesh_bbox=mesh.bounds, sampling=...)  # no material
state = propagate_source_rays(source_rays, mesh, material)  # alias: compute_refracted_face_field
```

---

## Implementation stages

These are **build order / evidence checkpoints**, not separate milestones. The shipped M3 capability is the full Stage 3 pipeline.

### Stage 1 — Source coverage (no refraction)

**API:** `accumulate_source_coverage`, `make_source_ray_bundle`, `first_visible_hits` on `SourceRayBundle`.

`SourceRayBundle → first_visible_hits → accumulate weights on entry faces`. `b_out` = mean **incident** direction (label as non-refracted baseline). Sample `face_energy` / `hit_count` / `b_out` through camera `hit_faces`.

### Stage 2 — Entry Snell

**API:** `refract_direction`, `refract_ray_bundle`.

`refract_direction(b_in, normal, n_from=1.0, n_to=n_refractive)` — directions are physical propagation; helper orients normals internally (**no caller double-negation** without synthetic proof). Mandatory synthetic tests: normal bend, grazing, TIR ⇒ `ok=False`.

### Stage 3 — Exit trace, outgoing `b_out`, direct image

**API:** `propagate_entry_exit_transport`, `propagate_source_rays`, `compute_refractive_direct_image`, `sample_face_state_to_camera`, `refractive_exit_view_coupling`.

`entry + ε·d_internal → first_visible_hits → exit Snell (n_to=1.0) → deposit on exit_faces`. Prefer raw weights; attenuation uses **internal** `exit_depth` only (not source-to-entry). Final `b_out` is energy-weighted mean **outgoing** direction; should differ from M2B geometric `b_f`. `compute_refractive_direct_image` adds exit–view coupling on the camera grid.

**Fallback:** Stage 1–2 alone still yield useful diagnostics; M4 can overlay coverage if Stage 3 is unstable.

---

## Camera bridge

```python
cam_valid, _, cam_hit_faces = first_visible_hits(mesh, camera_rays)
energy_img = sample_face_values_to_image(cam_hit_faces, state.face_energy, background_value=0.0)
# or sample_face_state_to_camera(state, cam_hit_faces)
```

Canonical diagnostics: `b_out` as **x/y/z scalar images**. M2B compose helpers remain for debug comparison only.

---

## Guardrails

```text
No source-to-camera single pass. No material in make_source_ray_bundle.
No change to first_visible_hits return arity. No CameraRayBundle refactor.
No renderer / ray hierarchy. No scattering. Raw face_energy default.
Synthetic refract_direction tests before mesh use.
```

---

## Implementation status

| Item | Stage | Status |
|------|-------|--------|
| `SourceRayBundle`, `RayBundleProtocol`, `OpticalMaterialConfig` | 1–3 | **Done** |
| `make_source_ray_bundle`, `accumulate_source_coverage` | 1 | **Done** |
| `refract_direction`, `refract_ray_bundle` | 2 | **Done** |
| `FaceOpticalState`, `propagate_entry_exit_transport`, `propagate_source_rays` | 3 | **Done** |
| `sample_face_state_to_camera`, `compute_refractive_direct_image` | 3 | **Done** |
| `tests/test_m3_face_transport.py`, `tests/test_m3_validation_helpers.py` | 1–3 | **Done** |
| Evidence notebooks `03_0` / `03_1` / `03_2` | 1–3 | **Done** |

Files: `rays/source.py`, `optics/{material,source_sampling,refraction,face_transport}.py`.

---

## Validation (summary)

| Stage | Must pass |
|-------|-----------|
| 1 | Bundle invariants; entry hits; `face_energy > 0`; M2B regression-free |
| 2 | Synthetic Snell/TIR; unit internal dirs; internal ≠ incident on oblique faces |
| 3 | Some exit faces; unit outgoing dirs; internal-length attenuation only; `b_out` ≠ M2B proxy |

---

## Handoffs

**From M2B — discard:** `L_proxy`, `T_face`, Beer–Lambert `I_proxy` as physics truth. **Keep:** `hit_faces`, face→pixel sampling, light configs.

**To M4+ — keep:** `FaceOpticalState`, `SourceRayBundle`, sampling bridge. **Add:** diffusion mesh, deposition, `Phi`, hybrid compose.

---

## Success criteria

| Criterion | Status |
|-----------|--------|
| Package implements full M3 transport (Stage 3) | **Yes** |
| Camera pass contract unchanged | **Yes** |
| Evidence notebooks | **Done** |
| Sequence-gen on full M3+ hybrid path | **Later** (M4–M6) |

---

*Bump `schema_version` / `preprocess_contract_version` on semantic change — do not silently rewrite meaning.*
