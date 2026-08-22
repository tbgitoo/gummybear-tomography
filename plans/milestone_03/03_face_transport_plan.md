# Milestone 3 — Face Energy and Mean Beam Direction

**Source:** `plans/00_architecture.md` §5.1  
**Upstream:** `GummyBearTomography/plans/milestone_03/implementation_plan.md` (shortened)  
**Role:** First **durable** face-level transport — source rays, Snell refraction, exit-face deposition. Replaces M2B Beer–Lambert proxies; not diffusion (M4) or full sequence compose (M5+).  
**Core:** `gummybear.rays.source`, `gummybear.optics.face_transport`, `refraction`, `source_sampling`, `material`  
**Evidence:** `notebooks/milestone_03/` — **Done** (`03_0` → `03_1` → `03_2`). **Status:** M3 Stage 3 shipped; M6 hybrid direct path **Later** (M4–M6).

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

**Docs:** [`docs/milestone_03_face_transport.md`](../docs/milestone_03_face_transport.md).

---

## Per-face ray accounting

Ray weights `w_i` from `SourceRayBundle.weights` (`make_source_ray_bundle`; no material). Each contributing ray scatter-adds on face `f`: `face_energy[f] += w_i`, `hit_count[f] += 1`, `direction_sum[f] += w_i · d_i`; then `b_out[f] = unit(direction_sum[f] / hit_count[f])`; zero hits ⇒ `valid=False`.

**Stage 1** (`accumulate_source_coverage`): first-surface **entry** hits; `d_i` = **incident source** direction (no Snell).

**Stage 3** (`propagate_entry_exit_transport`): entry Snell → internal trace → exit Snell; deposit on **exit** face; `d_i` = **outgoing** direction. Optional Beer–Lambert on internal chord only: `w_i ← w_i · exp(−μ_total · exit_depth)` when `apply_attenuation=True` (default **False** on transport; **True** in `compute_refractive_direct_image`).

**`b_out` note:** divisor is **hit count**, not `face_energy`. Equal weights ⇒ unweighted mean direction after normalization; varying weights (inverse-square point light) are not strictly `Σ w_i d_i / Σ w_i`.

**Density (diagnostic):** `energy_density[f] = face_energy[f] / area_faces[f]` — see experimental conclusions.

---

## Experimental conclusions

From M3 evidence on `proto_bear.stl` (upstream [`milestone_03_experimental_results.md`](../../GummyBearTomography/docs/results/milestone_03_experimental_results.md); reproduced in `notebooks/milestone_03/`):

1. **Area-normalize for interpretation** — raw `face_energy` is triangulation-dependent; **energy density** is the mesh-stable diagnostic (Stage 3 notebook plots log density).
2. **Transport dominates cost** — `propagate_entry_exit_transport` / optical-field formation is expensive; camera sampling of an existing `FaceOpticalState` is cheap. **Future:** cache one optical field, many camera poses (architecture supports this; M6 sequence-gen still recomputes direct transport per view today).
3. **M3 alone is refractive lensing** — only restricted exit regions contribute; bulk translucent appearance needs M4 diffusion + scatter deposition.
4. **Central representation** — `FaceOpticalState` links illumination physics to image formation; camera pass samples it through `hit_faces` only.

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

**Phase ownership:** ray generation — `LightConfig`, bbox, `SourceSamplingParams` → parallel rays + weights; transport — `OpticalMaterialConfig` → entry Snell, internal trace, exit deposit, optional Beer–Lambert on **internal** path only; camera — M2A/M2B visibility, **no changes**.

---

## Core types and production API

**`FaceOpticalState`** — `face_energy [n_faces]` (raw weight, not area-normalized by default), `b_out [n_faces,3]` (unit where valid), `hit_count`, `valid`. Zero hits ⇒ zero energy, zero `b_out`, `valid=False`. Optional: `energy_density(face_areas)`.

**`OpticalMaterialConfig`** — `n_refractive` (default 1.33) drives Snell; `mu_total = mu_absorption + mu_scatter` for attenuation **after** geometry is stable.

**`SourceRayBundle`** — frozen: `origins`, `directions` (unit), `weights ≥ 0`, optional `sample_shape`. **Do not merge with `CameraRayBundle`.** Satisfies `RayBundleProtocol`; `first_visible_hits_with_points` when positions needed — do not extend default 3-tuple return.

```python
source_rays = make_source_ray_bundle(light, mesh_bbox=mesh.bounds, sampling=...)  # no material
state = propagate_source_rays(source_rays, mesh, material)  # alias: compute_refracted_face_field
```

**Build order (evidence checkpoints, not separate milestones):** Stage 1 — `accumulate_source_coverage`, `make_source_ray_bundle`, `first_visible_hits`; Stage 2 — `refract_direction`, `refract_ray_bundle` (synthetic normal/grazing/TIR before mesh; **no caller double-negation** on normals); Stage 3 — `propagate_entry_exit_transport`, `propagate_source_rays`, `compute_refractive_direct_image`, `sample_face_state_to_camera`, `refractive_exit_view_coupling` (`entry + ε·d_internal → exit Snell → exit_faces`; raw weights default; internal `exit_depth` attenuation only; `b_out` ≠ M2B `b_f`; exit–view coupling on camera grid). Stage 1–2 alone remain useful diagnostics if Stage 3 is unstable.

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

## Handoffs

**From M2B — discard:** `L_proxy`, `T_face`, Beer–Lambert `I_proxy` as physics truth. **Keep:** `hit_faces`, face→pixel sampling, light configs.

**To M4+ — keep:** `FaceOpticalState`, `SourceRayBundle`, sampling bridge. **Add:** diffusion mesh, deposition, `Phi`, hybrid compose.

**Implementation:** `rays/source.py`, `optics/{material,source_sampling,refraction,face_transport}.py`; tests `test_m3_face_transport.py`, `test_m3_validation_helpers.py`; docs [`docs/milestone_03_face_transport.md`](../docs/milestone_03_face_transport.md). All listed APIs and evidence notebooks **Done**.

---

*Bump `schema_version` / `preprocess_contract_version` on semantic change — do not silently rewrite meaning.*
