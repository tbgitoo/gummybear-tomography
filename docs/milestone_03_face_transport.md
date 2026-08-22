# Milestone 3 — Face-level refractive transport

Normative plan: [`plans/milestone_03/03_face_transport_plan.md`](../plans/milestone_03/03_face_transport_plan.md).  
Broader physics ladder: [`physics_model.md`](physics_model.md) § M3.  
Evidence: [`notebooks/milestone_03/`](../notebooks/milestone_03/) (`03_0` → `03_1` → `03_2`).

---

## Read this first

**Per-face outputs only:**

```text
face_energy[f]   hit_count[f]   b_out[f]   valid[f]
```

Implemented as `FaceOpticalState` in `gummybear.optics.face_transport`.

**Not in scope:** wavefronts, packet histories, Monte Carlo rendering, source-to-camera path solving, scattering, ray-framework abstractions.

```text
LightConfig → SourceRayBundle (geometry only)
OpticalMaterialConfig + mesh → FaceOpticalState (refraction + exit deposit)
CameraRayBundle → first_visible_hits → hit_faces → sample state to image
```

Camera and source rays share structure (`origins`, `directions`, `sample_shape`) but **different semantics**. `first_visible_hits` is reused **unchanged** (3-tuple return). The illumination pass and camera pass join only via `hit_faces` and `sample_face_values_to_image` (or `sample_face_state_to_camera`). M2B `L_proxy` / `I_proxy` remain debug-only; **`FaceOpticalState`** is what M4+ builds on.

**Repo reality (checked):**

| Plan concept | Implementation |
|--------------|----------------|
| `SourceRayBundle` | `gummybear.rays.source` — separate from `CameraRayBundle` |
| Transport | `accumulate_source_coverage` (Stage 1 entry); `propagate_entry_exit_transport` (Stage 3 exit) |
| Production wrapper | `propagate_source_rays(..., refract=True)` — alias exit path via `compute_refracted_face_field` |
| Direct camera channel | `compute_refractive_direct_image` → `refractive_exit_view_coupling`; used in `datasets/sequence_generation.py` |
| Evidence notebooks | Thin wrappers in `gummybear_validation.helpers` / `plotting.m3_face_transport` |
| Stage 3 notebook | Shows face-state sampling + log **energy density**; does not plot `I_direct` (that path is in sequence gen) |

---

## Per-face ray accounting

Each ray carries weight `w_i` from `make_source_ray_bundle` (directional: uniform `intensity / n_rays`; point: uniform or inverse-square). Material is **not** applied at ray creation.

For every ray that successfully deposits on face `f`:

```text
face_energy[f]  += w_i
hit_count[f]    += 1
direction_sum[f] += w_i · d_i
b_out[f]         = unit( direction_sum[f] / hit_count[f] )
valid[f]         = hit_count[f] > 0
```

**Stage 1 — entry coverage** (`accumulate_source_coverage`): `first_visible_hits` on source rays; `d_i` = incident source direction; deposits on **entry** faces. Used in `03_0`.

**Stage 3 — exit transport** (`propagate_entry_exit_transport`): entry Snell (air → `n_refractive`) → internal ray from `entry + ε·d_internal` → exit hit → exit Snell (→ air); `d_i` = **outgoing** direction; deposits on **exit** faces. Optional internal Beer–Lambert: `w_i ← w_i · exp(−μ_total · exit_depth)` when `apply_attenuation=True`. Default on transport is **False**; `compute_refractive_direct_image` defaults **True**.

**Interpreting `b_out`:** division is by **hit count**, not accumulated weight. Equal weights ⇒ unweighted mean direction after normalization; unequal weights (point inverse-square) are not strictly energy-weighted.

**Energy density:** `FaceOpticalState.energy_density(face_areas)` or `face_energy / mesh.area_faces` — preferred diagnostic when comparing meshes or plotting camera panels (`03_2`).

---

## Experimental conclusions

From M3 evidence on `cad/proto_bear.stl` (`notebooks/milestone_03/`):

### Refractive lensing

The mesh acts as a refractive lensing object. Illumination becomes an exit-surface optical field (energy and mean outgoing direction per face). Camera images sample that field through visible faces — not by solving source-to-camera paths in one pass.

### 1. Prefer area-normalized quantities

Raw `face_energy` depends on how the surface is triangulated. **Energy density** (`face_energy / area`) is largely mesh-independent and is the physically meaningful scalar for interpretation and Stage 3 camera diagnostics.

### 2. Optical-field formation dominates cost

Expensive step:

```text
Light → entry refraction → internal transport → exit refraction → FaceOpticalState
```

Camera sampling of an existing state is comparatively cheap. A natural dataset strategy is: compute the optical field once, then many camera viewpoints. The data model supports this separation; current M6 sequence generation still runs direct transport per camera view when `alpha_direct ≠ 0`.

### 3. Diffusion is required for translucency

Without M4 diffusion, only relatively restricted exit regions contribute — the model behaves primarily as a refractive lens, not bulk translucent glow. Scattering and volume transport (M4+) distribute energy and produce broader, more natural images.

### Milestone outcome

```text
Light → entry refraction → internal transport → exit refraction
    → FaceOpticalState → camera sampling (hit_faces bridge)
```

`FaceOpticalState` is the durable link between illumination physics and image generation for M4+ hybrid compose.

---

## Key APIs (quick reference)

```python
from gummybear.optics import (
    make_source_ray_bundle,
    accumulate_source_coverage,
    propagate_entry_exit_transport,
    propagate_source_rays,
    sample_face_state_to_camera,
    compute_refractive_direct_image,
)

source_rays = make_source_ray_bundle(light, mesh_bbox=mesh.bounds, sampling=...)
state = propagate_source_rays(source_rays, mesh, material)  # full Stage 3

cam_valid, _, hit_faces = first_visible_hits(mesh, camera_rays)
energy_img, b_out_img, valid_img = sample_face_state_to_camera(state, hit_faces)
```

Guardrails: no material in `make_source_ray_bundle`; no change to `first_visible_hits` 3-tuple return; no merged source/camera bundle type.
