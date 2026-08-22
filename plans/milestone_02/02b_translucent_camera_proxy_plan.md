# Milestone 2B — Face-Centered Illumination / Translucent Camera Proxy

**Source:** `plans/00_architecture.md` §5.2  
**Role:** **Debugging / wiring milestone** — not the canonical forward model for datasets or ML.  
**Split:** M2B.1 directional (constant `b_f`) then M2B.2 point source — same API, different light config.  
**Core (package):** `gummybear.optics` — see [§ Durable vs transient](#durable-vs-transient).  
**Evidence:** three notebooks under `notebooks/milestone_02/` — **implemented**

| Notebook | Role |
|----------|------|
| `02b_1_illumination_basics.ipynb` | directional / constant `b_f` |
| `02b_2_lambert_beer.ipynb` | point light, `L_proxy` / Beer–Lambert |
| `02b_3_end_to_end.ipynb` | full compose pipeline |

**Read order:** `02b_1` → `02b_2_lambert_beer` → `02b_3_end_to_end` (last is the clean full-pipeline reference).

---

## Read this first — transient proxy, not repo physics

**`L_proxy`, `T_face`, and `I_proxy` from Beer–Lambert on upstream half-rays are preliminary debugging scaffolding.** They prove two-pass wiring and conventions before M3+ exists. They are **not** used by `sequence_generation` / the data catalog and **not** the ML observation contract.

The package keeps the APIs for notebooks and sampling-invariant tests. **Do not** extend `L_proxy` as if it were core physics.

---

## Do not map classical tomography onto M2B

| Classical tomography | This repo |
|----------------------|-----------|
| Attenuation **along the camera ray** | Camera rays → **first surface only** (M2A). `L_proxy` uses a **separate illumination half-ray** at face centroids. |
| Sinogram / line integral = observation | Canonical unit = **camera image under documented illumination**. |
| One unified projection pass | **Illumination on faces** + **camera on pixels**, joined by `hit_faces` only. |

---

## Architecture (explicit — what the notebooks exercise)

Two sampling spaces, one bridge:

```text
                    ┌── M2A (all notebooks) ──────────────────────────────┐
                    │  PinholeCameraConfig → make_pinhole_rays → rays      │
                    │  first_visible_hits(mesh, rays) → hit_faces[H×W]     │
                    └──────────────────────────┬──────────────────────────┘
                                               │ hit_faces[pixel] = f
         ┌─────────────────────────────────────┴─────────────────────────────────────┐
         │ ILLUMINATION PASS (faces f)          │ CAMERA SAMPLING (pixels)            │
         ├──────────────────────────────────────┼─────────────────────────────────────┤
 DURABLE │ face scalars / vectors[f]          │ sample_face_values_to_image         │
         │   (centroids, fake indices, b_f…)   │   (hit_faces, face_values) → image  │
         │                                      │                                     │
         │ illumination_directions_at_faces     │ camera_sampled_beam_vector_field    │
         │   → b_f[f]  (M2B.2)                  │   → beam_dirs, obs_dirs, hit_faces  │
 TRANSIENT│ compute_face_upstream_thickness    │ beam_vector_field_to_camera_image   │
  (debug)│   → L_proxy[f]                       │   → I_proxy  (uses T_face lookup)   │
         │ thickness_to_transmittance → T_face  │                                     │
         │ source_intensity_at_faces            │                                     │
         └──────────────────────────────────────┴─────────────────────────────────────┘
```

**Full debug pipeline** (steps 4–6 in `02b_2_*`; label outputs *transient*):

```text
L_proxy = compute_face_upstream_thickness(mesh, light)
T_face  = thickness_to_transmittance(L_proxy, μ)
I_src   = source_intensity_at_faces(mesh, light)
field   = camera_sampled_beam_vector_field(rays, mesh, light)   # durable shape
I_proxy = beam_vector_field_to_camera_image(field, T_face, I_src, mode="bdotv")
```

M3+ replaces face fields and drops Beer–Lambert; **keeps** `hit_faces`, `sample_face_values_to_image`, and beam-field sampling.

---

## Durable vs transient

| Durable (M3–M6) | Transient (debug only) |
|-----------------|------------------------|
| `hit_faces` bridge | `L_proxy`, `T_face` |
| `sample_face_values_to_image` | `compute_face_upstream_thickness` |
| `CameraSampledBeamVectorField`, `camera_sampled_beam_vector_field` | `beam_vector_field_to_camera_image` + Beer–Lambert |
| `DirectionalLightConfig` / `PointLightConfig`, `illumination_directions_at_faces` | Treating `μ` or `L_proxy` as “final” appearance knobs |

---

## Evidence notebooks (content map)

Shorten on import like M2A: `ROOT` walk, constrained `pip install ".[dev]" -c requirements.txt`, `display_path`; markdown banner **“debug proxy — not sequence-gen physics”**.

### `02b_1_illumination_basics.ipynb` — M2B.1 / **durable sampling only**

**Purpose:** Prove `hit_faces` is a valid face→pixel map **before** any Beer–Lambert.

| Step | Content |
|------|---------|
| Box smoke | `face_values = arange(n_faces)` → `sample_face_values_to_image(hit_faces, …)` → blocky silhouette |
| Bear + auto-framed pinhole | M2A `first_visible_hits` on `proto_bear.stl` via `pinhole_camera_framing_mesh` |
| Centroid panels | Sample face centroid X/Y/Z and face index through `hit_faces` → 4-panel figure |
| Camera basis | Print forward/right/up; ray direction stats (sanity only) |

**APIs:** `sample_face_values_to_image`, `first_visible_hits`, `make_pinhole_rays`. **No** `L_proxy` / `T_face`.

---

### `02b_2_lambert_beer.ipynb` — M2B.2 face fields + **transient** thickness

**Purpose:** Point-light `b_f`, upstream thickness proxy, source falloff — then repeats the end-to-end block.

| Step | Content |
|------|---------|
| Point `b_f` | `illumination_directions_at_faces` → unique direction count; sample `bx/by/bz` to image |
| **`L_proxy` / `T_face`** | `compute_face_upstream_thickness` + `thickness_to_transmittance(μ)`; plot sampled `L_img`, `T_img` |
| Source intensity | Compare directional vs point (`falloff="none"` vs `"inverse_square"`); sample `I_source_img` |

**APIs:** `PointLightConfig`, `DirectionalLightConfig`, `illumination_directions_at_faces`, `compute_face_upstream_thickness`, `thickness_to_transmittance`, `source_intensity_at_faces`. Full compose stack: see `02b_3_end_to_end.ipynb`.

---

### `02b_3_end_to_end.ipynb` — **canonical transient pipeline** (reference)

**Purpose:** Single clean walkthrough of the full M2B debug stack with factor decomposition.

| Step | Content |
|------|---------|
| 1–2 | Load `proto_bear.stl`; pinhole `(0,-40,0)` → `(0,0,0)`, FOV 35°, 256² |
| 3 | `PointLightConfig(position=(0,0,30), falloff="none")` — optional switch to directional or `inverse_square` |
| 4 | **`L_proxy`, `T_face`, `face_source_intensity`** — print min/max/mean |
| 5 | **`camera_sampled_beam_vector_field`** — durable beam/observation grids |
| 6 | **`beam_vector_field_to_camera_image(..., mode="bdotv")`** → `I_proxy` |
| 7–9 | Diagnostic panel: mask, sampled `L_proxy`/`T_face`/source, **`g(b·v)`**, `I_proxy`, beam x/y/z components |

Use this notebook to verify **I_proxy ≈ source × T × g** factorization, not physical realism.

---

## Light models (M2B.1 vs M2B.2)

| | M2B.1 directional | M2B.2 point |
|--|-------------------|-------------|
| Config | `DirectionalLightConfig(propagation_direction, intensity)` | `PointLightConfig(position, intensity, falloff, r_min)` |
| `b_f` | Constant for all faces | `normalize(P_f - position)` per face |
| Notebooks | `02b_1` (sampling); directional checks in `02b_2_lambert_beer` §source | `02b_2_lambert_beer`, `02b_3_end_to_end` |

**Probe for `L_proxy` (debug):** origin `P_f - ε·b_f`, direction `-b_f`.

---

## Implementation status

| Item | Durability | Status |
|------|------------|--------|
| Two-pass invariant + `hit_faces` | Durable | **Done** |
| `sample_face_values_to_image`, beam-field split | Durable | **Done** |
| Light configs + `illumination_directions_at_faces` | Durable | **Done** |
| `L_proxy` / `T_face` / `beam_vector_field_to_camera_image` | Transient | **Done** — not on sequence-gen path |
| Three evidence notebooks | Debug | **Done** |
| `tests/test_face_illumination.py` | Optional | **No** |

---

## M3+ handoff

| Discard (M2B debug) | Replace with |
|---------------------|--------------|
| `L_proxy`, `T_face`, Beer–Lambert compose | `FaceOpticalState`, refraction, deposition, diffusion, hybrid compose |

**Keep:** `hit_faces`, face→pixel sampling, light configs, `camera_sampled_beam_vector_field` shape.

---

## Success criteria

| Criterion | Here |
|-----------|------|
| `02b_1` proves face→pixel sampling without Beer–Lambert | **Yes** |
| `02b_2_*` run full debug pipeline + factor panel | **Yes** |
| Durable wiring in package | **Yes** |
| Sequence gen uses M2B proxy | **No** — by design |

---

*Bump `schema_version` / `preprocess_contract_version` on semantic change — do not silently rewrite meaning.*
