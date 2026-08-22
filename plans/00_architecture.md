# Gummybear Tomography — Architecture

**Audience:** Researchers and engineers reproducing or extending this work  
**Scope:** Project goals, data contracts, and milestone ladder  
**Companion:** `GummyBearTomography_Final_Report.ipynb` (description, hypothesis, test results and runnable pipeline)

This document is the architecture summary.

---

## 1. Purpose

**Gummybear Tomography** is a standalone research codebase for:

1. **Synthetic optical imaging** of a translucent phantom (STL mesh → multi-view camera images under controlled illumination).
2. **Particle localisation** from those images with compact CNN architectures, focusing on whether **Fourier pooling** preserves absolute spatial information better than global average pooling when observations are scarce.
3. **Controlled comparisons** across acquisition richness: single view → multi-view camera orbit → multi-illumination × multi-view.

The canonical phantom is a FreeCAD-designed gummybear mesh (`cad/`). Runtime APIs treat geometry as a generic watertight triangle mesh with optional spherical inclusions (particles).

**Research question:**  
Can physically informed low-spatial-frequency representations (Fourier pooling) compensate for spatial information lost in pooling, especially when only limited views or illuminations are available—improving localisation accuracy and/or parameter efficiency?

**Out of scope for this release:** classical μ-reconstruction from line integrals; production deployment.

---

## 2. Scientific spine

```text
STL phantom
  → camera / illumination geometry
  → refractive transport from illumination source
  → volumetric source deposition + diffusion (FEM)
  → particle-aware source perturbation
  → multi-role camera sequences (clean / particle / observed / …)
  → catalog + task datasets
  → localisation models (M8 → M9 → M10)
```

**Canonical observation:** a **camera intensity image** of a translucent object (linear float), not an attenuation projection / sinogram.

**Composition (approximate, camera domain):**

```text
observed ≈ clean ⊕ anomaly ⊕ corruption
```

Corruption models are outside the scope of the current release.

Localisation experiments in this release train on the **anomaly** (particle-attributable) signal; background optical properties are nuisance factors in the catalog, not network inputs.

---

## 3. Data contracts (normative)

### 3.1 Sequence unit (on disk)

Each sequence directory holds `manifest.json` and role folders (`observed/`, `clean/`, `anomaly/`, …). Frames follow:

```text
<sequence_id>_frame_<index>_angle_<angle>.raw.tif   # float32, authoritative
<sequence_id>_frame_<index>_angle_<angle>.jpg       # display only
```

Frame index is the ordering source of truth.

### 3.2 ML sample shapes

| Stage | Sample | Layout |
|-------|--------|--------|
| **M8** single view | one camera image | `[V=1, C, H, W]` |
| **M9** camera orbit | ordered views, fixed light | `[V, C, H, W]` |
| **M10** joint | lights × cameras | `[I, V, C, H, W]` (illumination-major) |
| **M10 / 10_1** subsample | fixed camera, all lights | `[I, C, H, W]` |

Batched tensors add a leading batch axis. Train / validation / test splits are by **particle / sequence identity**—never by random images from the same sequence or patches in single views.

### 3.3 Configuration

Generation is driven by Excel workbooks under `configs/` (optical setups, camera schedules, particles, splits). A **catalog** joins workbook jobs to on-disk manifests; a **task dataset** selects rows and fields for `x, y = dataset[i]` with lazy image load.

Default image representation for ML: float32 `.raw.tif`. Prefer **per-view z-score** normalisation when training localisation models (as frozen in the Final Report protocol).

Train, validation, and test assignments are stored in the catalog metadata and remain fixed throughout the experiments described in the Final Report.

---

## 4. Model idea (localisation)

Shared backbone for fair comparisons:

```text
Image [C,H,W] → CNN → feature maps [C_feat,H',W']
                         ├─ GAP ──────────► MLP → (x,y,z)
                         ├─ Fourier pool ► MLP → (x,y,z)
                         └─ Flatten ─────► MLP → (x,y,z)   # capacity upper bound
```

**Fourier pooling** multiplies each feature map by a fixed low-order sine/cosine spatial basis before averaging, so absolute position is encoded as a compact channel pattern without a full flatten head. Multi-view / multi-light stages reuse the same single-view trunk and fuse view (and optionally light) embeddings or predictions.

---

## 5. Milestone ladder

Milestones are **capabilities**. Numbers match the published code and Final Report.

### Forward model and data (M0–M7)

| Milestone | Capability |
|-----------|------------|
| **M0** | Package scaffold, install, smoke tests |
| **M1** | STL load / mesh trust |
| **M2** | Camera rays + approximate translucent appearance |
| **M3** | Refractive direct transport — [detail §5.1](#51-m3--refractive-direct-transport) |
| **M4** | Coarse tet mesh, source deposition, diffusion solve, diffuse camera sampling (`fem` extra / NGSolve) |
| **M5** | Analytic particles; clean vs particle source correction |
| **M6** | Multi-role sequence generation from Excel workbooks |
| **M7** | Catalog rows + lazy task datasets |

NGSolve is required only for diffusion (M4+) generation. Catalog / ML code imports without FEM.

#### 5.1 M3 — Refractive direct transport

M3 refines the **illumination pass** (not the camera pass): source rays enter the mesh, refract under a constant index, traverse to exit faces, and accumulate **`FaceOpticalState`** (`face_energy[f]`, `b_out[f]`, `hit_count`, `valid`). The camera pass from M2A is unchanged; diagnostics sample face state through `hit_faces` only.

```text
LightConfig → SourceRayBundle (geometry only)
           → entry/exit Snell + internal trace → FaceOpticalState
CameraRayBundle → first_visible_hits → sample_face_state_to_camera / I_direct
```

M2B `L_proxy` / `I_proxy` are debug scaffolding only — M3 replaces them for production forward-model paths. M3 was implemented in three **implementation stages** (coverage → entry Snell → exit transport); detail and API guardrails: [`plans/milestone_03/03_face_transport_plan.md`](milestone_03/03_face_transport_plan.md).

**Not in M3:** source-to-camera path solving in one pass, scattering, Monte Carlo rendering, gap-filled “from-face” backtracking refinements, or a generic ray framework.

### Localisation ladder (M8–M10)

A single scientific progression—**scarce → richer acquisition**—with a frozen single-view substrate:

```text
M8  Single-view foundation
      one camera image → (x,y,z)
      freeze architecture + representation + normalisation for later stages

M9  Camera-view fusion
      ordered orbit [V,C,H,W] → one (x,y,z)
      compare simple pooling / averaging of single-view experts vs learned fusion
      (optional geometry: known camera sinθ / cosθ)

M10 Lighting fusion
      vary illumination; joint unit [I,V,C,H,W]
      10_1: lights at fixed camera [I,C,H,W]
      10_2: hierarchical light-then-camera fusion on the full grid
      (optional flat light×camera baselines)
```

| Milestone | Question in one line |
|-----------|----------------------|
| **M8** | Does Fourier pooling beat GAP for localisation from a **single** diffuse view? |
| **M9** | How much does a **camera orbit** buy on top of the frozen M8 trunk? |
| **M10** | How much does **structured illumination diversity** buy beyond camera fusion? |

Optional later work (not required for this release): portable export of trained artefacts.

---

## 6. Repository layout

```text
gummybear-tomography/
├── README.md
├── LICENSE                 # Apache-2.0
├── pyproject.toml
├── requirements.txt
├── cad/                    # phantom STL (+ FreeCAD source if included)
├── configs/                # Excel generation / localisation workbooks
├── src/
│   ├── gummybear/          # geometry, rays, optics, particles, datasets
│   ├── gummybear_validation/
│   ├── tomography_ml/      # catalog, localisation, training helpers
│   └── tomography_ml_validation/
├── tests/
└── plans/
    └── 00_architecture.md  # this file
```

Full generated corpora and large checkpoints are **not** stored in git; publish them externally (e.g. Hugging Face) and link from the README.

---

## 7. Environment

| Extra | Purpose |
|-------|---------|
| (core) | NumPy, SciPy, trimesh, Pillow, matplotlib, pandas, openpyxl |
| `dl` | PyTorch / torchvision (M8+) |
| `fem` | NGSolve / Netgen (M4+ generation) |
| `dev` | pytest, ruff, mypy |

Python **3.12**. Prefer non-editable install:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
# pip install ".[dl,dev]" -c requirements.txt   # ML without FEM (fallback)
pip install ".[fem,dl,dev]" -c requirements.txt   # full generation
```

---

## 8. Design rules

1. **Sequence-first** — ordered multi-view (and multi-light) units; no random patch leakage across splits.  
2. **Stable contracts** — bump schema / preprocess versions rather than silently changing semantics.  
3. **Algorithms in `src/`** — notebooks stay thin.    
4. **Float images are authoritative** — JPEG/PNG are display copies.  
5. **No private dependencies** — the stack must run from public packages and this repository.

---

## 9. Relation to the Final Report

| Final Report section | Architecture coverage |
|----------------------|------------------------|
| Problem / hypothesis | §1, §4 |
| Optical simulation pipeline | §2, §5 (M0–M7) |
| M8 / M9 datasets and single- vs multi-view tasks | §3, §5 (M8–M9) |
| M10 multi-illumination | §3, §5 (M10) |
| Fourier pooling maths | §4 (summary); full derivation in the report |

Use this file for **structure and contracts**; use the Final Report for **motivation, equations, figures, and experimental protocol**.

---
