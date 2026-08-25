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
  → publish corpora / checkpoints (M11 Hugging Face artefacts)
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
| **M4** | Coarse tet mesh, source deposition, diffusion solve, diffuse camera sampling (`fem` extra / NGSolve) — [detail §5.2](#52-m4--volumetric-diffusion) |
| **M5** | Analytic particles; clean/dirty transport pairs → source delta — [detail §5.3](#53-m5--analytic-particle-artifacts) |
| **M6** | Multi-role sequence generation from Excel workbooks — [detail §5.4](#54-m6--factorized-sequence-generation) |
| **M7** | Catalog rows + lazy task datasets — [detail §5.5](#55-m7--sample-catalog-and-lazy-task-datasets) |

NGSolve is required only for diffusion (M4+) generation. Catalog / ML code imports without FEM.

#### 5.1 M3 — Refractive direct transport

M3 refines the **illumination pass** (not the camera pass): source rays enter the mesh, refract under a constant index, traverse to exit faces, and accumulate **`FaceOpticalState`** (`face_energy[f]`, `b_out[f]`, `hit_count`, `valid`). The camera pass from M2A is unchanged; diagnostics sample face state through `hit_faces` only.

```text
LightConfig → SourceRayBundle (geometry only)
           → entry/exit Snell + internal trace → FaceOpticalState
CameraRayBundle → first_visible_hits → sample_face_state_to_camera / I_direct
```

M2B `L_proxy` / `I_proxy` are debug scaffolding only — M3 replaces them for production forward-model paths. M3 was implemented in three **implementation stages** (coverage → entry Snell → exit transport); detail and API guardrails: [`plans/milestone_03/03_face_transport_plan.md`](milestone_03/03_face_transport_plan.md). Physics summary: [`docs/milestone_03_face_transport.md`](../docs/milestone_03_face_transport.md).

**Not in M3:** source-to-camera path solving in one pass, scattering, Monte Carlo rendering, gap-filled “from-face” backtracking refinements, or a generic ray framework.

#### 5.2 M4 — Volumetric diffusion

M4 adds bulk translucency: M3 internal losses → volumetric scatter source `S(x)` on a **coarse Netgen tet mesh** → NGSolve FEM fluence `Φ(x)` → diffuse camera sampling → hybrid compose with M3 direct. **NGSolve is the production solver**; early structured-grid NumPy FD prototypes were abandoned (wrong geometry, uncompetitive vs FEM speed). Detail: [`plans/milestone_04/04_diffusion_plan.md`](milestone_04/04_diffusion_plan.md).

```text
M3 segments → deposit_ray_source → solve_diffusion → sample_diffuse_image
I_direct (M3) + I_diffuse → I_total = alpha * I_direct + I_diffuse
```

**Not in M4:** particles (M5), sequence generation (M6), monolithic renderer, replacing `I_direct` with diffusion.

**Empirical default for ML PoP:** at practical coarse tet counts, the direct channel is relatively coarse vs bulk diffuse; first localization experiments should prefer **`alpha = 0`** (diffusion-only) unless denser mesh / hybrid lensing is required — see M4 plan § Experimental conclusions.

#### 5.3 M5 — Analytic particle artifacts

M5 adds analytic spherical inclusions on the M4 backbone. Particles perturb **ray transport and source deposition only** — they do **not** remesh or change the diffusion operator. The durable ledger is the **clean/dirty transport pair** (`AffectedTransportPair`): **one per affected transport path** (`path_id`), not one per intersection event and not particle records alone:

```text
segments + particles → one AffectedTransportPair per hit path_id
  → ΔE_background + ΔE_particle_scat → S_particle
  → same A, new RHS → Φ_particle → I_particle
I_anomaly = I_particle − I_clean
```

Validated particle-scatter assignment is Beer–Lambert **attenuated chord** with exact ray–tet distribution (not midpoint/uniform chord). Detail: [`plans/milestone_05/05_particle_plan.md`](milestone_05/05_particle_plan.md). Guidelines: [`docs/milestone_05_particle_guidelines.md`](../docs/milestone_05_particle_guidelines.md). Physics: [`docs/physics_model.md`](../docs/physics_model.md) § M5.

**Not in M5:** refractive particle deflection, sequence generation (M6), production-scale `run_m5d_simulation`.

#### 5.4 M6 — Factorized sequence generation

M6 turns the M5D path into a **workbook-driven, cache-aware, output-idempotent** generator. Motto: do expensive physics once; vary particles, then diffusion boundaries, then cameras; record everything. Three layers stay separate: **output delta** (`resolved_job_hash` / manifest) → **source-cache** plan → physics. Persist clean/particle `.npz`+`.json` sources; re-solve diffusion at runtime (no FEM-operator cache); reuse camera×mesh visibility and Phi localization, not finished role frames. Detail: [`plans/milestone_06/06_sequence_generation_plan.md`](milestone_06/06_sequence_generation_plan.md).

**Not in M6:** M5 physics redesign, silent overwrite of changed `sequence_id`, workbook-SHA-only identity, committed `data/generated/` corpora.

#### 5.5 M7 — Sample catalog and lazy task datasets

M7 joins workbook `SequenceJob`s to on-disk M6 manifests into catalog **rows** (a sample is a row, not a tensor). Tasks select fields into a pure-Python lazy dataset: **`x, y = dataset[i]`** (dicts; no `torch.utils.data.Dataset`). Role images load on demand to **`numpy.ndarray` `(V, C, H, W)`** (HWC is loader-internal only). Schedule-consistent subsets fix `V`/angles/resolution before multi-view ML. Detail: [`plans/milestone_07/07_catalog_task_dataset_plan.md`](milestone_07/07_catalog_task_dataset_plan.md).

**Not in M7:** training, global `x_train`/`y_train`, directory-scan membership, public HWC stacks.

#### 5.6 M8 — Single-view localisation foundation

M8 builds a **defensible single-view localiser** from Excel-specified multi-regime corpora (M6/M7): raw-float roles, lazy catalog tasks, then hypothesis-driven CNN ablations (avg-pool vs **Fourier-coded** readout vs Flatten). The release narrative and primary train/val/test evidence live in [`GummyBearTomography_Final_Report.ipynb`](../GummyBearTomography_Final_Report.ipynb) (§4 dataset, M8 Steps 1–4). Upstream `08_*` notebooks retain the full WIN 3B–3J ablation audit trail; thin `notebooks/milestone_08/` cells verify gaps not shown in the report. Detail: [`plans/milestone_08/08_single_view_localization_plan.md`](milestone_08/08_single_view_localization_plan.md).

**Not in M8:** multi-view fusion (M9), multi-illumination fusion (M10), predicting background optics, JPEG-as-training-store.

#### 5.6b M9 — Camera-view fusion

M9 asks how much a **camera orbit** buys on the frozen M8 trunk: one shared xyz from ordered views `[V,C,H,W]` (fixed illumination). Primary evidence: [`GummyBearTomography_Final_Report.ipynb`](../GummyBearTomography_Final_Report.ipynb) **M9 Steps 1–2** (frozen fusion, then e2e + sinθ/cosθ; compact vs large heads; Fourier vs GAP). Thin `notebooks/milestone_09/` cells keep ablation evidence omitted or condensed in the report (per-angle expert mean, packing/LR ladders, Fourier-vs-GAP overlays, fusion-capacity slice). Detail: [`plans/milestone_09/09_camera_view_fusion_plan.md`](milestone_09/09_camera_view_fusion_plan.md).

**Not in M9:** illumination diversity or joint camera×light fusion (M10); reopening WIN 3J.

### Localisation ladder (M8–M10) and publication (M11)

A single scientific progression—**scarce → richer acquisition**—with a frozen single-view substrate, then Hub packaging:

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

M11 Publish artefacts
      Hugging Face dataset card + parquet indexes
      corpora / checkpoints downloadable without regenerating
```

| Milestone | Question in one line |
|-----------|----------------------|
| **M8** | Does Fourier pooling beat GAP for localisation from a **single** diffuse view? |
| **M9** | How much does a **camera orbit** buy on top of the frozen M8 trunk? |
| **M10** | How much does **structured illumination diversity** buy beyond camera fusion? |
| **M11** | How do we **publish** corpora + checkpoints so others can browse and reproduce without regenerating? |

#### 5.7 M11 — Hugging Face repository artefacts

M11 is **publication plumbing**, not new science: turn on-disk M6/M8/M10 corpora and ML checkpoints into a Hub dataset that mirrors the repo layout and shows something reasonable in the Dataset Viewer. Detail: [`plans/milestone_11/11_huggingface_artifacts_plan.md`](milestone_11/11_huggingface_artifacts_plan.md). Notebook: [`notebooks/milestone_11/11_0_huggingface_export.ipynb`](../notebooks/milestone_11/11_0_huggingface_export.ipynb).

```text
data/generated/{m8_1,m10_illumination}.zip  +  checkpoints/
  → parquet indexes (JPEG/PNG preview bytes embedded) + dataset-card README
  → Hub Dataset Viewer works without extracted trees
```

**Not in M11:** regenerating optics, retraining, changing catalog schemas, embedding float `.raw.tif` in parquet.

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
├── data/                   # local / Hub (not in git except metadata indexes)
│   ├── generated/          # optical corpora (m8_1, m10_illumination, …)
│   └── huggingface_metadata/  # M11 parquet + dataset card (git-tracked)
├── checkpoints/            # ML weights (local / Hub; not in git)
├── src/
│   ├── gummybear/          # geometry, rays, optics, particles, datasets
│   ├── gummybear_validation/
│   ├── tomography_ml/      # catalog, localisation, training helpers
│   └── tomography_ml_validation/
├── tests/
└── plans/
    └── 00_architecture.md  # this file
```

Full generated corpora and large checkpoints are **not** stored in git; publish them externally (Hugging Face, M11) and link from the README.

---

## 7. Environment

| Extra | Purpose |
|-------|---------|
| (core) | NumPy, SciPy, trimesh, Pillow, matplotlib, pandas, openpyxl |
| `dl` | PyTorch / torchvision (M8+) |
| `fem` | NGSolve / Netgen (M4+ generation) |
| `hf` | PyArrow (M11 parquet metadata export) |
| `dev` | pytest, ruff, mypy |

Python **3.12**. Prefer non-editable install:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
# pip install ".[dl,dev]" -c requirements.txt   # ML without FEM (fallback)
pip install ".[fem,dl,dev]" -c requirements.txt   # full generation
# pip install ".[hf]" -c requirements.txt         # M11 Hub metadata only
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
| External corpora / checkpoints download | §5 (M11), §6 |
| Fourier pooling maths | §4 (summary); full derivation in the report |

Use this file for **structure and contracts**; use the Final Report for **motivation, equations, figures, and experimental protocol**.

---
