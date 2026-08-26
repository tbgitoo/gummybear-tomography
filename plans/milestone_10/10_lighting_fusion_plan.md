# Milestone 10 — Lighting / illumination fusion

Contract / index for the illumination ladder. Runnable narrative and results: [`GummyBearTomography_Final_Report.ipynb`](../../GummyBearTomography_Final_Report.ipynb) (§4 M10 dataset + **M10 Steps 1–3**). No local `notebooks/milestone_10/`.

**Source:** `plans/00_architecture.md` §5.6  
**Role:** Frozen M8 trunk → shared xyz from illumination diversity, then factorized illumination-angle-then-camera fusion  
**Install:** `pip install ".[dl,dev]" -c requirements.txt` (generation `.[fem]` when regenerating)  
**Core:** `tomography_ml.studies.m10_illumination_fusion` / `m10_hierarchical_fusion` · `localize_multiview` (`for_10_1_*`, `for_10_2`)  
**Labeling:** **Conclusion** = frozen protocol / repo contract.

---

## Lights are not extra cameras

M10 asks whether **illumination angle diversity** adds complementary localisation information, and whether camera and light should be fused as **structured acquisition axes**. It does not reopen WIN 3J.

```text
10_0  multi-illumination corpus              (data; Final Report §4)
10_1  illumination-only fusion, camera fixed ([I,C,H,W])
10_2  hierarchical fusion        (full grid; preferred joint architecture)
```


**Naming:** Step 1 = **10_1A** (frozen); Step 2 = **10_1B** (e2e); Step 3 = **10_2**. Inside 10_1, variants **A/B/C/D** = single-light / xyz-mean / learned / +angle FiLM (orthogonal to 10_1A vs 10_1B).

**Downstream:** M11 Hub packaging. Do not rework M9 conclusions; compare against them.

---

## Frozen substrate · joint sample (Conclusion)

```text
API            = m8_single_view_block_freeze()  (same WIN 3J as M9)
representation = anomaly_ref;  normalisation = per_image_zscore
corpus         = data/generated/m10_illumination
                 workbook configs/m10/localization_m10_illumination.xlsx
lights         = M10_LIGHT_ANGLES_DEG (I=6 on the full corpus)

On disk: one sequence per (particle, light) with a camera orbit
ML group: particle_id → {light → CatalogRow}   # particle-stratified splits

[I, V, C, H, W]     batched [B, I, V, C, H, W]
10_1  fixed camera  →  [I, C, H, W]
10_2  full grid; fuse illumination angles per camera, then cameras
# NOT inputs: bear_mu_a, bear_mu_s
```

---

## Ladder (Conclusion)

| Stage | Camera | Trains | Geometry | Library |
|---|---|---|---|---|
| **10_1A** (Step 1) | fixed 180° | fusion only; trunk frozen | D: light FiLM | `for_10_1_c_frozen` / `for_10_1_d_frozen` (+ pooled) |
| **10_1B** (Step 2) | fixed | trunk + fusion e2e | D: light FiLM | `for_10_1_c` / `for_10_1_d` (+ pooled) |
| **10_2** (Step 3) | orbit | e2e hierarchical | light then camera sin/cos | `HierarchicalLightThenCameraFusionLocalizer.for_10_2` |

**10_1 A–D:** A = single light; B = `mean_L(xyz_L)`; C = learned fusion, no light angle; D = same compact capacity as C + light-angle FiLM. Interpretations live in the Final Report (Steps 1–2).

**10_2:** preferred joint inductive bias — `run_m10_hierarchical_fusion` / `for_10_2`.

Checkpoints: `checkpoints/m10/m10_frozen_illumination_fusion.pt`, `m10_e2e_illumination_fusion.pt`, `m10_hierarchical_light_then_camera.pt`.

CI covers forward / pattern-id contracts (`tests/test_localize_multiview.py`), not full training. Helpers: `tomography_ml_validation.milestone_10`.

---

## Guardrails · handoffs

```text
Do not reopen WIN 3J architecture, representation, or per-view z-score.
Do not treat camera and light as interchangeable view dimensions.
Do not treat angle metadata as a substitute for image evidence.
Do not use relative-only (θ_light − θ_cam) as the sole geometry token.
Do not default 10_1/10_2 to the large 09_3 fusion MLP.
Do not train JPEG previews; float .raw.tif remains authoritative.
Do not add a local 10_0 generation notebook (report §4).
```

**From M9:** compact e2e + sin/cos is enough on the camera orbit; here, attempt to address remaining error through illumination angle.  
**To M11:** publish `m10_illumination` zip + `checkpoints/m10/` without regenerating.

*Bump experiment IDs / fusion-pattern records when 10_1/10_2 meaning changes — do not silently rewrite them.*
