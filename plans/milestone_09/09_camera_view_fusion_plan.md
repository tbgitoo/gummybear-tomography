# Milestone 9 — Camera-view fusion

This plan does not attempt to reproduce the full ML narrative from the Final Report.

Its purpose is to document the camera-orbit fusion ladder, freeze contracts, and supporting evidence that were condensed or omitted from the report.

**Source:** `plans/00_architecture.md` §5.6 (M8–M10 ladder)  
**Primary evidence:** [`GummyBearTomography_Final_Report.ipynb`](../../GummyBearTomography_Final_Report.ipynb) — §4 M8/M9 sample contract + **M9 Steps 1–2**  
**Role:** Frozen M8 trunk → **one shared xyz** from an ordered camera orbit `[V,C,H,W]` (fixed illumination)  
**Install:** `pip install ".[dl,dev]" -c requirements.txt`  
**Core:** `tomography_ml.localization.localize_multiview` · `tomography_ml.studies.m9_frozen_fusion` / `m9_e2e_geometry_fusion` / `m9_expert_xyz_mean`  
**Labeling:** unmarked = planned design. **Conclusion** = frozen protocol / repo contract.

---

## Read this first — one particle, many cameras, frozen M8

M9 asks how much a **camera orbit** buys on top of the frozen WIN 3J single-view block — not a new CNN search, and not illumination diversity (that is M10).

```text
M8 freeze (WIN 3J)  →  per-view h_i
09_0  mean of per-view expert xyz          (classical averaging)
09_1  frozen encoder + small fusion        (no explicit geometry)
09_2  e2e encoder + compact fusion + sinθ/cosθ
09_3  same e2e + geometry; larger fusion MLP (capacity upper bound)
```

**Conclusion — scientific story:** there is **one** particle position. Prefer `V` views → shared encoder → fusion → **one xyz**. Do not reopen CNN width, Fourier modes, downsampling, or the `h_i` cut-point. Geometry metadata is `sin(θ)/cos(θ)` only and **supplements** image evidence.

**Downstream (not M9):** M10 lighting fusion (`10_1`–`10_2`, Steps 1–3). See [`plans/milestone_10/10_lighting_fusion_plan.md`](../milestone_10/10_lighting_fusion_plan.md).

---

## Final Report vs local notebooks — coverage map

The Final Report is the **canonical runnable narrative**. Local notebooks under `notebooks/milestone_09/` keep **ablation evidence** omitted or condensed in the report. Algorithms stay in `src/`; notebooks stay thin.

| Local notebook | Stage | What it proves | Final Report | Notes |
|---|---|---|---|---|
| [`09_0_expert_xyz_mean.ipynb`](../../notebooks/milestone_09/09_0_expert_xyz_mean.ipynb) | 09_0 | One WIN 3J expert per sampled angle, then `mean_j(xyz_j)`; bias vs expert spread | **No** — report xyz-mean is a **shared-trunk** control | Gap fill |
| [`09_1A_frozen_fourier_fusion.ipynb`](../../notebooks/milestone_09/09_1A_frozen_fourier_fusion.ipynb) | 09_1A | Frozen **Fourier** Stage A + Stage B heads (mean-pool / DeepSets / ordered-concat) + LR grid | **Yes** — **M9 Step 1** Fourier half | Keep packing + LR ladder |
| [`09_1B_frozen_pooled_fusion.ipynb`](../../notebooks/milestone_09/09_1B_frozen_pooled_fusion.ipynb) | 09_1B | Same ladder on **GAP** trunk; Fourier vs pooled overlay | **Yes** — **M9 Step 1** pooled half | Combined figure is the gap |
| [`09_2A_e2e_fourier_geometry_fusion.ipynb`](../../notebooks/milestone_09/09_2A_e2e_fourier_geometry_fusion.ipynb) | 09_2A | E2e Fourier + camera sin/cos; **compact (09_2) and large (09_3)** heads | **Yes** — **M9 Step 2** Fourier | Keep e2e Fourier ladder |
| [`09_2B_e2e_pooled_geometry_fusion.ipynb`](../../notebooks/milestone_09/09_2B_e2e_pooled_geometry_fusion.ipynb) | 09_2B | E2e pooled + geometry; compact vs large; Fourier vs GAP overlay | **Yes** — **M9 Step 2** pooled | Combined figure is the gap |
| [`09_3_e2e_fourier_geometry_large_fusion.ipynb`](../../notebooks/milestone_09/09_3_e2e_fourier_geometry_large_fusion.ipynb) | 09_3 | Compact vs large fusion MLP on Fourier e2e (same protocol as 09_2) | **Yes** — large arm of Step 2 / 09_2A | Capacity-axis slice |

**Curation rule:** if a cell’s figure or table already exists in the Final Report, cite the report (or load CSV/checkpoint from `checkpoints/m9/`) instead of retraining. Fourier-vs-pooled RMSE bars live in `09_1B` / `09_2B`, not in the Fourier-only notebooks.

The report’s 09_0 control is **shared-trunk** `mean_i(Linear(h_i))`. The local 09_0 notebook is the **per-angle expert** baseline (independent WIN 3J models, then coordinate mean).

---

## Frozen substrate (Conclusion — do not retune)

```text
API            = m8_single_view_block_freeze() / SingleViewBlockFreezeRecord
architecture   = fourier_base_mlp → LocalizerSingleViewFourier
                 channels (16, 32, 64), downsample=base, MLP hidden 128
representation = anomaly_ref (delta capability path)
normalisation  = per_image_zscore (per-view)
corpus         = m8_1/single_particle · workbook configs/m8/localization_single_particle.xlsx
                 report ML filters opt_m8_high_001
schedule       = full orbit on disk (V=36); fusion studies may stride (report: 60° → V=6)
```

**Normative latent `h_i`:** after CNN + Fourier-coded pool + `Linear(C→128)→ReLU`, **before** the final `Linear(128,3)`. Fusion replaces that last layer. Pooled controls use the GAP `encode_latent` cut at the same hidden size.

```text
views  [B, V, 1, H, W]   # float; V>1 for M9
angles [B, V]            # degrees; 09_2/09_3 tokens use sinθ, cosθ
# NOT inputs: bear_mu_a, bear_mu_s
```

Mean-latent then Linear is **affine-equivalent** to xyz-mean under the frozen WIN 3J head — sanity check only, not a distinct hypothesis.

---

## Ladder (condensed)

| Stage | Trains | Geometry | Library (sketch) |
|---|---|---|---|
| **09_0** | per-angle experts (eval-time mean of xyz) | no | `ExpertXyzMeanLocalizer` |
| **09_1A** | fusion only; Fourier trunk frozen | no | `CompactLatentFusionLocalizer`, `FrozenEncoderDeepSetsLocalizer.for_09_1_fourier` |
| **09_1B** | fusion only; GAP trunk frozen | no | `for_09_1_no_fourier` / `*_frozen_pooled` |
| **09_2A** | trunk + compact fusion e2e | sin/cos | `GeometryAwareFourierFusionLocalizer.for_09_2` |
| **09_2B** | same, GAP trunk | sin/cos | `for_09_2_pooled` |
| **09_3** | same as 09_2; **larger** MLP | sin/cos | `for_09_3` / `for_09_3_pooled` (hidden 512, depth 2 vs 128/1) |

```text
09_1:  frozen h_i → {mean-pool | DeepSets φ-then-mean | ordered concat} → xyz
09_2:  token_i = [h_i, sin θ_i, cos θ_i] → compact fusion → xyz   (e2e)
09_3:  same tokens → large fusion → xyz                         (capacity bound)
```

**09_1 packing (Conclusion):** xyz-mean vs mean-pool MLP vs DeepSets vs ordered concat, **Fourier and pooled**, each learned head with its own Stage-B LR study. Report interpretation: Fourier advantage **persists** under frozen two-stage training but **shrinks** as fusion complexity grows; DeepSets > mean-pool, as expected.

**09_2 vs 09_3 (Conclusion):** isolates **fusion capacity** under the same e2e + geometry protocol. Report interpretation: e2e training **abolishes** the Fourier advantage on learned fusion heads (still present on single-view / xyz-mean controls). If 09_2 recovers most of 09_3, prefer compact fusion.

No transformers / attention (optional WIN 5 stretch ≠ M9). No per-view agreement loss as the main objective.

Checkpoints: `checkpoints/m9/m09_expert_xyz_mean.pt`, `m09_frozen_{fourier,pooled}_fusion.pt`, `m09_e2e_{fourier,pooled}_geometry_fusion.pt`.

---

## Evidence spine · notebook pattern

| Phase | Proves | Primary cite | Local after thinning |
|---|---|---|---|
| **M9.0** | Sample is `[V,C,H,W]` on the **same** M8 corpus; expert-mean vs single view | Report §4 (sample) | `09_0` diagnostics |
| **M9.1** | Frozen Fourier vs GAP across five fusion variants | Report **M9 Step 1** | `09_1A` / `09_1B` |
| **M9.2** | E2e + sin/cos; compact vs large; Fourier vs GAP | Report **M9 Step 2** | `09_2A` / `09_2B` |
| **M9.3** | Large-head Fourier e2e is capacity, not a new mechanism | Report Step 2 large arm | `09_3` |

Match M8: helpers/tests in `tomography_ml_validation/`; notebooks call `run_m9_*` + optional `run_installed_pytest_test(s)`; no absolute paths in outputs. Installable tests cover **forward / freeze / pattern-id** contracts (`tests/test_localize_multiview.py`) — not full training in CI.

---

## Guardrails · handoffs

```text
Do not reopen WIN 3J architecture, representation, or per-view z-score.
Do not treat angle metadata as a substitute for image evidence.
Do not train JPEG previews; float .raw.tif remains authoritative.
Do not duplicate Excel workbooks for fusion-only ablations.
Do not make 09_3 a different mechanism from 09_2 (capacity only).
Illumination / joint camera×light fusion → M10, not M9.
```

**From M8:** `m8_single_view_block_freeze()`; same `m8_1` sequences; V=1 stills vs V>1 orbits.  
**To M10:** if fusion-head capacity gains are limited, remaining error is more likely **physical information** than camera merging.

*Bump experiment IDs / freeze records when fusion-pattern meaning changes — do not silently rewrite 09_1/09_2/09_3.*
