# Milestone 8 — Single-view localisation (frozen foundation)

This plan does not attempt to reproduce the full ML narrative from the Final Report.

Its purpose is to document ablations, freeze decisions, and supporting evidence that were condensed or omitted from the report.

**Source:** `plans/00_architecture.md` §5.6  
**Primary evidence:** [`GummyBearTomography_Final_Report.ipynb`](../../GummyBearTomography_Final_Report.ipynb) — dataset pipeline (§4) + M8 ML steps 1–4 + conclusions  
**Role:** Excel-driven multi-regime corpora → **defensible single-view localiser** (Fourier-base primary) → **WIN 3J freeze** for M9/M10  
**Install:** `pip install ".[dl,dev]" -c requirements.txt` (catalog from M7; generation `.[fem]` when regenerating)  

**Labeling:** unmarked = planned design. **Conclusion** = frozen protocol / repo contract.

---

## Read this first — bottleneck isolation, not a CNN leaderboard

M8 asks whether a **single diffuse view** can localise a particle, and **which mechanisms** preserve spatial information well enough to freeze for multi-view work.

```text
Phase 0 — Excel → generate → raw-float catalog (multi optical regime, fixed instrument)
WIN 1–2 — training-scale corpus + shared CNN/torch spine
WIN 3A — spatial readout ladder on delta: avg-pool ≺ Fourier ≺ Flatten (parameter economy)
WIN 3B–3D — secondary RF / channel / head checks → retain Fourier-base triad
WIN 3E — formal architecture freeze (Fourier primary; Flatten positive; pool negative)
WIN 3F–3I — representation, normalisation, regime, per-axis observability
WIN 3J — freeze single-view block (architecture + delta + per-view z-score + protocol)
```

**Conclusion — scientific story (inscribed):** spatial representation is the dominant single-view bottleneck; **Fourier-coded pooling** recovers most Flatten benefit at far lower learned parameter cost; **per-view z-score** is the standard intensity normalisation for downstream work; **delta (`anomaly_ref`)** is the capability/oracle path frozen at 3J while **observed** remains the operational restoration target.

**Downstream (not M8):** M9 camera-view fusion (`09_0`–`09_3`); M10 lighting fusion (`10_0`–`10_2`). See those milestone plans.

---

## Final Report vs local notebooks — coverage map

The Final Report is the **canonical runnable narrative** for this repository release. Local notebooks under `notebooks/milestone_08/` reproduce **ablation evidence** omitted or condensed in the report. Re-run them here; CSV outputs land under `checkpoints/m8/m08_3*/`.

| Local notebook | WIN | What it proves | Final Report | Notes |
|---|---|---|---|---|
| [`08_0_regime_validation.ipynb`](../../notebooks/milestone_08/08_0_regime_validation.ipynb) | 0B–0E | Tiny `m8_0` corpus; raw-float roles; **regime intensity histograms**; catalog load | **Partial** — §4 pipeline + generation; **missing 0E regime panels** | Gap fill |
| — (cite report §4) | 1 | Training-scale `m8_1/single_particle` (~300×3 regimes) | **Yes** — §4 M8 fixed-illumination dataset | — |
| [`08_2_encoder_proof_of_concept.ipynb`](../../notebooks/milestone_08/08_2_encoder_proof_of_concept.ipynb) | 2 | Encoder forward + **single-sequence overfit proof of concept** | **Partial** — one forward pass in §5 setup only | Gap fill |
| — (cite report Step 1) | 3A.0 | LR grid; canonical LRs per readout | **Yes** — **M8 Step 1** | — |
| [`08_3a1_overfit_ladder.ipynb`](../../notebooks/milestone_08/08_3a1_overfit_ladder.ipynb) | 3A.1 | Memorisation ladder (1→N sequences) | **No** | Runnable locally |
| — (cite report Steps 2–4) | 3A.2 | Train→val→test triad; z then xyz; 3 repeats | **Yes** — **M8 Steps 2–3** (+ report **Step 4** split sensitivity) | — |
| [`08_3b_receptive_field.ipynb`](../../notebooks/milestone_08/08_3b_receptive_field.ipynb) | 3B | RF / downsampling grid (`win3b_*`) | **No** | Runnable locally |
| [`08_3c_channel_capacity.ipynb`](../../notebooks/milestone_08/08_3c_channel_capacity.ipynb) | 3C | Channel-width grid; **Fourier-base retention** | **No** | Runnable locally |
| [`08_3d_head_expressiveness.ipynb`](../../notebooks/milestone_08/08_3d_head_expressiveness.ipynb) | 3D | Linear vs MLP head on triad | **No** | Runnable locally |
| [`08_3e_architecture_freeze.ipynb`](../../notebooks/milestone_08/08_3e_architecture_freeze.ipynb) | 3E | `ArchitectureFreezeRecord`; confirmatory triad | **No** (assumed via `m8_single_view_block_freeze()` in M9) | Runnable locally |
| [`08_3f_representation.ipynb`](../../notebooks/milestone_08/08_3f_representation.ipynb) | 3F | delta vs clean vs observed (`win3f_*`) | **No** (report trains on anomaly only) | Runnable locally |
| [`08_3g_normalisation.ipynb`](../../notebooks/milestone_08/08_3g_normalisation.ipynb) | 3G | raw / global z / per-view z / min–max ablation | **Partial** — report **uses** per-view z-score; ablation tables absent | Runnable locally |
| [`08_3h_optical_regime.ipynb`](../../notebooks/milestone_08/08_3h_optical_regime.ipynb) | 3H | low / med / high regime sweep | **No** — report ML filters **`opt_m8_high_001` only** | Runnable locally |
| [`08_3i_observability.ipynb`](../../notebooks/milestone_08/08_3i_observability.ipynb) | 3I | RMSE_X/Y/Z consolidation + confirmatory bars | **Partial** — xyz totals in Steps 2–3; axis story not emphasised | Runnable locally |
| [`08_3j_single_view_freeze.ipynb`](../../notebooks/milestone_08/08_3j_single_view_freeze.ipynb) | 3J | One-shot train/val/**test**; JSON/CSV freeze artefact | **No** dedicated M8 section | Runnable locally |

**Also absent from Final Report (infra wins, cite M6 plan / manifests instead):** WIN **0F** camera×mesh visibility cache; WIN **0G** Phi sampling localization cache.

---

## Dataset contracts (Phase 0 — frozen; detail in report §4)

| Rule | Meaning |
|------|---------|
| Excel = data spec | `configs/m8/*.xlsx`; humans edit workbooks; Python executes |
| Variable background optics | `optical_setups.mu_a` / `mu_s` vary by regime; **not** model inputs |
| Fixed instrument | Shared light position, `source_intensity`, camera schedule across regimes |
| Authoritative intensity | Float `.raw.tif` (`raw_float`); JPG is display-only |
| ML normalisation (downstream) | **Per-view z-score** after WIN 3G — do not pin M7 `jpeg_uint8` |
| Splits | Train / val / test by **`sequence_id`** (particle identity), never random patches |
| Single-view consumption | Full orbit on disk; M8 ML uses **one fixed camera angle** (180° in report protocol) |

**Conclusion:** catalog rows + lazy task datasets (M7) → localisation `DatasetTaskSpec` with `anomaly_ref` → `(V=1,C,H,W)` torch batch in training helpers.

Workbooks under `configs/m8/`: `localization_single_particle.xlsx`, multi-particle templates, `m8_demo.xlsx` smoke. Scenario outputs: `data/generated/m8_0/` (tiny), `data/generated/m8_1/single_particle/` (training-scale). Do **not** duplicate workbooks for architecture-only ablations.

---

## WIN 3 — condensed mechanism ladder

**Primary question:** Can one view localise a particle, and which architectural choices preserve spatial information?

**Spatial readout ladder (WIN 3A — Conclusion):**

```text
global avg-pool  → spatially blind negative control
Fourier-coded    → compact fixed-basis readout (primary after 3C/3E)
Flatten          → flexible readout upper bound (~4000× learned params vs pool)
```

Evaluate architecture mechanism studies on **delta** first (3A–3E); reopen **observed / clean** only in 3F after freeze.

**Retained triad after 3C (Conclusion):**

```text
Primary:   Fourier-base — channels (16,32,64), downsample=base, MLP head hidden=128
Positive:  Flatten at same 3A geometry
Negative:  global avg-pool at same 3A geometry
Library:   LocalizerSingleViewFourier
```

**Frozen protocol at 3J (Conclusion — reused by M9):**

```text
representation  = anomaly_ref (delta capability path)
normalisation   = per_image_zscore (per-view z-score)
training        = Adam + MSE; early-stop on val; architecture-specific LRs
reference corpus = gummybear / high (opt_m8_high_001) for report Steps 1–4
API             = win3j_single_view_freeze() / SingleViewBlockFreezeRecord
```

Study CSVs are written under `checkpoints/m8/m08_3*/` when notebooks are executed locally (optical corpora remain under `data/generated/`).

---

## Evidence spine (this repo)

| Phase | Proves | Primary cite | Notebook |
|-------|--------|--------------|----------|
| **M8.0** | Raw-float + regime intensity span (low/med/high exemplars) | Report §4 (partial) | `notebooks/milestone_08/08_0_regime_validation.ipynb` |
| **M8.1** | Training-scale workbook → on-disk sequences | Report §4 | — |
| **M8.2** | Encoder loads single-view roles; can overfit one image | Report §5 setup (partial) | `notebooks/milestone_08/08_2_encoder_proof_of_concept.ipynb` |
| **M8.3A.0–2** | LR + train/val/test triad + split sensitivity | Report M8 Steps 1–4 | — |
| **M8.3A.1** | Overfitability ladder (smoke triad) | — | `notebooks/milestone_08/08_3a1_overfit_ladder.ipynb` |
| **M8.3B–3E** | RF, channels, head, formal freeze record | — | `08_3b` … `08_3e` under `notebooks/milestone_08/` |
| **M8.3F–3I** | Representation, norm ablation, regimes, per-axis RMSE | Report uses z-score + delta only (partial) | `08_3f` … `08_3i` |
| **M8.3J** | Freeze contract for M9 handoff | M9 imports `m8_single_view_block_freeze()` | `notebooks/milestone_08/08_3j_single_view_freeze.ipynb` |

Notebook pattern (match M7): algorithms in `src/tomography_ml/localization/` + helpers/tests in `tomography_ml_validation/milestone_08/`; notebooks call helpers + optional `run_installed_pytest_test(s)`; no absolute paths in outputs.

Installable tests cover **contracts** (freeze record fields, grid shapes) — not full training in CI. Helpers: `m8_corpus_paths`, `pick_regime_exemplars`, `win3_grids_summary`, `assert_win3j_freeze_contract`, `run_win3*_…_study`.

---

## Model interface (Conclusion)

```python
views:  [B, V, 1, H, W]   # float; V=1 for M8 single-view studies
angles: [B, V]            # required for M9+; fixed 180° slice in M8 report
# NOT inputs: bear_mu_a, bear_mu_s
```

**Primary metric:** Euclidean centre RMSE on held-out `sequence_id`s + **RMSE_X / RMSE_Y / RMSE_Z** (WIN 3I). Radius error reported separately if predicted.

Shared backbone comparison (report + local ablations):

```text
Image → CNN → feature maps → {GAP | Fourier pool | Flatten} → MLP → (x,y,z)
```

Literature anchors (conceptual only): Deep Sets / classical fusion (M9 baselines); attention (optional stretch WIN 5, not M8 core).

---

## Guardrails · handoffs

```text
Do not train on JPEG previews or auto-contrast displays.
Do not feed background μa/μs into nets.
Do not duplicate Excel workbooks per architecture variant.
Do not reopen CNN/Fourier/head search inside M9/M10 fusion notebooks.
Do not silently change post-3C Fourier-base retention or post-3G per-view z-score.
Delta is oracle/capability path at 3J; observed remains operational target for restoration.
```

**From M7:** `CatalogTaskDataset`, `field_status`, schedule-consistent subsets, lazy `(V,C,H,W)` numpy roles.  
**To M9:** import `m8_single_view_block_freeze()` / per-view experts; camera orbit on **same** M8 corpus; fusion ladder `09_0`–`09_3` — [`plans/milestone_09/`](../milestone_09/) (when present).  
**To M10:** multi-illumination corpora + lighting fusion — [`plans/milestone_10/`](../milestone_10/) (when present).

*Bump `schema_version` / experiment IDs when freeze semantics change — do not silently rewrite WIN 3J meaning.*
