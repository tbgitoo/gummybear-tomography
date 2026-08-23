# Milestone 6 — Factorized sequence generation

**Source:** `plans/00_architecture.md` §5.4 · [`docs/milestone_06_execution_planning.md`](../../docs/milestone_06_execution_planning.md)  
**Role:** Workbook-driven, cache-aware, **output-idempotent** multi-role sequences around M5D. **Not** new physics, remeshing, or FEM-operator persistence.  
**Core:** `gummybear.datasets.{generation_workbook,generation_plan,cache_keys,source_cache,output_plan,sequence_generation,sequence_writer,role_images,manifest_writer}` · `gummybear_validation.milestone_06`  
**Install:** `pip install ".[fem,dev]" -c requirements.txt`  
**Evidence:** `notebooks/milestone_06/` — **Done** (`06_1_dryrun` → `06_5_matrix_sweeps`). Optional CLI / parallelism / corruptions remain non-blocking.

**Labeling:** unmarked = planned design. **Conclusion** = hardened by timings, rejected alternatives, or API shape.

---

## Generation order (reuse expensive physics)

```text
1. Clean deposition (illumination + transport + S_clean) — single most expensive step
2. Particle delta (AffectedTransportPair → S_particle) — second; keyed to clean cache
   (different step-1 settings ⇒ new particle cache)
3. Diffusion solve (S → Φ) — cheap (NGSolve); re-solve when Robin / g / α change;
   Φ / operators are NOT persisted
4. Camera views — individually cheap, numerous. Mesh hits (pose×STL) + Phi localization
   (tet id + barycentric weights) dominate; cache those geometry artifacts, load when
   available, apply to Φ from step 3 (not to S). See M5 plan for interpolate=True.
   Do NOT cache finished role-frame images.
```

```text
workbook → SequenceJobs → reconcile outputs (skip complete; fail stale)
  → group missing jobs by cache keys
      → clean optical cache → particle cache
          → diffusion (runtime FEM) → visibility / localization → roles + manifest
```

**Three layers (do not conflate):** (1) **output delta**, (2) **source-cache** hit/miss, (3) physics. Cache hit ≠ output complete; complete output ≠ caches present on disk.

---

## Cost hierarchy — Conclusion (M6.2 smoke)

| Stage | Share (approx.) | Persist? |
|-------|-----------------|----------|
| Clean source | ~47% (~30 s) | **Yes** — `.npz` + `.json` |
| Particle source | ~17% (~11 s) | **Yes** — from clean cache |
| Camera capture | ~35% (~23 s) | Hits / Phi localization **yes**; role frames **no** |
| Diffusion | ≪1% (~0.3 s) | **No** operator cache |

Apriori FEM would look expensive; measured, it is the cheapest stage (consumes ~0.3s, e.g. <1% of runtime). Optimize deposition reuse + geometry caches, not operator persistence, precisely because diffusion is cheap.

---

## Caches, keys, output delta

**Clean optical key:** STL/mesh identity, illumination, `mu_s`/`mu_a`, n, source schedule/deposition, algorithm versions. **Not** invalidated by: camera, Robin, `g`, `alpha_direct`, corruption, JPG quality. (`g` / `alpha_direct` live on **diffusion_setups**; `D = OpticalMaterialConfig(...).diffusion_coefficient`.)

**Particle key:** clean cache ID + particle geometry/optics/placement/algorithm — same non-invalidators. Optional `particle_group_id` → ordered `ParticleSet` (order in key; non-overlap via `require_valid`).

**Payloads:** versioned `.npz` + `.json` (sidecar last). Hit = complete pair + matching key/schema + required arrays + **exact diffusion-mesh alignment**. No HDF5/Zarr/DB; never serialize live FEM.

**Camera geometry (also M8 WIN 0F/0G):** pose×STL visibility; mesh×hits Phi localization. Re-apply to `Phi_nodes` across optical/particle regimes.

**Output identity:** `resolved_job_hash` owns `sequence_id` completeness with manifest + role files. Excludes workbook path/SHA, absolute paths, timestamps, `max_workers`, and (schema `1.6-m6-draft`+) **`split` / workbook `seed`**. Statuses (≠ cache `hit`/`miss_*`): `output_missing`, `output_complete_current`, `output_incomplete`, `output_stale_*`, `output_orphaned_not_requested`. Fail before physics on stale/changed; never silent overwrite; disabled rows do not delete outputs. Layout: `data/generated/<scenario>/` + sibling `_cache/`.

---

## Workbooks, roles, phases

| Workbook | Role |
|----------|------|
| `configs/m6/m6_generation_plan.xlsx` | Smoke / vertical slice |
| `configs/m6/m6_matrix_plan.xlsx` | M6.5 sweeps + idempotency |
| `configs/m6/m6_multi_particle.xlsx` | Multi-particle planning dry-run |

Sheets: `sequences`, `optical_setups`, `particles`, `diffusion_setups`, `camera_schedules`, `corruptions`. Parser: pandas + openpyxl → typed `SequenceJob`. Excel = control surface only.

Roles: required `clean/` `particle/` `observed/`; recommended `anomaly/` = `particle − clean` in named domain. JPG default (`q=95`); composition checks **pre-JPEG** when available. Frame order = zero-padded indices. Manifests: portable paths; `schema_version`; setup sheet/row provenance. Smoke: 128×128, 3–6 views, one sphere, `max_workers=1`; dataset default 224×224.

| Phase | What | Notebook |
|-------|------|----------|
| M6.1 | Dry-run grouping (no physics) | `06_1_dryrun.ipynb` |
| M6.2 | Serial smoke, `use_persistent_cache=False` | `06_2_generate_sequences_no_cache.ipynb` |
| M6.3 | Role/manifest validation | `06_3_sequence_validation.ipynb` |
| M6.4 | Source caches; forced MISS → HIT | `06_4_generate_sequences.ipynb` |
| M6.5 | Matrix; shared source IDs; rerun no-op; added-row delta | `06_5_matrix_sweeps.ipynb` |

Import: `notebooks/milestone_06/`. **Conclusion:** keep M6.2 uncached under `m6_2/` for timing contract; M6.4 demos under `m6_4/`.

---

## Conclusions · guardrails · handoffs

**Conclusions:** logistics not physics; deposition dominates / diffusion does not; three planning layers; idempotency per `sequence_id`/`resolved_job_hash` (identical rerun = zero physics); workbook SHA = provenance only (i.e this is the Excel workbook SHA; for caching geometry sequenc characteristics are used instead). Since once camera rays and intersections are known, image rendering is relatively cheap, only geometry is cached; there are no separate image caches.

```text
No M5 redesign / remesh / FEM handle persistence / clean recompute per pose.
No silent overwrite of changed sequence_id; no delete on disabled rows.
No conflating cache hit with output completeness; no workbook-SHA-only identity.
No absolute paths in manifests; no unstated JPEG residual identity.
No committed corpora under data/generated/.
```

**From M5:** pairs / `ParticleSet` / `interpolate=True`; M5D is reference only. **To M7:** sequences + manifests; split by sequence identity. **To M8+:** dense orbits lean on visibility/localization; fix `source_intensity` across optical regimes when comparing.

*Bump `schema_version` / cache-key / payload / job-hash algorithm versions on semantic change.*
