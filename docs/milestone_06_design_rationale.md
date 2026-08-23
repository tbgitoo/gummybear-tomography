# M6 Design Rationale

**Milestone:** M6 — factorized synthetic camera-sequence generation and caching  
**Status:** Supporting design; M6.1–M6.5 implemented  
**Authority:** normative phase / cache / manifest / output-delta contracts live in  
[`plans/milestone_06/06_sequence_generation_plan.md`](../plans/milestone_06/06_sequence_generation_plan.md).  
This document explains **why** that factorization exists (measurements → cache tiers). It must not override the plan.

---

## Motto

```text
Do expensive physics once.
Reuse it honestly.
Vary particles next.
Vary diffusion boundary cheaply.
Vary cameras last (reuse pose×mesh hits when schedules recur).
Record everything.
```

Naive (wrong):

```text
for every frame:
    recompute clean deposition → particle delta → diffusion → one camera image
```

Factorized (M6):

```text
workbook → SequenceJobs → reconcile outputs (skip complete; fail stale)
  → group missing jobs by cache keys
      → clean optical/source cache
          → particle/dirty source cache
              → diffusion solves (runtime FEM; not operator-cached)
                  → camera×mesh visibility / Phi localization (geometry caches)
                      → role images + manifest
```

M6 is logistics: make the **output delta** and the **expensive calculation graph** explicit. An identical second run against complete outputs must perform **no** physics.

---

## Measured cost hierarchy (why cache what)

### Smoke wall-clock (M6.2, uncached serial smoke — order of magnitude)

| Stage | Share (approx.) | Persist in M6? |
|-------|-----------------|----------------|
| Clean source deposition | ~47% (~30 s) | **Yes** — `.npz` + `.json` |
| Particle source (affected pairs → `S_particle`) | ~17% (~11 s) | **Yes** — keyed to clean cache |
| Camera capture (many views) | ~35% (~23 s) | Hits / Phi localization **yes**; finished role frames **no** |
| Diffusion solve | ≪1% (~0.3 s) | **No** FEM/operator cache — re-solve when Robin / `g` / `α` change |

### Cache-hit second pass (M6.4, same scientific inputs)

| Stage | Forced miss | Cache hit |
|-------|-------------|-----------|
| Clean source | ~30 s | ~0.7 s |
| Particle source | ~11 s | ≪0.1 s |
| Diffusion + camera | still run | still run (fields + frames not image-cached) |

**Conclusion:** a priori FEM looks expensive; measured, deposition dominates and diffusion is cheapest. Optimize **source reuse** and **geometry caches**, not operator persistence.

### Dense-orbit follow-on (motivates geometry caches)

Once clean/particle sources hit, per-view cost shifts: **visibility cast** and **tet localization** dominate at large `V`; applying cached barycentric weights to `Φ` is cheap. Diffusion stays ~tens of ms. Do **not** cache finished JPGs.

---

## Cache tiers

### 1. Clean optical / source (most expensive)

**Depends on:** STL/mesh identity, illumination, `mu_s`/`mu_a`, refractive index, source schedule/deposition method, algorithm versions.

**Payload (serializable):** source/refracted rays, transport segments, `E_clean` / `S_clean`, diagnostics + sidecar. Exact diffusion-mesh alignment required before accepting element arrays. Never pickle Netgen/NGSolve.

**Must not invalidate:** camera pose/schedule, Robin / extrapolation, diffusion-setup `g`, `alpha_direct`, corruption, JPG quality, `max_workers`, workbook SHA.

### 2. Particle / dirty source (second)

**Depends on:** full clean cache ID + particle geometry/optics/placement/algorithm.

**Payload:** transport deltas, `E_particle` / `S_particle`, affected-path diagnostics; linked to parent clean ID + same mesh alignment.

**Must not invalidate:** same camera / Robin / `g` / `α` / corruption list as above.

### 3. Diffusion (cheap — re-solve, do not persist operators)

Robin / `g` / `alpha_direct` change may rebuild the live operator and re-solve `Φ`; they do **not** bust source caches. Record settings as **provenance** only. `D = OpticalMaterialConfig(...).diffusion_coefficient` (`g` on diffusion sheet so it cannot bust the clean optical key).

### 4. Camera×mesh visibility (geometry)

Key: STL identity + intrinsics + pose fingerprint (+ algorithm version).  
Payload: valid mask, depths, faces, hit points — **not** rendered frames.  
Independent of μ, particles, and `Φ`. Reuse across optical regimes / clean vs particle.

### 5. Phi sampling localization (geometry)

Key: diffusion-mesh identity + `camera_visibility_cache_id` + interpolation algorithm.  
Payload: tet id + barycentric weights `(N, 4)`.  
Independent of `Φ` values / optics. Localize once per pose; apply to clean and particle `Φ`.

### 6. Apply Phi / write roles (late, cheap per cached hits)

`I_*` composition after fields exist. **No** persistent role-frame image cache in M6.

```text
Visibility cache     = hit geometry
Localization cache   = tet weights
Apply to Phi_nodes   = late cheap step
Role JPG cache       = out of scope  (output idempotency ≠ image cache)
```

---

## Three planning layers (do not conflate)

1. **Output delta** — `sequence_id` + `resolved_job_hash` + manifest/role files → skip complete, fail stale, never silent overwrite.  
2. **Source-cache plan** — hit/miss only for jobs that still need generation.  
3. **Physics / images** — deposit / solve / capture / write.

A source-cache hit does **not** prove sequence completeness; a complete sequence does **not** require cache files on disk. Workbook SHA is **provenance only** — not a scientific cache key or sole output identity.

Layout: `data/generated/<scenario>/` with sibling `_cache/` (e.g. `m6_5/`).

---

## Invalidation cheat-sheet

| Change | Clean cache | Particle cache | Diffusion re-solve | Visibility / localization |
|--------|:-----------:|:--------------:|:------------------:|:-------------------------:|
| Light / bear μ / source schedule | bust | bust (new clean parent) | yes | no* |
| Particle geometry / optics | keep | bust | yes | no* |
| Robin / `g` / `α` | keep | keep | yes | no* |
| Camera pose / schedule | keep | keep | no† | bust if pose/STL/mesh identity changes |
| Unrelated workbook row / SHA | keep | keep | no | no |

\*Unless mesh generation identity changes.  
†Re-compose / re-sample from existing or re-solved `Φ` as needed; sources stay.

---

## Workbook control surface (brief)

Excel is human-facing only (`configs/m6/m6_generation_plan.xlsx` smoke; `m6_matrix_plan.xlsx` sweeps). Parsed once → typed `SequenceJob`s. Sheets: `sequences`, `optical_setups`, `particles`, `diffusion_setups`, `camera_schedules`, `corruptions`. Scientific cache keys come from **resolved** optical/particle/mesh inputs, not from the workbook filename.

---

## Guardrails

```text
No clean deposition recompute per camera pose.
No FEM handle / operator persistence.
No finished role-frame image cache as a primary tier.
No workbook-SHA-only identity; no conflating cache hit with output completeness.
No silent overwrite of changed sequence_id; no delete on disabled rows.
No committed corpora under data/generated/.
```

**Evidence notebooks:** `notebooks/milestone_06/` (`06_1` dry-run → `06_2` uncached timing → `06_4` miss/hit → `06_5` shared caches + idempotent delta).

*Full contracts, statuses, and payload schemas: see the M6 plan. Bump cache-key / payload versions on semantic change.*
