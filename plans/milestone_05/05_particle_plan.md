# Milestone 5 — Analytic particle artifacts

**Source:** `plans/00_architecture.md` §5 (M5)  
**Role:** Localized **particle inclusions** on top of M4 — perturb ray transport and volumetric source `S`, then reuse the same mesh/operator/camera pipeline. Produces clean vs particle images and forward-model anomaly `I_particle − I_clean`.  
**Core:** `gummybear.particles.{geometry,perturbation,placement}` · validation `gummybear_validation.milestone_05.{simulate,validation}`  
**Install:** `pip install ".[fem]" -c requirements.txt` (reuses M4 Netgen/NGSolve). Particles themselves need no extra install.  
**Evidence:** `notebooks/milestone_05/` — **Done** (`05a` → `05d`). **Status:** package APIs + validation helpers + evidence notebooks shipped.

**Guidelines:** [`docs/milestone_05_particle_guidelines.md`](../../docs/milestone_05_particle_guidelines.md) · physics summary [`docs/physics_model.md`](../../docs/physics_model.md) § M5. Package APIs match the contracts below (`AffectedTransportPair`, `build_affected_transport_pairs`, `compute_transport_source_correction`, `run_m5d_simulation`).

**Labeling:** Unmarked text records the intended design, including portions subsequently implemented. Items tagged Conclusion identify decisions specifically hardened through evidence, rejected alternatives, or the final API.

---

## Read this first

**Conclusion — central invariant:** path pairs, not particle records or per-hit pairs.

```text
M5 is a clean-vs-perturbed transport comparison layer.

ParticleIntersectionEvent constructs dirty intervals on a path.
The accounting object is AffectedTransportPair — ONE per affected transport path
  (RaySegmentBundle.ray_ids / path_id), not one per intersection event.
Multiple inclusions on the same ray → multiple events/intervals INSIDE that one pair.
Source and image deltas = dirty − clean transport contributions.
Particle records alone are NOT the source-correction ledger.
```

```text
M4 backbone (unchanged mesh / operator A)
  + particle intersections on in-object segments
  → one AffectedTransportPair per hit path_id
  → ΔE_background + ΔE_particle_scat → E_particle / S_particle
  → Φ_particle (same A, new RHS) → I_particle → I_anomaly
```

**Non-negotiable:** particles do **not** remesh; do **not** change `A`; do **not** introduce particle-scale tets or local `μ(x)` / `D(x)` fields.

**Not in scope:** refractive particle deflection, Monte Carlo particle optics, image-space painted blobs, sequence generation (M6), workbook/catalog labels, production-scale caching.

---

## Key design decision — delta trajectory

Particles are **analytic spheres** in continuous space. Impact appears only where illumination paths cross inclusions. That much was planned; the pairing grain below is a **conclusion**.

| Ledger | Role |
|--------|------|
| `ParticleIntersectionEvent` | Geometry construction input / diagnostic (may be many per path) |
| `AffectedTransportPair` | **Canonical** clean/dirty accounting — **exactly one per affected `path_id`** |
| `delta_E_*` | Element-wise source correction from the pair collection |

**Conclusion — cardinality (do not confuse):**

```text
N intersection events on path P     →  still 1 AffectedTransportPair for P
N distinct hit paths                →  N AffectedTransportPair objects
```

In-line dual-hit rays keep both chords in that pair’s `particle_events` / `dirty_intervals`, with dirty intensity continuing through the first inclusion into the second.

**Invalid:** inferring background source correction from particle records alone; setting `ray_id = segment_index` as path identity; treating “one event ⇒ one pair.”

**Lineage (M4B contract, required):** `RaySegmentBundle` carries `ray_ids` (transport-path identity), `starts`/`ends`/`intensities`, optional `path_order` / `segment_ids`. Distinguishes `n_rays` (segments) vs `n_transport_paths`.

---

## Sub-milestones (M5A–M5D)

| Stage | Responsibility |
|-------|----------------|
| **M5A** | `ParticleSphere` / `ParticleSet` + segment–sphere intersections |
| **M5B** | Dirty transport + `AffectedTransportPair` for hit paths |
| **M5C** | Transport-derived source delta → `E_particle` / `S_particle` |
| **M5D** | Reuse M4 solve/sample/compose → clean / particle / delta images |

**Modules:** M5A `particles/geometry.py` · M5B–C `particles/perturbation.py` · placement `particles/placement.py` · M5D `gummybear_validation.milestone_05.simulate` (`run_m5d_simulation`).

**Production stack (conceptual):**

```python
events = intersect_segments_with_particles(segments, particles)
pairs = build_affected_transport_pairs(segments, particles, events=events)
corr = compute_transport_source_correction(E_clean_elem, pairs, volumes=...)
# S_particle = corr.S_particle; same A → Φ_particle → sample → compose
# I_anomaly = I_particle - I_clean   # documented composition domain
```

**Conclusion:** end-to-end orchestration is **notebook / `run_m5d_simulation` first** (inspectable intermediates) — not M6 sequence gen.

---

## Physics conventions

**Extinction (attenuation):** `μ_total = μ_abs + μ_scat`; `I_after = I_before · exp(−μ_total · L)`. Not abs-then-scat sequencing.

**Bookkeeping split:** `E_abs = E_loss · μ_abs/μ_total` (irreversible, **not** in `S`); `E_scat = E_loss · μ_scat/μ_total` → local particle-scatter source.

**Replacement model:** inside dirty particle intervals, particle coeffs define the medium; do not double-count background scatter in the same chord volume. Outside intervals, background remains.

**Source correction:**

```text
ΔE_background = deposit(dirty) − deposit(clean)     # may be negative (shadow)
ΔE_particle   = attenuated_chord scatter on dirty   # ≥ 0; exact ray–tet BL
ΔE_transport  = ΔE_background + ΔE_particle
E_particle    = E_clean + ΔE_transport
S_particle    = E_particle / volumes
```

**Conclusion — scatter assignment:** validated path is `attenuated_chord` / `exact_ray_tet_beer_lambert`. **Rejected in implementation:** mid-chord lumping; uniform chord (conserve ∫E but wrong spatial pattern).

**Baseline direct channel:** same M3/M4E pipeline with particle-**attenuated** weights only — no bend/refract/new paths.

**Conclusion — camera sampling of Φ (M4D — do not confuse the two modes):** Diffuse images sample nodal fluence at STL **camera hit points** (`hit_points`), not by painting tet energy onto surface triangles. Production / M5D / M6 use `sample_diffuse_image(..., interpolate=True)` (API default).

```text
interpolate=True  (production):
  1. Prefer a tet that CONTAINS the hit (barycentric weights ≈ in [0,1];
     KD-tree of tet centroids → nearby candidates, then all tets).
  2. If the hit sits slightly outside the coarse volume (STL ≠ tet skin):
     take the NEAREST usable tet and still use barycentric weights
     (linear EXTRAPOLATION past the faces).
  3. Φ(p) = Σ_k w_k · Φ_nodes[tet_k]   # linear blend of the four nodes

interpolate=False  (legacy / historical M4 debug only):
  nearest-centroid tet, then Φ(p) = mean(Φ at four corners)
  — NO spatial weights; piecewise-constant per tet.
```

**Mnemonic:** `True` ≠ “nearest tet + average.”  
`True` = **containing tet if possible, else nearest, then linear blend of its four nodes.**  
`False` = nearest tet + average. M5D and sequence generation must not silently revert to `False`.

---

## Specifics

**M5A — geometry:** `ParticleSphere(center, radius, mu_abs, mu_scat)` is the inclusion atom. **Conclusion:** the **runtime / API unit is always `ParticleSet`**; the one-particle case is `ParticleSet.from_particles([sphere])` (length 1) — **not** a bare `ParticleSphere` on production paths — so single- and multi-particle jobs share types, overlap validation, manifest block (`count` / `items`), and M5B–M5D call sites. **`ParticleSet.from_particles(..., require_non_overlapping=True)`** (default) calls `require_valid()` and raises **`ParticleOverlapError`** when any pair has `distance(centers) < r_i + r_j` (touching at a point is allowed; `gap_tol` default `0`). Events: `entry_t`/`exit_t` ∈ [0,1], chord length, midpoint (diagnostic only). Multi-hit sort by path order — **sequential** hits on one ray are fine; **volume-overlapping** spheres are not.

**M5B — pairs:** **Conclusion:** `build_affected_transport_pairs` emits **one `AffectedTransportPair` per affected `path_id`** (`ray_ids`). Intersection events only mark which paths are affected and supply ordered chords; they are **not** the pairing grain. On each hit path: build clean/dirty intervals; track `I_in`/`I_out`, `E_abs`/`E_scat`, particle-scatter source events (all events for that path live on the same pair). Second guard: if two particle chords overlap on the same segment (`entry_t < previous_exit`), raise `ValueError` (`overlap_composition: "unsupported"`). **Conclusion:** full redeposition of all rays = validation fallback only — not default.

**M5C — correction:** Consume pairs + `E_clean_elem`; expose inspectable `delta_E_background_elem`, `delta_E_particle_scat_elem`, `delta_E_transport_elem`, `source_model="affected_transport_pair_delta"`. **Conclusion — spatial signature:** unchanged before entry → optional entry enhancement (strong `μ_s`) → attenuation through particle → downstream shadow (`dirty ≤ clean` background).

**M5D — integration:** Same mesh/`A`; new RHS only. Camera diffuse uses **`interpolate=True`** (containing-tet / else-nearest + linear nodal blend — see Physics conventions). Outputs: `I_*_clean`, `I_*_particle`, `ΔI_*`, `ΔΦ`, source-delta viz. **Conclusion:** helper favors inspectability over disk/batch/speed — **M6** owns production scaling.

**Cache:** particle change invalidates intersections, `S`/`Φ`/`I_*`. Does **not** invalidate STL, diffusion mesh, `A`, camera rays/hits (pose fixed).

---

## Non-overlapping particles (repo contract)

**Conclusion:** multi-particle realizations must not geometrically intersect. Mandatory in the package, not a soft style preference.

**Why (physics + bookkeeping):**

1. **Multiply-defined spatial regions** — Dirty transport uses a **replacement** inclusion model: inside a particle interval, that particle’s `μ_abs` / `μ_scat` define the medium. Two spheres that occupy the same volume would leave extinction and scatter bookkeeping undefined (which particle’s coeffs apply? sum? max?).
2. **Ambiguous clean/dirty segment assignment** — Pair construction splits each affected path into ordered background vs particle intervals from non-overlapping chords. Overlapping chords on one segment cannot be ordered into a unique interval partition without an overlap-composition rule the repo deliberately does **not** implement.

**What the code does:**

| Layer | Location | Behavior |
|-------|----------|----------|
| Geometry | `ParticleSet.from_particles` / `require_valid` | Default reject: `distance < r_i + r_j` → `ParticleOverlapError` (“physically impossible for distinct inclusions”) |
| Transport pairs | `build_affected_transport_pairs` | If clipped chords overlap on a segment → `ValueError: Overlapping particle intervals are not supported`; metadata `overlap_composition: "unsupported"` |

**Allowed:** touching at a point; multiple particles on one path **in sequence** (in-line dual hits); lateral placements with disjoint ray sets. Evidence notebooks assert `ParticleSet.validate()` and reject unexpected dual-hit / overlap cases.

**Not allowed without a new milestone:** additive / union / max extinction composition for overlapping active-particle intervals; bypassing validation except in explicit test fixtures (`require_non_overlapping=False`).

---

## Conclusions (evidence / architecture)

Settled from M5 evidence (`05a`–`05d`), package validation, and implementation hardening — same **Conclusion** sense as above:

1. **Path pairs carry the impact** — Particle geometry / intersection events alone do not define the source delta. The durable ledger is **one dirty−clean `AffectedTransportPair` per affected transport path**; many hits on that path stay inside the same pair. Downstream image anomalies inherit that path-level ledger.
2. **Detailed spatial evaluation** beats aggregate accounting — Midpoint and uniform-chord scatter deposition passed some ∫E conservation checks while producing incorrect local patterns (speckle, incoherent shadow structure). The validated path requires Beer-Lambert chord deposition, exact ray-tet placement, and explicit entry/shadow profile inspection. Aggregate sums can catch bookkeeping errors but not wrong spatial distribution; nonlinear Beer-Lambert makes placement part of the physics.
3. **No remesh / no new operator** — Particle dims ≪ coarse tet size; resolving inclusions in the mesh destroys M4 efficiency. Modify `S` only; reuse mesh and `A`.
4. **Multi-particle is physics-ready only for non-overlapping spheres** — Ordered `ParticleSet`; in-line dual hits and lateral disjoint rays are supported. Geometric sphere overlap is rejected (`ParticleOverlapError` / unsupported chord overlap). Workbook grouping / cache keys / `n_particles` labels belong to **M6/M7**, not M5.
5. **M5D proves the causal chain** — `particle → ΔS → ΔΦ → ΔI` without remeshing. Particles need not be directly silhouette-visible; camera sees smooth transport-induced perturbation. Reference helper ≠ production generator.

---

## Guardrails

```text
No particle remeshing or particle-scale FEM.
No local μ(x)/D(x) coefficient fields as baseline.
No refractive particle deflection in baseline M5.
No geometrically overlapping particles (ParticleOverlapError / unsupported chord overlap).
No source correction from particle records alone.
No ray_id = segment_index as path identity.
No dumping E_abs into S(x).
No full redeposition as default path.
No painted image-space anomalies.
No ParticleRenderer / plugin frameworks.
```

---

## Handoffs

**From M4 — keep:** `RaySegmentBundle` + `ray_ids` lineage, `deposit_ray_source`, `DiffusionMesh` + live `netgen_mesh`, `solve_diffusion` / operator cache keys, `sample_diffuse_image`, `compose_hybrid_image`, camera `hit_points`.

**To M6 — keep:** `ParticleSet`, pairs, `compute_transport_source_correction`, clean/particle/anomaly image roles, `requires_remeshing: false` / `changes_diffusion_operator: false` metadata. **Add:** workbook jobs, particle-source cache keys, batch/disk/runtime. **Do not** treat `run_m5d_simulation` as production-scalable.

**Forward-model id (when M5 tier selected):** `m5_refractive_diffusion_particle_perturbation`.

---

## Evidence notebooks (import target)

| Notebook | Exercises |
|----------|-----------|
| `05a_particle_geometry.ipynb` | M5A — spheres, intersections, multi-particle events |
| `05b_ray_particle.ipynb` | M5B — pairs; single / lateral / in-line configs |
| `05c_particle_delta.ipynb` | M5C — ΔE split; entry scatter / downstream shadow |
| `05d_particle_diffusion.ipynb` | M5D — ΔS → ΔΦ → ΔI; optional two-particle hybrid |

**Implementation:** core APIs **Done** (`src/gummybear/particles/`); validation/simulate **Done**; evidence notebooks **`notebooks/milestone_05/`** (`05a`–`05d`); tests `test_particle_optics_contracts.py`, `test_particle_placement.py`, package validation module.

---

*Bump `schema_version` / `preprocess_contract_version` on semantic change — do not silently rewrite meaning.*
