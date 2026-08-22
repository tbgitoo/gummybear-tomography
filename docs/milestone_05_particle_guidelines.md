# Milestone 5 — Particle artifact guidelines

Normative plan: [`plans/milestone_05/05_particle_plan.md`](../plans/milestone_05/05_particle_plan.md).  
Broader physics ladder: [`physics_model.md`](physics_model.md) § M5.  
Evidence: [`notebooks/milestone_05/`](../notebooks/milestone_05/) (`05a` → `05d`).

**Status:** M5A–M5D package APIs, validation helpers, and evidence notebooks are **Done** in this repo. M6 sequence generation consumes the same contracts.

---

## Purpose

M5 adds localized **analytic particle inclusions** on top of the completed M4 direct-plus-diffuse pipeline.

> Particles perturb ray transport and source deposition. They do **not** trigger diffusion mesh regeneration or change the FEM operator `A`.

```text
STL geometry
  → ray transport (M3 segments)
  → exact ray–tet deposition (M4B)
  → FEM diffusion solve (M4C)
  → camera hit-point sampling (M4D; nodal interpolation by default)
  → direct + diffuse composition (M4E)
```

M5 inserts particle intersections and a transport-derived source delta into that backbone; everything downstream reuses the same mesh, operator, and camera geometry.

---

## Central invariant — delta trajectory

**Path pairs carry the impact; particle records do not. Pairing grain is the transport path, not the intersection event.**

```text
M5 is a clean-vs-perturbed transport comparison layer.

ParticleIntersectionEvent  →  construction input / diagnostic (0..N per path)
AffectedTransportPair        →  canonical accounting: ONE per affected path_id
Source / image deltas        →  dirty − clean transport contributions
```

```text
clean path transport
  + particle intersections
  → one AffectedTransportPair per hit transport path
      (multiple in-line hits → multiple intervals inside that pair)
  → ΔE_background + ΔE_particle_scat
  → E_particle / S_particle
  → Φ_particle (same A, new RHS)
  → I_particle;  I_anomaly = I_particle − I_clean
```

**Mandatory contracts:**

- M5B **must** produce **one** `AffectedTransportPair` per affected path (`ray_ids`), not one pair per event.
- M5C **must** consume pairs + `E_clean_elem`; **must not** infer background correction from particle records alone.
- `ray_ids` on `RaySegmentBundle` is transport-path identity — **not** `segment_index`.

**Repo reality (checked):**

| Plan concept | Implementation |
|--------------|----------------|
| Analytic spheres | `gummybear.particles.geometry` — `ParticleSphere`, `ParticleSet`, `segment_sphere_intersection`, `intersect_segments_with_particles` |
| Clean/dirty pairs | `gummybear.particles.perturbation` — `build_affected_transport_pairs`, `AffectedTransportPair` |
| Source correction | `compute_transport_source_correction` → `TransportSourceCorrectionResult` (`source_model = "affected_transport_pair_delta"`) |
| Per-pair deposition inspect | `deposit_transport_pair_sources`, `deposit_particle_scatter_sources` |
| Segment lineage | `gummybear.optics.source_deposition.RaySegmentBundle` — `ray_ids`, optional `path_order` / `segment_ids`; `n_rays` = segment count |
| M5D reference run | `gummybear_validation.milestone_05.simulate.run_m5d_simulation`, `M5DSimulationConfig` |
| Executable checks | `gummybear_validation.milestone_05.validation` + `tests/test_particle_optics_contracts.py`, `tests/test_particle_placement.py` |
| Sequence-gen hook | `gummybear.datasets.sequence_generation` uses same pair/correction path when M5 tier is selected |
| Forward-model tier | `m5_refractive_diffusion_particle_perturbation` (workbook / manifest metadata) |

---

## Core architectural decisions

### Analytic particles — not meshed

Baseline atom: `ParticleSphere(center, radius, mu_abs, mu_scat)`.

**Runtime unit is always `ParticleSet`.** The one-particle case is a length-1 set (`ParticleSet.from_particles([sphere])`), not a bare sphere on production APIs — so single- and multi-particle paths share overlap validation, manifest `count`/`items`, and M5B–M5D call sites.

- Particles live in continuous space; they are **not** tetrahedralized.
- They do **not** trigger remeshing or local coefficient fields (`μ_a(x)`, `μ_s(x)`, `D(x)`).
- Particle size is far below coarse diffusion tet scale; resolving inclusions in the mesh would destroy M4 efficiency.

### Segment-level perturbation

Particles interact with existing in-object **ray segments** from M4B (`in_object_segments_from_rays` on M3 refracted rays).

They modify:

```text
Ray intensity along dirty intervals
Local scattered energy (particle scatter source)
Later (deferred): ray direction (refractive particle)
```

They do **not** modify the diffusion mesh or operator.

### Absorbing / scattering sphere

Total extinction drives attenuation:

```text
μ_particle_total = μ_abs + μ_scat
I_out = I_in · exp(−μ_particle_total · L)
E_loss = I_in − I_out
E_abs  = E_loss · μ_abs / μ_particle_total    # irreversible; NOT in S(x)
E_scat = E_loss · μ_scat / μ_particle_total   # local particle-scatter source
```

Pure absorber: attenuates only. Pure scatterer: attenuates ballistic channel **and** feeds `S`. Mixed: both, without double-counting or dumping absorption into `S`.

**Replacement model:** inside a dirty particle interval, particle coefficients define optical behavior; do not double-count background scatter in the same chord volume.

---

## Source deposition strategy

### Reuse M4 — do not recompute the world

Start from clean M4 deposition (`deposit_ray_source` → `E_clean_elem`, `S_clean`). Apply **localized** corrections for affected transport paths only. Full redeposition of all rays is a validation fallback, not the default.

### Transport-derived source delta

```text
ΔE_background_elem = deposit(dirty) − deposit(clean)     # may be negative (shadow)
ΔE_particle_scat_elem = attenuated_chord scatter on dirty path
ΔE_transport_elem = ΔE_background_elem + ΔE_particle_scat_elem
E_particle_elem = E_clean_elem + ΔE_transport_elem
S_particle = E_particle_elem / volumes
```

Keep `E_*` (integrated energy) and `S_*` (energy density) separate — same convention as M4.

### Validated particle-scatter assignment

```text
assignment   = "attenuated_chord"
distribution = "exact_ray_tet_beer_lambert"
```

Evaluates Beer–Lambert intensity along the entry→exit chord and distributes scatter through exact ray–tet intervals.

**Rejected models** (historical failure modes):

- **Mid-chord / representative-point** — can conserve ∫E while placing energy in wrong tets.
- **Uniform chord** — can conserve ∫E while hiding entry-side enhancement and producing incoherent shadows.

Aggregate accounting identities alone do **not** validate spatial placement; profile and 3D source-delta views were required to expose these bugs.

---

## Ray–sphere interaction contract

Per segment–particle overlap (after clipping to `t ∈ [0, 1]`):

```text
segment_index, particle_index
entry_t, exit_t, entry_point, exit_point
path_length_inside_particle
midpoint_inside_particle   # diagnostic only — NOT source assignment
```

Multiple hits on one path: sort by path order (sequential in-line hits are fine). Geometric sphere overlap is **forbidden**: `ParticleSet.from_particles` / `require_valid` raise `ParticleOverlapError` when `distance < r_i + r_j` (touch at a point allowed). Overlapping clipped chords on one segment also fail in `build_affected_transport_pairs` (`overlap_composition: "unsupported"`). Reason: replacement inclusion would otherwise leave μ multiply defined in shared volume, and clean/dirty interval partitioning would be ambiguous — see plan § Non-overlapping particles.

---

## Camera, diffusion, and cache policy

**Diffusion:** reuse mesh + operator `A`; particle runs change RHS `S` only. Re-solving `Φ` is acceptable — FEM is not the current bottleneck.

**Camera:** `sample_diffuse_image(..., interpolate=True)` by default. **Do not read this as nearest-tet + average.** Production mode is: containing tet if possible, else nearest usable tet, then **linear** barycentric blend `Φ(p)=Σ w_k Φ_k` (slight extrapolation when the STL hit sits outside the coarse volume). `interpolate=False` is the legacy nearest-centroid + mean-of-four path (historical M4 debug). Interpolation affects fluence→pixel mapping only.

**Direct channel (baseline M5):** same M3/M4E pipeline with particle-**attenuated** ray weights — no bend, refract, or new optical paths.

**Invalidate on particle change:** intersections, `S`/`Φ`, `I_diffuse`, `I_direct`, `I_total`.

**Do not invalidate:** STL, diffusion mesh, camera rays/hits (pose fixed), FEM operator (background coeffs unchanged).

---

## Validation layers

Evidence target: `notebooks/milestone_05/05c_particle_delta.ipynb` (algorithms in package, notebook inspects).

### 1. Accounting / construction

- Pairs drive correction; no inference from particle records alone.
- `ΔE_transport = ΔE_background + ΔE_particle_scat`; `E_particle = E_clean + ΔE_transport`.
- Pure absorber → zero particle-scatter source.
- Deterministic assignment; metadata `source_model == "affected_transport_pair_delta"`.

### 2. Physics-facing plausibility

Along affected paths:

- No source delta before particle entry.
- Strong `μ_s` → local entry enhancement possible.
- Attenuation through particle; downstream dirty background ≤ clean (shadow).
- 3D ΔS: localized positive particle-scatter + downstream negative shadow.

These are **contract and plausibility checks**, not a claim of general physical realism.

### M5D integrative check

`run_m5d_simulation(...)` + `05d_particle_diffusion.ipynb` demonstrate:

```text
particle → ΔS → ΔΦ → ΔI
```

without remeshing. Particles need not be directly silhouette-visible; camera sees smooth transport-induced perturbation. Helper favors inspectability — **M6** owns production disk/memory/batching.

---

## Experimental conclusions

From M5 evidence and package validation (aligned with [`05_particle_plan.md`](../plans/milestone_05/05_particle_plan.md) § Experimental conclusions):

1. **Ray pairs carry the impact** — durable signal is dirty−clean **transport trajectory** on affected paths; image anomalies inherit that ledger.
2. **Spatial placement beats aggregate accounting** — midpoint/uniform chord passed some ∫E checks with wrong local patterns.
3. **No remesh / no new operator** — modify `S` only; reuse M4 mesh and `A`.
4. **Multi-particle physics-ready** — ordered `ParticleSet`, lateral and in-line configs; workbook/cache/`n_particles` labels → M6/M7.
5. **M5D proves causal chain** — reference helper ≠ production sequence generator.

---

## Key lessons from M4

1. Exact geometry beats convenient approximations.
2. Do not remesh for particle changes.
3. Reuse the FEM backbone (`A` fixed, RHS changes).
4. Preserve inspectable intermediate artifacts (`delta_E_*` split, pair objects).
5. Prefer cache-aware, composable helpers over monolithic renderers.
6. Keep absorption vs scatter bookkeeping explicit.
7. For ML PoP at coarse tet budgets, M4 evidence favors diffusion-only (`alpha = 0`) unless hybrid lensing is required — see [`plans/milestone_04/04_diffusion_plan.md`](../plans/milestone_04/04_diffusion_plan.md).

---

## Key APIs (quick reference)

```python
from gummybear.particles import (
    ParticleSphere,
    ParticleSet,
    intersect_segments_with_particles,
    build_affected_transport_pairs,
    compute_transport_source_correction,
)
from gummybear_validation.milestone_05 import run_m5d_simulation, M5DSimulationConfig
```

**Deferred:** refractive particle deflection, Monte Carlo particle optics, particle-scale FEM, image-space painted anomalies, production M6 caching at M5D helper fidelity.

---

## Architectural summary

```text
M4:  clean translucent bear (bulk diffusion + optional direct)
M5:  same backbone + localized analytic perturbations
     anomaly = I_particle − I_clean in documented composition domain
```

Particles are ray-interaction objects that change transport and source terms. They do not regenerate the diffusion mesh and do not introduce particle-resolved tet geometry in baseline M5.
