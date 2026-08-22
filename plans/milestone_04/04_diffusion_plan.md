# Milestone 4 — Volumetric diffusion layer

**Source:** `plans/00_architecture.md` §5 (M4)  
**Role:** Bulk **translucent** appearance — deposit M3 scattering losses as volumetric source `S(x)`, solve for fluence `Φ(x)`, sample diffuse camera channel, compose with M3 direct. Does **not** replace `I_direct`; adds `I_diffuse`.  
**Core:** `gummybear.optics.{diffusion_mesh,source_deposition,diffusion_solve,diffuse_sampling,hybrid_compose}`  
**Install:** `pip install ".[fem]" -c requirements.txt` (Netgen + NGSolve). M0–M3 and catalog/ML imports must not require FEM.  
**Evidence:** `notebooks/milestone_04/` — **Done** (`04a` → `04e`). **Status:** package + tests + evidence notebooks shipped.

---

## Read this first

**M3 → M4 handoff:** M3 yields refractive/lensing **`I_direct`** and in-object ray segments with Beer–Lambert losses along internal chords. M4 deposits **scattering** into a coarse volume mesh, solves diffusion, samples **`I_diffuse`** at camera **hit points**, then composes.

```text
M3 segments + material  →  S(x) on coarse tets  →  Φ(x) FEM solve  →  I_diffuse at hit_points
I_direct (M3)           →  compose_hybrid_image  →  I_total = alpha * I_direct + I_diffuse
```

**STL** remains the high-res surface/ray authority; the **diffusion tet mesh is intentionally coarser** and not surface-authoritative.

**Not in scope:** particles (M5), sequence generation (M6), ML, Monte Carlo transport, full RTE, high-res tet of the visual STL, monolithic renderer abstraction, replacing direct with diffusion, face-energy smoothing as fake volume diffusion.

**Docs:** [`docs/physics_model.md`](../docs/physics_model.md) § M4.

---

## Key design decision — NGSolve, not NumPy FD

Early planning included a **structured-grid finite-difference** diffusion prototype (pure NumPy/SciPy). That route was **abandoned**: it cannot preserve the **unstructured** coarse tet geometry derived from STL and is not an equivalent replacement for volumetric fluence on the bear.

**NGSolve FEM on Netgen tet meshes is the production path.** In practice it is fast enough that NumPy trials looked pointless — sparse Cholesky solves on ~10³–10⁴ elements cost far less wall time than mesh generation and ray–tet deposition. There is **no** SciPy/trimesh meshing or FD fallback; missing `.[fem]` raises `ImportError` with install guidance.

**Superseded (do not revive):** projected-centroid deposition; structured-grid FD diffusion. Centroid projection assigns energy to tets the ray never traverses (visible noise, no meaningful speed gain — face-plane tests dominate either way).

---

## Sub-milestones (M4A–M4E)

| Stage | Responsibility |
|-------|----------------|
| **M4A** | Netgen STL → `DiffusionMesh` generation and caching |
| **M4B** | Exact ray–tet source deposition (`S_clean`) with energy conservation |
| **M4C** | NGSolve diffusion solve (`Φ`) with Robin boundary |
| **M4D** | Camera hit-point sampling of `Φ` (`I_diffuse`) |
| **M4E** | Hybrid composition (`I_total = alpha * I_direct + I_diffuse`) |

**Modules:** M4A `diffusion_mesh.py` · M4B `source_deposition.py` · M4C `diffusion_solve.py` · M4D `diffuse_sampling.py` · M4E `hybrid_compose.py`.

**Production stack (conceptual):**

```python
diff_mesh = generate_diffusion_mesh(stl_path, target_elements=..., cache_dir=...)
segments = in_object_segments_from_rays(surface_mesh, refracted_rays, ...)
dep = deposit_ray_source(diff_mesh, segments, material=...)
phi = solve_diffusion(diff_mesh, dep.S_clean, D=..., mu_a=..., extrapolation_length=...)
I_diffuse = sample_diffuse_image(diff_mesh, phi.Phi_nodes, hit_points, valid, sample_shape, ...).I_diffuse
result = compose_hybrid_image(I_direct, I_diffuse, alpha=...)
```

End-to-end orchestration remains **notebook-first** (inspectable intermediates: mesh, `S`, `Φ`, channels) — not a black-box renderer.

---

## Source and diffusion conventions

- **`E_scat_elem[e]`** — element-integrated scattered energy. **`S_clean[e] = E_scat_elem[e] / volume[e]`** — FEM source term (energy/volume).
- Along each ray interval of length `Δl`: `total_loss = I_local * (1 - exp(-μ_total * Δl))`; scatter fraction `μ_s / μ_total` → `S`; absorption tracked, not in `S`.
- Input segments: `RaySegmentBundle` (closed); from `in_object_segments_from_rays` on M3 refracted rays.
- **`D`** from `default_diffusion_coefficient(μ_s, μ_a)` when not overridden; requires `μ_total > 0`.

---

## Specifics

**M4C — boundary and solve:** `-div(D ∇Φ) + μ_a Φ = S` on unstructured tets. Robin BC model `effective_refractive_boundary`: `(D/L) u v` on the mesh surface; **`extrapolation_length` is macroscopic leakage, not a Fresnel geometric length**. Dirichlet `Φ = 0` debug-only (`use_dirichlet_zero_debug=True`). Operator reuse keyed via `operator_cache_keys()`.

**M4C — mesh handle:** `solve_diffusion()` requires a **live** `DiffusionMesh.netgen_mesh`. Reload from npz-only cache drops the Netgen handle → solve fails until remesh (M5 cache-protocol follow-up).

**M4D — sampling:** Requires camera **`hit_points` `[N, 3]`** — `hit_faces` alone are insufficient for tet location. Default **`interpolate=True`**: barycentric nodal `Φ` in containing tet; **`interpolate=False`**: legacy nearest-tet mean nodal (M4 evidence notebooks). `I_diffuse = exitance_scale * Φ(hits)` — surface fluence sampling, not full directional exitance/BRDF. Masked pixels exactly **zero**; new camera pose **resamples only** (no remesh / re-solve).

**M4E — compose:** `alpha` scales **`I_direct` only**; `alpha = 0` ⇒ pure diffuse FEM baseline. Tier tag `FORWARD_MODEL_TIER = "m4_refractive_diffusion"`. See experimental conclusions for when to prefer diffusion-only.

---

## Experimental conclusions

From M4 evidence on `cad/proto_bear.stl` (`notebooks/milestone_04/`; typical `target_elements` ~10³):

1. **Direct channel reads coarse at practical mesh budgets** — Unless the diffusion tet mesh is **really dense**, both ray–tet scatter deposition and the M3 **`I_direct`** face channel look **relatively coarse** next to the smooth bulk **`I_diffuse`** field. Hybrid panels need substantially finer tets and/or higher direct ray counts before `I_direct` and `I_diffuse` feel visually matched.
2. **ML proof-of-principle: prefer diffusion-only** — For first ML / localisation experiments, **`alpha = 0`** (pure `I_diffuse`; no refractive direct term) is the preferred forward-model tier: smoother training labels, lower generation cost (no full direct ray budget per camera view), and less coupling to face-quantized direct artifacts at default mesh sizes. Re-enable hybrid (`alpha > 0`) when mesh density or dataset goals require refractive lensing detail.
3. **Separate bulk solve from camera sampling** — Fluence `Φ` and deposition `S` are pose-stable; only `I_diffuse` resampling changes with camera. Matches the M3 “cache field, many views” pattern and is the natural M6 dataset strategy at diffusion-only tiers.

**Evidence default:** `04e` baseline run uses `alpha = 0.0`; the notebook also plots an `alpha` sweep for hybrid comparison only.

---

## Guardrails

```text
No particle perturbations (M5). No sequence gen (M6). No ML in M4 package path.
Exact ray-tet deposition only — no projected-centroid alternative.
NGSolve FEM only — no structured-grid FD fallback.
No high-res tet of visual STL. No blocking M0–M3 on FEM install.
Separate bulk solve from camera sampling — cache Φ, resample for new poses.
Alpha scales I_direct, not I_diffuse — record in metadata.
```

---

## Cache and artifacts

Regenerate diffusion mesh when any of: STL hash, geometry ID, scale/units, meshing method, Netgen version, `target_elements` / `maxh` / grading / preprocess params change.

Generated artifacts under `data/generated/<run>/` (gitignored): mesh metadata JSON, `S`/`Φ` npz, diagnostic images, run manifest.

---

## Sanity checks

Mesh bbox ≈ STL bbox; element count in validated range. `S ≥ 0`; `S = 0` when `μ_s = 0`. `Φ` decreases when `μ_a` increases (same source). `I_diffuse` zero on camera mask. `sum(S_clean * volumes) ≈ total_scattered`. FEM residual norms near machine zero on tests.

---

## Handoffs

**From M3 — keep:** refracted internal segments (`refract_ray_bundle`, `in_object_segments_from_rays`), `OpticalMaterialConfig`, `compute_refractive_direct_image` / `I_direct`, camera `hit_points` + mask from M2A visibility.

**To M5+ — keep:** `DiffusionMesh`, deposition, `solve_diffusion`, `sample_diffuse_image`, `compose_hybrid_image`. **Add:** particle dirty/clean source pairs, perturbed deposition, anomaly roles. **Fix:** FEM-compatible mesh cache for cross-pipeline reuse.

**To M6 / ML — default tier:** diffusion-only (`alpha = 0`) for first proof-of-principle localization datasets unless hybrid refractive detail is explicitly required; record `alpha`, mesh element count, and ray budgets in manifest metadata.

---

## Evidence notebooks (import target)

| Notebook | Exercises |
|----------|-----------|
| `04a_diffusion_mesh.ipynb` | M4A — mesh generation, cache keys |
| `04b_ray_to_volume_deposition.ipynb` | M4B — synthetic + M3 segments → `S(x)` |
| `04c_diffusion_solve.ipynb` | M4C — FEM, `Φ`, residuals |
| `04d_camera_capture.ipynb` | M4D — hit-point sampling, `I_diffuse` |
| `04e_end_to_end_light_diffusion_camera.ipynb` | M4E — hybrid compose, alpha sweep |

**Implementation:** core APIs **Done**; evidence notebooks **`notebooks/milestone_04/`** (`04a`–`04e`); tests `test_m4_diffusion.py`, `test_m4_validation_helpers.py`.

---

*Bump `schema_version` / `preprocess_contract_version` on semantic change — do not silently rewrite meaning.*
