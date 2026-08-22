# Physics Model

Normative architecture: [`plans/00_architecture.md`](../plans/00_architecture.md) (camera vs projection; face-centered illumination / camera sampling; M2→M5 appearance ladder).

## Sensing story

The canonical observation is a **camera image of a translucent object** under documented illumination — **not** a classical attenuation-projection / sinogram / shadow-map sample.

| Concept | Role |
|---------|------|
| Camera pass | Pixel → first visible mesh face (`hit_faces`, mask, depth; M2A; reused later) |
| Illumination pass | Transmittance on **mesh faces** first (M2B → M3) |
| \(L_{\mathrm{proxy}}[f]\) | Face-level thickness along illumination direction through face centroid |
| \(T_{\mathrm{face}}[f] = e^{-\mu L_{\mathrm{proxy}}[f]}\) | Beer–Lambert-like link (M2B) |
| Camera intensity \(I\) | \(I[\mathrm{pixel}] \approx I_{\mathrm{bg}}\, T_{\mathrm{face}}[\mathrm{hit\_faces}[\mathrm{pixel}]]\, g(b\cdot v)\) |
| \(\mathrm{OD} = -\log T\) | Optional diagnostic / additive domain |
| `b·v` | Coarse **global** beam↔view contrast — **not** full illumination |
| Surface normals | M2A diagnostics; M3 **interface** quantities on the illumination path |

## Two coupled passes (from M2B onward)

```text
illumination sampling space = mesh faces
camera sampling space       = camera pixels
```

```text
1. Camera pass (M2A):   pixel → ray → first hit → hit_faces / mask / depth
2. Illumination pass:   face centroid P_f → L_proxy along beam b → T_face[f]
3. Camera sampling:     I ≈ I_bg · T_face[hit_faces] · g(b·v)
```

**Core invariant:** M2B does **not** compute illumination thickness primarily in camera-pixel space. Illumination/transmittance is computed on mesh elements first; the camera pass only **samples** that face-level state.

**Warnings:**

- M2B does **not** claim the camera ray is the physical illumination path.
- A camera-ray chord / per-pixel thickness is only a **documented special-case fallback**, not the general model.
- Do **not** use raw camera→mesh ray directions in `g(b·v)`; observation directions are mesh→camera (typically `-ray_directions`).
- Face-level `T_face` may look faceted; smoothing is deferred.

## Staged forward model

### M2A — Camera pass

First-surface visibility: `hit_faces`, mask, depth, optional normals. Intensity may be constant on the object or documented `g(b·v)`. Trust the pipeline; photorealism is not required.

### M2B — Face-centered translucent proxy

```text
P_f = centroid(face f)
L_proxy[f] = mesh thickness along line(P_f, b_illumination)
T_face[f] = exp(-μ L_proxy[f])
```

Prefer split composition: sample beam/observation fields + `hit_faces` first, then form intensity from `T_face` and `g(b·v)`. Spatial variation should come mainly from `T_face`; `g(b·v)` is angular coupling. `L_proxy` is **not** the canonical ML observation. Documented limits: straight-line proxy, no Snell, no multiple paths, no caustics.

### M3 — Refractive direct transport

Refines the **illumination** pass (camera sampling/visibility unchanged):

```text
source rays → entry (point/normal) → refract (constant n) → propagate
           → exit (point/normal) → outgoing bookkeeping (I_direct / face_energy / b_out)
```

Internal beam direction before exit is generally **not** the external beam. Exit normals transform that direction into an outgoing contribution — they are **not** a license for Lambertian `n·b` / `n·v` shading. M3 alone is lensing / direct refractive contribution, **not** bulk translucent glow.

### M4 — Volumetric deposition + diffusion (FEM)

```text
M4A  Coarse tet mesh from STL (Netgen/NGSolve; `fem` extra; cached)
M4B  M3 segments → S(x) via exact ray–tet intervals
       mu_total = mu_s + mu_a;  dI/ds = -mu_total I
       ΔS ∝ μ_s · I · Δl on traversed tets; absorption irreversible (not in S)
M4C  Homogeneous solve: -div(D ∇Φ) + μ_a Φ = S
       Robin BC model = effective refractive boundary; extrapolation_length is macroscopic leakage
       (not Fresnel geometric thickness). Dirichlet Φ=0 is debug-only.
M4D  Camera hit → interpolate Φ → I_diffuse (default barycentric; nearest-tet = historical debug)
M4E  I_total = α I_direct + I_diffuse
```

**Warnings / non-obvious design decisions:**

- High-res **STL** is surface/ray authority; diffusion mesh is intentionally **coarser** and not authoritative.
- Netgen/NGSolve required only for M4+ generation (`pip install '.[fem]'`); M0–M3 and catalog/ML imports must not require FEM.
- Exact ray–tet intervals are the mainline deposition; projected-centroid deposition is historical only. It adds more noise and does not offer gains in execution speed.
- Particles / camera poses must **not** invalidate mesh or diffusion-operator caches by default.
- `interpolate=True` vs `False` changes only how scalar fluence maps to camera hits — not deposition, FEM solve, or transport physics.

### M5 — Particle-aware source correction

Particles come **after** M3+M4. Clean vs dirty transport pairs yield:

```text
ΔE_background = deposit(dirty) − deposit(clean)   # background tets
ΔE_particle   = scatter deposited along dirty particle chord
E_particle    = E_clean + ΔE_background + ΔE_particle
```

**Warnings / non-obvious design decisions:**

- Extinction = `mu_abs + mu_scat`; absorption does **not** enter `S`.
- Scatter uses Beer–Lambert intensity along the chord with exact ray–tet intervals — **not** midpoint lumping or uniform chord deposition (those can conserve energy but wrong spatial pattern).
- Do **not** remesh or change diffusion coefficients per particle distribution by default.
- Anomaly roles are `I_particle − I_clean` in the documented composition domain.
- Plausibility checks (entry enhancement, downstream dirty ≤ clean, …) are **not** a claim of full physical realism.
- `run_m5d_simulation` is an inspectable reference path, not a production M6 generator.

## What `b·v` is (and is not)

Alignment of illumination beam `b` with view `v` may give early contrast. That is **not** Lambertian surface shading and **not** a full illumination model. Do **not** treat `n·b` / `n·v` as the default physical story unless explicitly labeled as a debug proxy. In M2B, `g(b·v)` is global; `T_face` carries spatial variation.

## Units

FreeCAD / STL geometry dimensions are **millimetres**.

## Deferred physics

Face interpolation/smoothing; multiple paths; caustics; Monte Carlo / photon mapping; high-res tet of the visual STL; particle-resolved remeshing / inhomogeneous `D` as baseline; GPU path tracing; photoreal renderers. These must **not** block M0–M3. Sequence generation (M6) may use documented lower forward-model tiers.
