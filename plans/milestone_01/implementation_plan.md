# Milestone 01 — STL Inspection: Implementation Plan (Human-First Revision)

**Source:** `plans/00_architecture.md` Section 5 (Milestone 1)  
**Status:** **Implemented** in spirit (`gummybear.geometry`, inspect notebook). Keep as historical human-first plan.  
**Scope:** Establish trust in `cad/proto_bear.stl` before any projection work.

### As implemented in this repository

| Plan item | What landed |
|-----------|-------------|
| `load_stl` / `describe_mesh` | `src/gummybear/geometry/io.py`, `summary.py` |
| `validate_mesh` | Named `validate_mesh_for_projection` (+ `require_projection_ready`); same essential checks |
| Optional `read_mesh_summary` | `inspect_stl(path)` → `{mesh, summary, validation}` |
| Summary fields + `stl_sha256` / `units` | Present; `source_path` is repo-relative when supplied |
| Notebook | `notebooks/milestone_01/01_inspect_stl.ipynb` (M1 trust gate; M0 keeps a thinner install smoke notebook) |
| Units note in `docs/physics_model.md` | **Deferred** |
| Essential invariant tests | **Not present**. Visual inspection and downstream milestone validation were considered sufficient for this research project. |

Extra helpers beyond the plan (`face_centroids`, vector normalize) exist for later milestones; not in scope but useful for later milestones.

---

## Purpose

Milestone 01 is a mesh trust gate.

The goal is not to build infrastructure. The goal is to answer:

> Can this STL be used as a reliable input for Milestone 02 path-length projection?

By the end of the milestone we should know:

- the STL loads successfully,
- the mesh is watertight (or not),
- the mesh dimensions are plausible,
- the mesh appears visually correct,
- the units assumption is documented.

---

## Human-First Philosophy

This repository is primarily a research and learning project; plan layout for human implementation.

Optimize for:

- shortest path to new information,
- shortest path to validating assumptions,
- proof-of-principle implementations,
- scientific progress over software ceremony.

Avoid:

- enterprise architecture,
- excessive abstraction,
- speculative frameworks,
- tests that exist only to increase coverage.

When choosing between a theoretically cleaner solution and a simpler proof-of-principle implementation, prefer the simpler solution unless it would block later milestones.

---

## Non-Goals

Explicitly out of scope:

- ray tracing,
- path-length projection,
- attenuation models,
- datasets and manifests,
- anomaly generation,
- machine learning,
- decomposition models,
- FEM / NGSolve,
- export systems,
- plugin architectures,
- mesh repair tooling.

---

## Deliverables

1. STL load helper.
2. Deterministic mesh summary.
3. Watertight validation.
4. Basic STL inspection notebook.
5. Units note in `docs/physics_model.md`.
6. Small set of meaningful tests.

---

## Proposed API

```python
load_stl(path)
describe_mesh(mesh)
validate_mesh(mesh)
```

Optional convenience helper:

```python
read_mesh_summary(path)
```

Keep the API intentionally small.

---

## Mesh Summary Contract

Suggested fields:

```text
vertices
faces
bbox_min
bbox_max
bbox_size
centroid
area
volume
is_watertight
is_winding_consistent
stl_sha256
units
```

Values should be JSON-serializable.

No sidecar file is required in Milestone 01.

The notebook printing and inspecting the summary is sufficient.

Persisted artifacts should only be introduced when a later milestone has a demonstrated need.

---

## Validation Checks

### Essential

- STL file loads.
- Vertex count > 0.
- Face count > 0.
- Bounding-box dimensions are non-zero.
- Mesh is watertight.
- SHA-256 hash recorded.

### Human Validation

The researcher should compare at least one STL dimension against FreeCAD.

This is the most reliable protection against hidden unit mistakes.

### Informational Only

- winding consistency,
- area,
- volume.

Useful to inspect but not necessarily blockers.

---

## Notebook Outline

`notebooks/01_inspect_stl.ipynb`

1. Load STL.
2. Print summary.
3. Run validation checks.
4. Render mesh.
5. Compare dimensions against FreeCAD.
6. Record observations.
7. State readiness for Milestone 02.

The notebook should call package APIs.

Core logic should not live in notebook cells.

---

## Testing Philosophy

Tests are scientific invariants.

A test should normally protect one of:

- reproducibility,
- geometry correctness,
- projection readiness,
- a future regression.

Avoid tests that merely increase coverage.

### Essential Tests

1. Watertight mesh passes validation.
2. Non-watertight mesh fails validation.
3. Summary generation is deterministic.
4. Summary contract contains required keys.

### Optional Tests

- loader smoke test,
- hash consistency checks.

---

## Recommended Implementation Sequence

### Phase 1 — Load and Inspect

Implement:

```python
load_stl()
describe_mesh()
```

Goal:

Produce the first mesh summary.

### Phase 2 — Validation

Implement:

```python
validate_mesh()
```

Goal:

Determine if the mesh is projection-ready.

### Phase 3 — Notebook

Render the bear and inspect dimensions.

Goal:

Human confidence.

### Phase 4 — Documentation

Document units assumptions and findings.

Goal:

Knowledge preservation.

---

## Success Criteria

Milestone 01 is complete when:

- STL loading works.
- Mesh summary is generated.
- Watertightness is verified.
- Mesh has been visually inspected.
- At least one dimension has been cross-checked against FreeCAD.
- Units assumptions are documented.
- Minimal invariant tests pass.

---

## Deferred Items

Do not implement unless later milestones require them:

- mesh repair tools,
- geometry format expansion,
- CAD automation,
- advanced geometry metrics,
- persistent summary sidecars,
- heavy test suites.

---

## Handoff to Milestone 02

Milestone 02 may assume:

- STL loading works,
- mesh bounds are known,
- units assumptions are documented,
- mesh validation status is known.

Milestone 02 must still implement:

- ray generation,
- ray intersections,
- path-length accumulation,
- attenuation rendering,
- analytic primitive validation.

---

*Principle: get trustworthy geometry information as quickly as possible, then move on to projections.*
