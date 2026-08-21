# Milestone 0 — Project scaffold

> Historical planning document.
> Written before implementation of Milestone 0.
> Preserved for provenance and project history.

**Architecture:** [`00_architecture.md`](../00_architecture.md) §5 (M0 = package scaffold, install, smoke tests)  
**Companion notebook:** `notebooks/01_inspect_stl.ipynb` (thin; imports separately in simplified form)  
**Audience:** Future implementors / agents recreating or auditing M0

M0 ends when the repo is **installable, importable, and structurally correct**. Geometry logic that the STL notebook calls is **M1**; M0 only reserves the layout and notebook conventions so that notebook can stay thin.

---

## Conventions (M0 historical decisions)

| Rule | Detail |
|------|--------|
| Dist / import | Dist `gummybear-tomography`; import package `gummybear` under `src/` |
| Python | `==3.12.*` (`requires-python` in `pyproject.toml`) |
| Install | **No editable installs.** From repo root: `pip install ".[dev]" -c requirements.txt` (combine extras as needed, e.g. `".[dl,dev]"`). Never `pip install -e` or unconstrained `pip install <pkg>`. |
| Version | Same literal in `pyproject.toml` and `gummybear` (e.g. `0.0.1.dev0`) |
| Algorithms | Live in `src/`; notebooks stay thin (ROOT resolve → quiet install → call package APIs) |
| Generated data | `data/generated/` is gitignored; do not commit large artefacts |
| FreeCAD | CAD lives in `cad/`; FreeCAD is **not** a runtime dependency |

Optional extras at M0: `dl` (torch), `fem` (ngsolve), `dev` (pytest, …). No mandatory `all` extra.


Historical note:
Editable installs (-e) were intentionally forbidden during development
to ensure notebooks exercised the package exactly as a fresh user would.

-e = -evil :)

---

## Deliverables checklist

| Artefact | Requirement |
|----------|-------------|
| `pyproject.toml` | Name, version, Python 3.12, core deps, extras, `packages.find` with `where = ["src"]`, `include = ["gummybear*"]` |
| `.gitignore` | At least `data/generated/`, `.venv/`, `*.egg-info/`, caches, coverage |
| `README.md` | Pitch, local install matching the rule above, pointer to `plans/00_architecture.md` |
| `src/gummybear/` | Version export; stub subpackages importable (see smoke tests) |
| `schemas/` | At least a skeleton `sequence_manifest.schema.json` (empty-object schema is fine) |
| `docs/` | Short stubs only (`project_plan`, `physics_model`, `data_contract`, `experiment_log`) |
| `data/fixtures/` | Reserved (empty / `.gitkeep`); `data/generated/` ignored |
| `tests/test_smoke.py` | Import + layout checks below |
| `notebooks/01_inspect_stl.ipynb` | Thin notebook contract below (body may no-op on geometry until M1) |

Stub package tree (empty `__init__.py` is enough for M0):

```text
src/gummybear/
  __init__.py          # version only
  geometry/            # M1 fills inspect_stl / load_stl / …
  rays/  optics/  particles/  datasets/  models/  export_contracts/
```

---

## Tests M0 must pass

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install ".[dev]" -c requirements.txt
pytest tests/test_smoke.py
```

Minimum smoke coverage:

| Test | Asserts |
|------|---------|
| Import package | `import gummybear` succeeds; version string matches scaffold |
| Import subpackages | `geometry`, `rays`, `optics`, `particles`, `datasets`, `models`, `export_contracts` import without error |
| Layout | Repo-root `pyproject.toml` exists relative to `tests/` |

Done when those pass and `data/generated/` is ignored (`git check-ignore -v data/generated/…`).

---

## Notebook contract — `01_inspect_stl.ipynb`

1. **Resolve `ROOT`** by walking parents until `pyproject.toml` exists (no absolute paths in committed cells).
2. **Quiet install from `ROOT`**, non-editable, constrained — align with AGENTS / README (e.g. `pip install "{ROOT}[dev]" -c …` or equivalent from an activated venv). No `-e`.
3. **Call package APIs only** — no mesh maths in the notebook.

Intended call surface (implemented under **M1**, exercised by this notebook):

```text
gummybear.geometry.inspect_stl(path)   → mesh + summary + validation
gummybear.geometry.load_stl(path)
gummybear.geometry.describe_mesh(mesh)
gummybear.geometry.validate_mesh_for_projection(…)
```

Canonical phantom: `cad/proto_bear.stl`. Notebook may print watertight / bounds / `mesh.show()`; all loading and trust checks stay in `src/gummybear/geometry/`.

**M0 responsibility for the notebook:** directory reserved, install cell correct, imports point at the package. **M1 responsibility:** make `inspect_stl` / `load_stl` / … real so the notebook runs end-to-end.

---

## Out of scope for M0

STL trust / mesh stats (M1); rays and appearance (M2+); sequence generation and manifests (M6+); catalog / ML datasets (M7+); localisation models (M8+); CI workflows; real fixture binaries; FEM beyond declaring the optional `fem` extra.

---

## Verification summary

| Check | Pass means |
|-------|------------|
| Install | `pip install ".[dev]" -c requirements.txt` from a clean venv |
| Smoke | `pytest tests/test_smoke.py` green |
| Imports | `python -c "import gummybear; print(gummybear…version…)"` works |
| Hygiene | No large generated files tracked; `data/generated/` ignored |
| Notebook shape | `01_inspect_stl.ipynb` resolves ROOT, installs without `-e`, only calls `gummybear.geometry` |

When those hold, M0 is complete. Next capability is **M1 — STL load / mesh trust**, which turns the inspect notebook into a real run.
