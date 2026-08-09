# AGENTS.md — gummybear-tomography

Short project conventions for contributors and coding agents. Long-form plan: [`plans/00_architecture.md`](plans/00_architecture.md).

---

## Install rules (mandatory)

**No editable installs.** Any `pip install -e` is a failure.

Always install from the repository root and constrain with `requirements.txt`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install ".[fem]" -c requirements.txt
# ML:        pip install ".[dl,dev]" -c requirements.txt
# Extended DEV + FEM/M4+: pip install ".[fem,dl,dev]" -c requirements.txt
```

After changing packaged code under `src/`, reinstall the same way before re-running imports/tests.

Forbidden: unconstrained `pip install <pkg>` (omitting `-c requirements.txt`), and any `-e` form.

---

## Names and versions

| Item | Value |
|------|--------|
| Dist | `gummybear-tomography` |
| Imports | `gummybear`, `gummybear_validation`, `tomography_ml`, … under `src/` |
| Python | `==3.12.*` |
| Packaging | `pyproject.toml` (source of truth); pin via `requirements.txt` |

Phantom mesh: `cad/` STL. FreeCAD is not a runtime dependency.

---

## Hard contracts agents must not break

1. **Sequence-first** — canonical ML units are ordered multi-view (and multi-light) samples; split by sequence / particle identity, never by random patches from the same sequence.
2. **Float images are authoritative** — `.raw.tif` (float32) for numerics; JPEG/PNG are display copies.
3. **Bump versions on semantic change** — `schema_version` / `preprocess_contract_version` in manifests and export sidecars; do not silently rewrite meaning.
4. **Algorithms live in `src/`** — notebooks stay thin.
5. **Do not commit** large generated data, caches, or checkpoints under `data/generated/` (or equivalent); publish corpora elsewhere and link.
6. **No absolute path leakage** in committed notebook outputs (prefer repo-relative paths; quiet `!pip install`).

---

## Verify

```bash
pytest
```

FEM-backed generation needs the `fem` extra; catalog / ML paths should not require NGSolve unless generating diffusion data.
