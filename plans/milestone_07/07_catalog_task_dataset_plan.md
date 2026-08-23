# Milestone 7 — Sample catalog and lazy task datasets

**Source:** `plans/00_architecture.md` §5.5 · upstream `plans/milestone_07/implementation_plan.md`  
**Role:** Workbook-defined **sample catalog** over M6 sequence artifacts + minimal **task-dataset** extraction: `catalog → schedule-consistent subset → field selection → lazy x, y = dataset[i]`. **Not** training, not physics, not global `x_train`/`y_train`.  
**Core:** `tomography_ml.gummybear_data_catalog` · installable tests `tomography_ml_validation.milestone_07`  
**Install:** `pip install ".[dl,dev]" -c requirements.txt` (catalog/ML path; FEM not required for catalog contracts). Generation artifacts come from M6 (`.[fem]`).  
**Evidence:** `notebooks/milestone_07/` — **Done** (`07_1_workbook_loading` → `07_6_catalog_subset_validation`). **Status:** package APIs + validation helpers shipped.

**Labeling:** unmarked = planned design. **Conclusion** = hardened API / contract.

---

## Read this first — a sample is not a tensor

```text
A sample is a catalog row.
The observation array is one possible field on that row.

Tasks decide:  X = selected fields    Y = selected fields
Fields are not permanently “features” or “labels.”

For example, a localization task, emulated forward simulation task, and reconstruction task
can all be different views of the same catalog rows - with different assignment of the features and labels:
- localization: features=image tensors, labels=coordinates
- forward emulation: features = coordinates + optical settings, labels=image tensors
- reconstruction task: features = image tensors, labels = optical properties

The catalog stores observations (or references to observations for lazy loading if large) and metadata.
Tasks define how those fields are assigned to inputs (X=features) and targets (Y=labels).
```

```text
Workbook (enabled SequenceJobs)
  → SampleCatalog          # membership = workbook, not directory scan
  → schedule-consistent subset   # fixed V, angles, resolution
  → DatasetTaskSpec        # which rows + which x_fields / y_fields
  → CatalogTaskDataset     # lazy: x, y = dataset[i]
```

**Conclusion — dataset formalism (repo contract):**

| Rule | Meaning |
|------|---------|
| `x, y = dataset[i]` | Public access pattern; `x`/`y` are **dicts** of selected field names → values |
| No `torch.utils.data.Dataset` | Pure Python `__len__` / `__getitem__`; **no** PyTorch dependency in M7 core |
| Lazy load | Only index `i` loads role images; other rows stay as `RoleRef` / scalars |
| Image layout | Public role arrays are **`numpy.ndarray` shape `(V, C, H, W)`** — never HWC |
| Roles ≠ views | Roles = image types (`observed`/`clean`/…); views = ordered camera frames |

Loader path (role field): PIL → `(H,W,C)` → transpose `(C,H,W)` → stack frames → `(V,C,H,W)`.  
**Conclusion (local):** default representation is float32 **`.raw.tif`** (`raw_float`); `jpeg_uint8` is the legacy display path. M8 may add batch dim → `(B,V,C,H,W)`.

---

## Package boundary

```text
gummybear / gummybear_validation     → generate + validate sequences (M6)
tomography_ml.gummybear_data_catalog → catalog, RoleRef, task datasets (read M6 only)
tomography_ml_validation.milestone_07 → installable contract pytest
```

M7 does **not** generate images, run FEM, write caches, or train. Notebooks use `gummybear_validation.notebook_tools.run_installed_pytest_test(s)`.

Public imports:

```python
from tomography_ml.gummybear_data_catalog import (
    RoleRef, CatalogRow, build_catalog_rows,
    DatasetTaskSpec, build_task_dataset, CatalogTaskDataset,
    filter_schedule_consistent, load_role_array,
)
x, y = dataset[i]  # dicts; role values are np.ndarray (V, C, H, W)
```

---

## M6 adapters (membership & paths)

| Call | M7 use |
|------|--------|
| `load_generation_workbook` + `validate_generation_plan` | Catalog membership = `plan.jobs` (enabled only) |
| `resolve_output_root(job) / job.sequence_id` | Sequence dir (**not** `resolve_output_root` alone) |
| Read `manifest.json` | Authoritative ordered `frames[].filenames[role]` |
| Optional `validate_generated_sequence` | Deep artifact check (opens images) — not default reconciliation |

Do **not** scan directories for membership. Missing artifacts → `field_status`, rows stay in catalog.

---

## Catalog model (essence)

**`RoleRef(manifest_path, role_name)`** — self-contained lazy handle; not a bare role string; not first-frame paths.

**`CatalogRow`** — the sample: ids, split, schedule/angles/`angles_hash`, `*_ref` role handles, optical/particle/diffusion scalars, `field_status` (`complete` / `directory_missing` / `manifest_missing` / `incomplete_catalog`).

**Schedule-consistent subset** (required before fixed-shape multi-view ML): same `camera_schedule_id`, same `V`, same **ordered** angles, same resolution. Mixed schedules (e.g. matrix workbook V=6 and V=12) are valid catalogs but **not** one ML-ready `(V,C,H,W)` pool — filter first.

**Multi-particle:** `n_particles` + ordered `particles` labels; scalar `particle_x/y/z/radius` = first particle only when `N==1`, else `None`. Do not pack multi-particle geometry into four scalars.

---

## Task dataset API

```python
task = DatasetTaskSpec(
    name="particle_localization",
    row_filter={"split": "train", "field_status": "complete"},  # keep simple
    x_fields=("observed_ref",),  # or anomaly_ref, …
    y_fields=("particle_x", "particle_y", "particle_z", "particle_radius"),
)
dataset = build_task_dataset(subset_or_rows, task)
x, y = dataset[i]
# x["observed_ref"].shape == (V, C, H, W)
# y["particle_x"] == float (etc.)
```

Same catalog → different tasks (e.g. `observed_ref`→`clean_ref` inpainting; localization; optical regression). Incomplete rows may be filtered for a task; they remain in the catalog.

**Conclusion:** `CatalogTaskDataset` stores selected rows eagerly; **image bytes** load only on `__getitem__` for requested `RoleRef` fields. No batch API in M7 core. Later wrappers (`IlluminationOnlyDataset`, `HierarchicalCameraLightDataset`) compose on this spine for M9/M10 joint grids — they are not a substitute for the `x,y=dataset[i]` contract.

---

## Phases (evidence spine)

| Phase | Proves | Notebook |
|-------|--------|----------|
| **M7.1** | Enabled workbook jobs → catalog skeleton (no dir scan) | `07_1_workbook_loading.ipynb` |
| **M7.2** | Lightweight manifest reconcile; membership unchanged | `07_2_acquisition_shape_consistency.ipynb` |
| **M7.3** | Schedule-consistent subset → common V / angles / resolution | `07_3_files_and_manifest_consistency.ipynb` |
| **M7.4** | Full field join + `RoleRef`s; still no image arrays | `07_4_flat_catalog.ipynb` |
| **M7.5** | `x, y = dataset[i]`; only index `i` loads `(V,C,H,W)` | `07_5_task_dataset.ipynb` |
| **M7.6** | Status view + dual task views; installable pytest | `07_6_catalog_subset_validation.ipynb` |

---

## Tensor / array dimensions (exact)

| Stage | Shape | Type |
|-------|-------|------|
| One RGB frame (raw loader) | `(H, W, C)` | intermediate only |
| One frame (public CHW) | `(C, H, W)` | after transpose |
| Multi-view role (public) | **`(V, C, H, W)`** | `numpy.ndarray` |
| M8+ batched | `(B, V, C, H, W)` | training adapters |
| M10 joint (later) | `(I, V, C, H, W)` | hierarchical/illumination datasets |

`C` is normally 3 for RGB. `V` = ordered acquisition count from schedule/manifest — acquisition order is part of observation identity (`angles_hash`).

---

## Conclusions · guardrails · handoffs

**Conclusions:** sample = catalog row; tasks select fields into lazy `(x,y)`; workbook defines membership; schedule filter precedes fixed-shape ML; public images are numpy `(V,C,H,W)` with HWC→CHW inside the loader; no PyTorch Dataset base class in M7.

```text
No directory-scan membership; no drop incomplete rows from catalog.
No global x_train/y_train; no batch API as M7 core; no training in M7.
No torch.utils.data.Dataset inheritance; no public HWC stacks.
No treating roles as views; no bare role-name strings instead of RoleRef.
No FEM/physics in tomography_ml catalog path.
```

**From M6:** sequence dirs + manifests + roles; `resolved_job_hash` / cache IDs as provenance.  
**To M8+:** pick X/Y from the same catalog; batch `(V,C,H,W)` → `(B,V,C,H,W)`; split by **sequence / particle identity**, never random patches from one sequence.

*Bump catalog/task schema or image-representation defaults on semantic change — do not silently rewrite meaning.*
