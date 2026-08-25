# Milestone 11 — Hugging Face repository artefacts

**Source:** `plans/00_architecture.md` §5.7  
**Role:** Publish **optical corpora + checkpoint trees** as a Hub dataset that mirrors the git repo layout and drives a usable Dataset Viewer — without regenerating physics or retraining.  
**Core:** `tomography_ml_validation.huggingface_metadata` · thin notebook `notebooks/milestone_11/`  
**Install:** `pip install ".[hf]" -c requirements.txt` (PyArrow). Catalog load for indexes also needs the usual ML stack when resolving workbooks: `.[dl,hf]`.  
**Evidence:** `notebooks/milestone_11/11_0_huggingface_export.ipynb`.

**Labeling:** unmarked = planned design. **Conclusion** = repo / Hub contract.

---

## Read this first — indexes, not a second corpus

```text
M6/M8/M10 on-disk sequences (authoritative float .raw.tif)
  → M11 parquet rows (one sequence = one row; JPEG/PNG preview @ angle 0° **embedded**)
  → dataset-card README (YAML configs: m8_1, m10_illumination)
  → Hub: zips + parquet indexes + checkpoints (no extracted trees)
```

**Conclusion — publication contract:**

| Rule | Meaning |
|------|---------|
| Float TIFF authoritative | Parquet stores **paths** to `.raw.tif`; does not embed float pixels |
| Preview for the Viewer | Image structs with **JPEG/PNG bytes**; `path` is the **filename** only (not a `data/generated/...` tree path, which is missing on the Hub) |
| Zip corpora on the Hub | Publish `data/generated/m8_1.zip` and `data/generated/m10_illumination.zip` — not extracted directories |
| Layout parity | After unzip, paths = repo-relative (`data/generated/...`); parquet lives at `data/huggingface_metadata/` |
| Configs, not one blob | Hub subsets `m8_1` and `m10_illumination` with train / validation / test shards |
| Git tracks indexes only | `data/huggingface_metadata/` may be committed; `data/generated/` and `checkpoints/` stay external |

---

## Scope

| In M11 | Out of M11 |
|--------|------------|
| Dataset-card README with `configs` + `dataset_info` | New localisation science |
| Parquet indexes for `m8_1` and `m10_illumination` | Changing M6 manifests / catalog schemas |
| Document Hub zip upload (`m8_1.zip`, `m10_illumination.zip`) + checkpoints | Embedding full multi-view stacks in parquet |
| Thin export notebook / CLI | Automating Hub auth / CI push (manual upload OK) |

---

## Artefacts

```text
data/huggingface_metadata/
├── README.md                          # Hub dataset card (upload as dataset root README)
├── m8_1/{train,validation,test}.parquet
└── m10_illumination/{train,validation,test}.parquet
```

Hub hosts zip archives (not extracted trees) so upload/download stays tractable:

```text
data/generated/m8_1.zip                # unzip → data/generated/m8_1/single_particle/
data/generated/m10_illumination.zip    # unzip → data/generated/m10_illumination/
checkpoints/{m8,m9,m10}/               # optional companion download
```

The Dataset Viewer uses **embedded preview bytes** in parquet; it does not resolve paths inside those zips.

---

## API

```python
from tomography_ml_validation.huggingface_metadata import export_huggingface_metadata
export_huggingface_metadata(ROOT)
# or: python -m tomography_ml_validation.huggingface_metadata
```

```python
from datasets import load_dataset
ds = load_dataset("tbhugging/gummybear-tomography", "m8_1")
```

---

## Notebooks

| Notebook | What it does |
|----------|--------------|
| [`11_0_huggingface_export.ipynb`](../../notebooks/milestone_11/11_0_huggingface_export.ipynb) | Install `.[hf]`, export parquet + README, smoke-check one row from **embedded** preview bytes |

---

## Not in M11

Regenerating sequences, retraining, rewriting workbook identity, JPEG-as-training-store, private upstream mirrors.
