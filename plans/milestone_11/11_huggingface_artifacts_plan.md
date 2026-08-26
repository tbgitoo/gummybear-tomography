# Milestone 11 — Hugging Face artefacts

**Source:** `plans/00_architecture.md` §5.7  
**Role:** Publish (1) **optical corpora + checkpoint trees** as a Hub **dataset**, and (2) selected **localisation weights** as Hub **model** repos — without regenerating physics or retraining.  
**Core:** `tomography_ml_validation.huggingface_metadata` · thin notebook(s) under `notebooks/milestone_11/`  
**Install:** `pip install ".[hf]" -c requirements.txt` (PyArrow). Catalog load for indexes also needs the usual ML stack: `.[dl,hf]`. Hub upload uses the standalone [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (agent skill: [HF CLI for AI Agents](https://huggingface.co/docs/hub/en/agents-cli)).  
**Evidence:** [`notebooks/milestone_11/11_0_huggingface_export.ipynb`](../../notebooks/milestone_11/11_0_huggingface_export.ipynb) (dataset); [`notebooks/milestone_11/11_1_singleview_cnn_fourier_export.ipynb`](../../notebooks/milestone_11/11_1_singleview_cnn_fourier_export.ipynb) (M8 Fourier model → local Hub clone).

**Labeling:** unmarked = planned design. **Conclusion** = repo / Hub contract.

---

## Read this first — two Hub surfaces

```text
A) Dataset  https://huggingface.co/datasets/tbhugging/gummybear-tomography
   id: tbhugging/gummybear-tomography
   M6/M8/M10 on-disk sequences → parquet indexes + zips + optional checkpoint trees
   Viewer: embedded JPEG/PNG preview @ angle 0°

B) Model   https://huggingface.co/tbhugging/singleview_cnn_fourier
   id: tbhugging/singleview_cnn_fourier
   Extract a single trained architecture from a study .pt → Hub model card + weights
   First ship: M8 Step 3 Fourier xyz only
   Stage in a local clone of that Hub repo (path in gitignored configs/hf/local.toml), then upload
```

**Conclusion — publication contract (dataset):**

| Rule | Meaning |
|------|---------|
| Float TIFF authoritative | Parquet stores **paths** to `.raw.tif`; does not embed float pixels |
| Preview for the Viewer | Image structs with **JPEG/PNG bytes**; `path` is the **filename** only |
| Zip corpora on the Hub | Publish `data/generated/m8_1.zip` and `data/generated/m10_illumination.zip` — not extracted directories |
| Layout parity | After unzip, paths = repo-relative (`data/generated/...`); parquet at `data/huggingface_metadata/` |
| Configs, not one blob | Hub subsets `m8_1` and `m10_illumination` with train / validation / test shards |
| Git tracks indexes only | `data/huggingface_metadata/` may be committed; `data/generated/` and `checkpoints/` stay external |

**Conclusion — publication contract (models):**

| Rule | Meaning |
|------|---------|
| One architecture per model repo | Do not upload the full multi-arch study blob as “the” model |
| Source checkpoint | `checkpoints/m8/m08_train_val_test_xyz.pt` → key `final_state_by_arch["fourier"]` |
| Meaning frozen | WIN 3J trunk; `anomaly_ref`; `per_image_zscore`; single view 180°; targets `(x,y,z)` |
| Card documents | Input tensor shape, preprocess, LR used, link to Final Report M8 Step 3 + companion dataset |
| Reproducible load | Small loader / `from_pretrained`-style helper in `tomography_ml` (or card snippet) that rebuilds the Fourier localizer and loads the state dict |

---

## Scope

| In M11 | Out of M11 |
|--------|------------|
| Dataset-card README with `configs` + `dataset_info` | New localisation science |
| Parquet indexes for `m8_1` and `m10_illumination` | Changing M6 manifests / catalog schemas |
| Document Hub zip upload + checkpoint trees | Embedding full multi-view stacks in parquet |
| **M8 Step 3 Fourier xyz → [`tbhugging/singleview_cnn_fourier`](https://huggingface.co/tbhugging/singleview_cnn_fourier)** | Publishing pooled / flatten / M9 / M10 models in the same first pass |
| Thin export notebook / CLI | Automating Hub auth / CI push (manual `hf` upload OK) |

Later model repos (optional, not this pass): M8 pooled/flatten controls, M9 fusion heads, M10 hierarchical — each as its own Hub model if useful.

---

## A — Dataset artefacts (existing)

```text
data/huggingface_metadata/
├── README.md                          # Hub dataset card (upload as dataset root README)
├── m8_1/{train,validation,test}.parquet
└── m10_illumination/{train,validation,test}.parquet
```

Hub hosts zip archives (not extracted trees):

```text
data/generated/m8_1.zip
data/generated/m10_illumination.zip
checkpoints/{m8,m9,m10}/               # optional companion download
```

```python
from tomography_ml_validation.huggingface_metadata import export_huggingface_metadata
export_huggingface_metadata(ROOT)

from datasets import load_dataset
ds = load_dataset("tbhugging/gummybear-tomography", "m8_1")
```

| Notebook | What it does |
|----------|--------------|
| [`11_0_huggingface_export.ipynb`](../../notebooks/milestone_11/11_0_huggingface_export.ipynb) | Install `.[hf]`, export parquet + README, smoke-check embedded preview |
| [`11_1_singleview_cnn_fourier_export.ipynb`](../../notebooks/milestone_11/11_1_singleview_cnn_fourier_export.ipynb) | Export M8 Step 3 Fourier weights + card into local clone of `tbhugging/singleview_cnn_fourier` (path from `configs/hf/local.toml`) |
| [`11_1_test_singleview_cnn_fourier.ipynb`](../../notebooks/milestone_11/11_1_test_singleview_cnn_fourier.ipynb) | Download published Hub model; run inference on packaged M8 demo anomaly @ 180° |

---

## B — Model artefact (planned): M8 Step 3 Fourier xyz

**Scientific object:** the **Fourier** branch of Final Report **M8 Step 3** (train → val/test on `(particle_x, particle_y, particle_z)`), not the z-only Step 2 study and not the split-sensitivity Step 4 ensemble.

**Local source of truth:**

```text
checkpoints/m8/m08_train_val_test_xyz.pt
  final_state_by_arch["fourier"]   # state_dict to publish
  lr_by_arch["fourier"]            # record on the card
  comparison / session_summary     # cite metrics; do not require uploading CSVs
```

**Hub model (public):** [`tbhugging/singleview_cnn_fourier`](https://huggingface.co/tbhugging/singleview_cnn_fourier)

**Local staging path:** gitignored [`configs/hf/local.toml`](../../configs/hf/local.toml) key `models.singleview_cnn_fourier.local_clone` (copy from [`configs/hf/local.toml.example`](../../configs/hf/local.toml.example)). Do not put machine paths in this plan.

Stage export artefacts in that clone, then push:

```text
<local_clone>/
├── README.md                         # model card
├── config.json                       # arch id, y_fields, x_field, image_normalize, keep_angles
├── model.safetensors  (or pytorch_model.bin)
└── (optional) example inference snippet
```

**Export steps:** run [`11_1_singleview_cnn_fourier_export.ipynb`](../../notebooks/milestone_11/11_1_singleview_cnn_fourier_export.ipynb) (calls `export_singleview_cnn_fourier`). Summary:

1. Load study checkpoint; extract `final_state_by_arch["fourier"]` only.
2. Write `config.json` matching the WIN 3J / M8 Step 3 contract (representation, normalisation, single-view angle, target fields).
3. Serialize weights (`pytorch_model.bin`) into `local_clone` from `configs/hf/local.toml`.
4. Author model card `README.md`: Final Report M8 Step 3, companion dataset, metrics from the study blob.
5. Upload from the staging clone (manual):

```bash
# local_clone := configs/hf/local.toml → models.singleview_cnn_fourier.local_clone
cd "$local_clone"
hf upload tbhugging/singleview_cnn_fourier . .
# or: git add -A && git commit && git push
```

**Do not:** rename experiment IDs; silently change preprocess meaning; publish JPEG as training input; claim the Hub model is multi-view or multi-illumination.

---

## Agent / CLI workflow

```text
# Dataset (existing)
https://huggingface.co/datasets/tbhugging/gummybear-tomography

# Model (first ship)
https://huggingface.co/tbhugging/singleview_cnn_fourier
# local_clone from gitignored configs/hf/local.toml

hf auth login
cd "$local_clone"
hf upload tbhugging/singleview_cnn_fourier . .
```

Skill install (once per machine): see [Hugging Face CLI for AI Agents](https://huggingface.co/docs/hub/en/agents-cli) — `hf skills add --global` (or project-local `hf skills add`).

---

## Not in M11

Regenerating sequences, retraining, rewriting workbook identity, JPEG-as-training-store, private upstream mirrors, bundling all M8–M10 architectures into one undifferentiated model repo.
