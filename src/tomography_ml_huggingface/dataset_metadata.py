"""Build Hugging Face dataset-card metadata (parquet + README configs).

Exports browseable indexes for the optical corpora under ``data/generated/``:

* ``m8_1`` — Milestone 8/9 single-particle sequences
* ``m10_illumination`` — Milestone 10 multi-illumination sequences

Each parquet row is one **sequence**, with JPEG/PNG preview images for the
primary camera view (angle 0°). Preview pixels are **embedded as bytes** so
the Hub Dataset Viewer works when the corpora are published as zip archives
(``data/generated/m8_1.zip``, ``data/generated/m10_illumination.zip``) rather
than extracted trees. Float ``.raw.tif`` paths stay as strings for ML loaders
after unzip; they are not embedded.

Outputs land in ``data/huggingface_metadata/``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from gummybear.paths import repo_relative_path
from tomography_ml.gummybear_data_catalog import build_catalog_rows, load_catalog_jobs
from tomography_ml.gummybear_data_catalog.catalog import CatalogRow

METADATA_SUBDIR = Path("data") / "huggingface_metadata"
M8_WORKBOOK = Path("configs") / "m8" / "localization_single_particle.xlsx"
M8_OUTPUT = Path("data") / "generated" / "m8_1" / "single_particle"
M10_WORKBOOK = Path("configs") / "m10" / "localization_m10_illumination.xlsx"
M10_OUTPUT = Path("data") / "generated" / "m10_illumination"

REGIME_BY_SETUP = {
    "opt_m8_low_001": "low",
    "opt_m8_med_001": "medium",
    "opt_m8_high_001": "high",
}

PREVIEW_ROLES = ("observed", "clean", "particle", "anomaly")
SPLITS = ("train", "validation", "test")


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Writing Hugging Face metadata parquet requires pyarrow. "
            'Install with: pip install ".[hf]" -c requirements.txt'
        ) from exc
    return pa, pq


def _image_type(pa):
    return pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def _image_value(
    path: str | None,
    *,
    data: bytes | None = None,
) -> dict[str, Any] | None:
    """Return a Hugging Face Image struct.

    When ``data`` is set, ``path`` is stored as a **basename** only (plus
    extension). A repo-relative ``data/generated/...`` path looks local to
    the Hub viewer and can stall first-rows while it hunts inside
    ``m8_1.zip`` / ``m10_illumination.zip``. Unzipped locations stay in
    ``sequence_dir`` / ``*_raw_path``.
    """
    if not path and data is None:
        return None
    viewer_path = path
    if data is not None and path:
        viewer_path = Path(path).name
    return {"bytes": data, "path": viewer_path}


def _embedded_preview(repo_root: Path, repo_rel: str | None) -> dict[str, Any] | None:
    if not repo_rel:
        return None
    source = Path(repo_root) / repo_rel
    data = source.read_bytes() if source.is_file() else None
    return _image_value(repo_rel, data=data)


def _repo_rel(repo_root: Path, path: Path | str) -> str:
    del repo_root  # resolved via pyproject.toml walk in repo_relative_path
    return repo_relative_path(Path(path))


def _load_manifest(sequence_dir: Path) -> dict[str, Any]:
    path = sequence_dir / "manifest.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _preview_frame(manifest: Mapping[str, Any], *, angle_deg: float = 0.0) -> dict[str, Any]:
    frames = list(manifest.get("frames") or [])
    if not frames:
        raise ValueError("manifest has no frames")
    for frame in frames:
        if float(frame.get("angle_deg", float("nan"))) == float(angle_deg):
            return frame
    return frames[0]


def _role_paths(
    repo_root: Path,
    sequence_dir: Path,
    frame: Mapping[str, Any],
) -> dict[str, str | None]:
    filenames = dict(frame.get("filenames") or {})
    out: dict[str, str | None] = {}
    for role in PREVIEW_ROLES:
        rel = filenames.get(role)
        if not rel:
            out[role] = None
            out[f"{role}_raw"] = None
            continue
        display = sequence_dir / str(rel)
        out[role] = _repo_rel(repo_root, display) if display.is_file() else None
        raw_key = f"{role}_raw"
        raw_rel = filenames.get(raw_key)
        if raw_rel:
            raw = sequence_dir / str(raw_rel)
            out[raw_key] = _repo_rel(repo_root, raw) if raw.is_file() else None
        else:
            out[raw_key] = None
    return out


def _illumination_angle_deg(optical_setup_id: str | None) -> float | None:
    if not optical_setup_id:
        return None
    match = re.search(r"illum_(\d+)$", str(optical_setup_id))
    if match:
        return float(match.group(1))
    match = re.search(r"_m10_(\d+)_", str(optical_setup_id))
    if match:
        return float(match.group(1))
    return None


def _optical_regime(row: CatalogRow) -> str | None:
    setup = str(row.optical_setup_id or "")
    if setup in REGIME_BY_SETUP:
        return REGIME_BY_SETUP[setup]
    sid = row.sequence_id.lower()
    if "low" in sid:
        return "low"
    if "med" in sid:
        return "medium"
    if "high" in sid:
        return "high"
    return None


def catalog_rows_for_corpus(
    repo_root: Path,
    *,
    workbook: Path,
    output_root: Path,
) -> tuple[CatalogRow, ...]:
    jobs = load_catalog_jobs(workbook, root_path=repo_root, stl_root=repo_root)
    jobs = [
        replace(job, output_root=repo_relative_path(output_root))
        for job in jobs
    ]
    return build_catalog_rows(jobs)


def row_to_metadata_record(
    row: CatalogRow,
    *,
    repo_root: Path,
    corpus: str,
    preview_angle_deg: float = 0.0,
) -> dict[str, Any] | None:
    """Flatten one complete catalog row into a Hub-viewer metadata record."""
    if row.field_status != "complete":
        return None
    sequence_dir = Path(row.sequence_dir)
    if not sequence_dir.is_dir():
        return None
    manifest = _load_manifest(sequence_dir)
    frame = _preview_frame(manifest, angle_deg=preview_angle_deg)
    paths = _role_paths(repo_root, sequence_dir, frame)
    record: dict[str, Any] = {
        "corpus": corpus,
        "sequence_id": row.sequence_id,
        "split": row.split,
        "optical_setup_id": row.optical_setup_id,
        "optical_regime": _optical_regime(row),
        "illumination_angle_deg": _illumination_angle_deg(row.optical_setup_id),
        "frame_count": int(row.frame_count or 0),
        "preview_angle_deg": float(frame.get("angle_deg", preview_angle_deg)),
        "preview_frame_index": int(frame.get("frame_index", 0)),
        "particle_x": row.particle_x,
        "particle_y": row.particle_y,
        "particle_z": row.particle_z,
        "particle_radius": row.particle_radius,
        "bear_mu_s": row.bear_mu_s,
        "bear_mu_a": row.bear_mu_a,
        "sequence_dir": _repo_rel(repo_root, sequence_dir),
        "manifest_path": _repo_rel(repo_root, sequence_dir / "manifest.json"),
        "observed": _embedded_preview(repo_root, paths["observed"]),
        "clean": _embedded_preview(repo_root, paths["clean"]),
        "particle": _embedded_preview(repo_root, paths["particle"]),
        "anomaly": _embedded_preview(repo_root, paths["anomaly"]),
        "observed_raw_path": paths["observed_raw"],
        "anomaly_raw_path": paths["anomaly_raw"],
    }
    return record


def build_corpus_records(
    repo_root: Path | str,
    *,
    corpus: str,
    workbook: Path | str,
    output_root: Path | str,
    preview_angle_deg: float = 0.0,
) -> list[dict[str, Any]]:
    root = Path(repo_root).resolve()
    rows = catalog_rows_for_corpus(
        root,
        workbook=root / workbook if not Path(workbook).is_absolute() else Path(workbook),
        output_root=(
            root / output_root if not Path(output_root).is_absolute() else Path(output_root)
        ),
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        rec = row_to_metadata_record(
            row,
            repo_root=root,
            corpus=corpus,
            preview_angle_deg=preview_angle_deg,
        )
        if rec is not None:
            records.append(rec)
    return records


_HF_VALUE = {
    "string": {"dtype": "string", "_type": "Value"},
    "float64": {"dtype": "float64", "_type": "Value"},
    "int32": {"dtype": "int32", "_type": "Value"},
}
_HF_IMAGE = {"_type": "Image"}


def _parquet_huggingface_metadata() -> dict[str, str]:
    """Arrow schema metadata so the Hub treats preview columns as Image."""
    features = {
        "corpus": _HF_VALUE["string"],
        "sequence_id": _HF_VALUE["string"],
        "split": _HF_VALUE["string"],
        "optical_setup_id": _HF_VALUE["string"],
        "optical_regime": _HF_VALUE["string"],
        "illumination_angle_deg": _HF_VALUE["float64"],
        "frame_count": _HF_VALUE["int32"],
        "preview_angle_deg": _HF_VALUE["float64"],
        "preview_frame_index": _HF_VALUE["int32"],
        "particle_x": _HF_VALUE["float64"],
        "particle_y": _HF_VALUE["float64"],
        "particle_z": _HF_VALUE["float64"],
        "particle_radius": _HF_VALUE["float64"],
        "bear_mu_s": _HF_VALUE["float64"],
        "bear_mu_a": _HF_VALUE["float64"],
        "sequence_dir": _HF_VALUE["string"],
        "manifest_path": _HF_VALUE["string"],
        "observed": _HF_IMAGE,
        "clean": _HF_IMAGE,
        "particle": _HF_IMAGE,
        "anomaly": _HF_IMAGE,
        "observed_raw_path": _HF_VALUE["string"],
        "anomaly_raw_path": _HF_VALUE["string"],
    }
    return {"huggingface": json.dumps({"info": {"features": features}})}


def _records_to_table(records: Sequence[Mapping[str, Any]]):
    pa, _ = _require_pyarrow()
    image_type = _image_type(pa)

    def col_image(name: str):
        return [r.get(name) for r in records]

    arrays = {
        "corpus": pa.array([r["corpus"] for r in records], type=pa.string()),
        "sequence_id": pa.array([r["sequence_id"] for r in records], type=pa.string()),
        "split": pa.array([r["split"] for r in records], type=pa.string()),
        "optical_setup_id": pa.array(
            [r["optical_setup_id"] for r in records], type=pa.string()
        ),
        "optical_regime": pa.array([r["optical_regime"] for r in records], type=pa.string()),
        "illumination_angle_deg": pa.array(
            [r["illumination_angle_deg"] for r in records], type=pa.float64()
        ),
        "frame_count": pa.array([r["frame_count"] for r in records], type=pa.int32()),
        "preview_angle_deg": pa.array(
            [r["preview_angle_deg"] for r in records], type=pa.float64()
        ),
        "preview_frame_index": pa.array(
            [r["preview_frame_index"] for r in records], type=pa.int32()
        ),
        "particle_x": pa.array([r["particle_x"] for r in records], type=pa.float64()),
        "particle_y": pa.array([r["particle_y"] for r in records], type=pa.float64()),
        "particle_z": pa.array([r["particle_z"] for r in records], type=pa.float64()),
        "particle_radius": pa.array(
            [r["particle_radius"] for r in records], type=pa.float64()
        ),
        "bear_mu_s": pa.array([r["bear_mu_s"] for r in records], type=pa.float64()),
        "bear_mu_a": pa.array([r["bear_mu_a"] for r in records], type=pa.float64()),
        "sequence_dir": pa.array([r["sequence_dir"] for r in records], type=pa.string()),
        "manifest_path": pa.array([r["manifest_path"] for r in records], type=pa.string()),
        "observed": pa.array(col_image("observed"), type=image_type),
        "clean": pa.array(col_image("clean"), type=image_type),
        "particle": pa.array(col_image("particle"), type=image_type),
        "anomaly": pa.array(col_image("anomaly"), type=image_type),
        "observed_raw_path": pa.array(
            [r["observed_raw_path"] for r in records], type=pa.string()
        ),
        "anomaly_raw_path": pa.array(
            [r["anomaly_raw_path"] for r in records], type=pa.string()
        ),
    }
    table = pa.table(arrays)
    return table.replace_schema_metadata(_parquet_huggingface_metadata())


def write_split_parquets(
    records: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
) -> dict[str, Path]:
    """Write ``train`` / ``validation`` / ``test`` parquet shards."""
    _, pq = _require_pyarrow()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    by_split: dict[str, list[Mapping[str, Any]]] = {s: [] for s in SPLITS}
    for rec in records:
        split = str(rec.get("split") or "train")
        by_split.setdefault(split, []).append(rec)
    for split, rows in by_split.items():
        if not rows:
            continue
        path = out_dir / f"{split}.parquet"
        table = _records_to_table(rows)
        pq.write_table(table, path, compression="zstd")
        written[split] = path
    return written


def dataset_card_markdown(*, counts: Mapping[str, Mapping[str, int]]) -> str:
    """Return a Hugging Face dataset card with YAML ``configs`` for m8_1 / m10."""
    m8 = counts.get("m8_1", {})
    m10 = counts.get("m10_illumination", {})
    m8_total = sum(m8.values())
    m10_total = sum(m10.values())
    return f"""---
license: apache-2.0
language:
- en
pretty_name: Gummybear Tomography Dataset
tags:
- tomography
- optical-tomography
- synthetic-data
- computer-vision
- localization
- physics-simulation
- pytorch
size_categories:
- 1K<n<10K
configs:
- config_name: m8_1
  default: true
  data_files:
  - split: train
    path: data/huggingface_metadata/m8_1/train.parquet
  - split: validation
    path: data/huggingface_metadata/m8_1/validation.parquet
  - split: test
    path: data/huggingface_metadata/m8_1/test.parquet
- config_name: m10_illumination
  data_files:
  - split: train
    path: data/huggingface_metadata/m10_illumination/train.parquet
  - split: validation
    path: data/huggingface_metadata/m10_illumination/validation.parquet
  - split: test
    path: data/huggingface_metadata/m10_illumination/test.parquet
dataset_info:
- config_name: m8_1
  features:
  - name: corpus
    dtype: string
  - name: sequence_id
    dtype: string
  - name: split
    dtype: string
  - name: optical_setup_id
    dtype: string
  - name: optical_regime
    dtype: string
  - name: illumination_angle_deg
    dtype: float64
  - name: frame_count
    dtype: int32
  - name: preview_angle_deg
    dtype: float64
  - name: preview_frame_index
    dtype: int32
  - name: particle_x
    dtype: float64
  - name: particle_y
    dtype: float64
  - name: particle_z
    dtype: float64
  - name: particle_radius
    dtype: float64
  - name: bear_mu_s
    dtype: float64
  - name: bear_mu_a
    dtype: float64
  - name: sequence_dir
    dtype: string
  - name: manifest_path
    dtype: string
  - name: observed
    dtype: image
  - name: clean
    dtype: image
  - name: particle
    dtype: image
  - name: anomaly
    dtype: image
  - name: observed_raw_path
    dtype: string
  - name: anomaly_raw_path
    dtype: string
  splits:
  - name: train
    num_examples: {int(m8.get("train", 0))}
  - name: validation
    num_examples: {int(m8.get("validation", 0))}
  - name: test
    num_examples: {int(m8.get("test", 0))}
- config_name: m10_illumination
  features:
  - name: corpus
    dtype: string
  - name: sequence_id
    dtype: string
  - name: split
    dtype: string
  - name: optical_setup_id
    dtype: string
  - name: optical_regime
    dtype: string
  - name: illumination_angle_deg
    dtype: float64
  - name: frame_count
    dtype: int32
  - name: preview_angle_deg
    dtype: float64
  - name: preview_frame_index
    dtype: int32
  - name: particle_x
    dtype: float64
  - name: particle_y
    dtype: float64
  - name: particle_z
    dtype: float64
  - name: particle_radius
    dtype: float64
  - name: bear_mu_s
    dtype: float64
  - name: bear_mu_a
    dtype: float64
  - name: sequence_dir
    dtype: string
  - name: manifest_path
    dtype: string
  - name: observed
    dtype: image
  - name: clean
    dtype: image
  - name: particle
    dtype: image
  - name: anomaly
    dtype: image
  - name: observed_raw_path
    dtype: string
  - name: anomaly_raw_path
    dtype: string
  splits:
  - name: train
    num_examples: {int(m10.get("train", 0))}
  - name: validation
    num_examples: {int(m10.get("validation", 0))}
  - name: test
    num_examples: {int(m10.get("test", 0))}
---

# GummyBear Tomography Dataset

Synthetic optical tomography corpora and reproducibility artefacts for the
[gummybear-tomography](https://github.com/tbgitoo/gummybear-tomography) project.

This Hub package sits **beside** the Git repository layout. Generated corpora
are stored as **zip archives** so upload/download stays tractable (many small
files). The Dataset Viewer does **not** need those zips extracted: preview
JPEG/PNG bytes are embedded in the parquet Image columns.

Hub layout:

- `data/generated/m8_1.zip` — full single-particle optical corpus (Milestone 8/9)
- `data/generated/m10_illumination.zip` — multi-illumination corpus (Milestone 10)
- `data/huggingface_metadata/` — parquet indexes + this dataset card (viewer)
- `checkpoints/` — trained weights (optional download)

Unzip the archives locally if you need the full multi-view float TIFF stacks.
After unzip, paths match the git-repo layout (`data/generated/m8_1/…`,
`data/generated/m10_illumination/…`).

Source code, notebooks, and generation configs live in the GitHub repository.
Float32 ``.raw.tif`` sidecars under each sequence remain the **authoritative**
ML image representation; JPEG/PNG previews exist for display and for the Hub
Dataset Viewer.

## Configurations

| Config | Sequences | Workbook | Hub archive | After unzip |
|--------|-----------|----------|-------------|-------------|
| `m8_1` | {m8_total} | `configs/m8/localization_single_particle.xlsx` | `data/generated/m8_1.zip` | `data/generated/m8_1/single_particle/` |
| `m10_illumination` | {m10_total} | `configs/m10/localization_m10_illumination.xlsx` | `data/generated/m10_illumination.zip` | `data/generated/m10_illumination/` |

### `m8_1` (default)

Single embedded particle, three optical regimes (`low` / `medium` / `high`),
36 camera views per sequence. The parquet index shows **one preview row per
sequence** (camera angle 0°) with labels `particle_x/y/z` and regime metadata.

Split counts: train={m8.get("train", 0)}, validation={m8.get("validation", 0)},
test={m8.get("test", 0)}.

### `m10_illumination`

Same localisation task with six illumination setups
(`opt_m10_illum_{{000,060,120,180,240,300}}`). Use the Hub config dropdown to
switch corpora in the Dataset Viewer.

Split counts: train={m10.get("train", 0)}, validation={m10.get("validation", 0)},
test={m10.get("test", 0)}.

## Columns (viewer)

| Column | Meaning |
|--------|---------|
| `observed` / `clean` / `particle` / `anomaly` | Preview images (JPEG/PNG **bytes** embedded; `path` is the filename only, not a Hub tree path) |
| `particle_x` / `particle_y` / `particle_z` | Ground-truth particle centre (mm) |
| `optical_regime` | M8 regime label (`low` / `medium` / `high`) when applicable |
| `illumination_angle_deg` | M10 light angle decoded from `optical_setup_id` |
| `observed_raw_path` / `anomaly_raw_path` | Repo-relative float TIFF paths (valid after unzip) |
| `sequence_dir` | Repo-relative sequence folder (valid after unzip) |

## Load in Python

```python
from datasets import load_dataset

ds = load_dataset("tbhugging/gummybear-tomography", "m8_1")
row = ds["train"][0]
print(row["sequence_id"], row["particle_x"], row["particle_y"], row["particle_z"])
row["observed"]  # PIL image from embedded bytes (no unzip required)
```

```python
ds_m10 = load_dataset("tbhugging/gummybear-tomography", "m10_illumination")
```

`load_dataset` reads the parquet indexes only. Unzip
`data/generated/m8_1.zip` / `m10_illumination.zip` locally if you need the
float ``.raw.tif`` stacks referenced by `observed_raw_path`.

## Regenerate metadata locally

From the repository root (after optical corpora exist on disk):

```bash
pip install ".[hf]" -c requirements.txt
python -m tomography_ml_huggingface.dataset_metadata
```

Upload `data/huggingface_metadata/` together with the zip archives
`data/generated/m8_1.zip` and `data/generated/m10_illumination.zip`. Do **not**
upload extracted trees: preview Image columns already contain JPEG/PNG bytes
for the Dataset Viewer. Unzip locally only when you need the full stacks.

## License

Apache-2.0 (same as the source repository).
"""


def export_huggingface_metadata(
    repo_root: Path | str,
    *,
    out_dir: Path | str | None = None,
    write_readme: bool = True,
) -> dict[str, Any]:
    """Build parquet shards + dataset card under ``data/huggingface_metadata``."""
    root = Path(repo_root).resolve()
    meta_root = Path(out_dir) if out_dir is not None else root / METADATA_SUBDIR
    meta_root.mkdir(parents=True, exist_ok=True)

    corpora = (
        ("m8_1", M8_WORKBOOK, M8_OUTPUT),
        ("m10_illumination", M10_WORKBOOK, M10_OUTPUT),
    )
    counts: dict[str, dict[str, int]] = {}
    written: dict[str, dict[str, str]] = {}

    for name, workbook, output in corpora:
        records = build_corpus_records(
            root,
            corpus=name,
            workbook=workbook,
            output_root=output,
        )
        split_counts = Counter(str(r["split"]) for r in records)
        counts[name] = {s: int(split_counts.get(s, 0)) for s in SPLITS}
        paths = write_split_parquets(records, out_dir=meta_root / name)
        written[name] = {k: v.as_posix() for k, v in paths.items()}
        print(
            f"{name}: {len(records)} sequences "
            f"(train={counts[name]['train']}, "
            f"validation={counts[name]['validation']}, "
            f"test={counts[name]['test']})"
        )

    readme_path = meta_root / "README.md"
    if write_readme:
        readme_path.write_text(dataset_card_markdown(counts=counts), encoding="utf-8")
        print(f"Wrote {readme_path}")

    return {
        "out_dir": str(meta_root),
        "counts": counts,
        "written": written,
        "readme": str(readme_path) if write_readme else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export Hugging Face parquet metadata for m8_1 and m10_illumination."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: walk up from CWD to pyproject.toml).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/data/huggingface_metadata).",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Only write parquet shards (keep an existing README).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = args.repo_root
    if root is None:
        root = Path.cwd()
        while not (root / "pyproject.toml").exists():
            if root == root.parent:
                raise SystemExit("Could not locate repository root (pyproject.toml).")
            root = root.parent

    export_huggingface_metadata(
        root,
        out_dir=args.out_dir,
        write_readme=not args.skip_readme,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
