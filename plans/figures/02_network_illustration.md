# Network inference illustration — POV-Ray graphical abstract

**Audience:** Contributors adding scientific architecture figures  
**Scope:** Extend `gummybear_illustration` with a second POV-Ray 3.7 scene: M8 **neural-network inference**, in the same visual language as `plans/01_gummybear_illustration.md` / `figures/notebooks/render_m8_physical_scene.ipynb`.  
**Companion notebook:** `figures/notebooks/render_m8_network_scene.ipynb`

This is a schematic **along a left-to-right pipeline**, not a catalog millimetre reconstruction of the lab. Geometry still uses the same POV conventions (z-up, gray-blue sky, dark materials, captions on the PNG after render).

---

## 1. Purpose

Show how a **single-view (or stacked multi-view) z-score projection** is encoded by the M8 CNN (`Encode`: 16 → 32 → 64 channels, spatial size **128×128** throughout), then reduced by **three alternative readouts** (average pooling, Fourier pooling, flatten), then mapped by an **MLP / head** to a **3D localization** panel.

Feature-volume colours come from **actual activations** of the fully trained M8 **Fourier xyz** system when a checkpoint and `torch` are available. Tests and missing-checkpoint runs use an explicit **FALLBACK** synthetic bundle (same shapes, warned).

---

## 2. In scope / out of scope

| In scope | Out of scope |
|----------|----------------|
| POV schematic of CNN + three readouts + MLP + coord panel | Retraining, FEM, NGSolve |
| Hook trained `Encode` on one 180° `anomaly_raw` | Loading optical caches |
| Optional `povray` + 2D PNG captions | Requiring POV-Ray or `torch` for `pytest` |
| Gitignored `outputs/pov/`, `outputs/renders/` | Sunset, landscape, people, “AI brain” icons, decorative scenery |

---

## 3. Scientific mapping (authoritative)

| Figure element | Code / data |
|----------------|-------------|
| Input plates | Same greyscale z-score pipeline as the physical-scene inset (`per_image_zscore`, clip knob). One or more 180° (and optional neighbouring orbit) `anomaly_raw` float `.raw.tif` files. |
| CNN volumes | `tomography_ml.localization.encoder.Encode`, `CHANNEL_PRESETS["base"] = (16, 32, 64)`, `downsample="base"` → **no MaxPool**, maps stay `[B, C, 128, 128]`. |
| Trained weights | `checkpoints/m8/m08_train_val_test_xyz.pt` → `final_state_by_arch["fourier"]` into `make_m8_single_view_model("fourier", n_outputs=3, …)` (`LocalizeSingleViewFourier` + **linear** head). |
| Average pooling | Channel-wise mean of the last map (`AdaptiveAvgPool2d` / GAP). Draw a **few** channels as colormap chips. |
| Fourier pooling | `FourierCodedPool2d`: 64 real modes `(kx, ky, const\|cos\|sin)` from `enumerate_fourier_modes`. Draw a compact set of planes with **basis sinusoids** tinted by the pooled scalar. |
| Flatten | Conceptual `64×128×128 = 1_048_576` vector: a **large** dense slab / cell grid (not one cube per element). Emphasizes parameter burden vs compact pooling. |
| MLP | Dark layered blocks. Trained Fourier xyz uses a **linear** `Linear(64, 3)` head; flatten uses `LazyLinear(128)→ReLU→Linear`. The figure shows a shared **head stack** after the three branches (schematic), not three separate POV copies of every head. |
| Output panel | Same triad language as the physical-scene localization inset (xy plate, RGB axes, green sphere). Predicted location from the Fourier model; faint second marker for ground-truth particle when labels exist. |

Do **not** load M8 xyz weights into `LocalizerSingleViewFourier` (MLP head) — state-dict mismatch.

Input protocol for trained activations: `keep_angles_deg=180`, `image_normalize="per_image_zscore"`, `x_fields=("anomaly_ref",)`, task `localization_xyz`. Default sample: first M8 sequence (`sample_index=0`) like the physical figure, or an explicit `manifest_path`.

---

## 4. Visual style (match physical abstract)

- `#version 3.7;`, `assumed_gamma 1.0`, `sky <0,0,1>`
- Gray-blue `sky_sphere` + faint floor (reuse `sky_and_horizon` / similar plane). **No** outdoor landscape, sunsets, or people.
- Scene lights independent of catalog illumination (key + fill).
- Dark gray camera-body / volume materials; z-score plates as `image_map` quads (same as inset stack).
- Activation textures: matplotlib colormap (e.g. turbo) on PNG `image_map`, not rainbow emissive gimmicks.
- Illustration camera: side overview (same *family* as physical: yaw ~80°, FOV ~60°), looking at the pipeline centre.
- **Typography:** 2D overlay after POV (PIL), not POV `text`. Sober sans-serif, black.

---

## 5. Pipeline layout (schematic world)

Left → right is obtained by laying the schematic out along **+x** and then rotating that union **−90° about world z** through the illustration look-at, so the camera (yaw ~80°) sees the flow across the frame rather than end-on.

1. **Input stack** — thin upright greyscale plates (physical-scene plate language).
2. **CNN** — three axis-aligned boxes, **same xy footprint** (128 spatial units, scaled to scene mm), **thickness ∝ channel count** (16 / 32 / 64). Front face: colormap of a representative channel (or 3-channel composite). Side face: channel-vs-space strip so depth is readable.
3. **Split** — three rows (or a shallow y-fan):
   - Average pooling chips
   - Fourier mode planes (sinusoidal textures)
   - Oversized flatten slab
4. **MLP** — stacked dark blocks shrinking toward the right.
5. **Localization panel** — triad + predicted green sphere + faint GT.

Arrows: same cylinder+cone language as the physical inset.

---

## 6. Two-stage render (same as physical)

```text
activations (+ optional torch/checkpoint)
        │
        ▼
  textures PNG + build_network_pov_scene  →  .pov
        │
        ▼  render=True
  render_pov_file  →  <stem>_plain.png
        │
        ▼
  overlay_network_captions  →  <stem>.png
```

Caption strings (defaults; positions notebook-tunable as width/height fractions, y from top):

| Label | Role |
|-------|------|
| `single-view or multi-view input` | under input stack |
| `CNN` | under feature volumes |
| `average pooling` | branch 1 |
| `Fourier pooling` | branch 2 |
| `flatten` | branch 3 |
| `MLP` | head stack |
| `3D localization` | coord panel |

---

## 7. Package layout (additions)

```text
src/gummybear_illustration/
    network_activations.py      # bundle + optional trained hooks + FALLBACK
    network_textures.py         # z-score / colormap / Fourier-basis PNGs
    network_pov_scene.py        # schematic POV string
    network_captions.py         # PIL overlay
    export_m8_network_scene.py  # public exporter
```

Reuse: `render_pov_file`, `anomaly_zscore`, `pov_primitives`, `sky_and_horizon`, `load_m8_physical_setup` (for the input TIFF and GT particle).

`torch` / `tomography_ml` are imported **inside** the activation collector. Core `pytest` must pass without the `dl` extra and without `checkpoints/m8/*.pt`.

---

## 8. Fallback

| Situation | Behaviour |
|-----------|-----------|
| No `torch` or no xyz checkpoint | Synthetic `[C,128,128]` maps from the z-scored input (tiled/filtered). `UserWarning` FALLBACK. POV comment. |
| No anomaly TIFF | Tests supply a tiny array; production export raises if neither manifest nor `setup` is given. |
| POV-Ray missing | Write `.pov` only. |

Never silently pretend synthetic maps are trained activations (`bundle.source` is `"trained"` or `"fallback"`).

---

## 9. Tests

- Synthetic bundle → `.pov` contains CNN / pooling / flatten / MLP / localization comments; no caption strings in the POV file.
- Texture PNGs exist next to the `.pov`.
- Caption overlay on a dummy RGB writes dark pixels near configured fractions.
- `render=True` without `povray` does not fail export.

---

## 10. Notebook

`figures/notebooks/render_m8_network_scene.ipynb`: knobs (sample, checkpoint path, camera, caption xy), `export_m8_network_scene(...)`, caption-only last cell. Install: `pip install ".[dl]" -c requirements.txt` for trained activations; illustration geometry still works with the base extra + FALLBACK.

---

## 11. Non-goals

- Do not change `Encode`, pooling, or training code to “look better” in POV.
- Do not commit renders, activation PNGs, or checkpoints.
- Do not leak absolute paths in notebook outputs or POV logs (`display_path`).
