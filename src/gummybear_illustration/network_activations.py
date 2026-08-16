"""Activations for the M8 network illustration (trained or FALLBACK)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from gummybear.paths import checkpoint_dir, display_path

from .anomaly_zscore import per_image_zscore
from .load_sample import PhysicalSetup
from .network_textures import fourier_term_multiplied

SPATIAL = 128
CHANNEL_DEPTHS = (16, 32, 64)


@dataclass(frozen=True)
class NetworkActivationBundle:
    """CNN maps and readouts for one illustration sample.

    ``source`` is ``\"trained\"`` or ``\"fallback\"``. Spatial maps are
    ``[C, H, W]`` with ``H = W = 128``.
    """

    input_zscore: np.ndarray
    conv_maps: tuple[np.ndarray, ...]
    gap: np.ndarray
    fourier_prepool: np.ndarray
    fourier_pooled: np.ndarray
    gap_conv_maps: tuple[np.ndarray, ...]
    gap_prepool: np.ndarray
    gap_mlp_stages: tuple[tuple[str, np.ndarray], ...]
    flatten_conv_maps: tuple[np.ndarray, ...]
    flatten_pre: np.ndarray
    flatten_vec: np.ndarray
    flatten_mlp_stages: tuple[tuple[str, np.ndarray], ...]
    mlp_stages: tuple[tuple[str, np.ndarray], ...]
    y_pred: np.ndarray
    y_pred_pooled: np.ndarray
    y_pred_flatten: np.ndarray
    y_true: np.ndarray | None
    source: str


def _resize_2d(arr: np.ndarray, size: int = SPATIAL) -> np.ndarray:
    img = Image.fromarray(np.asarray(arr, dtype=np.float32), mode="F")
    out = img.resize((size, size), resample=Image.Resampling.BILINEAR)
    return np.asarray(out, dtype=np.float32)


def load_input_zscore(setup: PhysicalSetup, *, angle_deg: float = 180.0) -> np.ndarray:
    """Z-score the anomaly frame nearest ``angle_deg``."""
    if not setup.frame_anomaly_raw:
        raise ValueError("PhysicalSetup has no anomaly_raw frames")
    target = float(angle_deg)
    best = min(setup.frame_anomaly_raw, key=lambda p: abs(float(p[0]) - target))
    with Image.open(best[1]) as im:
        arr = np.asarray(im, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return _resize_2d(per_image_zscore(arr))


def fallback_activation_bundle(
    input_zscore: np.ndarray,
    *,
    y_true: np.ndarray | None = None,
) -> NetworkActivationBundle:
    """Deterministic maps derived from the input (not trained weights)."""
    base = _resize_2d(np.asarray(input_zscore, dtype=np.float32))
    maps: list[np.ndarray] = []
    for nch in CHANNEL_DEPTHS:
        vol = np.empty((nch, SPATIAL, SPATIAL), dtype=np.float32)
        for i in range(nch):
            rolled = np.roll(base, shift=i * 3, axis=1)
            scale = 0.25 + 0.75 * float(i + 1) / float(nch)
            vol[i] = rolled * scale
        maps.append(vol)
    feat = maps[-1]
    gap = feat.mean(axis=(1, 2))
    pre, fourier = fourier_term_multiplied(feat)
    pred = np.array([0.0, 0.0, 4.0], dtype=float)
    true = None if y_true is None else np.asarray(y_true, dtype=float).reshape(3)
    if true is not None:
        pred = true + np.array([0.12, -0.08, 0.06], dtype=float)
        pooled_pred = true + np.array([2.6, -2.1, 1.5], dtype=float)
    else:
        pooled_pred = pred + np.array([2.6, -2.1, 1.5], dtype=float)
    convs = tuple(maps)
    flat_vec = np.asarray(feat, dtype=np.float32).reshape(-1)
    return NetworkActivationBundle(
        input_zscore=base,
        conv_maps=convs,
        gap=gap,
        fourier_prepool=pre,
        fourier_pooled=fourier,
        gap_conv_maps=convs,
        gap_prepool=np.asarray(feat, dtype=np.float32),
        gap_mlp_stages=(
            ("GAP pooling in  [64]", gap.astype(np.float32)),
            ("GAP flatten after pool  [64]", gap.astype(np.float32)),
            ("GAP head Linear out  [3]", pooled_pred.astype(np.float32)),
        ),
        flatten_conv_maps=convs,
        flatten_pre=np.asarray(feat, dtype=np.float32),
        flatten_vec=flat_vec,
        flatten_mlp_stages=(
            ("Flatten MLP Linear  [128]", np.zeros(128, dtype=np.float32)),
            ("Flatten MLP ReLU  [128]", np.zeros(128, dtype=np.float32)),
            ("Flatten MLP out  [3]", pred.astype(np.float32)),
        ),
        mlp_stages=(
            ("Fourier head Linear in  [64]", fourier.astype(np.float32)),
            ("Fourier head Linear out  [3]", pred.astype(np.float32)),
        ),
        y_pred=pred,
        y_pred_pooled=pooled_pred,
        y_pred_flatten=pred,
        y_true=None if true is None else true,
        source="fallback",
    )


def collect_m8_network_activations(
    setup: PhysicalSetup,
    *,
    repo_root_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    angle_deg: float = 180.0,
    device: str = "cpu",
) -> NetworkActivationBundle:
    """Hook the trained Fourier xyz encoder, or FALLBACK with a warning."""
    zimg = load_input_zscore(setup, angle_deg=angle_deg)
    y_true = np.asarray(setup.particle_center, dtype=float).reshape(3)
    try:
        import torch

        from tomography_ml.studies.single_view_m8 import make_m8_single_view_model
        from tomography_ml.studies.study_checkpoints import (
            M08_TRAIN_VAL_TEST_XYZ,
            load_study_checkpoint,
        )
    except ImportError as exc:
        warnings.warn(
            f"FALLBACK network activations (torch/tomography_ml unavailable: {exc})",
            UserWarning,
            stacklevel=2,
        )
        return fallback_activation_bundle(zimg, y_true=y_true)

    from .paths import repo_root as _repo_root

    root = _repo_root(repo_root_path) if repo_root_path is not None else _repo_root()
    ckpt = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else checkpoint_dir(root, "m8") / M08_TRAIN_VAL_TEST_XYZ
    )
    if not ckpt.is_file():
        warnings.warn(
            "FALLBACK network activations "
            f"(missing checkpoint {display_path(ckpt)})",
            UserWarning,
            stacklevel=2,
        )
        return fallback_activation_bundle(zimg, y_true=y_true)

    blob = load_study_checkpoint(ckpt)
    weights = blob["final_state_by_arch"]["fourier"]
    model = make_m8_single_view_model("fourier", n_outputs=3, device=device)
    model.load_state_dict(weights)
    model.eval()
    x = torch.from_numpy(zimg[np.newaxis, np.newaxis, ...]).to(device)
    captured: dict[str, np.ndarray] = {}

    def _hook(name: str):
        def inner(_mod, _inp, out):
            captured[name] = out.detach().cpu().numpy()[0]

        return inner

    handles = [
        model.encoder.blocks[0].register_forward_hook(_hook("c16")),
        model.encoder.blocks[1].register_forward_hook(_hook("c32")),
        model.encoder.blocks[2].register_forward_hook(_hook("c64")),
    ]
    with torch.no_grad():
        y = model(x)
    for h in handles:
        h.remove()
    feat = captured["c64"]
    convs = (captured["c16"], captured["c32"], captured["c64"])
    gap = feat.mean(axis=(1, 2))
    feat_t = torch.from_numpy(feat[np.newaxis, ...]).to(device)
    pooled = np.asarray(model.pool(feat_t).detach().cpu().numpy()[0], dtype=np.float32)
    pre, _pooled_from_basis = fourier_term_multiplied(feat)
    y_np = y.detach().cpu().numpy().reshape(3).astype(float)
    mlp_stages: list[tuple[str, np.ndarray]] = [
        ("Fourier head Linear in  [64]", pooled),
        ("Fourier head Linear out  [3]", y_np.astype(np.float32)),
    ]
    flatten_conv = convs
    flatten_pre = np.asarray(feat, dtype=np.float32)
    flatten_vec = flatten_pre.reshape(-1)
    y_flatten = y_np.copy()
    flatten_mlp: list[tuple[str, np.ndarray]] = [
        ("Flatten MLP Linear  [128]", np.zeros(128, dtype=np.float32)),
        ("Flatten MLP ReLU  [128]", np.zeros(128, dtype=np.float32)),
        ("Flatten MLP out  [3]", y_flatten.astype(np.float32)),
    ]
    flatten_w = blob.get("final_state_by_arch", {}).get("flatten")
    if flatten_w is not None:
        try:
            flat_m = make_m8_single_view_model("flatten", n_outputs=3, device=device)
            flat_m.load_state_dict(flatten_w)
            flat_m.eval()
            captured_mlp: dict[str, np.ndarray] = {}

            def _mlp_hook(name: str):
                def inner(_mod, _inp, out):
                    captured_mlp[name] = out.detach().cpu().numpy()[0]

                return inner

            hs = [
                flat_m.encoder.blocks[0].register_forward_hook(_mlp_hook("c16")),
                flat_m.encoder.blocks[1].register_forward_hook(_mlp_hook("c32")),
                flat_m.encoder.blocks[2].register_forward_hook(_mlp_hook("c64")),
                flat_m.flat.register_forward_hook(_mlp_hook("flat")),
                flat_m.head[0].register_forward_hook(_mlp_hook("h0")),
                flat_m.head[1].register_forward_hook(_mlp_hook("h1")),
                flat_m.head[2].register_forward_hook(_mlp_hook("h2")),
            ]
            with torch.no_grad():
                y_flat = flat_m(x)
            for h in hs:
                h.remove()
            flatten_conv = (
                captured_mlp["c16"],
                captured_mlp["c32"],
                captured_mlp["c64"],
            )
            flatten_pre = np.asarray(captured_mlp["c64"], dtype=np.float32)
            flatten_vec = np.asarray(captured_mlp["flat"], dtype=np.float32).reshape(-1)
            y_flatten = y_flat.detach().cpu().numpy().reshape(3).astype(float)
            flatten_mlp = [
                ("Flatten MLP Linear  [128]", captured_mlp["h0"].astype(np.float32)),
                ("Flatten MLP ReLU  [128]", captured_mlp["h1"].astype(np.float32)),
                ("Flatten MLP out  [3]", captured_mlp["h2"].astype(np.float32)),
            ]
        except (RuntimeError, KeyError, ValueError) as exc:
            warnings.warn(
                f"Could not hook flatten MLP on the same sample: {exc}",
                UserWarning,
                stacklevel=2,
            )
    gap_conv = convs
    gap_pre = np.asarray(feat, dtype=np.float32)
    gap_vec = gap.astype(np.float32)
    y_pooled = y_np.copy()
    gap_mlp: list[tuple[str, np.ndarray]] = [
        ("GAP head Linear in  [64]", gap_vec),
        ("GAP head Linear out  [3]", y_pooled.astype(np.float32)),
    ]
    pooled_w = blob.get("final_state_by_arch", {}).get("pooled")
    if pooled_w is not None:
        try:
            gap_m = make_m8_single_view_model("pooled", n_outputs=3, device=device)
            gap_m.load_state_dict(pooled_w)
            gap_m.eval()
            captured_gap: dict[str, np.ndarray] = {}

            def _gap_hook(name: str):
                def inner(_mod, _inp, out):
                    captured_gap[name] = out.detach().cpu().numpy()[0]

                return inner

            ghs = [
                gap_m.encoder.blocks[0].register_forward_hook(_gap_hook("c16")),
                gap_m.encoder.blocks[1].register_forward_hook(_gap_hook("c32")),
                gap_m.encoder.blocks[2].register_forward_hook(_gap_hook("c64")),
                gap_m.encoder.flat.register_forward_hook(_gap_hook("flat")),
                gap_m.encoder.lin.register_forward_hook(_gap_hook("embed")),
                gap_m.lin.register_forward_hook(_gap_hook("out")),
            ]
            with torch.no_grad():
                y_gap = gap_m(x)
            for h in ghs:
                h.remove()
            gap_conv = (
                captured_gap["c16"],
                captured_gap["c32"],
                captured_gap["c64"],
            )
            gap_pre = np.asarray(captured_gap["c64"], dtype=np.float32)
            gap_vec = gap_pre.mean(axis=(1, 2)).astype(np.float32)
            y_pooled = y_gap.detach().cpu().numpy().reshape(3).astype(float)
            gap_mlp = [
                ("GAP pooling in  [64]", gap_vec),
                (
                    "GAP flatten after pool  [64]",
                    np.asarray(captured_gap["flat"], dtype=np.float32).reshape(-1),
                ),
                ("GAP Encode Linear  [128]", captured_gap["embed"].astype(np.float32)),
                ("GAP head Linear out  [3]", captured_gap["out"].astype(np.float32)),
            ]
        except (RuntimeError, KeyError, ValueError) as exc:
            warnings.warn(
                f"Could not hook pooled GAP model on the same sample: {exc}",
                UserWarning,
                stacklevel=2,
            )
    return NetworkActivationBundle(
        input_zscore=zimg,
        conv_maps=convs,
        gap=gap_vec,
        fourier_prepool=pre,
        fourier_pooled=pooled,
        gap_conv_maps=gap_conv,
        gap_prepool=gap_pre,
        gap_mlp_stages=tuple(gap_mlp),
        flatten_conv_maps=flatten_conv,
        flatten_pre=flatten_pre,
        flatten_vec=flatten_vec,
        flatten_mlp_stages=tuple(flatten_mlp),
        mlp_stages=tuple(mlp_stages),
        y_pred=y_np,
        y_pred_pooled=y_pooled,
        y_pred_flatten=y_flatten,
        y_true=y_true,
        source="trained",
    )
