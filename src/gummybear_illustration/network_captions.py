"""2D captions for the network-inference PNG."""

from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

FONT_HEIGHT_FRAC = 0.028

Rgb = tuple[int, int, int]

# Dark, readable branch colours (PIL RGB).
COLOR_DEFAULT: Rgb = (0, 0, 0)
COLOR_GAP: Rgb = (148, 18, 22)
COLOR_FOURIER: Rgb = (18, 42, 118)
COLOR_TARGET: Rgb = (12, 102, 48)

# Fractions of width/height; PIL y from the top.
CAPTION_DEFAULTS: dict[str, tuple[float, float] | None] = {
    "input": (0.12, 0.90),
    "cnn": (0.28, 0.90),
    "layer1": (0.34, 0.90),
    "layer2": (0.40, 0.90),
    "layer3": (0.46, 0.90),
    "layer1_gap": (0.34, 0.84),
    "layer2_gap": (0.40, 0.84),
    "layer3_gap": (0.46, 0.84),
    "cnn_readout": (0.52, 0.90),
    "embedding": (0.62, 0.22),
    "embedding_gap": (0.48, 0.22),
    "mlp": (0.78, 0.90),
    "pred_gap": (0.82, 0.78),
    "pred_fourier": (0.82, 0.84),
    "fourier_branch": (0.55, 0.08),
    "gap_branch": (0.38, 0.08),
    "target": (0.90, 0.90),
}

CAPTION_TEXTS: dict[str, str] = {
    "input": "Input\nImage",
    "cnn": "CNN",
    "layer1": "Layer 1",
    "layer2": "Layer 2",
    "layer3": "Layer 3",
    "layer1_gap": "Layer 1",
    "layer2_gap": "2",
    "layer3_gap": "3",
    "cnn_readout": "CNN Readout",
    "embedding": "Embedding",
    "embedding_gap": "Embedding",
    "mlp": "MLP",
    "pred_gap": "predicted",
    "pred_fourier": "predicted",
    "fourier_branch": "Fourier Pooling Branch",
    "gap_branch": "Global Average Pooling Branch",
    "target": "Target position",
}

CAPTION_COLORS: dict[str, Rgb] = {
    "input": COLOR_DEFAULT,
    "cnn": COLOR_DEFAULT,
    "layer1": COLOR_FOURIER,
    "layer2": COLOR_FOURIER,
    "layer3": COLOR_FOURIER,
    "layer1_gap": COLOR_GAP,
    "layer2_gap": COLOR_GAP,
    "layer3_gap": COLOR_GAP,
    "cnn_readout": COLOR_DEFAULT,
    "embedding": COLOR_FOURIER,
    "embedding_gap": COLOR_GAP,
    "mlp": COLOR_DEFAULT,
    "pred_gap": COLOR_GAP,
    "pred_fourier": COLOR_FOURIER,
    "fourier_branch": COLOR_FOURIER,
    "gap_branch": COLOR_GAP,
    "target": COLOR_TARGET,
}


def _sans_font(size: int) -> ImageFont.FreeTypeFont:
    path = font_manager.findfont("DejaVu Sans")
    return ImageFont.truetype(path, size=size)


def overlay_network_captions(
    png_path: str | Path,
    *,
    source: str | Path | None = None,
    positions: dict[str, tuple[float, float] | None] | None = None,
    colors: dict[str, Rgb] | None = None,
) -> Path:
    dest = Path(png_path)
    src = Path(source) if source is not None else dest
    pos = dict(CAPTION_DEFAULTS)
    if positions:
        pos.update(positions)
    fill_map = dict(CAPTION_COLORS)
    if colors:
        fill_map.update(colors)
    image = Image.open(src).convert("RGB")
    width, height = image.size
    size = max(11, int(round(FONT_HEIGHT_FRAC * height)))
    font = _sans_font(size)
    draw = ImageDraw.Draw(image)
    for key, text in CAPTION_TEXTS.items():
        xy = pos.get(key, CAPTION_DEFAULTS.get(key))
        if xy is None:
            continue
        fill = fill_map.get(key, COLOR_DEFAULT)
        extra = {}
        if "\n" in text:
            extra = dict(
                anchor="mm",
                align="center",
                spacing=int(round(0.12 * size)),
            )
        else:
            extra = dict(anchor="ms")
        draw.text(
            (float(xy[0]) * width, float(xy[1]) * height),
            text,
            font=font,
            fill=fill,
            **extra,
        )
    image.save(dest)
    return dest
