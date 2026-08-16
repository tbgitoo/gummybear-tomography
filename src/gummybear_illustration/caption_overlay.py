"""2D captions drawn on the rendered illustration PNG (not in POV-Ray)."""

from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

# Fractions of image width/height. PIL y is from the top.
# Views and localization share a baseline so they line up horizontally.
VIEWS_XY = (0.40, 0.94)
LOCALIZATION_XY = (0.78, 0.88)
DEEP_LEARNING_XY = (0.60, 0.76)
FONT_HEIGHT_FRAC = 0.032


def _sans_font(size: int) -> ImageFont.FreeTypeFont:
    path = font_manager.findfont("DejaVu Sans")
    return ImageFont.truetype(path, size=size)


def _xy_pair(
    value: tuple[float, float] | None,
    default: tuple[float, float],
) -> tuple[float, float]:
    if value is None:
        return default
    return (float(value[0]), float(value[1]))


def overlay_workflow_captions(
    png_path: str | Path,
    *,
    source: str | Path | None = None,
    views_xy: tuple[float, float] | None = None,
    deep_learning_xy: tuple[float, float] | None = None,
    localization_xy: tuple[float, float] | None = None,
) -> Path:
    """Draw the three workflow labels on an already-rendered PNG.

    Positions are fractions of width/height (PIL y from the top). ``source``
    is the caption-free render. When omitted, ``png_path`` is both input and
    output.
    """
    dest = Path(png_path)
    src = Path(source) if source is not None else dest
    views = _xy_pair(views_xy, VIEWS_XY)
    deep = _xy_pair(deep_learning_xy, DEEP_LEARNING_XY)
    loc = _xy_pair(localization_xy, LOCALIZATION_XY)
    image = Image.open(src).convert("RGB")
    width, height = image.size
    size = max(12, int(round(FONT_HEIGHT_FRAC * height)))
    font = _sans_font(size)
    draw = ImageDraw.Draw(image)
    fill = (0, 0, 0)

    def _xy(frac: tuple[float, float]) -> tuple[float, float]:
        return frac[0] * width, frac[1] * height

    draw.text(
        _xy(views),
        "single-view or multi-view input",
        font=font,
        fill=fill,
        anchor="ms",
    )
    draw.text(
        _xy(deep),
        "Deep\nLearning",
        font=font,
        fill=fill,
        anchor="mm",
        align="center",
        spacing=int(round(0.12 * size)),
    )
    draw.text(
        _xy(loc),
        "3D localization",
        font=font,
        fill=fill,
        anchor="ms",
    )
    image.save(dest)
    return dest
