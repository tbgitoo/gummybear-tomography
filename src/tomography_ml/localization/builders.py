"""Single-view model builders, architecture grids, and freeze records.

Factory helpers for convolutional neural network (CNN) encoders, spatial
readout heads (pooled, flatten, Fourier), parameter counting, and predefined
architecture / capability grids used in single-view localisation studies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

from tomography_ml.localization.encoder import (
    CHANNEL_PRESETS,
    DOWNSAMPLE_MODES,
    Encode,
    resolve_channels,
)
from tomography_ml.localization.alternative_localizer import (
    FlattenHeadType,
    LocalizeSingleView,
    LocalizeSingleViewFlatten,
    LocalizeSingleViewFourier,
)

HeadKind = Literal["pooled", "fourier", "flatten"]


@dataclass(frozen=True)
class SingleViewArchConfig:
    """One architecture variant for mechanism comparison studies.

    Describes head type, convolutional neural network (CNN) depth, downsampling,
    and readout width. Passed
    to :func:`build_from_config` so experiments can sweep design axes without
    duplicating constructor wiring. ``input_representation`` and
    ``normalisation`` are logged for traceability — wire ``x_field`` /
    ``image_normalize`` on the catalog dataset separately.

    Attributes:
        arch_name: Stable experiment label (e.g. ``fourier_base_mlp``).
        head_type: Spatial readout family: ``pooled``, ``fourier``, or ``flatten``.
        encoder_channels: Per-block channel widths or a :data:`CHANNEL_PRESETS` key.
        downsample: Downsampling preset (``base`` = three conv blocks, no MaxPool).
        pre_flatten_channels: Optional 1×1 conv width before flatten/Fourier readout.
        embed_dim: Encoder embedding width (unused by pooled head path).
        flatten_hidden: Hidden width for flatten/Fourier multilayer perceptron
            (MLP) heads; ``1`` selects a single linear layer.
        flatten_head: ``linear`` or ``mlp`` for flatten/Fourier readouts only.
        input_representation: Catalog ``x_field`` label for logging (not applied
            by builders — set on the dataset / subset).
        normalisation: ``image_normalize`` label for logging (not applied by builders).
    """

    arch_name: str
    head_type: HeadKind
    encoder_channels: tuple[int, ...] = (16, 32, 64)
    downsample: str = "base"
    pre_flatten_channels: int | None = None
    embed_dim: int = 128
    flatten_hidden: int = 128
    flatten_head: FlattenHeadType | str = "mlp"
    input_representation: str = "anomaly_ref"
    normalisation: str = "none"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict; ``encoder_channels`` is a list, not a tuple."""
        payload = asdict(self)
        payload["encoder_channels"] = list(self.encoder_channels)
        return payload


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters for architecture comparison tables.

    Only tensors with ``requires_grad=True`` are included (frozen encoder
    blocks in fusion experts are excluded automatically).

    Returns:
        Total element count across all trainable parameters.
    """
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def materialize_lazy_modules(
    model: nn.Module,
    dummy: torch.Tensor,
) -> nn.Module:
    """Materialise lazy layers with one no-grad forward pass.

    Flatten and Fourier heads use ``LazyLinear`` whose input size depends on
    feature-map geometry. Call this **before** :func:`count_parameters` or
    logging flatten length in capability-stage diagnostics.

    Args:
        model: Localiser under test (modified in place).
        dummy: Single-sample input matching training shape, typically
            ``[1, 1, H, W]`` on the target device.

    Returns:
        The same ``model`` instance (for chaining).
    """
    model.eval()
    with torch.no_grad():
        _ = model(dummy)
    return model


def describe_feature_geometry(
    encoder: Encode,
    *,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Summarise encoder output geometry for a fixed input resolution.

    Relates head expressiveness to spatial bandwidth alongside parameter
    count in architecture comparison tables.

    Args:
        encoder: Built :class:`~tomography_ml.localization.encoder.Encode` module.
        height: Input image height in pixels.
        width: Input image width in pixels.

    Returns:
        Dict with keys ``encoder_channels``, ``downsample``,
        ``pre_flatten_channels``, ``out_channels``, ``feature_map_hw``
        ``(H_out, W_out)``, ``flatten_length``, and ``embed_dim``.
    """
    h_out, w_out = encoder.feature_map_size(height, width)
    flat_len = encoder.flatten_length(height, width)
    return {
        "encoder_channels": list(encoder.channels),
        "downsample": encoder.downsample,
        "pre_flatten_channels": encoder.pre_flatten_channels,
        "out_channels": encoder.out_channels,
        "feature_map_hw": (h_out, w_out),
        "flatten_length": flat_len,
        "embed_dim": encoder.embed_dim,
    }


def make_encode(
    *,
    channels: tuple[int, ...] | str | None = None,
    downsample: str = "base",
    pre_flatten_channels: int | None = None,
    embed_dim: int = 128,
    in_channels: int = 1,
) -> Encode:
    """Build the shared convolutional neural network (CNN) encoder backbone (conv blocks + optional pools).

    Args:
        channels: Explicit per-block widths or a :data:`CHANNEL_PRESETS` name
            resolved by :func:`~tomography_ml.localization.encoder.resolve_channels`.
        downsample: One of :data:`DOWNSAMPLE_MODES` (``base`` = three conv
            blocks, no MaxPool).
        pre_flatten_channels: Optional 1×1 channel squeeze before readout.
        embed_dim: Bottleneck embedding dimension on the encoder tail.
        in_channels: Input image channels (catalog single-view uses ``1``).

    Returns:
        Uninitialised :class:`~tomography_ml.localization.encoder.Encode`.

    Typically used with :func:`make_pooled`, :func:`make_fourier`, or
    :func:`make_flatten` to attach a spatial readout head.
    """
    return Encode(
        channels=channels,
        downsample=downsample,
        pre_flatten_channels=pre_flatten_channels,
        embed_dim=embed_dim,
        in_channels=in_channels,
    )


def make_pooled(
    *,
    n_outputs: int,
    encoder_channels: tuple[int, ...] | str | None = None,
    downsample: str = "base",
    pre_flatten_channels: int | None = None,
    embed_dim: int = 128,
    device: torch.device | str | None = None,
) -> LocalizeSingleView:
    """Build a globally pooled single-view localiser (spatially blind control).

    CNN → global average pooling (GAP) → linear → ``n_outputs`` coordinates.
    Useful as a negative control when comparing Fourier or flatten spatial
    readouts.

    Args:
        n_outputs: Target dimension (typically 3 for particle coordinates
            (x, y, z)).
        encoder_channels: Backbone width preset or explicit tuple.
        downsample: Spatial resolution preset on the encoder.
        pre_flatten_channels: Optional pre-pool 1×1 squeeze.
        embed_dim: Encoder embedding width.
        device: If set, move the module to this device before returning.

    Returns:
        :class:`~tomography_ml.localization.alternative_localizer.LocalizeSingleView`.
    """
    model = LocalizeSingleView(
        make_encode(
            channels=encoder_channels,
            downsample=downsample,
            pre_flatten_channels=pre_flatten_channels,
            embed_dim=embed_dim,
        ),
        n_outputs=n_outputs,
    )
    if device is not None:
        model = model.to(device)
    return model


def make_flatten(
    *,
    n_outputs: int,
    hidden: int = 128,
    encoder_channels: tuple[int, ...] | str | None = None,
    downsample: str = "medium",
    pre_flatten_channels: int | None = None,
    embed_dim: int = 128,
    head_type: FlattenHeadType | str = "mlp",
    device: torch.device | str | None = None,
) -> LocalizeSingleViewFlatten:
    """Build a flatten-readout single-view localiser.

    CNN → feature map → flatten → linear or multilayer perceptron (MLP) →
    ``n_outputs`` coordinates.
    Default ``downsample="medium"`` (two 2× pools) keeps flatten length
    tractable; use ``downsample="base"`` only for deliberate full-resolution
    diagnostics.

    Args:
        n_outputs: Target dimension (typically 3 for particle coordinates
            (x, y, z)).
        hidden: MLP hidden width; ``1`` with ``head_type="linear"`` selects
            a single linear layer.
        encoder_channels: Backbone width preset or explicit tuple.
        downsample: Spatial resolution preset on the encoder.
        pre_flatten_channels: Optional pre-readout 1×1 squeeze.
        embed_dim: Encoder embedding width.
        head_type: ``linear`` or ``mlp`` readout after flatten.
        device: If set, move the module to this device before returning.

    Returns:
        :class:`~tomography_ml.localization.alternative_localizer.LocalizeSingleViewFlatten`.
    """
    model = LocalizeSingleViewFlatten(
        make_encode(
            channels=encoder_channels,
            downsample=downsample,
            pre_flatten_channels=pre_flatten_channels,
            embed_dim=embed_dim,
        ),
        n_outputs=n_outputs,
        hidden=hidden,
        head_type=head_type,
    )
    if device is not None:
        model = model.to(device)
    return model


def make_fourier(
    *,
    n_outputs: int,
    hidden: int = 128,
    encoder_channels: tuple[int, ...] | str | None = None,
    downsample: str = "base",
    pre_flatten_channels: int | None = None,
    embed_dim: int = 128,
    head_type: FlattenHeadType | str = "mlp",
    device: torch.device | str | None = None,
) -> LocalizeSingleViewFourier:
    """Build a Fourier-coded single-view localiser.

    CNN → Fourier-coded spatial pool → linear/multilayer perceptron (MLP) →
    ``n_outputs`` coordinates.
    Default geometry matches the retained library class
    ``LocalizerSingleViewFourier`` (``downsample="base"``, MLP head).

    Args:
        n_outputs: Target dimension (typically 3 for particle coordinates
            (x, y, z)).
        hidden: MLP hidden width; ``1`` with ``head_type="linear"`` selects
            a single linear readout.
        encoder_channels: Backbone width preset or explicit tuple.
        downsample: Spatial resolution preset on the encoder.
        pre_flatten_channels: Optional pre-readout 1×1 squeeze.
        embed_dim: Encoder embedding width.
        head_type: ``linear`` or ``mlp`` after Fourier pooling.
        device: If set, move the module to this device before returning.

    Returns:
        :class:`~tomography_ml.localization.alternative_localizer.LocalizeSingleViewFourier`.
    """
    model = LocalizeSingleViewFourier(
        make_encode(
            channels=encoder_channels,
            downsample=downsample,
            pre_flatten_channels=pre_flatten_channels,
            embed_dim=embed_dim,
        ),
        n_outputs=n_outputs,
        hidden=hidden,
        head_type=head_type,
    )
    if device is not None:
        model = model.to(device)
    return model


def build_from_config(
    config: SingleViewArchConfig,
    *,
    n_outputs: int,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Instantiate a single-view localiser from a :class:`SingleViewArchConfig`.

    Dispatches on ``config.head_type`` to :func:`make_pooled`,
    :func:`make_fourier`, or :func:`make_flatten`. Representation and
    normalisation on the config are for experiment logging only — wire
    ``x_field`` / ``image_normalize`` on the catalog dataset separately.

    Args:
        config: Architecture row from a predefined mechanism grid.
        n_outputs: Coordinate dimension passed to the head.
        device: Optional torch device for the returned module.

    Returns:
        One of ``LocalizeSingleView``, ``LocalizeSingleViewFourier``, or
        ``LocalizeSingleViewFlatten``.

    Raises:
        ValueError: If ``head_type`` is not recognised.
    """
    if config.head_type == "pooled":
        return make_pooled(
            n_outputs=n_outputs,
            encoder_channels=config.encoder_channels,
            downsample=config.downsample,
            pre_flatten_channels=config.pre_flatten_channels,
            embed_dim=config.embed_dim,
            device=device,
        )
    if config.head_type == "fourier":
        return make_fourier(
            n_outputs=n_outputs,
            hidden=config.flatten_hidden,
            encoder_channels=config.encoder_channels,
            downsample=config.downsample,
            pre_flatten_channels=config.pre_flatten_channels,
            embed_dim=config.embed_dim,
            head_type=config.flatten_head,
            device=device,
        )
    if config.head_type == "flatten":
        return make_flatten(
            n_outputs=n_outputs,
            hidden=config.flatten_hidden,
            encoder_channels=config.encoder_channels,
            downsample=config.downsample,
            pre_flatten_channels=config.pre_flatten_channels,
            embed_dim=config.embed_dim,
            head_type=config.flatten_head,
            device=device,
        )
    raise ValueError(f"unknown head_type {config.head_type!r}")


def win3b_receptive_field_grid() -> tuple[SingleViewArchConfig, ...]:
    """Receptive-field and spatial-resolution architecture grid.

    Small predefined set for comparing CNN depth and downsampling while holding
    the Fourier readout family fixed; includes one flatten variant as an
    absolute-performance reference. ``base`` downsampling means three conv
    blocks with no MaxPool.

    Protocol: WIN 3B receptive-field grid.
    """
    base_ch = CHANNEL_PRESETS["base"]
    return (
        # Depth / receptive-field axis (downsample fixed at base = 3A / no pool).
        SingleViewArchConfig(
            arch_name="fourier_shallow_base",
            head_type="fourier",
            encoder_channels=CHANNEL_PRESETS["shallow"],
            downsample="base",
            flatten_hidden=1,
            flatten_head="linear",
        ),
        SingleViewArchConfig(
            arch_name="fourier_base_base",
            head_type="fourier",
            encoder_channels=base_ch,
            downsample="base",
            flatten_hidden=1,
            flatten_head="linear",
        ),
        SingleViewArchConfig(
            arch_name="fourier_deeper_base",
            head_type="fourier",
            encoder_channels=CHANNEL_PRESETS["deeper"],
            downsample="base",
            flatten_hidden=1,
            flatten_head="linear",
        ),
        # Spatial-resolution axis via downsampling (channels fixed at base).
        SingleViewArchConfig(
            arch_name="fourier_base_low",
            head_type="fourier",
            encoder_channels=base_ch,
            downsample="low",
            flatten_hidden=1,
            flatten_head="linear",
        ),
        SingleViewArchConfig(
            arch_name="fourier_base_medium",
            head_type="fourier",
            encoder_channels=base_ch,
            downsample="medium",
            flatten_hidden=1,
            flatten_head="linear",
        ),
        SingleViewArchConfig(
            arch_name="fourier_base_high",
            head_type="fourier",
            encoder_channels=base_ch,
            downsample="high",
            flatten_hidden=1,
            flatten_head="linear",
        ),
        # Absolute-performance reference (one Flatten confirmation at 3A geometry).
        SingleViewArchConfig(
            arch_name="flatten_base_base",
            head_type="flatten",
            encoder_channels=base_ch,
            downsample="base",
            flatten_hidden=128,
            flatten_head="mlp",
        ),
    )


def win3c_channel_capacity_grid() -> tuple[SingleViewArchConfig, ...]:
    """Channel-capacity architecture grid (narrow / base / wide Fourier).

    Sweeps encoder channel width while fixing downsampling at ``base`` (no
    MaxPool). Includes pooled (spatially blind) and flatten (high-capacity)
    controls at the same geometry. With Fourier-coded pooling, channel count
    approximates available spatial-mode bandwidth.

    Protocol: WIN 3C channel-capacity grid.
    """
    base_ch = CHANNEL_PRESETS["base"]
    return (
        SingleViewArchConfig(
            arch_name="pooled_base_base",
            head_type="pooled",
            encoder_channels=base_ch,
            downsample="base",
        ),
        SingleViewArchConfig(
            arch_name="fourier_narrow_base",
            head_type="fourier",
            encoder_channels=CHANNEL_PRESETS["narrow"],
            downsample="base",
            flatten_hidden=1,
            flatten_head="linear",
        ),
        SingleViewArchConfig(
            arch_name="fourier_base_base",
            head_type="fourier",
            encoder_channels=base_ch,
            downsample="base",
            flatten_hidden=1,
            flatten_head="linear",
        ),
        SingleViewArchConfig(
            arch_name="fourier_wide_base",
            head_type="fourier",
            encoder_channels=CHANNEL_PRESETS["wide"],
            downsample="base",
            flatten_hidden=1,
            flatten_head="linear",
        ),
        SingleViewArchConfig(
            arch_name="flatten_base_base",
            head_type="flatten",
            encoder_channels=base_ch,
            downsample="base",
            flatten_hidden=128,
            flatten_head="mlp",
        ),
    )


def win3d_head_expressiveness_grid() -> tuple[SingleViewArchConfig, ...]:
    """Head-expressiveness grid on the retained architecture triad.

    All variants use base CNN geometry (channels base, downsample base).
    Primary axis: Fourier linear vs small multilayer perceptron (MLP).
    Positive: flatten linear vs small MLP. Negative: pooled (single linear
    head after global average pooling (GAP)).

    Protocol: WIN 3D head-expressiveness grid.
    """
    base_ch = CHANNEL_PRESETS["base"]
    return (
        SingleViewArchConfig(
            arch_name="pooled_base_base",
            head_type="pooled",
            encoder_channels=base_ch,
            downsample="base",
        ),
        SingleViewArchConfig(
            arch_name="fourier_base_linear",
            head_type="fourier",
            encoder_channels=base_ch,
            downsample="base",
            flatten_hidden=1,
            flatten_head="linear",
        ),
        SingleViewArchConfig(
            arch_name="fourier_base_mlp",
            head_type="fourier",
            encoder_channels=base_ch,
            downsample="base",
            flatten_hidden=128,
            flatten_head="mlp",
        ),
        SingleViewArchConfig(
            arch_name="flatten_base_linear",
            head_type="flatten",
            encoder_channels=base_ch,
            downsample="base",
            flatten_hidden=1,
            flatten_head="linear",
        ),
        SingleViewArchConfig(
            arch_name="flatten_base_mlp",
            head_type="flatten",
            encoder_channels=base_ch,
            downsample="base",
            flatten_hidden=128,
            flatten_head="mlp",
        ),
    )


@dataclass(frozen=True)
class ArchitectureFreezeRecord:
    """Formal single-view architecture freeze after mechanism studies.

    After architecture sweeps, further capability work (representation,
    normalisation, regime) holds convolutional neural network (CNN) geometry
    and spatial readout fixed at the selected primary. Flatten and pooled variants remain documented
    controls only.

    Attributes:
        selected_variant: Primary arch name (``fourier_base_mlp``).
        spatial_readout_type: Readout mechanism label for reports.
        depth: Number of conv blocks in the frozen backbone.
        widths: Channel widths per block.
        downsampling: Downsampling preset (``base`` = no MaxPool).
        head: Head nonlinearity family (``mlp`` multilayer perceptron on the retained primary).
        head_hidden: Multilayer perceptron (MLP) hidden width on the primary.
        positive_baseline: Flatten reference arch name.
        negative_control: Pooled reference arch name.
        library_class: Canonical import/class string for reproducibility.
        selection_rationale: Human-readable freeze justification.
        accuracy_complexity_notes: Short comparison notes for reports.
        freeze_fields: Tuple of axis names that must not be re-tuned silently.

    Protocol: WIN 3E architecture freeze.
    """

    selected_variant: str = "fourier_base_mlp"
    spatial_readout_type: str = "fourier_coded_pool"
    depth: int = 3
    widths: tuple[int, ...] = (16, 32, 64)
    downsampling: str = "base"
    head: str = "mlp"
    head_hidden: int = 128
    positive_baseline: str = "flatten_base_mlp"
    negative_control: str = "pooled_base_base"
    library_class: str = "LocalizerSingleViewFourier"
    selection_rationale: str = (
        "parameter_economy_vs_performance_after_win3a_3d;"
        "spatial_readout_is_dominant_bottleneck;"
        "fourier_recovers_most_flatten_benefit_at_~4000x_fewer_params;"
        "mlp_head_retained_after_win3d"
    )
    accuracy_complexity_notes: str = (
        "pool_negative_control;"
        "fourier_primary_compact_token;"
        "flatten_highest_performance_baseline_not_default"
    )
    freeze_fields: tuple[str, ...] = (
        "cnn_depth",
        "channel_widths",
        "downsampling_geometry",
        "spatial_readout_type",
        "head_type",
    )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable freeze record; tuple fields become lists."""
        payload = asdict(self)
        payload["widths"] = list(self.widths)
        payload["freeze_fields"] = list(self.freeze_fields)
        return payload

    def primary_config(self) -> SingleViewArchConfig:
        """``SingleViewArchConfig`` matching the frozen primary architecture."""
        return SingleViewArchConfig(
            arch_name=self.selected_variant,
            head_type="fourier",
            encoder_channels=self.widths,
            downsample=self.downsampling,
            flatten_hidden=self.head_hidden,
            flatten_head=self.head,
        )


def win3e_architecture_freeze() -> ArchitectureFreezeRecord:
    """Return the formal architecture freeze record.

    Selected primary after mechanism sweeps: Fourier-base + multilayer
    perceptron (MLP) (library default ``LocalizerSingleViewFourier``). Flatten and pooling remain
    interpretive controls, not unbounded search axes.

    Protocol: WIN 3E architecture freeze.
    """
    return ArchitectureFreezeRecord()


def win3e_control_configs() -> tuple[SingleViewArchConfig, ...]:
    """Return the interpretive triad at frozen convolutional neural network (CNN) geometry.

    Order: (1) Fourier primary from :func:`win3e_architecture_freeze`,
    (2) flatten positive baseline, (3) pooled negative control. Used whenever
    three reference architectures are needed without re-deriving names from
    the head-expressiveness grid.

    Returns:
        Three :class:`SingleViewArchConfig` instances — primary, positive,
        negative — all at ``downsample="base"`` channel preset ``base``.

    Protocol: WIN 3E control triad.
    """
    freeze = win3e_architecture_freeze()
    by_name = {cfg.arch_name: cfg for cfg in win3d_head_expressiveness_grid()}
    return (
        freeze.primary_config(),
        by_name[freeze.positive_baseline],
        by_name[freeze.negative_control],
    )


@dataclass(frozen=True)
class RepresentationSpec:
    """One input representation choice (label → catalog ``x_field``).

    Attributes:
        name: Short ladder label (``delta``, ``clean``, ``observed``).
        x_field: Catalog tensor key passed to ``DatasetTaskSpec.x_fields``.
        description: Human-readable role mapping for reports and CSVs.
    """

    name: str
    x_field: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Plain dict copy suitable for JSON / DataFrame cells."""
        return asdict(self)


def win3f_representation_grid() -> tuple[RepresentationSpec, ...]:
    """Input representation ladder on the frozen single-view architecture.

    ``delta`` uses the stored anomaly role (``particle − clean``), the
    oracle particle-specific signal. Under no corruption, ``observed`` matches
    the particle/dirty camera image.

    Protocol: WIN 3F representation grid.
    """
    return (
        RepresentationSpec(
            name="delta",
            x_field="anomaly_ref",
            description="particle_minus_clean (stored anomaly role)",
        ),
        RepresentationSpec(
            name="clean",
            x_field="clean_ref",
            description="clean simulator camera image",
        ),
        RepresentationSpec(
            name="observed",
            x_field="observed_ref",
            description="particle/dirty observed camera image",
        ),
    )


def win3f_selected_representation() -> RepresentationSpec:
    """Representation pinned after the representation ladder study.

    Delta (anomaly role) recovers localisation on the frozen Fourier primary;
    observed is informative but weaker; clean collapses. Capability studies
    (normalisation, regime, observability) continue on **delta** (oracle)
    until a formal representation freeze elevates observed / restoration.

    Protocol: WIN 3F selected representation.
    """
    by_name = {r.name: r for r in win3f_representation_grid()}
    return by_name["delta"]


@dataclass(frozen=True)
class NormalisationSpec:
    """One intensity-normalisation mode for catalog loading.

    Attributes:
        name: Short ladder label (``raw``, ``train_split_zscore``, …).
        image_normalize: Value for ``DatasetTaskSpec.image_normalize``.
        description: Scientific intent for reports.
        diagnostic: If ``True``, per-image modes that remove absolute intensity
            — useful for diagnosis but not the default scientific path unless
            formally elevated.
    """

    name: str
    image_normalize: str
    description: str
    diagnostic: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Plain dict copy suitable for JSON / DataFrame cells."""
        return asdict(self)


def win3g_normalisation_grid() -> tuple[NormalisationSpec, ...]:
    """Normalisation ladder on the frozen architecture and representation.

    Train-split global z-score preserves cross-sample absolute intensity.
    Per-image modes are **diagnostic** (dynamic-range removal) and must not
    silently become the main scientific path unless later elevated.

    Protocol: WIN 3G normalisation grid.
    """
    return (
        NormalisationSpec(
            name="raw",
            image_normalize="none",
            description="raw float, no normalisation",
            diagnostic=False,
        ),
        NormalisationSpec(
            name="train_split_zscore",
            image_normalize="train_split_zscore",
            description="training-split global z-score (mean/std from train)",
            diagnostic=False,
        ),
        NormalisationSpec(
            name="per_image_zscore",
            image_normalize="per_image_zscore",
            description="per-image z-score (diagnostic dynamic-range removal)",
            diagnostic=True,
        ),
        NormalisationSpec(
            name="per_image_minmax",
            image_normalize="per_image_minmax",
            description="per-image min/max to [0,1] (diagnostic)",
            diagnostic=True,
        ),
    )


def win3g_selected_normalisation() -> NormalisationSpec:
    """Normalisation pinned after the normalisation ladder study.

    Elevated standard: ``per_image_zscore`` (per-view z-score). Raw collapses;
    train-split global helps; per-view z-score is best on the frozen Fourier
    primary; per-view min–max is weaker.

    Protocol: WIN 3G selected normalisation.
    """
    by_name = {n.name: n for n in win3g_normalisation_grid()}
    return by_name["per_image_zscore"]


@dataclass(frozen=True)
class OpticalRegimeSpec:
    """One background optical regime (catalog filter + reported μ).

    Background ``μs`` / ``μa`` are catalog metadata for stratified studies —
    never model inputs. Filter rows by ``optical_setup_id`` only.

    Attributes:
        name: Ladder label (``low``, ``medium``, ``high``).
        optical_setup_id: Catalog ``optical_setup_id`` filter value.
        mu_s: Reported scattering coefficient (documentation only).
        mu_a: Reported absorption coefficient (documentation only).
        description: Short regime summary for tables.
    """

    name: str
    optical_setup_id: str
    mu_s: float
    mu_a: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Plain dict copy suitable for JSON / DataFrame cells."""
        return asdict(self)


def win3h_optical_regime_grid() -> tuple[OpticalRegimeSpec, ...]:
    """Low / medium / high optical-regime ladder for stratified studies.

    Values match ``configs/m8/localization_single_particle.xlsx`` sheet
    ``optical_setups``. Background ``μa`` / ``μs`` are nuisance factors for
    localisation — catalog provenance / reported columns only, never model
    inputs.

    Protocol: WIN 3H optical-regime grid.
    """
    return (
        OpticalRegimeSpec(
            name="low",
            optical_setup_id="opt_m8_low_001",
            mu_s=0.01,
            mu_a=0.003,
            description="low attenuation background optics",
        ),
        OpticalRegimeSpec(
            name="medium",
            optical_setup_id="opt_m8_med_001",
            mu_s=0.03,
            mu_a=0.01,
            description="medium attenuation background optics",
        ),
        OpticalRegimeSpec(
            name="high",
            optical_setup_id="opt_m8_high_001",
            mu_s=0.1,
            mu_a=0.03,
            description="high attenuation background optics",
        ),
    )


def win3i_key_result_sources() -> tuple[dict[str, str], ...]:
    """Relative CSV paths for key capability-study result tables.

    Paths are relative to ``data/generated/m8_1/``. Confirmatory training uses
    the elevated per-view z-score standard.

    Protocol: WIN 3I observability result sources.
    """
    return (
        {
            "win": "3F",
            "label": "representation",
            "relative_csv": "_win3f_representation/win3f_representation_study.csv",
        },
        {
            "win": "3G",
            "label": "normalisation",
            "relative_csv": "_win3g_normalisation/win3g_normalisation_study.csv",
        },
        {
            "win": "3H",
            "label": "optical_regime",
            "relative_csv": "_win3h_optical_regime/win3h_optical_regime_study.csv",
        },
    )


@dataclass(frozen=True)
class SingleViewBlockFreezeRecord:
    """Formal single-view block freeze before multi-view fusion work.

    Combines the architecture freeze with selected representation,
    normalisation, and training protocol. Edit only with a new written
    hypothesis — do not silently retune for multi-view.

    Attributes:
        architecture: Nested :class:`ArchitectureFreezeRecord`.
        representation_name: Selected representation label (``delta``).
        x_field: Catalog input field (``anomaly_ref``).
        normalisation_name: Selected normalisation label.
        image_normalize: Catalog normalisation string.
        phantom_family_reference: Reference corpus tag for reports.
        optical_regime_reference: Optical regime label used in observability work.
        optical_setup_id_reference: Catalog filter for the reference regime.
        keep_angles_deg: Single retained camera view (degrees).
        batch_size: Default training minibatch size.
        num_epochs_max: Upper epoch budget before early stopping.
        early_stop_patience: Validation root-mean-square error (RMSE) patience
            (epochs).
        optimizer: Optimiser name (``adam``).
        loss: Training loss (``mean squared error (MSE)``).
        lr_primary: Stage-3 learning rate (LR) for the Fourier primary.
        lr_positive_baseline: Stage-3 LR for the flatten control.
        lr_negative_control: Stage-3 LR for the pooled control.
        library_class: Canonical class string for the frozen primary.
        positive_baseline / negative_control: Control arch names.
        restoration_note: Observed-vs-delta guidance for restoration decisions.
        observability_note: Require total + per-axis root-mean-square error
            (RMSE) in reports.
        selection_rationale: Human-readable freeze justification.
        freeze_fields: Axes that must not change without a new hypothesis.

    Protocol: WIN 3J single-view block freeze.
    """

    architecture: ArchitectureFreezeRecord = ArchitectureFreezeRecord()
    representation_name: str = "delta"
    x_field: str = "anomaly_ref"
    normalisation_name: str = "per_image_zscore"
    image_normalize: str = "per_image_zscore"
    phantom_family_reference: str = "gummybear"
    optical_regime_reference: str = "high"
    optical_setup_id_reference: str = "opt_m8_high_001"
    keep_angles_deg: float = 180.0
    batch_size: int = 16
    num_epochs_max: int = 200
    early_stop_patience: int = 40
    optimizer: str = "adam"
    loss: str = "mse"
    lr_primary: float = 0.03
    lr_positive_baseline: float = 0.0003
    lr_negative_control: float = 0.001
    library_class: str = "LocalizerSingleViewFourier"
    positive_baseline: str = "flatten_base_mlp"
    negative_control: str = "pooled_base_base"
    restoration_note: str = (
        "win3f_delta_excellent_observed_degraded;"
        "restoration_strategically_important;"
        "multi_view_geometry_studies_may_use_delta_to_isolate_view_integration;"
        "observed_remains_operational_target_for_restoration_decisions"
    )
    observability_note: str = (
        "report_RMSE_total_and_per_axis_X_Y_Z;"
        "single_view_axis_difficulty_is_not_isotropic;"
        "do_not_hide_axis_structure_in_one_scalar"
    )
    selection_rationale: str = (
        "architecture_from_win3e;"
        "normalisation_per_view_zscore_from_win3g;"
        "representation_delta_oracle_from_win3f_capability_path;"
        "training_protocol_early_stopping_best_val_weights;"
        "freeze_before_multi_view_so_win4_5_test_view_integration_not_encoder_repair"
    )
    freeze_fields: tuple[str, ...] = (
        "architecture",
        "representation",
        "normalisation",
        "training_protocol",
        "keep_angles_deg",
        "reference_corpus",
    )

    def to_dict(self) -> dict[str, Any]:
        """Nested JSON-serialisable freeze; nested architecture is expanded."""
        payload = asdict(self)
        payload["architecture"] = self.architecture.to_dict()
        payload["freeze_fields"] = list(self.freeze_fields)
        return payload

    def representation(self) -> RepresentationSpec:
        """Selected representation spec (must match ``win3f_selected_representation``)."""
        return win3f_selected_representation()

    def normalisation(self) -> NormalisationSpec:
        """Selected normalisation spec (must match ``win3g_selected_normalisation``)."""
        return win3g_selected_normalisation()

    def lr_by_role(self) -> dict[str, float]:
        """Stage-3 learning rates (LR) keyed by interpretive role."""
        return {
            "primary": float(self.lr_primary),
            "positive_baseline": float(self.lr_positive_baseline),
            "negative_control": float(self.lr_negative_control),
        }

    def primary_config(self) -> SingleViewArchConfig:
        """Primary :class:`SingleViewArchConfig` from the nested architecture freeze."""
        return self.architecture.primary_config()


def win3j_single_view_freeze() -> SingleViewBlockFreezeRecord:
    """Return the formal single-view block freeze for multi-view studies.

    Asserts consistency with :func:`win3e_architecture_freeze` and the
    selected representation / normalisation helpers.

    Protocol: WIN 3J single-view freeze.

    See also:
        :class:`~tomography_ml.localization.localizer.LocalizerSingleViewFourier` — frozen trunk reused by fusion models.
    """
    record = SingleViewBlockFreezeRecord()
    arch = win3e_architecture_freeze()
    rep = win3f_selected_representation()
    norm = win3g_selected_normalisation()
    if record.architecture.selected_variant != arch.selected_variant:
        raise ValueError("3J architecture disagrees with win3e_architecture_freeze()")
    if record.representation_name != rep.name or record.x_field != rep.x_field:
        raise ValueError("3J representation disagrees with win3f_selected_representation()")
    if (
        record.normalisation_name != norm.name
        or record.image_normalize != norm.image_normalize
    ):
        raise ValueError("3J normalisation disagrees with win3g_selected_normalisation()")
    return record


def default_mechanism_grid() -> tuple[SingleViewArchConfig, ...]:
    """Union of receptive-field, channel-capacity, and head-expressiveness grids.

    De-duplicates by ``arch_name``. Prefer the stage-specific helpers
    (:func:`win3b_receptive_field_grid`, :func:`win3c_channel_capacity_grid`,
    :func:`win3d_head_expressiveness_grid`) when sweeping one axis at a time.

    Returns:
        De-duplicated tuple of :class:`SingleViewArchConfig` rows (order
        follows first appearance across B → C → D).
    """
    by_name: dict[str, SingleViewArchConfig] = {}
    for cfg in (
        *win3b_receptive_field_grid(),
        *win3c_channel_capacity_grid(),
        *win3d_head_expressiveness_grid(),
    ):
        by_name[cfg.arch_name] = cfg
    return tuple(by_name.values())


__all__ = [
    "ArchitectureFreezeRecord",
    "CHANNEL_PRESETS",
    "DOWNSAMPLE_MODES",
    "HeadKind",
    "NormalisationSpec",
    "OpticalRegimeSpec",
    "RepresentationSpec",
    "SingleViewArchConfig",
    "SingleViewBlockFreezeRecord",
    "build_from_config",
    "count_parameters",
    "default_mechanism_grid",
    "describe_feature_geometry",
    "make_encode",
    "make_flatten",
    "make_fourier",
    "make_pooled",
    "materialize_lazy_modules",
    "resolve_channels",
    "win3b_receptive_field_grid",
    "win3c_channel_capacity_grid",
    "win3d_head_expressiveness_grid",
    "win3e_architecture_freeze",
    "win3e_control_configs",
    "win3f_representation_grid",
    "win3f_selected_representation",
    "win3g_normalisation_grid",
    "win3g_selected_normalisation",
    "win3h_optical_regime_grid",
    "win3i_key_result_sources",
    "win3j_single_view_freeze",
]
