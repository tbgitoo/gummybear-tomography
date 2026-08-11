"""Multi-view localisation: fuse per-view predictions or latents into one coordinate.

Fusion ladder (increasing complexity):

**Expert coordinate averaging** — Run a separately trained single-view model per
acquisition angle; average per-view particle coordinates (x, y, z) with no
learned fusion module.

**Compact latent fusion** — Shared (typically frozen) trunk → latents ``h_i``;
pack via ordered concat or mean-pool → small two-layer multilayer perceptron
(MLP) → xyz coordinates.

**DeepSets latent fusion** — Same frozen trunk; permutation-invariant
``ρ(mean_i φ(h_i))`` instead of ordered concat.

**Geometry-aware end-to-end (e2e) fusion** — Jointly train trunk + fusion MLP;
attach camera or light ``sin θ / cos θ`` as concat tokens or Feature-wise
Linear Modulation (FiLM) before ordered-concat fusion. Compact or large fusion
MLP capacity.

**Flat joint camera×illumination fusion** — End-to-end trunk; each token carries
latent plus camera **and** light sin/cos; single ordered-concat fusion MLP.

**Illumination-only fusion** — Fixed camera; view axis is the light orbit.
Variants omit angle tokens (compact MLP only) or add light sin/cos (concat or
FiLM).

**Hierarchical light-then-camera fusion** — Fuse lights within each camera to a
camera-level latent, then fuse cameras across viewpoints.

**Mean-latent sanity path** — Mean locked latents then affine coordinate head;
equivalent to coordinate averaging under a linear head; not the learned compact
fusion baseline.

Single-view trunk architecture (Fourier-base or pooled global average pooling
(GAP)) is fixed here — do not retune channels, Fourier modes, or head width.

Notebook / protocol map:

- Expert averaging: 09_0
- Compact latent fusion: 09_1; illumination-only without angles: 10_1-C
- DeepSets: 09_1
- Geometry-aware e2e (camera orbit): 09_2 / 09_3; pooled control: 09_2B
- Flat camera×light joint: 10_baseline
- Illumination + light angles: 10_1-D
- Hierarchical: 10_2
- Mean-latent sanity: demoted control
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from tomography_ml.localization.alternative_localizer import LocalizeSingleView
from tomography_ml.localization.localizer import LocalizerSingleViewFourier

FUSION_PATTERN_09_0 = "expert_xyz_mean"
# Per-angle expert predictions averaged; no learned fusion. Protocol: 09_0.
FUSION_PATTERN_09_1 = "compact_fusion_mlp_frozen_fourier"
# Ordered-concat compact fusion MLP over frozen Fourier trunk. Protocol: 09_1.
FUSION_PATTERN_09_1_MEAN_POOL = "compact_fusion_mlp_mean_pool_frozen_fourier"
# Mean-pool latents before the same compact fusion MLP. Protocol: 09_1.
# Compact fusion with pooled (GAP) frozen trunk. Protocol: 09_1B.
FUSION_PATTERN_09_1_POOLED = "compact_fusion_mlp_frozen_pooled"
FUSION_PATTERN_09_1_MEAN_POOL_POOLED = (
    "compact_fusion_mlp_mean_pool_frozen_pooled"
)
# Latent packing modes for compact fusion (structural fork only).
PACKING_ORDERED_CONCAT = "ordered_concat"
PACKING_MEAN_POOL = "mean_pool"
# DeepSets ρ(mean φ(h)) with frozen Fourier trunk. Protocol: 09_1.
FUSION_PATTERN_09_1_DEEPSETS_FOURIER = "deepsets_fourier"
FUSION_PATTERN_09_1_DEEPSETS_NO_FOURIER = "deepsets_no_fourier"
# DeepSets φ and ρ hidden widths (match compact fusion MLP width 128).
M9_1_DEEPSETS_PHI_HIDDEN = 128
M9_1_DEEPSETS_RHO_HIDDEN = 128
# E2e trunk + camera sin/cos + compact fusion MLP. Protocol: 09_2.
FUSION_PATTERN_09_2 = "e2e_fourier_geometry_fusion"
# Same geometry wiring; larger fusion MLP capacity. Protocol: 09_3.
FUSION_PATTERN_09_3 = "e2e_fourier_geometry_large_fusion"
# Pooled (GAP) trunk e2e + geometry analogues. Protocol: 09_2B / 09_3.
FUSION_PATTERN_09_2_POOLED = "e2e_pooled_geometry_fusion"
FUSION_PATTERN_09_3_POOLED = "e2e_pooled_geometry_large_fusion"
# Flat joint camera + light sin/cos tokens. Protocol: 10_baseline.
FUSION_PATTERN_10_BASELINE = "e2e_fourier_illumination_geometry_fusion"
FUSION_PATTERN_10_BASELINE_POOLED = (
    "e2e_pooled_illumination_geometry_fusion"
)
# Illumination-only e2e fusion without angle tokens. Protocol: 10_1-C.
FUSION_PATTERN_10_1_C = "e2e_fourier_illumination_fusion"
FUSION_PATTERN_10_1_C_POOLED = "e2e_pooled_illumination_fusion"
# Frozen-trunk illumination-only fusion. Protocol: 10_1A-C.
FUSION_PATTERN_10_1_C_FROZEN = "frozen_illumination_fusion"
FUSION_PATTERN_10_1_C_FROZEN_POOLED = "frozen_pooled_illumination_fusion"
# Illumination-only e2e fusion with light sin/cos. Protocol: 10_1-D.
FUSION_PATTERN_10_1_D = "e2e_fourier_illumination_angle_fusion"
FUSION_PATTERN_10_1_D_POOLED = "e2e_pooled_illumination_angle_fusion"
# Frozen-trunk illumination fusion with light sin/cos. Protocol: 10_1A-D.
FUSION_PATTERN_10_1_D_FROZEN = "frozen_illumination_angle_fusion"
FUSION_PATTERN_10_1_D_FROZEN_POOLED = "frozen_pooled_illumination_angle_fusion"
# Append sin/cos as tokens vs FiLM-modulate latents before fusion.
GEOMETRY_MODE_CONCAT = "concat"
GEOMETRY_MODE_FILM = "film"
# Two-stage light-then-camera hierarchical fusion. Protocol: 10_2.
FUSION_PATTERN_10_2 = "hierarchical_light_then_camera_fusion"
FUSION_PATTERN_10_2_POOLED = "hierarchical_pooled_light_then_camera_fusion"
# Mean locked latents → affine head (demoted sanity control).
FUSION_PATTERN_MEAN_LATENT_SANITY = "shared_xyz_mean_sanity"

# Compact ordered-concat fusion MLP width and depth. Protocol: 09_2 family.
M9_2_FUSION_HIDDEN = 128
M9_2_FUSION_DEPTH = 1
# Large fusion MLP capacity upper bound. Protocol: 09_3.
M9_3_FUSION_HIDDEN = 512
M9_3_FUSION_DEPTH = 2
# Flat joint and illumination-only paths reuse compact fusion capacity.
M10_BASELINE_FUSION_HIDDEN = M9_2_FUSION_HIDDEN
M10_BASELINE_FUSION_DEPTH = M9_2_FUSION_DEPTH
M10_1_FUSION_HIDDEN = M9_2_FUSION_HIDDEN
M10_1_FUSION_DEPTH = M9_2_FUSION_DEPTH
# Both hierarchical fusion stages use compact MLP capacity.
M10_2_FUSION_HIDDEN = M9_2_FUSION_HIDDEN
M10_2_FUSION_DEPTH = M9_2_FUSION_DEPTH
M10_2_CAMERA_LATENT_DIM = 128

# Canonical illumination orbit (absolute azimuth about z).
M10_LIGHT_ANGLES_DEG = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
M10_LIGHT_RADIUS_XY = 20.0
M10_LIGHT_Z = 10.0
M10_OPTICAL_SETUP_PREFIX = "opt_m10_illum_"


def make_angle_features(angles_deg: torch.Tensor) -> torch.Tensor:
    """Map acquisition angles in degrees to ``(sin θ, cos θ)``.

    Accepts ``[...]`` shaped tensors; returns ``[..., 2]``. Used wherever
    fusion models encode orbit geometry as sin/cos features.
    """
    if angles_deg.ndim < 1:
        raise ValueError(
            f"angles_deg must have at least one dimension; got shape "
            f"{tuple(angles_deg.shape)}"
        )
    rad = angles_deg.to(dtype=torch.float32) * (math.pi / 180.0)
    return torch.stack((torch.sin(rad), torch.cos(rad)), dim=-1)


def angle_key(angle_deg: float, *, ndigits: int = 6) -> str:
    """Stable ``ModuleDict`` key for an acquisition angle in degrees.

    PyTorch module names cannot contain ``'.'``, so the fractional part uses
    ``p`` (e.g. ``180p000000``).
    """
    rounded = round(float(angle_deg), int(ndigits))
    as_fixed = f"{rounded:.{int(ndigits)}f}".replace(".", "p").replace("-", "m")
    return f"ang_{as_fixed}"


def match_expert_index(
    angle_deg: float,
    expert_angles_deg: Sequence[float],
    *,
    atol_deg: float = 1e-3,
) -> int:
    """Map one acquisition angle to its expert index in ``expert_angles_deg``.

    Routes each view slot to the angle-specific expert trained at that orbit
    position in :class:`ExpertXyzMeanLocalizer`.

    Args:
        angle_deg: Query angle in degrees.
        expert_angles_deg: Sorted unique expert registry angles.
        atol_deg: Maximum absolute difference (degrees) for a match.

    Returns:
        Integer index into ``expert_angles_deg``.

    Raises:
        KeyError: No expert within ``atol_deg``.
        ValueError: More than one expert matches (non-unique registry).
    """
    matches = [
        i
        for i, theta in enumerate(expert_angles_deg)
        if abs(float(theta) - float(angle_deg)) <= float(atol_deg)
    ]
    if not matches:
        raise KeyError(
            f"No expert for angle_deg≈{float(angle_deg)} "
            f"(atol={float(atol_deg)}); known={list(expert_angles_deg)}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple experts match angle_deg≈{float(angle_deg)}; "
            f"indices={matches}"
        )
    return int(matches[0])


def mean_coordinates(xyz_per_view: torch.Tensor) -> torch.Tensor:
    """Average per-view particle coordinates (x, y, z) predictions with uniform weights.

    No learned fusion — each view contributes equally. Evaluation rule for
    :class:`ExpertXyzMeanLocalizer` and the coordinate-level analogue of
    :func:`shared_xyz_mean` / :class:`MeanLatentFusionLocalizer` under an
    affine single-view head. :data:`fuse_coordinates` is a backward-compatible
    alias for this function.

    Args:
        xyz_per_view: Per-view predictions, shape ``[B, V, 3]``.

    Returns:
        Fused coordinates, shape ``[B, 3]``.

    Notebook / protocol: 09_0 expert coordinate averaging.
    """
    if xyz_per_view.ndim != 3 or xyz_per_view.shape[-1] != 3:
        raise ValueError(
            "xyz_per_view must have shape [B, V, 3]; "
            f"got {tuple(xyz_per_view.shape)}"
        )
    return xyz_per_view.mean(dim=1)


# Backward-compatible alias used by older call sites / notebooks.
fuse_coordinates = mean_coordinates


def new_frozen_single_view_expert(
    *,
    n_outputs: int = 3,
    hidden: int = 128,
    in_channels: int = 1,
) -> LocalizerSingleViewFourier:
    """Construct one Fourier-base single-view expert (Fourier trunk + multilayer perceptron (MLP)).

    Architecture is fixed — do not retune channels, Fourier modes, or head
    width here.
    """
    return LocalizerSingleViewFourier(
        n_outputs=n_outputs,
        hidden=hidden,
        in_channels=in_channels,
    )


def new_frozen_pooled_single_view_expert(
    *,
    n_outputs: int = 3,
    embed_dim: int = 128,
) -> LocalizeSingleView:
    """Construct a pooled global average pooling (GAP) single-view expert (no Fourier features).

    Negative-control trunk for Fourier-vs-pooled comparisons; matches
    ``pooled_base_base`` / ``make_pooled`` defaults.

    Notebook / protocol: 09_1B pooled backbone.
    """
    from tomography_ml.localization.builders import make_pooled

    return make_pooled(n_outputs=n_outputs, embed_dim=embed_dim, downsample="base")


def _unique_sorted_angles(
    angles_deg: Sequence[float],
    *,
    atol_deg: float,
) -> tuple[float, ...]:
    """Return sorted unique angles for expert registry keys."""
    angles = tuple(sorted(float(a) for a in angles_deg))
    if not angles:
        raise ValueError("angles_deg must be non-empty")
    for i in range(1, len(angles)):
        if abs(angles[i] - angles[i - 1]) <= float(atol_deg):
            raise ValueError(
                f"angles are not unique within atol={atol_deg}: "
                f"{angles[i - 1]} vs {angles[i]}"
            )
    return angles


def resolve_view_angles(
    angles_deg: torch.Tensor | Sequence[float] | None,
    *,
    batch: int,
    n_views: int,
    default_angles_deg: Sequence[float],
) -> torch.Tensor:
    """Broadcast per-view acquisition angles to batch shape ``[B, V]``.

    Normalises the several angle conventions used across fusion models: a
    shared orbit ``[V]``, an explicit per-batch schedule ``[B, V]``, or
    ``None`` to fall back to a registered default (e.g.
    ``ExpertXyzMeanLocalizer.expert_angles_deg``,
    ``GeometryAwareFourierFusionLocalizer.view_angles_deg``).

    Args:
        angles_deg: ``None``, ``[V]``, or ``[B, V]`` angles in degrees.
        batch: Batch size ``B``.
        n_views: View count ``V``.
        default_angles_deg: Orbit used when ``angles_deg`` is ``None``; must
            have length ``V``.

    Returns:
        Float tensor of shape ``[B, V]`` (degrees).
    """
    if angles_deg is None:
        if n_views != len(default_angles_deg):
            raise ValueError(
                "angles_deg is required when V does not equal "
                f"len(default_angles_deg)={len(default_angles_deg)}; got V={n_views}"
            )
        base = torch.tensor(list(default_angles_deg), dtype=torch.float32)
        return base.unsqueeze(0).expand(batch, -1)

    if isinstance(angles_deg, torch.Tensor):
        ang = angles_deg.detach().float()
    else:
        ang = torch.tensor(list(angles_deg), dtype=torch.float32)

    if ang.ndim == 1:
        if ang.numel() != n_views:
            raise ValueError(
                f"angles_deg length {ang.numel()} must equal V={n_views}"
            )
        return ang.unsqueeze(0).expand(batch, -1)
    if ang.ndim == 2:
        if ang.shape != (batch, n_views):
            raise ValueError(
                f"angles_deg shape {tuple(ang.shape)} must be "
                f"[B, V]=[{batch}, {n_views}]"
            )
        return ang
    raise ValueError(
        f"angles_deg must have shape [V] or [B, V]; got {tuple(ang.shape)}"
    )


def ensure_multi_view_batch(views: torch.Tensor) -> torch.Tensor:
    """Normalise single- or multi-view image batches to ``[B, V, C, H, W]``.

    Accepts ``[B, C, H, W]`` (implicit ``V=1``) or an already batched
    ``[B, V, C, H, W]`` tensor. All fusion localizers call this at the
    boundary so callers may pass either layout.

    Args:
        views: Camera images, 4-D or 5-D as above.

    Returns:
        Multi-view batch with shape ``[B, V, C, H, W]``.
    """
    if views.ndim == 4:
        return views.unsqueeze(1)
    if views.ndim != 5:
        raise ValueError(
            "views must have shape [B, C, H, W] or [B, V, C, H, W]; "
            f"got ndim={views.ndim} shape={tuple(views.shape)}"
        )
    return views


def shared_per_view_xyz(
    backbone: nn.Module,
    views: torch.Tensor,
) -> torch.Tensor:
    """Apply one shared single-view model to each view → ``[B, V, 3]``.

    Shared-weight control paired with compact latent fusion (contrast
    angle-specific experts in :class:`ExpertXyzMeanLocalizer`).
    """
    views = ensure_multi_view_batch(views)
    batch, n_views, channels, height, width = views.shape
    flat = views.reshape(batch * n_views, channels, height, width)
    pred = backbone(flat)
    n_out = int(getattr(backbone, "n_outputs", pred.shape[-1]))
    if pred.shape[-1] != n_out:
        raise ValueError(
            f"backbone must output {n_out} coords; got {tuple(pred.shape)}"
        )
    return pred.reshape(batch, n_views, n_out)


def shared_xyz_mean(
    backbone: nn.Module,
    views: torch.Tensor,
) -> torch.Tensor:
    """Shared single-view model on each view, then mean of xyz coordinates → ``[B, 3]``.

    Under an affine final head this equals
    :class:`MeanLatentFusionLocalizer` applied to the same views.
    """
    return mean_coordinates(shared_per_view_xyz(backbone, views))


class ExpertXyzMeanLocalizer(nn.Module):
    """Average per-view xyz coordinates from angle-specific single-view experts.

    Experts are independent modules (typically separately trained
    ``LocalizerSingleViewFourier`` instances). Forward runs each view through
    its matching expert, then returns the uniform mean of per-view coordinates.

    No learned fusion module — evaluation-time coordinate averaging only.

    See also:
        :class:`CompactLatentFusionLocalizer` — learned ordered-concat fusion baseline.
        :class:`GeometryAwareFourierFusionLocalizer` — end-to-end geometry-aware fusion.

    Notebook / protocol: 09_0.
    """

    fusion_pattern: str = FUSION_PATTERN_09_0

    def __init__(
        self,
        experts: Mapping[float, nn.Module],
        *,
        angle_atol_deg: float = 1e-3,
    ):
        """Register one single-view expert per acquisition angle.

        Model structure: ``ModuleDict`` keyed by stable :func:`angle_key` strings;
        no shared weights and no fusion multilayer perceptron (MLP). At inference
        each view is routed to the expert whose training angle matches the slot
        angle, then :func:`mean_coordinates` fuses the per-view xyz coordinate
        predictions.

        Args:
            experts: Mapping ``angle_deg → nn.Module``; each module must output
                ``[B, 3]`` coordinates (typically
                ``LocalizerSingleViewFourier``). Angles must be unique within
                ``angle_atol_deg``.
            angle_atol_deg: Tolerance for angle lookup and de-duplication.

        Notebook / protocol: 09_0.
        """
        super().__init__()
        if not experts:
            raise ValueError("experts must be a non-empty mapping angle_deg → module")
        angles = _unique_sorted_angles(
            list(experts.keys()), atol_deg=float(angle_atol_deg)
        )
        self.expert_angles_deg = angles
        self.angle_atol_deg = float(angle_atol_deg)
        provided = {float(k): v for k, v in experts.items()}
        self.experts = nn.ModuleDict(
            {angle_key(theta): provided[theta] for theta in angles}
        )

    def expert_for_angle(self, angle_deg: float) -> nn.Module:
        """Look up the angle-specific expert module.

        Args:
            angle_deg: Acquisition angle in degrees.

        Returns:
            The single-view expert registered at the matching orbit angle.

        Raises:
            KeyError: No registered expert within ``self.angle_atol_deg``.
        """
        idx = match_expert_index(
            angle_deg,
            self.expert_angles_deg,
            atol_deg=self.angle_atol_deg,
        )
        return self.experts[angle_key(self.expert_angles_deg[idx])]

    def predict_per_view(
        self,
        views: torch.Tensor,
        angles_deg: torch.Tensor | Sequence[float] | None = None,
    ) -> torch.Tensor:
        """Map multi-view batch → per-view coordinates ``[B, V, 3]``.

        ``views``: ``[B, V, C, H, W]`` (or ``[B, C, H, W]`` with ``V=1``).
        ``angles_deg``: ``[V]``, ``[B, V]``, or ``None`` (defaults to the
        registered ``expert_angles_deg`` when ``V`` matches).
        """
        views = ensure_multi_view_batch(views)
        batch, n_views, channels, height, width = views.shape
        angles = resolve_view_angles(
            angles_deg,
            batch=batch,
            n_views=n_views,
            default_angles_deg=self.expert_angles_deg,
        )

        slot_to_expert: dict[int, list[int]] = {
            i: [] for i in range(len(self.expert_angles_deg))
        }
        # Use the first batch row's angles for slot→expert assignment; require
        # all batch items share the same orbit ordering (standard catalog).
        angles0 = angles[0].detach().cpu().tolist()
        for v, theta in enumerate(angles0):
            for b in range(batch):
                if abs(float(angles[b, v]) - float(theta)) > self.angle_atol_deg:
                    raise ValueError(
                        "All batch items must share the same per-slot angles; "
                        f"slot {v}: batch0={theta} vs batch{b}={float(angles[b, v])}"
                    )
            idx = match_expert_index(
                float(theta),
                self.expert_angles_deg,
                atol_deg=self.angle_atol_deg,
            )
            slot_to_expert[idx].append(v)

        out = views.new_zeros((batch, n_views, 3))
        for expert_idx, slots in slot_to_expert.items():
            if not slots:
                continue
            expert = self.experts[angle_key(self.expert_angles_deg[expert_idx])]
            gathered = views[:, slots]
            s = len(slots)
            pred = expert(
                gathered.reshape(batch * s, channels, height, width)
            )
            if pred.shape[-1] != 3:
                raise ValueError(
                    f"expert at angle {self.expert_angles_deg[expert_idx]} "
                    f"must output 3 coordinates; got {tuple(pred.shape)}"
                )
            pred = pred.reshape(batch, s, 3)
            for j, slot in enumerate(slots):
                out[:, slot] = pred[:, j]
        return out

    def forward(
        self,
        views: torch.Tensor,
        angles_deg: torch.Tensor | Sequence[float] | None = None,
    ) -> torch.Tensor:
        """Route each view to its angle expert, then average xyz coordinates.

        Pipeline: :meth:`predict_per_view` → :func:`mean_coordinates`. No
        gradient flows through a fusion module because none exists; this is
        pure evaluation-time averaging of independently trained experts.

        Args:
            views: ``[B, V, C, H, W]`` or ``[B, C, H, W]`` (``V=1``).
            angles_deg: Optional per-slot angles; defaults to
                ``self.expert_angles_deg`` when ``V`` matches.

        Returns:
            Fused particle centre, shape ``[B, 3]``.

        Notebook / protocol: 09_0.
        """
        return mean_coordinates(self.predict_per_view(views, angles_deg))

    def learned_parameter_count(self) -> int:
        """Count trainable parameters across all angle-specific experts.

        Equals the sum of full expert sizes when experts are loaded pre-trained
        and left trainable; zero when all experts are frozen.
        """
        return int(
            sum(p.numel() for p in self.parameters() if p.requires_grad)
        )

    def describe(self) -> dict[str, Any]:
        """Return compact metadata for experiment logs and run history.

        Includes fusion pattern, registered orbit, expert count, and whether
        any trainable parameters remain (usually all experts frozen at eval).

        Notebook / protocol: 09_0.
        """
        return {
            "variant_id": "m09_0_expert_xyz_mean",
            "fusion_pattern": self.fusion_pattern,
            "n_experts": len(self.expert_angles_deg),
            "expert_angles_deg": list(self.expert_angles_deg),
            "learned_parameter_count": self.learned_parameter_count(),
            "learned_fusion_module": False,
        }


def freeze_backbone_parameters(backbone: nn.Module) -> None:
    """Set ``requires_grad=False`` on every parameter of a single-view trunk.

    Used by frozen-encoder paths so only the fusion head (compact multilayer
    perceptron (MLP), DeepSets ``ρ∘φ``, or geometry fusion MLP) trains while
    locked latents ``h_i = backbone.encode_latent(view_i)`` stay fixed.
    """
    for p in backbone.parameters():
        p.requires_grad = False


def encode_view_latents(
    backbone: nn.Module,
    views: torch.Tensor,
) -> torch.Tensor:
    """Encode multi-view batch → locked latents ``[B, V, hidden]``.

    Requires ``backbone.encode_latent`` on a Fourier or pooled single-view
    trunk.
    """
    encode = getattr(backbone, "encode_latent", None)
    if not callable(encode):
        raise TypeError(
            "backbone must implement encode_latent(x) → [B, hidden]; "
            f"got {type(backbone)!r}"
        )
    views = ensure_multi_view_batch(views)
    batch, n_views, channels, height, width = views.shape
    flat = views.reshape(batch * n_views, channels, height, width)
    h = encode(flat)
    return h.reshape(batch, n_views, -1)


def pack_geometry_tokens(
    latents: torch.Tensor,
    angles_deg: torch.Tensor,
) -> torch.Tensor:
    """Build per-view geometry tokens ``[B, V, hidden+2]`` = ``[h_i, sin θ_i, cos θ_i]``.

    ``latents``: ``[B, V, hidden]``. ``angles_deg``: ``[B, V]`` degrees.
    Sin/cos supplements Fourier trunk features for orbit-aware fusion.
    """
    if latents.ndim != 3:
        raise ValueError(
            "latents must have shape [B, V, hidden]; "
            f"got {tuple(latents.shape)}"
        )
    if angles_deg.ndim != 2:
        raise ValueError(
            "angles_deg must have shape [B, V]; "
            f"got {tuple(angles_deg.shape)}"
        )
    if angles_deg.shape[:2] != latents.shape[:2]:
        raise ValueError(
            "angles_deg batch/view dims must match latents; "
            f"got angles={tuple(angles_deg.shape)} vs "
            f"latents={tuple(latents.shape)}"
        )
    ang = make_angle_features(angles_deg.to(device=latents.device))
    return torch.cat((latents, ang), dim=-1)


def resolve_light_angles(
    light_angles_deg: torch.Tensor | Sequence[float] | float | None,
    *,
    batch: int,
    n_views: int,
) -> torch.Tensor:
    """Broadcast illumination azimuth angles to ``[B, V]`` degrees.

    Illumination-aware models treat the view axis as lights at a fixed camera
    or as the light index within a camera×light grid. Accepts scalar,
    per-batch, per-view, or full ``[B, V]`` / ``[B, 1]`` schedules.

    Args:
        light_angles_deg: Absolute light azimuth in degrees (several shapes;
            see implementation for accepted layouts).
        batch: Batch size ``B``.
        n_views: View or light count ``V``.

    Returns:
        Float tensor of shape ``[B, V]``.
    """
    if light_angles_deg is None:
        raise ValueError("light_angles_deg is required for illumination-aware fusion")
    if isinstance(light_angles_deg, (int, float)):
        base = torch.full((batch, n_views), float(light_angles_deg), dtype=torch.float32)
        return base
    if isinstance(light_angles_deg, torch.Tensor):
        ang = light_angles_deg.detach().float()
    else:
        ang = torch.tensor(list(light_angles_deg), dtype=torch.float32)
    if ang.ndim == 0:
        return ang.reshape(1, 1).expand(batch, n_views)
    if ang.ndim == 1:
        if ang.numel() == batch:
            return ang.unsqueeze(1).expand(batch, n_views)
        if ang.numel() == n_views:
            return ang.unsqueeze(0).expand(batch, -1)
        raise ValueError(
            "1-D light_angles_deg must have length B or V; "
            f"got {ang.numel()} for B={batch}, V={n_views}"
        )
    if ang.ndim == 2:
        if ang.shape == (batch, n_views):
            return ang
        if ang.shape == (batch, 1):
            return ang.expand(batch, n_views)
        raise ValueError(
            f"light_angles_deg shape {tuple(ang.shape)} must be "
            f"[B, V]=[{batch}, {n_views}] or [B, 1]"
        )
    raise ValueError(
        f"light_angles_deg must be scalar / [B] / [V] / [B,V]; got {tuple(ang.shape)}"
    )


def pack_illumination_geometry_tokens(
    latents: torch.Tensor,
    camera_angles_deg: torch.Tensor,
    light_angles_deg: torch.Tensor,
) -> torch.Tensor:
    """Build flat-joint tokens ``[B, V, hidden+4]`` with camera and light angles.

    ``[h_i, sin θ_cam, cos θ_cam, sin θ_light, cos θ_light]``.
    Absolute angles only — not relative camera–light angle.

    Notebook / protocol: 10_baseline flat camera×illumination fusion.
    """
    cam_tokens = pack_geometry_tokens(latents, camera_angles_deg)
    light = resolve_light_angles(
        light_angles_deg,
        batch=latents.shape[0],
        n_views=latents.shape[1],
    ).to(device=latents.device)
    light_feat = make_angle_features(light)
    return torch.cat((cam_tokens, light_feat), dim=-1)


def light_angle_deg_from_optical_setup_id(optical_setup_id: str) -> float:
    """Parse illumination setup id ``opt_m10_illum_XXX`` to azimuth in degrees."""
    text = str(optical_setup_id).strip()
    prefix = M10_OPTICAL_SETUP_PREFIX
    if not text.startswith(prefix):
        raise ValueError(
            f"optical_setup_id {optical_setup_id!r} does not start with {prefix!r}"
        )
    suffix = text[len(prefix) :]
    try:
        return float(suffix)
    except ValueError as exc:
        raise ValueError(
            f"cannot parse light angle from optical_setup_id={optical_setup_id!r}"
        ) from exc


def light_xy_from_angle_deg(
    light_angle_deg: float,
    *,
    radius_xy: float = M10_LIGHT_RADIUS_XY,
    z: float = M10_LIGHT_Z,
) -> tuple[float, float, float]:
    """Map absolute light azimuth to Cartesian point-light position ``(x, y, z)``."""
    rad = math.radians(float(light_angle_deg))
    return (
        float(radius_xy) * math.cos(rad),
        float(radius_xy) * math.sin(rad),
        float(z),
    )


class MeanLatentFusionLocalizer(nn.Module):
    """Mean locked latents ``h_i``, then apply affine coordinate head.

    ```text
    view_i → encode_latent → h_i ∈ R^{hidden}
    h_bar = mean_i(h_i)
    xyz = predict_from_latent(h_bar)   # final Linear only
    ```

    Output ``xyz`` is the particle coordinate triple (x, y, z). Mathematically
    equivalent to :func:`shared_xyz_mean` under an affine final head. **Not**
    the learned compact fusion baseline — see :class:`CompactLatentFusionLocalizer`.

    Notebook / protocol: demoted sanity control (not 09_1).
    """

    fusion_pattern: str = FUSION_PATTERN_MEAN_LATENT_SANITY

    def __init__(
        self,
        backbone: LocalizerSingleViewFourier | None = None,
        *,
        freeze_encoder: bool = True,
        n_outputs: int = 3,
        hidden: int = 128,
        in_channels: int = 1,
    ):
        """Build mean-latent sanity path with optional frozen encoder.

        Model structure::

            view_i → encode_latent → h_i ∈ R^{hidden}
            h̄ = mean_i(h_i)
            xyz = predict_from_latent(h̄)   # final Linear only

        When ``freeze_encoder=True`` (default), convolutional neural network
        (CNN) / Fourier / first MLP blocks are frozen and only the final affine
        head trains. Under an affine head this is equivalent to
        :func:`shared_xyz_mean` / coordinate averaging — use
        :class:`CompactLatentFusionLocalizer` for the learned compact fusion
        baseline.

        Args:
            backbone: Fourier single-view trunk; constructed via
                :func:`new_frozen_single_view_expert` when ``None``.
            freeze_encoder: If True, freeze encoder through first MLP projection.
            n_outputs: Coordinate dimension (default 3).
            hidden: Trunk latent / MLP hidden width.
            in_channels: Input image channels.

        Notebook / protocol: demoted sanity control (not 09_1).
        """
        super().__init__()
        self.backbone = backbone or new_frozen_single_view_expert(
            n_outputs=n_outputs,
            hidden=hidden,
            in_channels=in_channels,
        )
        if not isinstance(self.backbone, LocalizerSingleViewFourier):
            raise TypeError(
                "backbone must be LocalizerSingleViewFourier; "
                f"got {type(self.backbone)!r}"
            )
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            # Freeze CNN / Fourier / first MLP; leave final Linear trainable.
            for module in (self.backbone.encoder, self.backbone.pool):
                for p in module.parameters():
                    p.requires_grad = False
            for p in self.backbone.head[0].parameters():
                p.requires_grad = False

    def encode_view_latents(self, views: torch.Tensor) -> torch.Tensor:
        """Encode multi-view batch → locked latents ``[B, V, hidden]``."""
        return encode_view_latents(self.backbone, views)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """Mean-pool locked latents, then apply the backbone coordinate head.

        Under an affine final Linear this is equivalent to averaging per-view
        expert xyz coordinates — a demoted control after compact fusion showed
        no gain over averaging.

        Args:
            views: ``[B, V, C, H, W]`` (or ``[V, C, H, W]`` / ``[B, C, H, W]``).

        Returns:
            ``[B, n_outputs]`` predicted particle coordinates.
        """
        h = self.encode_view_latents(views)
        h_bar = h.mean(dim=1)
        return self.backbone.predict_from_latent(h_bar)

    def learned_parameter_count(self) -> int:
        """Trainable parameter count (final Linear only when encoder frozen)."""
        return int(
            sum(p.numel() for p in self.parameters() if p.requires_grad)
        )

    def describe(self) -> dict[str, Any]:
        """Return demoted-sanity metadata for experiment logs.

        Documents the affine-equivalence note, latent cut, and that no separate
        fusion multilayer perceptron (MLP) exists — only the shared trunk's
        final Linear may train.

        Notebook / protocol: demoted sanity control (not 09_1).
        """
        return {
            "variant_id": "mean_latent_linear_sanity",
            "fusion_pattern": self.fusion_pattern,
            "latent_cut": "after_first_mlp_proj_relu",
            "latent_dim": int(self.backbone.hidden),
            "freeze_encoder": self.freeze_encoder,
            "learned_parameter_count": self.learned_parameter_count(),
            "learned_fusion_module": False,
            "note": (
                "demoted; mean(h)→Linear ≡ shared mean(xyz) under affine head"
            ),
        }


class CompactLatentFusionLocalizer(nn.Module):
    """Compact fusion multilayer perceptron (MLP) over frozen (or end-to-end (e2e)) single-view latents.

    Also used for illumination-only fusion without angle tokens via
    :meth:`for_10_1_c` with ``freeze_encoder=False``.

    Packing is the only structural fork (same MLP depth/width otherwise):

    ```text
    view_i → backbone.encode_latent → h_i
    ordered_concat:  concat(h_1…h_V) → Linear → ReLU → Linear → xyz
    mean_pool:       mean(h_i)       → Linear → ReLU → Linear → xyz
    ```

    Accepts any backbone exposing ``encode_latent`` and ``.hidden``
    (Fourier primary or pooled global average pooling (GAP) negative control).

    Default: encoder frozen; only the fusion MLP trains.
    End-to-end variant: encoder trainable; views may be illuminations at fixed
    camera.

    See also:
        :class:`ExpertXyzMeanLocalizer` — coordinate mean without learned fusion.
        :class:`GeometryAwareFourierFusionLocalizer` — sin/cos tokens or FiLM geometry.
        :class:`FrozenEncoderDeepSetsLocalizer` — permutation-invariant ``ρ(mean φ(h))`` alternative.

    Notebook / protocol: 09_1 compact latent fusion; also 10_1-C illumination-only.
    """

    fusion_pattern: str = FUSION_PATTERN_09_1

    def __init__(
        self,
        backbone: nn.Module,
        *,
        n_views: int,
        fusion_hidden: int = 128,
        n_outputs: int = 3,
        freeze_encoder: bool = True,
        packing: str = PACKING_ORDERED_CONCAT,
        fusion_pattern: str | None = None,
        backbone_kind: str | None = None,
    ):
        """Wire trunk latents into a compact two-layer fusion multilayer perceptron (MLP).

        Model structure::

            view_i → backbone.encode_latent → h_i ∈ R^{latent_dim}
            ordered_concat:  concat(h_1…h_V) → Linear(fusion_in, fusion_hidden)
                             → ReLU → Linear → xyz coordinates
            mean_pool:       mean(h_i)       → same MLP (fusion_in = latent_dim)

        ``fusion_in = n_views * latent_dim`` for ordered concat, else
        ``latent_dim``. When ``freeze_encoder=True``, :func:`freeze_backbone_parameters`
        locks the trunk; only ``self.fusion`` trains. When
        ``freeze_encoder=False``, encoder and fusion train jointly.

        Args:
            backbone: Single-view module with ``encode_latent`` and ``.hidden``
                (Fourier or pooled GAP negative control).
            n_views: Expected view count ``V`` (≥ 2).
            fusion_hidden: Hidden width of the two-layer fusion MLP (default
                128, matching compact fusion capacity).
            n_outputs: Output coordinate dimension (default 3).
            freeze_encoder: If True, freeze backbone (fusion-only training).
            packing: ``ordered_concat`` (default) or ``mean_pool``.
            fusion_pattern: Override autodetected pattern string for logging.
            backbone_kind: ``fourier`` vs ``pooled`` / ``no_fourier`` for
                metadata; inferred from backbone type when ``None``.
        """
        super().__init__()
        if not callable(getattr(backbone, "encode_latent", None)):
            raise TypeError(
                "backbone must implement encode_latent(x) → [B, hidden]; "
                f"got {type(backbone)!r}"
            )
        latent_dim = int(getattr(backbone, "hidden", 0) or 0)
        if latent_dim < 1:
            raise ValueError(
                "backbone must expose .hidden latent width; "
                f"got {type(backbone)!r}"
            )
        if int(n_views) < 2:
            raise ValueError(f"n_views must be >= 2; got {n_views}")
        if int(fusion_hidden) < 1:
            raise ValueError(f"fusion_hidden must be >= 1; got {fusion_hidden}")
        if int(n_outputs) < 1:
            raise ValueError(f"n_outputs must be >= 1; got {n_outputs}")
        packing_key = str(packing).strip().lower()
        if packing_key not in (PACKING_ORDERED_CONCAT, PACKING_MEAN_POOL):
            raise ValueError(
                "packing must be "
                f"{PACKING_ORDERED_CONCAT!r} or {PACKING_MEAN_POOL!r}; "
                f"got {packing!r}"
            )

        if backbone_kind is not None:
            kind = str(backbone_kind).strip().lower()
        elif isinstance(backbone, LocalizerSingleViewFourier):
            kind = "fourier"
        elif isinstance(backbone, LocalizeSingleView):
            kind = "pooled"
        else:
            kind = "fourier"
        if kind not in ("fourier", "pooled", "no_fourier"):
            raise ValueError(
                "backbone_kind must be 'fourier' or 'pooled'/'no_fourier'; "
                f"got {backbone_kind!r}"
            )
        if kind == "no_fourier":
            kind = "pooled"

        self.backbone = backbone
        self.n_views = int(n_views)
        self.fusion_hidden = int(fusion_hidden)
        self.n_outputs = int(n_outputs)
        self.freeze_encoder = bool(freeze_encoder)
        self.packing = packing_key
        self.backbone_kind = kind
        if fusion_pattern is not None:
            self.fusion_pattern = str(fusion_pattern)
        elif packing_key == PACKING_MEAN_POOL and freeze_encoder:
            self.fusion_pattern = (
                FUSION_PATTERN_09_1_MEAN_POOL
                if kind == "fourier"
                else FUSION_PATTERN_09_1_MEAN_POOL_POOLED
            )
        elif packing_key == PACKING_ORDERED_CONCAT and freeze_encoder:
            self.fusion_pattern = (
                FUSION_PATTERN_09_1
                if kind == "fourier"
                else FUSION_PATTERN_09_1_POOLED
            )
        else:
            # e2e / custom callers set fusion_pattern explicitly (e.g. 10_1-C).
            self.fusion_pattern = FUSION_PATTERN_09_1
        if packing_key == PACKING_MEAN_POOL:
            fusion_in = latent_dim
        else:
            fusion_in = self.n_views * latent_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, self.fusion_hidden),
            nn.ReLU(),
            nn.Linear(self.fusion_hidden, self.n_outputs),
        )
        if self.freeze_encoder:
            freeze_backbone_parameters(self.backbone)

    @classmethod
    def for_10_1_c(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_views: int,
        **kwargs: Any,
    ) -> CompactLatentFusionLocalizer:
        """End-to-end (e2e) illumination fusion without light-angle tokens.

        Fixed camera; view axis = light orbit. Same compact multilayer perceptron (MLP) as frozen
        latent fusion but ``freeze_encoder=False`` so the Fourier trunk
        co-trains with fusion. No ``sin/cos`` geometry — contrast with
        :meth:`for_10_1_d` (via :class:`GeometryAwareFourierFusionLocalizer`).

        Args:
            backbone: Optional pre-built Fourier trunk.
            n_views: Number of illumination views ``V``.
            **kwargs: Forwarded to ``__init__`` (``fusion_hidden``, etc.).

        Notebook / protocol: 10_1-C e2e illumination-only.
        """
        kwargs.pop("freeze_encoder", None)
        kwargs.pop("fusion_pattern", None)
        kwargs.setdefault("packing", PACKING_ORDERED_CONCAT)
        n_outputs = int(kwargs.pop("n_outputs", 3))
        fusion_hidden = int(kwargs.pop("fusion_hidden", M10_1_FUSION_HIDDEN))
        trunk = backbone or new_frozen_single_view_expert(
            n_outputs=n_outputs,
            hidden=int(kwargs.pop("hidden", 128)),
        )
        return cls(
            trunk,
            n_views=n_views,
            fusion_hidden=fusion_hidden,
            n_outputs=n_outputs,
            freeze_encoder=False,
            fusion_pattern=FUSION_PATTERN_10_1_C,
            backbone_kind=kwargs.pop("backbone_kind", "fourier"),
            **kwargs,
        )

    @classmethod
    def for_10_1_c_pooled(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_views: int,
        **kwargs: Any,
    ) -> CompactLatentFusionLocalizer:
        """End-to-end (e2e) pooled trunk + illumination fusion without angle tokens.

        Global average pooling (GAP) negative-control trunk with
        illumination-only views at fixed camera. Encoder and fusion train
        jointly.

        Notebook / protocol: 10_1-C pooled e2e illumination-only.
        """
        kwargs.pop("freeze_encoder", None)
        kwargs.pop("fusion_pattern", None)
        kwargs.pop("backbone_kind", None)
        kwargs.setdefault("packing", PACKING_ORDERED_CONCAT)
        n_outputs = int(kwargs.pop("n_outputs", 3))
        fusion_hidden = int(kwargs.pop("fusion_hidden", M10_1_FUSION_HIDDEN))
        embed_dim = int(kwargs.pop("embed_dim", kwargs.pop("hidden", 128)))
        trunk = backbone or new_frozen_pooled_single_view_expert(
            n_outputs=n_outputs,
            embed_dim=embed_dim,
        )
        return cls(
            trunk,
            n_views=n_views,
            fusion_hidden=fusion_hidden,
            n_outputs=n_outputs,
            freeze_encoder=False,
            fusion_pattern=FUSION_PATTERN_10_1_C_POOLED,
            backbone_kind="pooled",
            **kwargs,
        )

    @classmethod
    def for_10_1_c_frozen(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_views: int,
        **kwargs: Any,
    ) -> CompactLatentFusionLocalizer:
        """Frozen Fourier trunk + compact fusion over illumination views.

        Trunk locked via :func:`freeze_backbone_parameters`; only the
        ordered-concat fusion multilayer perceptron (MLP) trains over illumination views at fixed
        camera. No light-angle tokens.

        Notebook / protocol: 10_1A-C frozen illumination-only.
        """
        kwargs.pop("freeze_encoder", None)
        kwargs.pop("fusion_pattern", None)
        kwargs.setdefault("packing", PACKING_ORDERED_CONCAT)
        n_outputs = int(kwargs.pop("n_outputs", 3))
        fusion_hidden = int(kwargs.pop("fusion_hidden", M10_1_FUSION_HIDDEN))
        trunk = backbone or new_frozen_single_view_expert(
            n_outputs=n_outputs,
            hidden=int(kwargs.pop("hidden", 128)),
        )
        return cls(
            trunk,
            n_views=n_views,
            fusion_hidden=fusion_hidden,
            n_outputs=n_outputs,
            freeze_encoder=True,
            fusion_pattern=FUSION_PATTERN_10_1_C_FROZEN,
            backbone_kind=kwargs.pop("backbone_kind", "fourier"),
            **kwargs,
        )

    @classmethod
    def for_10_1_c_frozen_pooled(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_views: int,
        **kwargs: Any,
    ) -> CompactLatentFusionLocalizer:
        """Frozen pooled trunk + compact fusion over illumination views.

        Frozen global average pooling (GAP) negative-control trunk; fusion-only
        training over illumination views at fixed camera. No light-angle tokens.

        Notebook / protocol: 10_1A-C frozen pooled illumination-only.
        """
        kwargs.pop("freeze_encoder", None)
        kwargs.pop("fusion_pattern", None)
        kwargs.pop("backbone_kind", None)
        kwargs.setdefault("packing", PACKING_ORDERED_CONCAT)
        n_outputs = int(kwargs.pop("n_outputs", 3))
        fusion_hidden = int(kwargs.pop("fusion_hidden", M10_1_FUSION_HIDDEN))
        embed_dim = int(kwargs.pop("embed_dim", kwargs.pop("hidden", 128)))
        trunk = backbone or new_frozen_pooled_single_view_expert(
            n_outputs=n_outputs,
            embed_dim=embed_dim,
        )
        return cls(
            trunk,
            n_views=n_views,
            fusion_hidden=fusion_hidden,
            n_outputs=n_outputs,
            freeze_encoder=True,
            fusion_pattern=FUSION_PATTERN_10_1_C_FROZEN_POOLED,
            backbone_kind="pooled",
            **kwargs,
        )

    def encode_view_latents(self, views: torch.Tensor) -> torch.Tensor:
        """Encode multi-view batch → latents ``[B, V, hidden]``."""
        return encode_view_latents(self.backbone, views)

    def pack_latents(self, h: torch.Tensor) -> torch.Tensor:
        """Pack ``[B, V, hidden]`` latents into fusion multilayer perceptron (MLP) input ``[B, in]``."""
        if h.ndim != 3:
            raise ValueError(
                f"latents must have shape [B, V, hidden]; got {tuple(h.shape)}"
            )
        if h.shape[1] != self.n_views:
            raise ValueError(
                f"expected V={self.n_views} views; got V={h.shape[1]}"
            )
        if self.packing == PACKING_MEAN_POOL:
            return h.mean(dim=1)
        return h.reshape(h.shape[0], self.n_views * h.shape[-1])

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """Encode views, pack latents, and fuse through compact multilayer perceptron (MLP) → xyz coordinates.

        Packing mode (ordered concat or mean pool) is fixed at construction.
        Supports camera-orbit multi-view stacks and illumination-only stacks
        without angle tokens.

        Args:
            views: ``[B, V, C, H, W]`` (camera orbit or light stack).

        Returns:
            ``[B, n_outputs]`` predicted particle coordinates.
        """
        h = self.encode_view_latents(views)
        return self.fusion(self.pack_latents(h))

    def learned_parameter_count(self) -> int:
        """Trainable parameter count (fusion only when encoder frozen)."""
        return int(
            sum(p.numel() for p in self.parameters() if p.requires_grad)
        )

    def describe(self) -> dict[str, Any]:
        """Return variant metadata for experiment logs.

        Dispatches on ``fusion_pattern``, ``packing``, ``backbone_kind``, and
        ``freeze_encoder`` to produce ``variant_id``, latent cut, fusion
        capacity, and a human-readable configuration note.

        Notebook / protocol: 09_1 compact fusion; 10_1-C illumination-only.
        """
        pooled = getattr(self, "backbone_kind", "fourier") == "pooled"
        if self.fusion_pattern in (
            FUSION_PATTERN_10_1_C,
            FUSION_PATTERN_10_1_C_POOLED,
            FUSION_PATTERN_10_1_C_FROZEN,
            FUSION_PATTERN_10_1_C_FROZEN_POOLED,
        ):
            frozen = self.freeze_encoder or self.fusion_pattern in (
                FUSION_PATTERN_10_1_C_FROZEN,
                FUSION_PATTERN_10_1_C_FROZEN_POOLED,
            )
            if pooled or self.fusion_pattern in (
                FUSION_PATTERN_10_1_C_POOLED,
                FUSION_PATTERN_10_1_C_FROZEN_POOLED,
            ):
                if frozen:
                    variant_id = "m10_1a_c_frozen_pooled_illumination_fusion"
                    note = (
                        "M10 10_1A-C pooled; frozen GAP trunk + compact fusion "
                        "over illuminations at fixed camera; no light-angle tokens"
                    )
                    latent_cut = "gap_embed_relu"
                else:
                    variant_id = "m10_1c_e2e_pooled_illumination_fusion"
                    note = (
                        "M10 10_1B-C pooled; e2e GAP trunk + compact fusion over "
                        "illuminations at fixed camera; no light-angle tokens"
                    )
                    latent_cut = "gap_embed_relu"
            else:
                if frozen:
                    variant_id = "m10_1a_c_frozen_fourier_illumination_fusion"
                    note = (
                        "M10 10_1A-C; frozen Fourier trunk + compact fusion over "
                        "illuminations at fixed camera; no light-angle tokens"
                    )
                    latent_cut = "after_first_mlp_proj_relu"
                else:
                    variant_id = "m10_1c_e2e_fourier_illumination_fusion"
                    note = (
                        "M10 10_1B-C; e2e Fourier trunk + compact fusion over "
                        "illuminations at fixed camera; no light-angle tokens"
                    )
                    latent_cut = "after_first_mlp_proj_relu"
        elif self.packing == PACKING_MEAN_POOL:
            if pooled:
                variant_id = (
                    "m09_1_compact_fusion_mlp_mean_pool_frozen_pooled"
                )
                note = (
                    "09_1B: frozen pooled (GAP) trunk; mean-pool latents "
                    "then compact MLP"
                )
                latent_cut = "gap_embed_relu"
            else:
                variant_id = "m09_1_compact_fusion_mlp_mean_pool_frozen_fourier"
                note = (
                    "09_1A: frozen Fourier trunk; mean-pool latents then "
                    "compact MLP (intermediate vs ordered_concat)"
                )
                latent_cut = "after_first_mlp_proj_relu"
        else:
            if pooled:
                variant_id = "m09_1_compact_fusion_mlp_frozen_pooled"
                note = (
                    "09_1B: frozen pooled (GAP) trunk; ordered-concat "
                    "compact fusion"
                )
                latent_cut = "gap_embed_relu"
            else:
                variant_id = "m09_1_compact_fusion_mlp_frozen_fourier"
                note = (
                    "09_1A: frozen Fourier trunk; ordered-concat compact "
                    "fusion only"
                )
                latent_cut = "after_first_mlp_proj_relu"
        return {
            "variant_id": variant_id,
            "fusion_pattern": self.fusion_pattern,
            "latent_cut": latent_cut,
            "latent_dim": int(self.backbone.hidden),
            "n_views": self.n_views,
            "fusion_hidden": self.fusion_hidden,
            "packing": self.packing,
            "backbone_kind": getattr(self, "backbone_kind", "fourier"),
            "freeze_encoder": self.freeze_encoder,
            "end_to_end": not self.freeze_encoder,
            "learned_parameter_count": self.learned_parameter_count(),
            "learned_fusion_module": True,
            "note": note,
        }


class DeepSetsFusionHead(nn.Module):
    """Permutation-invariant fusion: ``ρ(mean_i φ(h_i))`` over view latents.

    Model structure::

        φ: Linear(latent_dim → phi_hidden) → ReLU [× phi_depth blocks]
        pool: mean over view axis
        ρ: Linear(phi_hidden → rho_hidden) → ReLU → Linear → n_outputs

    Input ``H``: ``[B, V, latent_dim]``. Output: ``[B, n_outputs]``.
    Default ``phi_hidden=rho_hidden=128`` matches compact fusion multilayer
    perceptron (MLP) capacity.

    See also:
        :class:`CompactLatentFusionLocalizer` — ordered-concat fusion baseline (position-sensitive).

    Notebook / protocol: 09_1 DeepSets alternative to ordered concat.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        phi_hidden: int = M9_1_DEEPSETS_PHI_HIDDEN,
        rho_hidden: int = M9_1_DEEPSETS_RHO_HIDDEN,
        n_outputs: int = 3,
        phi_depth: int = 1,
    ):
        """Build φ (per-view) and ρ (post-pool) multilayer perceptrons (MLPs) for DeepSets fusion.

        Args:
            latent_dim: Locked trunk width ``D`` (``backbone.hidden``).
            phi_hidden: Hidden width inside φ (default 128).
            rho_hidden: Hidden width inside ρ (default 128).
            n_outputs: Coordinate dimension (default 3).
            phi_depth: Number of ``Linear→ReLU`` blocks in φ (≥ 1).
        """
        super().__init__()
        if int(latent_dim) < 1:
            raise ValueError(f"latent_dim must be >= 1; got {latent_dim}")
        if int(phi_hidden) < 1:
            raise ValueError(f"phi_hidden must be >= 1; got {phi_hidden}")
        if int(rho_hidden) < 1:
            raise ValueError(f"rho_hidden must be >= 1; got {rho_hidden}")
        if int(n_outputs) < 1:
            raise ValueError(f"n_outputs must be >= 1; got {n_outputs}")
        if int(phi_depth) < 1:
            raise ValueError(f"phi_depth must be >= 1; got {phi_depth}")

        self.latent_dim = int(latent_dim)
        self.phi_hidden = int(phi_hidden)
        self.rho_hidden = int(rho_hidden)
        self.n_outputs = int(n_outputs)
        self.phi_depth = int(phi_depth)

        phi_layers: list[nn.Module] = [
            nn.Linear(self.latent_dim, self.phi_hidden),
            nn.ReLU(),
        ]
        for _ in range(self.phi_depth - 1):
            phi_layers.extend(
                [nn.Linear(self.phi_hidden, self.phi_hidden), nn.ReLU()]
            )
        self.phi = nn.Sequential(*phi_layers)
        self.rho = nn.Sequential(
            nn.Linear(self.phi_hidden, self.rho_hidden),
            nn.ReLU(),
            nn.Linear(self.rho_hidden, self.n_outputs),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Permutation-invariant DeepSets: ``ρ(mean_i φ(h_i))``.

        Args:
            h: Locked per-view latents ``[B, V, latent_dim]``.

        Returns:
            ``[B, n_outputs]`` predicted particle coordinates.
        """
        if h.ndim != 3:
            raise ValueError(
                f"H must have shape [B, V, latent_dim]; got {tuple(h.shape)}"
            )
        if h.shape[-1] != self.latent_dim:
            raise ValueError(
                f"expected latent_dim={self.latent_dim}; got {h.shape[-1]}"
            )
        batch, n_views, dim = h.shape
        phi_h = self.phi(h.reshape(batch * n_views, dim))
        phi_h = phi_h.reshape(batch, n_views, self.phi_hidden)
        z = phi_h.mean(dim=1)
        return self.rho(z)


class FrozenEncoderDeepSetsLocalizer(nn.Module):
    """Frozen single-view encoder + DeepSets head ``ρ(mean_i φ(h_i))``.

    Fourier variant uses the Fourier trunk; pooled variant uses the global
    average pooling (GAP) negative-control trunk. Only the DeepSets head trains.

    See also:
        :class:`CompactLatentFusionLocalizer` — ordered-concat fusion at the same latent cut.

    Notebook / protocol: 09_1 DeepSets.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        n_views: int,
        phi_hidden: int = M9_1_DEEPSETS_PHI_HIDDEN,
        rho_hidden: int = M9_1_DEEPSETS_RHO_HIDDEN,
        n_outputs: int = 3,
        phi_depth: int = 1,
        freeze_encoder: bool = True,
        fusion_pattern: str | None = None,
        backbone_kind: str = "fourier",
    ):
        """Frozen trunk + trainable :class:`DeepSetsFusionHead`.

        Model structure::

            view_i → backbone.encode_latent → h_i
            xyz = DeepSetsFusionHead([h_1…h_V])

        When ``freeze_encoder=True`` (default), only φ and ρ train; latents
        are locked at the trunk ``encode_latent`` cut (Fourier or pooled).

        Args:
            backbone: Single-view trunk with ``encode_latent`` and ``.hidden``.
            n_views: Expected view count ``V``.
            phi_hidden, rho_hidden, phi_depth: DeepSets head hyperparameters.
            n_outputs: Output coordinate dimension.
            freeze_encoder: If True, freeze backbone.
            fusion_pattern: Override pattern string for logging.
            backbone_kind: ``fourier`` or ``pooled`` / ``no_fourier`` for
                metadata.
        """
        super().__init__()
        if int(n_views) < 2:
            raise ValueError(f"n_views must be >= 2; got {n_views}")
        if not callable(getattr(backbone, "encode_latent", None)):
            raise TypeError(
                "backbone must implement encode_latent; "
                f"got {type(backbone)!r}"
            )
        kind = str(backbone_kind).strip().lower()
        if kind not in ("fourier", "pooled", "no_fourier"):
            raise ValueError(
                "backbone_kind must be 'fourier' or 'pooled'/'no_fourier'; "
                f"got {backbone_kind!r}"
            )
        if kind == "no_fourier":
            kind = "pooled"

        self.backbone = backbone
        self.n_views = int(n_views)
        self.freeze_encoder = bool(freeze_encoder)
        self.backbone_kind = kind
        latent_dim = int(getattr(backbone, "hidden", 0) or 0)
        if latent_dim < 1:
            raise ValueError(
                "backbone must expose .hidden latent width; "
                f"got {type(backbone)!r}"
            )
        self.latent_dim = latent_dim
        if fusion_pattern is not None:
            self.fusion_pattern = str(fusion_pattern)
        elif kind == "fourier":
            self.fusion_pattern = FUSION_PATTERN_09_1_DEEPSETS_FOURIER
        else:
            self.fusion_pattern = FUSION_PATTERN_09_1_DEEPSETS_NO_FOURIER

        self.head = DeepSetsFusionHead(
            self.latent_dim,
            phi_hidden=phi_hidden,
            rho_hidden=rho_hidden,
            n_outputs=n_outputs,
            phi_depth=phi_depth,
        )
        if self.freeze_encoder:
            freeze_backbone_parameters(self.backbone)

    @classmethod
    def for_09_1_fourier(
        cls,
        backbone: LocalizerSingleViewFourier | None = None,
        *,
        n_views: int,
        **kwargs: Any,
    ) -> FrozenEncoderDeepSetsLocalizer:
        """DeepSets fusion with frozen Fourier trunk.

        Same frozen trunk as compact latent fusion.

        Notebook / protocol: 09_1 DeepSets Fourier.
        """
        kwargs.pop("backbone_kind", None)
        kwargs.pop("fusion_pattern", None)
        trunk = backbone or new_frozen_single_view_expert(
            n_outputs=int(kwargs.get("n_outputs", 3)),
            hidden=int(kwargs.pop("hidden", 128)),
        )
        return cls(
            trunk,
            n_views=n_views,
            backbone_kind="fourier",
            fusion_pattern=FUSION_PATTERN_09_1_DEEPSETS_FOURIER,
            **kwargs,
        )

    @classmethod
    def for_09_1_no_fourier(
        cls,
        backbone: LocalizeSingleView | None = None,
        *,
        n_views: int,
        **kwargs: Any,
    ) -> FrozenEncoderDeepSetsLocalizer:
        """DeepSets fusion with frozen global average pooling (GAP) trunk.

        Pooled negative-control backbone — same ``encode_latent`` cut as
        :class:`CompactLatentFusionLocalizer` with ``backbone_kind='pooled'``.
        Only the DeepSets φ/ρ head trains; permutation-invariant alternative
        to ordered-concat compact fusion.

        Notebook / protocol: 09_1B DeepSets pooled.
        """
        kwargs.pop("backbone_kind", None)
        kwargs.pop("fusion_pattern", None)
        kwargs.pop("hidden", None)
        trunk = backbone or new_frozen_pooled_single_view_expert(
            n_outputs=int(kwargs.get("n_outputs", 3)),
            embed_dim=int(kwargs.pop("embed_dim", 128)),
        )
        return cls(
            trunk,
            n_views=n_views,
            backbone_kind="pooled",
            fusion_pattern=FUSION_PATTERN_09_1_DEEPSETS_NO_FOURIER,
            **kwargs,
        )

    def encode_view_latents(self, views: torch.Tensor) -> torch.Tensor:
        """Encode multi-view batch → latents ``[B, V, hidden]``."""
        return encode_view_latents(self.backbone, views)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """Encode each view with the frozen trunk, then DeepSets-fuse to xyz coordinates.

        Args:
            views: ``[B, V, C, H, W]`` with ``V == n_views``.

        Returns:
            ``[B, n_outputs]`` predicted particle coordinates.
        """
        h = self.encode_view_latents(views)
        if h.shape[1] != self.n_views:
            raise ValueError(
                f"expected V={self.n_views} views; got V={h.shape[1]}"
            )
        return self.head(h)

    def learned_parameter_count(self) -> int:
        """Trainable parameter count (DeepSets head only when encoder frozen)."""
        return int(
            sum(p.numel() for p in self.parameters() if p.requires_grad)
        )

    def describe(self) -> dict[str, Any]:
        """Return DeepSets variant metadata for experiment logs.

        Reports φ/ρ capacities, latent cut, packing mode ``deepsets_mean_phi``,
        and trainable parameter count (head only when encoder frozen).

        Notebook / protocol: 09_1 DeepSets.
        """
        if self.backbone_kind == "fourier":
            variant_id = "m09_1_deepsets_fourier"
            label = "DeepSets Fourier"
            note = (
                "09_1 DeepSets; frozen M8 Fourier trunk; "
                "rho(mean_i phi(h_i))"
            )
        else:
            variant_id = "m09_1_deepsets_no_fourier"
            label = "DeepSets no-Fourier"
            note = (
                "09_1 DeepSets; frozen M8 pooled (non-Fourier) trunk; "
                "rho(mean_i phi(h_i))"
            )
        return {
            "variant_id": variant_id,
            "display_label": label,
            "fusion_pattern": self.fusion_pattern,
            "latent_cut": "after_first_mlp_proj_relu"
            if self.backbone_kind == "fourier"
            else "gap_embed_relu",
            "latent_dim": self.latent_dim,
            "n_views": self.n_views,
            "phi_hidden": self.head.phi_hidden,
            "rho_hidden": self.head.rho_hidden,
            "phi_depth": self.head.phi_depth,
            "packing": "deepsets_mean_phi",
            "backbone_kind": self.backbone_kind,
            "freeze_encoder": self.freeze_encoder,
            "end_to_end": not self.freeze_encoder,
            "learned_parameter_count": self.learned_parameter_count(),
            "learned_fusion_module": True,
            "note": note,
        }


def build_geometry_fusion_mlp(
    in_features: int,
    *,
    fusion_hidden: int,
    n_outputs: int,
    fusion_depth: int,
) -> nn.Sequential:
    """Build ordered-concat fusion multilayer perceptron (MLP) for geometry-aware localizers.

    ``fusion_depth`` is the number of ``Linear→ReLU`` hidden blocks before the
    final ``Linear → n_outputs`` layer. Default compact capacity uses hidden
    width 128 × depth 1; large variant uses 512 × depth 2.

    Notebook / protocol: 09_2 / 09_3 geometry-aware fusion family.
    """
    if int(in_features) < 1:
        raise ValueError(f"in_features must be >= 1; got {in_features}")
    if int(fusion_hidden) < 1:
        raise ValueError(f"fusion_hidden must be >= 1; got {fusion_hidden}")
    if int(n_outputs) < 1:
        raise ValueError(f"n_outputs must be >= 1; got {n_outputs}")
    if int(fusion_depth) < 1:
        raise ValueError(f"fusion_depth must be >= 1; got {fusion_depth}")

    layers: list[nn.Module] = []
    dim = int(in_features)
    for _ in range(int(fusion_depth)):
        layers.append(nn.Linear(dim, int(fusion_hidden)))
        layers.append(nn.ReLU())
        dim = int(fusion_hidden)
    layers.append(nn.Linear(dim, int(n_outputs)))
    return nn.Sequential(*layers)


class AngleConditionFiLM(nn.Module):
    """Apply Feature-wise Linear Modulation (FiLM) to latents using sin/cos acquisition angles.

    ```text
    (sin θ, cos θ) → Linear → (γ, β)
    h' = γ ⊙ h + β
    ```

    Initialised to identity (γ=1, β=0) so early training matches angle-unaware
    latents. Keeps fusion input at latent width instead of appending angle
    tokens (contrast ``geometry_mode='concat'``).

    See also:
        :class:`GeometryAwareFourierFusionLocalizer` — parent fusion model selecting ``geometry_mode='film'``.

    Notebook / protocol: optional ``geometry_mode='film'`` on geometry-aware fusion.
    """

    def __init__(self, feature_dim: int, *, cond_dim: int = 2):
        """Map sin/cos angles to per-feature scale and shift (γ, β).

        Model structure::

            (sin θ, cos θ) → Linear(cond_dim → 2·feature_dim) → (γ, β)
            h' = γ ⊙ h + β

        Weights and bias are zero-initialised except γ-bias set to 1 and
        β-bias to 0, giving identity Feature-wise Linear Modulation (FiLM) at
        step zero. Modulates latents *before* ordered-concat fusion, avoiding
        token widening (contrast ``geometry_mode='concat'``).

        Args:
            feature_dim: Latent width ``D`` (``backbone.hidden``).
            cond_dim: Conditioning size (default 2 for sin/cos).
        """
        super().__init__()
        if int(feature_dim) < 1:
            raise ValueError(f"feature_dim must be >= 1; got {feature_dim}")
        if int(cond_dim) < 1:
            raise ValueError(f"cond_dim must be >= 1; got {cond_dim}")
        self.feature_dim = int(feature_dim)
        self.cond_dim = int(cond_dim)
        self.to_gb = nn.Linear(self.cond_dim, 2 * self.feature_dim)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)
        # Identity FiLM: γ=1, β=0.
        self.to_gb.bias.data[: self.feature_dim].fill_(1.0)

    def forward(
        self,
        latents: torch.Tensor,
        angles_deg: torch.Tensor,
    ) -> torch.Tensor:
        """Apply identity-init Feature-wise Linear Modulation (FiLM) using acquisition angles.

        Converts ``angles_deg`` to ``(sin θ, cos θ)``, predicts ``(γ, β)`` per
        view, and returns ``γ ⊙ h + β``. At initialisation this is the identity
        on ``h``. Output tokens feed ordered-concat fusion at width
        ``feature_dim`` (not ``feature_dim + 2``).

        Args:
            latents: Per-view latents ``[B, V, D]``.
            angles_deg: Matching angles in degrees, ``[B, V]``.

        Returns:
            Modulated latents ``[B, V, D]``.
        """
        if latents.ndim != 3:
            raise ValueError(
                "latents must have shape [B, V, hidden]; "
                f"got {tuple(latents.shape)}"
            )
        if angles_deg.ndim != 2:
            raise ValueError(
                "angles_deg must have shape [B, V]; "
                f"got {tuple(angles_deg.shape)}"
            )
        if angles_deg.shape[:2] != latents.shape[:2]:
            raise ValueError(
                "angles_deg batch/view dims must match latents; "
                f"got angles={tuple(angles_deg.shape)} vs "
                f"latents={tuple(latents.shape)}"
            )
        if latents.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected latent dim {self.feature_dim}; "
                f"got {latents.shape[-1]}"
            )
        cond = make_angle_features(angles_deg.to(device=latents.device))
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"expected cond_dim={self.cond_dim}; got {cond.shape[-1]}"
            )
        gamma, beta = self.to_gb(cond).chunk(2, dim=-1)
        return gamma * latents + beta


class GeometryAwareFourierFusionLocalizer(nn.Module):
    """End-to-end (e2e) trunk + sin/cos geometry + ordered-concat fusion multilayer perceptron (MLP).

    Geometry can be attached either by concatenating ``sin θ / cos θ`` tokens or
    by Feature-wise Linear Modulation (FiLM) of each latent before fusion.

    ```text
    view_i → backbone.encode_latent → h_i
    concat mode:  token_i = [h_i, sin θ_i, cos θ_i]  (+ light sin/cos for flat joint)
    film mode:    h'_i = FiLM(sin θ_i, cos θ_i)(h_i); token_i = h'_i
    concat_ordered(token_1…token_V) → fusion MLP → one xyz coordinate triple
    ```

    Primary path uses a Fourier trunk; pooled analogues use the global average
    pooling (GAP) ``encode_latent`` cut. Compact or large fusion MLP capacity.
    Encoder and fusion train jointly by default; frozen-trunk factories lock the
    encoder.

    See also:
        :class:`CompactLatentFusionLocalizer` — latent-only fusion without angle tokens.
        :class:`HierarchicalLightThenCameraFusionLocalizer` — two-stage light-then-camera fusion.
        :class:`AngleConditionFiLM` — used when ``geometry_mode='film'``.

    Notebook / protocol: 09_2/09_3 camera orbit; 10_baseline flat joint;
    10_1-D illumination-only with light angles.
    """

    fusion_pattern: str = FUSION_PATTERN_09_2

    def __init__(
        self,
        backbone: nn.Module | None = None,
        *,
        n_views: int,
        view_angles_deg: Sequence[float],
        fusion_hidden: int = M9_2_FUSION_HIDDEN,
        fusion_depth: int = M9_2_FUSION_DEPTH,
        fusion_pattern: str | None = None,
        include_illumination: bool = False,
        n_outputs: int = 3,
        hidden: int = 128,
        in_channels: int = 1,
        backbone_kind: str | None = None,
        freeze_encoder: bool = False,
        geometry_mode: str = GEOMETRY_MODE_CONCAT,
    ):
        """Wire end-to-end (e2e) trunk, geometry tokens, and ordered-concat fusion multilayer perceptron (MLP).

        Model structure::

            view_i → encode_latent → h_i
            concat:  token_i = [h_i, sin θ, cos θ] (+ light sin/cos if baseline)
            film:    h'_i = Feature-wise Linear Modulation (FiLM)(sin θ, cos θ)(h_i); token_i = h'_i
            flat = concat(token_1…token_V) → fusion MLP → xyz coordinates

        Covers camera-orbit geometry fusion, flat joint camera×illumination
        tokens, and illumination-only fusion with light sin/cos at fixed
        camera. ``fusion_hidden`` / ``fusion_depth`` select compact (default
        128×1) vs large (512×2) capacity. ``freeze_encoder=True`` locks the
        trunk for fusion-only training.

        Args:
            backbone: Fourier or pooled trunk with ``encode_latent``; built
                from defaults when ``None``.
            n_views: View count ``V``.
            view_angles_deg: Registered per-slot angles (camera or light orbit).
            fusion_hidden, fusion_depth: Fusion MLP capacity.
            fusion_pattern: Override autodetected pattern for logging.
            include_illumination: If True, append light sin/cos (flat joint);
                requires ``geometry_mode='concat'``.
            n_outputs, hidden, in_channels: Trunk / output hyperparameters.
            backbone_kind: ``fourier`` vs ``pooled`` metadata and defaults.
            freeze_encoder: If True, freeze trunk (fusion-only training).
            geometry_mode: ``concat`` (append sin/cos tokens) or ``film``
                (FiLM-modulate ``h`` then fuse at latent width).

        Notebook / protocol: 09_2/09_3; 10_baseline; 10_1-D.
        """
        super().__init__()
        angles = tuple(float(a) for a in view_angles_deg)
        if int(n_views) < 2:
            raise ValueError(f"n_views must be >= 2; got {n_views}")
        if len(angles) != int(n_views):
            raise ValueError(
                f"len(view_angles_deg)={len(angles)} must equal "
                f"n_views={int(n_views)}"
            )
        if int(fusion_hidden) < 1:
            raise ValueError(f"fusion_hidden must be >= 1; got {fusion_hidden}")
        if int(fusion_depth) < 1:
            raise ValueError(f"fusion_depth must be >= 1; got {fusion_depth}")
        if int(n_outputs) < 1:
            raise ValueError(f"n_outputs must be >= 1; got {n_outputs}")
        mode = str(geometry_mode).strip().lower()
        if mode not in (GEOMETRY_MODE_CONCAT, GEOMETRY_MODE_FILM):
            raise ValueError(
                "geometry_mode must be "
                f"{GEOMETRY_MODE_CONCAT!r} or {GEOMETRY_MODE_FILM!r}; "
                f"got {geometry_mode!r}"
            )
        if mode == GEOMETRY_MODE_FILM and bool(include_illumination):
            raise ValueError(
                "geometry_mode='film' does not support include_illumination=True "
                "(camera+light FiLM is not wired); use concat tokens for "
                "10_baseline or film with a single angle schedule (10_1-D / 09_2)"
            )

        self.backbone = backbone or new_frozen_single_view_expert(
            n_outputs=n_outputs,
            hidden=hidden,
            in_channels=in_channels,
        )
        if not callable(getattr(self.backbone, "encode_latent", None)):
            raise TypeError(
                "backbone must implement encode_latent(x) → [B, hidden]; "
                f"got {type(self.backbone)!r}"
            )
        latent_dim = int(getattr(self.backbone, "hidden", 0) or 0)
        if latent_dim < 1:
            raise ValueError(
                "backbone must expose .hidden latent width; "
                f"got {type(self.backbone)!r}"
            )

        if backbone_kind is not None:
            kind = str(backbone_kind).strip().lower()
        elif isinstance(self.backbone, LocalizerSingleViewFourier):
            kind = "fourier"
        elif isinstance(self.backbone, LocalizeSingleView):
            kind = "pooled"
        else:
            kind = "fourier"
        if kind not in ("fourier", "pooled", "no_fourier"):
            raise ValueError(
                "backbone_kind must be 'fourier' or 'pooled'/'no_fourier'; "
                f"got {backbone_kind!r}"
            )
        if kind == "no_fourier":
            kind = "pooled"
        self.backbone_kind = kind

        self.n_views = int(n_views)
        self.view_angles_deg = angles
        self.fusion_hidden = int(fusion_hidden)
        self.fusion_depth = int(fusion_depth)
        self.n_outputs = int(n_outputs)
        self.include_illumination = bool(include_illumination)
        self.geometry_mode = mode
        if fusion_pattern is not None:
            pattern = str(fusion_pattern)
        elif self.include_illumination:
            pattern = (
                FUSION_PATTERN_10_BASELINE_POOLED
                if kind == "pooled"
                else FUSION_PATTERN_10_BASELINE
            )
        elif (
            self.fusion_hidden > M9_2_FUSION_HIDDEN
            or self.fusion_depth > M9_2_FUSION_DEPTH
        ):
            pattern = (
                FUSION_PATTERN_09_3_POOLED
                if kind == "pooled"
                else FUSION_PATTERN_09_3
            )
        else:
            pattern = (
                FUSION_PATTERN_09_2_POOLED
                if kind == "pooled"
                else FUSION_PATTERN_09_2
            )
        self.fusion_pattern = pattern
        self.freeze_encoder = bool(freeze_encoder)
        if self.geometry_mode == GEOMETRY_MODE_FILM:
            token_dim = latent_dim
            self.angle_film: AngleConditionFiLM | None = AngleConditionFiLM(
                latent_dim, cond_dim=2
            )
        else:
            geom_dim = 4 if self.include_illumination else 2
            token_dim = latent_dim + geom_dim
            self.angle_film = None
        self.fusion = build_geometry_fusion_mlp(
            self.n_views * token_dim,
            fusion_hidden=self.fusion_hidden,
            n_outputs=self.n_outputs,
            fusion_depth=self.fusion_depth,
        )
        if self.freeze_encoder:
            freeze_backbone_parameters(self.backbone)

    @classmethod
    def for_09_2(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_views: int,
        view_angles_deg: Sequence[float],
        **kwargs: Any,
    ) -> GeometryAwareFourierFusionLocalizer:
        """Compact end-to-end (e2e) fusion with camera sin/cos geometry tokens.

        Jointly trains Fourier trunk + fusion multilayer perceptron (MLP). Token width ``hidden + 2``;
        default fusion 128 hidden × depth 1.

        Notebook / protocol: 09_2 camera-orbit e2e.
        """
        kwargs.pop("include_illumination", None)
        return cls(
            backbone,
            n_views=n_views,
            view_angles_deg=view_angles_deg,
            fusion_hidden=int(kwargs.pop("fusion_hidden", M9_2_FUSION_HIDDEN)),
            fusion_depth=int(kwargs.pop("fusion_depth", M9_2_FUSION_DEPTH)),
            fusion_pattern=FUSION_PATTERN_09_2,
            include_illumination=False,
            backbone_kind=kwargs.pop("backbone_kind", "fourier"),
            **kwargs,
        )

    @classmethod
    def for_09_3(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_views: int,
        view_angles_deg: Sequence[float],
        **kwargs: Any,
    ) -> GeometryAwareFourierFusionLocalizer:
        """Large fusion multilayer perceptron (MLP) capacity with same camera sin/cos geometry.

        Same end-to-end (e2e) + camera sin/cos wiring as :meth:`for_09_2` but fusion
        MLP uses 512 hidden × depth 2 for capacity-axis comparison.

        Notebook / protocol: 09_3 camera-orbit e2e upper bound.
        """
        kwargs.pop("include_illumination", None)
        return cls(
            backbone,
            n_views=n_views,
            view_angles_deg=view_angles_deg,
            fusion_hidden=int(kwargs.pop("fusion_hidden", M9_3_FUSION_HIDDEN)),
            fusion_depth=int(kwargs.pop("fusion_depth", M9_3_FUSION_DEPTH)),
            fusion_pattern=FUSION_PATTERN_09_3,
            include_illumination=False,
            backbone_kind=kwargs.pop("backbone_kind", "fourier"),
            **kwargs,
        )

    @classmethod
    def for_09_2_pooled(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_views: int,
        view_angles_deg: Sequence[float],
        **kwargs: Any,
    ) -> GeometryAwareFourierFusionLocalizer:
        """Compact end-to-end (e2e) + geometry with pooled global average pooling (GAP) trunk.

        Pooled negative-control backbone; same camera sin/cos tokens and
        compact fusion as :meth:`for_09_2`.

        Notebook / protocol: 09_2B pooled camera-orbit e2e.
        """
        kwargs.pop("include_illumination", None)
        kwargs.pop("backbone_kind", None)
        n_outputs = int(kwargs.pop("n_outputs", 3))
        embed_dim = int(kwargs.pop("embed_dim", kwargs.pop("hidden", 128)))
        trunk = backbone or new_frozen_pooled_single_view_expert(
            n_outputs=n_outputs,
            embed_dim=embed_dim,
        )
        return cls(
            trunk,
            n_views=n_views,
            view_angles_deg=view_angles_deg,
            fusion_hidden=int(kwargs.pop("fusion_hidden", M9_2_FUSION_HIDDEN)),
            fusion_depth=int(kwargs.pop("fusion_depth", M9_2_FUSION_DEPTH)),
            fusion_pattern=FUSION_PATTERN_09_2_POOLED,
            include_illumination=False,
            backbone_kind="pooled",
            n_outputs=n_outputs,
            **kwargs,
        )

    @classmethod
    def for_09_3_pooled(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_views: int,
        view_angles_deg: Sequence[float],
        **kwargs: Any,
    ) -> GeometryAwareFourierFusionLocalizer:
        """Large fusion multilayer perceptron (MLP) + pooled global average pooling (GAP) trunk.

        Pooled analogue of :meth:`for_09_3` for Fourier-vs-GAP comparison.

        Notebook / protocol: 09_3 pooled camera-orbit e2e.
        """
        kwargs.pop("include_illumination", None)
        kwargs.pop("backbone_kind", None)
        n_outputs = int(kwargs.pop("n_outputs", 3))
        embed_dim = int(kwargs.pop("embed_dim", kwargs.pop("hidden", 128)))
        trunk = backbone or new_frozen_pooled_single_view_expert(
            n_outputs=n_outputs,
            embed_dim=embed_dim,
        )
        return cls(
            trunk,
            n_views=n_views,
            view_angles_deg=view_angles_deg,
            fusion_hidden=int(kwargs.pop("fusion_hidden", M9_3_FUSION_HIDDEN)),
            fusion_depth=int(kwargs.pop("fusion_depth", M9_3_FUSION_DEPTH)),
            fusion_pattern=FUSION_PATTERN_09_3_POOLED,
            include_illumination=False,
            backbone_kind="pooled",
            n_outputs=n_outputs,
            **kwargs,
        )

    @classmethod
    def for_10_baseline(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_views: int,
        view_angles_deg: Sequence[float],
        **kwargs: Any,
    ) -> GeometryAwareFourierFusionLocalizer:
        """Flat joint camera×light fusion with Fourier trunk.

        Weak flat baseline — concatenates absolute camera **and** light sin/cos
        into each token (``hidden + 4``). Compact fusion (128×1);
        ``include_illumination=True``.

        Notebook / protocol: 10_baselineA flat camera×illumination end-to-end (e2e).
        """
        kwargs.pop("include_illumination", None)
        return cls(
            backbone,
            n_views=n_views,
            view_angles_deg=view_angles_deg,
            fusion_hidden=int(kwargs.pop("fusion_hidden", M10_BASELINE_FUSION_HIDDEN)),
            fusion_depth=int(kwargs.pop("fusion_depth", M10_BASELINE_FUSION_DEPTH)),
            fusion_pattern=FUSION_PATTERN_10_BASELINE,
            include_illumination=True,
            backbone_kind=kwargs.pop("backbone_kind", "fourier"),
            **kwargs,
        )

    @classmethod
    def for_10_baseline_pooled(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_views: int,
        view_angles_deg: Sequence[float],
        **kwargs: Any,
    ) -> GeometryAwareFourierFusionLocalizer:
        """Flat joint camera×light fusion with pooled global average pooling (GAP) trunk.

        Pooled negative-control analogue of :meth:`for_10_baseline`; same
        absolute camera + light sin/cos flat pooling layout.

        Notebook / protocol: 10_baselineB flat pooled e2e.
        """
        kwargs.pop("include_illumination", None)
        kwargs.pop("backbone_kind", None)
        n_outputs = int(kwargs.pop("n_outputs", 3))
        embed_dim = int(kwargs.pop("embed_dim", kwargs.pop("hidden", 128)))
        trunk = backbone or new_frozen_pooled_single_view_expert(
            n_outputs=n_outputs,
            embed_dim=embed_dim,
        )
        return cls(
            trunk,
            n_views=n_views,
            view_angles_deg=view_angles_deg,
            fusion_hidden=int(kwargs.pop("fusion_hidden", M10_BASELINE_FUSION_HIDDEN)),
            fusion_depth=int(kwargs.pop("fusion_depth", M10_BASELINE_FUSION_DEPTH)),
            fusion_pattern=FUSION_PATTERN_10_BASELINE_POOLED,
            include_illumination=True,
            backbone_kind="pooled",
            n_outputs=n_outputs,
            **kwargs,
        )

    @classmethod
    def for_10_1_d(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_views: int,
        light_angles_deg: Sequence[float],
        **kwargs: Any,
    ) -> GeometryAwareFourierFusionLocalizer:
        """End-to-end (e2e) illumination fusion with light sin/cos at fixed camera.

        ``light_angles_deg`` is the ordered illumination schedule (length V).
        ``geometry_mode='concat'`` packs ``[h_i, sin θ_L, cos θ_L]``;
        ``geometry_mode='film'`` Feature-wise Linear Modulation (FiLM)-modulates
        ``h_i`` from sin/cos then fuses.

        Notebook / protocol: 10_1-D e2e illumination-only with light angles.
        """
        kwargs.pop("include_illumination", None)
        kwargs.pop("view_angles_deg", None)
        return cls(
            backbone,
            n_views=n_views,
            view_angles_deg=light_angles_deg,
            fusion_hidden=int(kwargs.pop("fusion_hidden", M10_1_FUSION_HIDDEN)),
            fusion_depth=int(kwargs.pop("fusion_depth", M10_1_FUSION_DEPTH)),
            fusion_pattern=FUSION_PATTERN_10_1_D,
            include_illumination=False,
            backbone_kind=kwargs.pop("backbone_kind", "fourier"),
            freeze_encoder=kwargs.pop("freeze_encoder", False),
            geometry_mode=str(
                kwargs.pop("geometry_mode", GEOMETRY_MODE_CONCAT)
            ),
            **kwargs,
        )

    @classmethod
    def for_10_1_d_pooled(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_views: int,
        light_angles_deg: Sequence[float],
        **kwargs: Any,
    ) -> GeometryAwareFourierFusionLocalizer:
        """Pooled trunk + light sin/cos at fixed camera (end-to-end (e2e)).

        Global average pooling (GAP) negative-control trunk with
        illumination-angle geometry.

        Notebook / protocol: 10_1-D pooled e2e illumination-only.
        """
        kwargs.pop("include_illumination", None)
        kwargs.pop("view_angles_deg", None)
        kwargs.pop("backbone_kind", None)
        n_outputs = int(kwargs.pop("n_outputs", 3))
        embed_dim = int(kwargs.pop("embed_dim", kwargs.pop("hidden", 128)))
        trunk = backbone or new_frozen_pooled_single_view_expert(
            n_outputs=n_outputs,
            embed_dim=embed_dim,
        )
        return cls(
            trunk,
            n_views=n_views,
            view_angles_deg=light_angles_deg,
            fusion_hidden=int(kwargs.pop("fusion_hidden", M10_1_FUSION_HIDDEN)),
            fusion_depth=int(kwargs.pop("fusion_depth", M10_1_FUSION_DEPTH)),
            fusion_pattern=FUSION_PATTERN_10_1_D_POOLED,
            include_illumination=False,
            backbone_kind="pooled",
            n_outputs=n_outputs,
            freeze_encoder=False,
            geometry_mode=str(
                kwargs.pop("geometry_mode", GEOMETRY_MODE_CONCAT)
            ),
            **kwargs,
        )

    @classmethod
    def for_10_1_d_frozen(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_views: int,
        light_angles_deg: Sequence[float],
        **kwargs: Any,
    ) -> GeometryAwareFourierFusionLocalizer:
        """Frozen Fourier trunk + light sin/cos at fixed camera.

        Fusion-only training over illumination views with angle geometry.

        Notebook / protocol: 10_1A-D frozen illumination-only.
        """
        kwargs.pop("include_illumination", None)
        kwargs.pop("view_angles_deg", None)
        kwargs.pop("freeze_encoder", None)
        return cls(
            backbone,
            n_views=n_views,
            view_angles_deg=light_angles_deg,
            fusion_hidden=int(kwargs.pop("fusion_hidden", M10_1_FUSION_HIDDEN)),
            fusion_depth=int(kwargs.pop("fusion_depth", M10_1_FUSION_DEPTH)),
            fusion_pattern=FUSION_PATTERN_10_1_D_FROZEN,
            include_illumination=False,
            backbone_kind=kwargs.pop("backbone_kind", "fourier"),
            freeze_encoder=True,
            geometry_mode=str(
                kwargs.pop("geometry_mode", GEOMETRY_MODE_CONCAT)
            ),
            **kwargs,
        )

    @classmethod
    def for_10_1_d_frozen_pooled(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_views: int,
        light_angles_deg: Sequence[float],
        **kwargs: Any,
    ) -> GeometryAwareFourierFusionLocalizer:
        """Frozen pooled trunk + light sin/cos at fixed camera.

        Fusion-only training with global average pooling (GAP) negative-control
        trunk.

        Notebook / protocol: 10_1A-D frozen pooled illumination-only.
        """
        kwargs.pop("include_illumination", None)
        kwargs.pop("view_angles_deg", None)
        kwargs.pop("backbone_kind", None)
        kwargs.pop("freeze_encoder", None)
        n_outputs = int(kwargs.pop("n_outputs", 3))
        embed_dim = int(kwargs.pop("embed_dim", kwargs.pop("hidden", 128)))
        trunk = backbone or new_frozen_pooled_single_view_expert(
            n_outputs=n_outputs,
            embed_dim=embed_dim,
        )
        return cls(
            trunk,
            n_views=n_views,
            view_angles_deg=light_angles_deg,
            fusion_hidden=int(kwargs.pop("fusion_hidden", M10_1_FUSION_HIDDEN)),
            fusion_depth=int(kwargs.pop("fusion_depth", M10_1_FUSION_DEPTH)),
            fusion_pattern=FUSION_PATTERN_10_1_D_FROZEN_POOLED,
            include_illumination=False,
            backbone_kind="pooled",
            n_outputs=n_outputs,
            freeze_encoder=True,
            geometry_mode=str(
                kwargs.pop("geometry_mode", GEOMETRY_MODE_CONCAT)
            ),
            **kwargs,
        )

    def encode_view_latents(self, views: torch.Tensor) -> torch.Tensor:
        """Encode multi-view batch → latents ``[B, V, hidden]``."""
        return encode_view_latents(self.backbone, views)

    def forward(
        self,
        views: torch.Tensor,
        angles_deg: torch.Tensor | Sequence[float] | None = None,
        light_angles_deg: torch.Tensor | Sequence[float] | float | None = None,
    ) -> torch.Tensor:
        """Encode views, attach geometry, and fuse through multilayer perceptron (MLP) → ``[B, n_outputs]``.

        When ``angles_deg`` is ``None``, uses the registered ``view_angles_deg``.
        When ``include_illumination`` is True, ``light_angles_deg`` is required
        (scalar / ``[B]`` / ``[B, V]`` absolute light azimuth in degrees).
        """
        h = self.encode_view_latents(views)
        if h.shape[1] != self.n_views:
            raise ValueError(
                f"expected V={self.n_views} views; got V={h.shape[1]}"
            )
        angles = resolve_view_angles(
            angles_deg,
            batch=h.shape[0],
            n_views=self.n_views,
            default_angles_deg=self.view_angles_deg,
        )
        if self.geometry_mode == GEOMETRY_MODE_FILM:
            if light_angles_deg is not None:
                raise ValueError(
                    "light_angles_deg was provided but geometry_mode='film' "
                    "uses the single view-angle schedule only"
                )
            assert self.angle_film is not None
            tokens = self.angle_film(h, angles)
        elif self.include_illumination:
            tokens = pack_illumination_geometry_tokens(
                h, angles, light_angles_deg
            )
        else:
            if light_angles_deg is not None:
                raise ValueError(
                    "light_angles_deg was provided but include_illumination=False"
                )
            tokens = pack_geometry_tokens(h, angles)
        flat = tokens.reshape(h.shape[0], self.n_views * tokens.shape[-1])
        return self.fusion(flat)

    def learned_parameter_count(self) -> int:
        """Count all trainable parameters (backbone + fusion [+ Feature-wise Linear Modulation (FiLM)])."""
        return int(
            sum(p.numel() for p in self.parameters() if p.requires_grad)
        )

    def fusion_parameter_count(self) -> int:
        """Count trainable fusion multilayer perceptron (MLP) parameters (+ FiLM when enabled).

        Useful when the backbone is frozen and only fusion (+ optional
        :class:`AngleConditionFiLM`) trains.
        """
        n = int(
            sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        )
        if self.angle_film is not None:
            n += int(
                sum(
                    p.numel()
                    for p in self.angle_film.parameters()
                    if p.requires_grad
                )
            )
        return n

    def describe(self) -> dict[str, Any]:
        """Return variant metadata for experiment logs.

        Derives ``variant_id``, geometry feature list, ``token_dim``,
        ``geometry_mode``, fusion capacity, and fusion-only parameter count
        from ``fusion_pattern``, ``include_illumination``, and freeze flags.

        Notebook / protocol: 09_2/09_3; 10_baseline; 10_1-D.
        """
        kind = getattr(self, "backbone_kind", "fourier")
        pooled = kind == "pooled"
        if self.fusion_pattern in (
            FUSION_PATTERN_10_BASELINE,
            FUSION_PATTERN_10_BASELINE_POOLED,
        ):
            geom = ("sin_camera", "cos_camera", "sin_light", "cos_light")
            token_dim = int(self.backbone.hidden) + 4
            if pooled or self.fusion_pattern == FUSION_PATTERN_10_BASELINE_POOLED:
                variant_id = (
                    "m10_baseline_e2e_pooled_illumination_geometry_fusion"
                )
                note = (
                    "M10 10_baselineB flat joint; pooled GAP trunk + absolute "
                    "camera + light sin/cos geometry tokens"
                )
                latent_cut = "gap_embed_relu"
            else:
                variant_id = (
                    "m10_baseline_e2e_fourier_illumination_geometry_fusion"
                )
                note = (
                    "M10 10_baselineA flat joint; Fourier trunk + absolute "
                    "camera + light sin/cos geometry tokens"
                )
                latent_cut = "after_first_mlp_proj_relu"
        elif self.fusion_pattern in (
            FUSION_PATTERN_10_1_D,
            FUSION_PATTERN_10_1_D_POOLED,
            FUSION_PATTERN_10_1_D_FROZEN,
            FUSION_PATTERN_10_1_D_FROZEN_POOLED,
        ):
            geom = ("sin_light", "cos_light")
            film = getattr(self, "geometry_mode", GEOMETRY_MODE_CONCAT) == (
                GEOMETRY_MODE_FILM
            )
            token_dim = (
                int(self.backbone.hidden)
                if film
                else int(self.backbone.hidden) + 2
            )
            frozen = getattr(self, "freeze_encoder", False) or self.fusion_pattern in (
                FUSION_PATTERN_10_1_D_FROZEN,
                FUSION_PATTERN_10_1_D_FROZEN_POOLED,
            )
            angle_note = (
                "light sin/cos FiLM on latents"
                if film
                else "light sin/cos concat tokens"
            )
            if pooled or self.fusion_pattern in (
                FUSION_PATTERN_10_1_D_POOLED,
                FUSION_PATTERN_10_1_D_FROZEN_POOLED,
            ):
                if frozen:
                    variant_id = "m10_1a_d_frozen_pooled_illumination_angle_fusion"
                    note = (
                        f"M10 10_1A-D pooled; frozen GAP trunk + {angle_note} "
                        "at fixed camera"
                    )
                    latent_cut = "gap_embed_relu"
                else:
                    variant_id = "m10_1d_e2e_pooled_illumination_angle_fusion"
                    note = (
                        f"M10 10_1B-D pooled; e2e GAP trunk + {angle_note} at "
                        "fixed camera (09_2 analogue; view dim = illumination)"
                    )
                    latent_cut = "gap_embed_relu"
            else:
                if frozen:
                    variant_id = "m10_1a_d_frozen_fourier_illumination_angle_fusion"
                    note = (
                        f"M10 10_1A-D; frozen Fourier trunk + {angle_note} "
                        "at fixed camera"
                    )
                    latent_cut = "after_first_mlp_proj_relu"
                else:
                    variant_id = "m10_1d_e2e_fourier_illumination_angle_fusion"
                    note = (
                        f"M10 10_1B-D; e2e Fourier + {angle_note} at fixed camera "
                        "(09_2 analogue; view dim = illumination)"
                    )
                    latent_cut = "after_first_mlp_proj_relu"
        elif self.fusion_pattern in (
            FUSION_PATTERN_09_3,
            FUSION_PATTERN_09_3_POOLED,
        ):
            if pooled or self.fusion_pattern == FUSION_PATTERN_09_3_POOLED:
                variant_id = "m09_3_e2e_pooled_geometry_large_fusion"
                note = (
                    "09_2B/09_3 pooled; e2e GAP trunk + camera sin/cos + large "
                    "fusion MLP"
                )
                latent_cut = "gap_embed_relu"
            else:
                variant_id = "m09_3_e2e_fourier_geometry_large_fusion"
                note = (
                    "M9/M10 09_3; same e2e + camera sin/cos geometry as 09_2 with "
                    "larger fusion MLP (capacity upper bound)"
                )
                latent_cut = "after_first_mlp_proj_relu"
            geom = ("sin_theta", "cos_theta")
            token_dim = int(self.backbone.hidden) + 2
        else:
            if pooled or self.fusion_pattern == FUSION_PATTERN_09_2_POOLED:
                variant_id = "m09_2_e2e_pooled_geometry_fusion"
                note = (
                    "09_2B; jointly trained pooled (GAP) trunk + camera sin/cos "
                    "+ compact fusion"
                )
                latent_cut = "gap_embed_relu"
            else:
                variant_id = "m09_2_e2e_fourier_geometry_fusion"
                note = (
                    "M9/M10 09_2; jointly trained Fourier trunk + camera sin/cos "
                    "geometry + compact fusion"
                )
                latent_cut = "after_first_mlp_proj_relu"
            geom = ("sin_theta", "cos_theta")
            token_dim = int(self.backbone.hidden) + 2
        frozen_enc = bool(getattr(self, "freeze_encoder", False))
        geom_mode = str(getattr(self, "geometry_mode", GEOMETRY_MODE_CONCAT))
        packing = (
            "ordered_concat_film_latents"
            if geom_mode == GEOMETRY_MODE_FILM
            else "ordered_concat_geometry_tokens"
        )
        return {
            "variant_id": variant_id,
            "fusion_pattern": self.fusion_pattern,
            "latent_cut": latent_cut,
            "latent_dim": int(self.backbone.hidden),
            "backbone_kind": kind,
            "geometry_features": geom,
            "geometry_mode": geom_mode,
            "token_dim": token_dim,
            "include_illumination": self.include_illumination,
            "n_views": self.n_views,
            "view_angles_deg": list(self.view_angles_deg),
            "fusion_hidden": self.fusion_hidden,
            "fusion_depth": self.fusion_depth,
            "fusion_parameter_count": self.fusion_parameter_count(),
            "packing": packing,
            "encoder_frozen": frozen_enc,
            "freeze_encoder": frozen_enc,
            "end_to_end": not frozen_enc,
            "learned_parameter_count": self.learned_parameter_count(),
            "learned_fusion_module": True,
            "note": note,
        }


def ensure_camera_light_views(
    views: torch.Tensor,
    *,
    n_cameras: int,
    n_lights: int,
    layout: str = "light_major",
) -> torch.Tensor:
    """Normalize observations to cam-major ``[B, V, I, C, H, W]`` for fusion.

    Canonical external sample layout is illumination-major
    ``[B, I, V, C, H, W]`` (``I`` = lights, ``V`` = cameras). Hierarchical
    light-then-camera fusion indexes the transposed cam-major grid
    ``[B, V, I, …]``.

    Accepts:

    - 6-D canonical ``[B, I, V, C, H, W]`` (permuted to cam-major)
    - 6-D already cam-major ``[B, V, I, C, H, W]`` when
      ``layout="camera_major_6d"`` (legacy / internal)
    - flat ``[B, I·V, C, H, W]`` with ``layout``:

      - ``light_major``: ``for light: for cam`` (canonical flatten)
      - ``camera_major``: ``for cam: for light`` (legacy flatten)
    """
    n_cameras = int(n_cameras)
    n_lights = int(n_lights)
    expected_v = n_cameras * n_lights
    layout_key = str(layout).strip().lower()
    if layout_key in {"illumination_major", "canonical"}:
        layout_key = "light_major"

    if views.ndim == 6:
        # Detect by shape: canonical [B, I, V, ...] vs cam-major [B, V, I, ...].
        if views.shape[1] == n_lights and views.shape[2] == n_cameras:
            return views.permute(0, 2, 1, 3, 4, 5).contiguous()
        if views.shape[1] == n_cameras and views.shape[2] == n_lights:
            return views
        raise ValueError(
            "6-D views must be canonical [B, I, V, C, H, W] "
            f"(I={n_lights}, V={n_cameras}) or cam-major [B, V, I, …]; "
            f"got {tuple(views.shape)}"
        )

    if views.ndim != 5:
        raise ValueError(
            "views must be [B, I, V, C, H, W] or "
            f"[B, I·V, C, H, W]; got {tuple(views.shape)}"
        )
    batch, n_views, channels, height, width = views.shape
    if n_views != expected_v:
        raise ValueError(
            f"flat views V={n_views} must equal I·V="
            f"{expected_v} (I={n_lights}, V={n_cameras})"
        )
    if layout_key == "light_major":
        # [B, I, V, ...] → [B, V, I, ...]
        reshaped = views.reshape(batch, n_lights, n_cameras, channels, height, width)
        return reshaped.permute(0, 2, 1, 3, 4, 5).contiguous()
    if layout_key == "camera_major":
        return views.reshape(batch, n_cameras, n_lights, channels, height, width)
    raise ValueError(
        "layout must be 'light_major' (canonical) or 'camera_major'; "
        f"got {layout!r}"
    )


class HierarchicalLightThenCameraFusionLocalizer(nn.Module):
    """Fuse lights within each camera, then fuse cameras across viewpoints.

    External sample layout is canonical illumination-major
    ``[B, I, V, C, H, W]``. Fusion permutes to cam-major internally:

    ```text
    image[light, cam] → (permute) → image[cam, light]
        → shared CNN → Fourier → MLP → h
    within each camera:
        token_light = [h, sin θ_L, cos θ_L]
        fuse over lights → z_cam
    across cameras:
        token_cam = [z_cam, sin θ_C, cos θ_C]
        fuse over cameras → xyz coordinates
    ```

    Compact fusion heads at both stages (default 128 hidden × depth 1).
    Full end-to-end (e2e) — shared Fourier trunk trains with both fusion stages.

    See also:
        :class:`GeometryAwareFourierFusionLocalizer` — flat joint camera×illumination alternative.

    Notebook / protocol: 10_2 hierarchical light-then-camera fusion.
    """

    fusion_pattern: str = FUSION_PATTERN_10_2

    def __init__(
        self,
        backbone: nn.Module | None = None,
        *,
        n_cameras: int,
        n_lights: int,
        camera_angles_deg: Sequence[float],
        light_angles_deg: Sequence[float],
        fusion_hidden: int = M10_2_FUSION_HIDDEN,
        fusion_depth: int = M10_2_FUSION_DEPTH,
        camera_latent_dim: int = M10_2_CAMERA_LATENT_DIM,
        flat_layout: str = "light_major",
        n_outputs: int = 3,
        hidden: int = 128,
        in_channels: int = 1,
        backbone_kind: str | None = None,
    ):
        """Two-stage hierarchical fusion over camera×light observations.

        Model structure::

            canonical views [B, I, V, C, H, W]
            → cam-major grid [B, V, I, C, H, W]
            → shared encode_latent → h  (latent_dim)
            Stage 1 (per camera):
                token_light = [h, sin θ_L, cos θ_L]
                light_fusion: concat over lights → z_cam ∈ R^{camera_latent_dim}
            Stage 2 (across cameras):
                token_cam = [z_cam, sin θ_C, cos θ_C]
                camera_fusion: concat over cameras → xyz

        Preferred path vs flat joint camera×illumination pooling. Both fusion
        stages use compact multilayer perceptrons (MLPs) (default 128 hidden ×
        depth 1). Full end-to-end (e2e) — shared trunk trains with both fusion
        heads. Fourier is the primary trunk; pooled GAP is the negative control.

        Args:
            backbone: Fourier or pooled trunk with ``encode_latent``; default from
                :func:`new_frozen_single_view_expert`.
            n_cameras, n_lights: Acquisition grid size ``V``, ``I``.
            camera_angles_deg, light_angles_deg: Registered orbit angles.
            fusion_hidden, fusion_depth: Capacity at **both** fusion stages.
            camera_latent_dim: Width of intermediate ``z_cam`` (default 128).
            flat_layout: How flat ``[B, I·V, …]`` inputs reshape —
                ``light_major`` (canonical) or ``camera_major`` (legacy).
            n_outputs, hidden, in_channels: Trunk hyperparameters.
            backbone_kind: ``fourier`` or ``pooled`` metadata (inferred from
                trunk type when omitted).

        Notebook / protocol: 10_2.
        """
        super().__init__()
        cam_angles = tuple(float(a) for a in camera_angles_deg)
        light_angles = tuple(float(a) for a in light_angles_deg)
        if int(n_cameras) < 1:
            raise ValueError(f"n_cameras must be >= 1; got {n_cameras}")
        if int(n_lights) < 2:
            raise ValueError(f"n_lights must be >= 2; got {n_lights}")
        if len(cam_angles) != int(n_cameras):
            raise ValueError(
                f"len(camera_angles_deg)={len(cam_angles)} must equal "
                f"n_cameras={int(n_cameras)}"
            )
        if len(light_angles) != int(n_lights):
            raise ValueError(
                f"len(light_angles_deg)={len(light_angles)} must equal "
                f"n_lights={int(n_lights)}"
            )
        if int(fusion_hidden) < 1:
            raise ValueError(f"fusion_hidden must be >= 1; got {fusion_hidden}")
        if int(fusion_depth) < 1:
            raise ValueError(f"fusion_depth must be >= 1; got {fusion_depth}")
        if int(camera_latent_dim) < 1:
            raise ValueError(
                f"camera_latent_dim must be >= 1; got {camera_latent_dim}"
            )
        layout_key = str(flat_layout).strip().lower()
        if layout_key in {"illumination_major", "canonical"}:
            layout_key = "light_major"
        if layout_key not in {"camera_major", "light_major"}:
            raise ValueError(
                "flat_layout must be 'light_major' (canonical) or 'camera_major'; "
                f"got {flat_layout!r}"
            )

        if backbone_kind is not None:
            kind = str(backbone_kind).strip().lower()
        elif backbone is None:
            kind = "fourier"
        elif isinstance(backbone, LocalizerSingleViewFourier):
            kind = "fourier"
        else:
            kind = "pooled"
        if kind not in ("fourier", "pooled", "no_fourier"):
            raise ValueError(
                "backbone_kind must be 'fourier' or 'pooled'/'no_fourier'; "
                f"got {backbone_kind!r}"
            )
        if kind == "no_fourier":
            kind = "pooled"

        if backbone is None:
            if kind == "pooled":
                backbone = new_frozen_pooled_single_view_expert(
                    n_outputs=n_outputs, embed_dim=hidden
                )
            else:
                backbone = new_frozen_single_view_expert(
                    n_outputs=n_outputs,
                    hidden=hidden,
                    in_channels=in_channels,
                )
        if not callable(getattr(backbone, "encode_latent", None)):
            raise TypeError(
                "backbone must implement encode_latent(x) → [B, hidden]; "
                f"got {type(backbone)!r}"
            )
        if not hasattr(backbone, "hidden"):
            raise TypeError(
                "backbone must expose .hidden latent width; "
                f"got {type(backbone)!r}"
            )
        self.backbone = backbone
        self.backbone_kind = kind
        self.n_cameras = int(n_cameras)
        self.n_lights = int(n_lights)
        self.camera_angles_deg = cam_angles
        self.light_angles_deg = light_angles
        self.fusion_hidden = int(fusion_hidden)
        self.fusion_depth = int(fusion_depth)
        self.camera_latent_dim = int(camera_latent_dim)
        self.flat_layout = layout_key
        self.n_outputs = int(n_outputs)
        self.fusion_pattern = (
            FUSION_PATTERN_10_2_POOLED if kind == "pooled" else FUSION_PATTERN_10_2
        )

        latent_dim = int(self.backbone.hidden)
        light_token_dim = latent_dim + 2
        cam_token_dim = self.camera_latent_dim + 2
        self.light_fusion = build_geometry_fusion_mlp(
            self.n_lights * light_token_dim,
            fusion_hidden=self.fusion_hidden,
            n_outputs=self.camera_latent_dim,
            fusion_depth=self.fusion_depth,
        )
        self.camera_fusion = build_geometry_fusion_mlp(
            self.n_cameras * cam_token_dim,
            fusion_hidden=self.fusion_hidden,
            n_outputs=self.n_outputs,
            fusion_depth=self.fusion_depth,
        )

    @classmethod
    def for_10_2(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_cameras: int,
        n_lights: int,
        camera_angles_deg: Sequence[float],
        light_angles_deg: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> HierarchicalLightThenCameraFusionLocalizer:
        """Hierarchical light-then-camera fusion factory (Fourier trunk).

        Defaults light orbit to :data:`M10_LIGHT_ANGLES_DEG` when
        ``light_angles_deg`` is omitted. End-to-end shared trunk + two compact
        fusion stages (see class docstring).

        Notebook / protocol: 10_2.
        """
        lights = (
            tuple(float(a) for a in light_angles_deg)
            if light_angles_deg is not None
            else tuple(M10_LIGHT_ANGLES_DEG)
        )
        kwargs.pop("backbone_kind", None)
        return cls(
            backbone,
            n_cameras=n_cameras,
            n_lights=n_lights,
            camera_angles_deg=camera_angles_deg,
            light_angles_deg=lights,
            fusion_hidden=int(kwargs.pop("fusion_hidden", M10_2_FUSION_HIDDEN)),
            fusion_depth=int(kwargs.pop("fusion_depth", M10_2_FUSION_DEPTH)),
            camera_latent_dim=int(
                kwargs.pop("camera_latent_dim", M10_2_CAMERA_LATENT_DIM)
            ),
            flat_layout=str(kwargs.pop("flat_layout", "light_major")),
            backbone_kind="fourier",
            **kwargs,
        )

    @classmethod
    def for_10_2_pooled(
        cls,
        backbone: nn.Module | None = None,
        *,
        n_cameras: int,
        n_lights: int,
        camera_angles_deg: Sequence[float],
        light_angles_deg: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> HierarchicalLightThenCameraFusionLocalizer:
        """Hierarchical light-then-camera fusion with pooled (GAP) trunk.

        Same two-stage light→camera wiring as :meth:`for_10_2`; negative-control
        backbone for Fourier vs pooled comparison.

        Notebook / protocol: 10_2 pooled control.
        """
        lights = (
            tuple(float(a) for a in light_angles_deg)
            if light_angles_deg is not None
            else tuple(M10_LIGHT_ANGLES_DEG)
        )
        kwargs.pop("backbone_kind", None)
        n_outputs = int(kwargs.pop("n_outputs", 3))
        embed_dim = int(kwargs.pop("embed_dim", kwargs.pop("hidden", 128)))
        trunk = backbone or new_frozen_pooled_single_view_expert(
            n_outputs=n_outputs, embed_dim=embed_dim
        )
        return cls(
            trunk,
            n_cameras=n_cameras,
            n_lights=n_lights,
            camera_angles_deg=camera_angles_deg,
            light_angles_deg=lights,
            fusion_hidden=int(kwargs.pop("fusion_hidden", M10_2_FUSION_HIDDEN)),
            fusion_depth=int(kwargs.pop("fusion_depth", M10_2_FUSION_DEPTH)),
            camera_latent_dim=int(
                kwargs.pop("camera_latent_dim", M10_2_CAMERA_LATENT_DIM)
            ),
            flat_layout=str(kwargs.pop("flat_layout", "light_major")),
            n_outputs=n_outputs,
            hidden=embed_dim,
            backbone_kind="pooled",
            **kwargs,
        )

    def encode_view_latents(self, views: torch.Tensor) -> torch.Tensor:
        """Encode flat multi-view batch → latents ``[B, V, hidden]``."""
        return encode_view_latents(self.backbone, views)

    def forward(
        self,
        views: torch.Tensor,
        camera_angles_deg: torch.Tensor | Sequence[float] | None = None,
        light_angles_deg: torch.Tensor | Sequence[float] | None = None,
        *,
        layout: str | None = None,
    ) -> torch.Tensor:
        """Fuse illuminations per camera, then fuse cameras → ``[B, n_outputs]``.

        ``views``: canonical ``[B, I, V, C, H, W]`` or flat ``[B, I·V, C, H, W]``
        with ``layout`` / ``self.flat_layout`` (``light_major`` default).
        """
        layout_key = str(layout).strip().lower() if layout is not None else self.flat_layout
        grid = ensure_camera_light_views(
            views,
            n_cameras=self.n_cameras,
            n_lights=self.n_lights,
            layout=layout_key,
        )
        batch, n_cam, n_light, channels, height, width = grid.shape
        flat = grid.reshape(batch, n_cam * n_light, channels, height, width)
        h = self.encode_view_latents(flat)
        h_grid = h.reshape(batch, n_cam, n_light, -1)

        cam_angles = resolve_view_angles(
            camera_angles_deg,
            batch=batch,
            n_views=self.n_cameras,
            default_angles_deg=self.camera_angles_deg,
        ).to(device=h.device)
        light_angles = resolve_view_angles(
            light_angles_deg,
            batch=batch,
            n_views=self.n_lights,
            default_angles_deg=self.light_angles_deg,
        ).to(device=h.device)

        light_feat = make_angle_features(light_angles)  # [B, N_light, 2]
        light_feat = light_feat.unsqueeze(1).expand(batch, n_cam, n_light, 2)
        light_tokens = torch.cat((h_grid, light_feat), dim=-1)
        light_flat = light_tokens.reshape(batch * n_cam, n_light * light_tokens.shape[-1])
        z_cam = self.light_fusion(light_flat).reshape(batch, n_cam, self.camera_latent_dim)

        cam_feat = make_angle_features(cam_angles)  # [B, N_cam, 2]
        cam_tokens = torch.cat((z_cam, cam_feat), dim=-1)
        cam_flat = cam_tokens.reshape(batch, n_cam * cam_tokens.shape[-1])
        return self.camera_fusion(cam_flat)

    def learned_parameter_count(self) -> int:
        """Count all trainable parameters (trunk + both fusion stages)."""
        return int(
            sum(p.numel() for p in self.parameters() if p.requires_grad)
        )

    def fusion_parameter_count(self) -> int:
        """Count trainable parameters in ``light_fusion`` and ``camera_fusion``."""
        return int(
            sum(
                p.numel()
                for module in (self.light_fusion, self.camera_fusion)
                for p in module.parameters()
                if p.requires_grad
            )
        )

    def describe(self) -> dict[str, Any]:
        """Return hierarchical fusion metadata for experiment logs.

        Documents two-stage structure, camera/light orbits, intermediate
        ``camera_latent_dim``, flat tensor layout, and fusion-only parameter
        count separate from full e2e trainable count.

        Notebook / protocol: 10_2.
        """
        pooled = self.backbone_kind == "pooled"
        return {
            "variant_id": (
                "m10_2_hierarchical_pooled_light_then_camera_fusion"
                if pooled
                else "m10_2_hierarchical_light_then_camera_fusion"
            ),
            "fusion_pattern": self.fusion_pattern,
            "backbone_kind": self.backbone_kind,
            "latent_cut": "after_first_mlp_proj_relu",
            "latent_dim": int(self.backbone.hidden),
            "camera_latent_dim": self.camera_latent_dim,
            "geometry_features": (
                "sin_light",
                "cos_light",
                "sin_camera",
                "cos_camera",
            ),
            "n_cameras": self.n_cameras,
            "n_lights": self.n_lights,
            "camera_angles_deg": list(self.camera_angles_deg),
            "light_angles_deg": list(self.light_angles_deg),
            "fusion_hidden": self.fusion_hidden,
            "fusion_depth": self.fusion_depth,
            "fusion_parameter_count": self.fusion_parameter_count(),
            "packing": "hierarchical_light_then_camera",
            "flat_layout": self.flat_layout,
            "encoder_frozen": False,
            "end_to_end": True,
            "learned_parameter_count": self.learned_parameter_count(),
            "learned_fusion_module": True,
            "note": (
                "M10 10_2 pooled; GAP trunk + light-then-camera hierarchical fusion"
                if pooled
                else (
                    "M9/M10 10_2; fuse illuminations within each camera, then fuse "
                    "camera-level latents across viewpoints"
                )
            ),
        }


__all__ = [
    "FUSION_PATTERN_09_0",
    "FUSION_PATTERN_09_1",
    "FUSION_PATTERN_09_1_MEAN_POOL",
    "FUSION_PATTERN_09_1_POOLED",
    "FUSION_PATTERN_09_1_MEAN_POOL_POOLED",
    "FUSION_PATTERN_09_1_DEEPSETS_FOURIER",
    "FUSION_PATTERN_09_1_DEEPSETS_NO_FOURIER",
    "FUSION_PATTERN_09_2",
    "FUSION_PATTERN_09_2_POOLED",
    "FUSION_PATTERN_09_3",
    "FUSION_PATTERN_09_3_POOLED",
    "FUSION_PATTERN_10_BASELINE",
    "FUSION_PATTERN_10_BASELINE_POOLED",
    "FUSION_PATTERN_10_1_C",
    "FUSION_PATTERN_10_1_C_POOLED",
    "FUSION_PATTERN_10_1_C_FROZEN",
    "FUSION_PATTERN_10_1_C_FROZEN_POOLED",
    "FUSION_PATTERN_10_1_D",
    "FUSION_PATTERN_10_1_D_POOLED",
    "FUSION_PATTERN_10_1_D_FROZEN",
    "FUSION_PATTERN_10_1_D_FROZEN_POOLED",
    "FUSION_PATTERN_10_2",
    "FUSION_PATTERN_10_2_POOLED",
    "FUSION_PATTERN_MEAN_LATENT_SANITY",
    "PACKING_MEAN_POOL",
    "PACKING_ORDERED_CONCAT",
    "M9_1_DEEPSETS_PHI_HIDDEN",
    "M9_1_DEEPSETS_RHO_HIDDEN",
    "M9_2_FUSION_DEPTH",
    "M9_2_FUSION_HIDDEN",
    "M9_3_FUSION_DEPTH",
    "M9_3_FUSION_HIDDEN",
    "M10_BASELINE_FUSION_DEPTH",
    "M10_BASELINE_FUSION_HIDDEN",
    "M10_1_FUSION_DEPTH",
    "M10_1_FUSION_HIDDEN",
    "M10_2_CAMERA_LATENT_DIM",
    "M10_2_FUSION_DEPTH",
    "M10_2_FUSION_HIDDEN",
    "M10_LIGHT_ANGLES_DEG",
    "M10_LIGHT_RADIUS_XY",
    "M10_LIGHT_Z",
    "M10_OPTICAL_SETUP_PREFIX",
    "CompactLatentFusionLocalizer",
    "DeepSetsFusionHead",
    "ExpertXyzMeanLocalizer",
    "FrozenEncoderDeepSetsLocalizer",
    "GeometryAwareFourierFusionLocalizer",
    "HierarchicalLightThenCameraFusionLocalizer",
    "MeanLatentFusionLocalizer",
    "angle_key",
    "build_geometry_fusion_mlp",
    "encode_view_latents",
    "ensure_camera_light_views",
    "ensure_multi_view_batch",
    "freeze_backbone_parameters",
    "fuse_coordinates",
    "light_angle_deg_from_optical_setup_id",
    "light_xy_from_angle_deg",
    "make_angle_features",
    "match_expert_index",
    "mean_coordinates",
    "new_frozen_pooled_single_view_expert",
    "new_frozen_single_view_expert",
    "pack_geometry_tokens",
    "pack_illumination_geometry_tokens",
    "resolve_light_angles",
    "resolve_view_angles",
    "shared_per_view_xyz",
    "shared_xyz_mean",
]
