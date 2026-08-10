"""Shared matplotlib helpers for localisation and illumination-fusion notebooks."""

from .anomaly_orbit_preview import (
    DEFAULT_CAMERA_LIGHT_GRID_FPS,
    DEFAULT_DISPLAY_MEAN,
    DEFAULT_DISPLAY_STD,
    anomaly_still_vs_orbit_html,
    display_anomaly_camera_light_grid,
    display_anomaly_still_vs_orbit,
)
from .illumination_fusion import (
    IlluminationFusionPlotConfig,
    PLOT_CONFIG_10_1A,
    PLOT_CONFIG_10_1B,
    plot_illumination_fusion_results,
    plot_stage_b_lr_study,
)
from .single_view_studies import (
    plot_error_histograms,
    plot_learning_rate_study,
    plot_rmse_summary_bars,
)

__all__ = [
    "DEFAULT_CAMERA_LIGHT_GRID_FPS",
    "DEFAULT_DISPLAY_MEAN",
    "DEFAULT_DISPLAY_STD",
    "IlluminationFusionPlotConfig",
    "PLOT_CONFIG_10_1A",
    "PLOT_CONFIG_10_1B",
    "anomaly_still_vs_orbit_html",
    "display_anomaly_camera_light_grid",
    "display_anomaly_still_vs_orbit",
    "plot_error_histograms",
    "plot_illumination_fusion_results",
    "plot_learning_rate_study",
    "plot_rmse_summary_bars",
    "plot_stage_b_lr_study",
]
