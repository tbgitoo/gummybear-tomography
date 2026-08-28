"""Hugging Face Hub upload/download helpers for gummybear-tomography artefacts."""

from tomography_ml_huggingface.dataset_metadata import (
    METADATA_SUBDIR,
    export_huggingface_metadata,
    main as export_cli_main,
)
from tomography_ml_huggingface.m10_hierarchical_export import (
    export_gummybear_hierarchical_fusion,
    resolve_gummybear_hierarchical_fusion_paths,
)
from tomography_ml_huggingface.m10_hierarchical_inference import (
    download_gummybear_hierarchical_fusion,
    load_gummybear_hierarchical_fusion,
    load_gummybear_hierarchical_fusion_from_hub,
    run_hub_contract_smoke_inference,
)
from tomography_ml_huggingface.m9_compact_export import (
    export_camera_orbit_compact_09_2b,
    resolve_camera_orbit_compact_09_2b_paths,
)
from tomography_ml_huggingface.m9_compact_inference import (
    download_camera_orbit_compact_09_2b,
    load_camera_orbit_compact_09_2b,
    load_camera_orbit_compact_09_2b_from_hub,
    load_packaged_m9_demo_multiview_example,
    run_packaged_m9_demo_inference,
)
from tomography_ml_huggingface.model_export import (
    export_singleview_cnn_fourier,
    resolve_singleview_cnn_fourier_paths,
)
from tomography_ml_huggingface.model_inference import (
    DEFAULT_HUB_DOWNLOAD_TIMEOUT_S,
    HubDownloadError,
    download_singleview_cnn_fourier,
    load_singleview_cnn_fourier,
    load_singleview_cnn_fourier_from_hub,
    run_packaged_demo_inference,
)

__all__ = [
    "METADATA_SUBDIR",
    "DEFAULT_HUB_DOWNLOAD_TIMEOUT_S",
    "HubDownloadError",
    "download_camera_orbit_compact_09_2b",
    "download_gummybear_hierarchical_fusion",
    "download_singleview_cnn_fourier",
    "export_camera_orbit_compact_09_2b",
    "export_cli_main",
    "export_gummybear_hierarchical_fusion",
    "export_huggingface_metadata",
    "export_singleview_cnn_fourier",
    "load_camera_orbit_compact_09_2b",
    "load_camera_orbit_compact_09_2b_from_hub",
    "load_gummybear_hierarchical_fusion",
    "load_gummybear_hierarchical_fusion_from_hub",
    "load_packaged_m9_demo_multiview_example",
    "load_singleview_cnn_fourier",
    "load_singleview_cnn_fourier_from_hub",
    "resolve_camera_orbit_compact_09_2b_paths",
    "resolve_gummybear_hierarchical_fusion_paths",
    "resolve_singleview_cnn_fourier_paths",
    "run_hub_contract_smoke_inference",
    "run_packaged_demo_inference",
    "run_packaged_m9_demo_inference",
]
