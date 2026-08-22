"""Matplotlib helpers for particle geometry and source-deposition validation plots."""

from .intersection_geometry import (
    plot_sphere, 
    set_axes_equal, 
    plot_intersection_event, 
    plot_affected_transport_pairs,
    plot_particle_intersections_in_mesh,
    plot_segment)

from .energy_deposition import (
    plot_element_scalar, plot_transport_pair_deposition_barcodes
)


from .source_delta_3d import (    plot_active_element_scalar_3d,    source_active_element_mask,    source_delta_plot_panels,)

from .refractive_transport_3d import (
    add_transparent_surface_mesh,
    plot_point_light,
    plot_refractive_illumination_scene,
)

from .pinhole_camera import plot_pinhole_wireframe
from .face_sampling import (
    face_centroid_channels,
    plot_face_sampled_row,
    sample_face_scalars_to_images,
)
from .m2b_proxy import (
    collect_m2b_proxy_diagnostics,
    plot_beam_direction_components,
    plot_m2b_proxy_diagnostics,
)
from .m3_face_transport import (
    plot_entry_internal_directions_3d,
    plot_face_state_camera_panel,
    plot_face_state_on_mesh_3d,
    plot_source_rays_with_bbox,
    sample_face_state_camera_images,
)
from .m4_diffusion import (
    plot_camera_scalar,
    plot_deposition_scene,
    plot_diffusion_centroids_3d,
    plot_hybrid_panels,
    plot_phi_on_nodes,
    plot_profile_along_ray,
    plot_surface_with_diffusion_centroids,
)


__all__ = [
    "plot_sphere", 
    "set_axes_equal",  
    "plot_intersection_event", 
    "plot_affected_transport_pairs",
    "plot_particle_intersections_in_mesh",
    "plot_segment",
    "plot_element_scalar",
    "plot_transport_pair_deposition_barcodes",
    "plot_active_element_scalar_3d",
    "source_active_element_mask",
    "source_delta_plot_panels",
    "add_transparent_surface_mesh",
    "plot_point_light",
    "plot_refractive_illumination_scene",
    "plot_pinhole_wireframe",
    "face_centroid_channels",
    "sample_face_scalars_to_images",
    "plot_face_sampled_row",
    "collect_m2b_proxy_diagnostics",
    "plot_m2b_proxy_diagnostics",
    "plot_beam_direction_components",
    "plot_source_rays_with_bbox",
    "plot_face_state_on_mesh_3d",
    "sample_face_state_camera_images",
    "plot_face_state_camera_panel",
    "plot_entry_internal_directions_3d",
    "plot_diffusion_centroids_3d",
    "plot_surface_with_diffusion_centroids",
    "plot_phi_on_nodes",
    "plot_profile_along_ray",
    "plot_deposition_scene",
    "plot_camera_scalar",
    "plot_hybrid_panels",
]