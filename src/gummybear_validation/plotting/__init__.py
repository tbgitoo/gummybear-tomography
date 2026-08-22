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
]