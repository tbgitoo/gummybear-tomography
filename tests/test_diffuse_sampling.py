"""Focused tests for nodal tetrahedral diffuse sampling."""

from __future__ import annotations

import numpy as np

from gummybear.optics import (
    DiffusionMesh,
    DiffusionMeshMetadata,
    interpolate_phi_nodes_to_points,
    sample_diffuse_image,
)


def _single_tet_mesh() -> DiffusionMesh:
    nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    tets = np.asarray([[0, 1, 2, 3]])
    return DiffusionMesh(
        nodes=nodes,
        tets=tets,
        centroids=nodes[tets].mean(axis=1),
        volumes=np.asarray([1.0 / 6.0]),
        metadata=DiffusionMeshMetadata(
            stl_hash="synthetic",
            geometry_id="single_tet",
            num_elements=1,
            num_nodes=4,
        ),
    )


def test_interpolate_phi_nodes_at_vertices_centroid_and_known_point():
    mesh = _single_tet_mesh()
    phi_nodes = np.asarray([2.0, 4.0, 8.0, 10.0])
    known_weights = np.asarray([0.1, 0.2, 0.3, 0.4])
    known_point = known_weights @ mesh.nodes
    points = np.vstack(
        [
            mesh.nodes,
            mesh.nodes.mean(axis=0),
            known_point,
        ]
    )

    sampled = interpolate_phi_nodes_to_points(mesh, phi_nodes, points)

    np.testing.assert_allclose(sampled[:4], phi_nodes)
    np.testing.assert_allclose(sampled[4], np.mean(phi_nodes))
    np.testing.assert_allclose(sampled[5], known_weights @ phi_nodes)


def test_interpolation_reproduces_linear_nodal_field():
    mesh = _single_tet_mesh()
    phi_nodes = 1.5 + 2.0 * mesh.nodes[:, 0] - mesh.nodes[:, 1]
    points = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.6, 0.1, 0.1],
            [0.2, 0.7, 0.05],
        ]
    )

    sampled = interpolate_phi_nodes_to_points(mesh, phi_nodes, points)

    np.testing.assert_allclose(sampled, 1.5 + 2.0 * points[:, 0] - points[:, 1])
    assert not np.allclose(sampled, np.mean(phi_nodes))


def test_interpolation_uses_nearest_tet_extrapolation_outside_mesh():
    mesh = _single_tet_mesh()
    phi_nodes = mesh.nodes[:, 0]
    point = np.asarray([[1.0 + 1e-8, 0.0, 0.0]])

    sampled = interpolate_phi_nodes_to_points(mesh, phi_nodes, point)

    np.testing.assert_allclose(sampled, point[:, 0])


def test_sample_diffuse_image_preserves_mask_shape_and_exitance_scale():
    mesh = _single_tet_mesh()
    phi_nodes = 1.0 + mesh.nodes[:, 0]
    hit_points = np.asarray(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [0.3, 0.1, 0.1],
            [np.nan, np.nan, np.nan],
        ]
    )
    camera_mask = np.asarray([True, False, True, False])

    base = sample_diffuse_image(
        mesh,
        phi_nodes,
        hit_points,
        camera_mask,
        sample_shape=(2, 2),
        exitance_scale=1.0,
    )
    scaled = sample_diffuse_image(
        mesh,
        phi_nodes,
        hit_points,
        camera_mask,
        sample_shape=(2, 2),
        exitance_scale=2.5,
    )

    assert base.I_diffuse.shape == (2, 2)
    assert np.all(base.I_diffuse[~camera_mask.reshape(2, 2)] == 0.0)
    np.testing.assert_allclose(scaled.I_diffuse, 2.5 * base.I_diffuse)


def test_sample_diffuse_image_can_use_legacy_nearest_tet_sampling():
    mesh = _single_tet_mesh()
    phi_nodes = mesh.nodes[:, 0]
    hit_points = np.asarray([[0.1, 0.1, 0.1], [0.7, 0.1, 0.1]])
    camera_mask = np.ones(2, dtype=bool)

    interpolated = sample_diffuse_image(
        mesh,
        phi_nodes,
        hit_points,
        camera_mask,
        sample_shape=(1, 2),
    )
    legacy = sample_diffuse_image(
        mesh,
        phi_nodes,
        hit_points,
        camera_mask,
        sample_shape=(1, 2),
        interpolate=False,
    )

    np.testing.assert_allclose(interpolated.I_diffuse, [[0.1, 0.7]])
    np.testing.assert_allclose(legacy.I_diffuse, np.mean(phi_nodes))
