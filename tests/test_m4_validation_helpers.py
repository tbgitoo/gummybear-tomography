"""Tests for M4 validation notebook helpers."""

import numpy as np

from gummybear.optics.source_deposition import SourceDepositionResult
from gummybear_validation.helpers.m4_diffusion import (
    assert_deposition_conservation,
    assert_deposition_sanity,
    make_centroid_axis_ray,
)


def test_make_centroid_axis_ray_endpoints():
    class _Mesh:
        centroids = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 1.0]], dtype=float)

    ray, p0, p1 = make_centroid_axis_ray(_Mesh(), axis="x", intensity=2.0)
    assert p0[0] < p1[0]
    assert ray.intensities.shape == (1,)


def test_assert_deposition_conservation():
    volumes = np.array([1.0, 2.0, 3.0])
    result = SourceDepositionResult(
        S_clean=np.array([0.5, 0.0, 0.25]),
        E_scat_elem=np.array([0.5, 0.0, 0.75]),
        total_ballistic_input=2.0,
        total_scattered=1.25,
        total_absorbed=0.5,
        remaining_direct_energy=0.25,
        mu_s=0.2,
        mu_a=0.1,
    )
    assert_deposition_sanity(result, n_tets=3)
    assert_deposition_conservation(result, volumes)
