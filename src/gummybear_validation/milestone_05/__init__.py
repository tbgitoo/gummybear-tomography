"""Particle transport validation helpers and the inspectable simulation runner."""

from .validation import test_affected_pair_indices_match_intersection_events
from .simulate import M5DSimulationConfig, run_m5d_simulation

__all__ = [
    "test_affected_pair_indices_match_intersection_events",
    "run_m5d_simulation",
    "M5DSimulationConfig",
]
