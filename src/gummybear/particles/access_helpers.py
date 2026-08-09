"""Loose attribute access and notebook assertion helpers for particle transport.

Utilities tolerate dict-like records, dataclasses, and plain objects when
inspecting M5 transport pairs in notebooks and tests. Not used on the hot
generation path.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from typing import Any, Sequence


def as_mapping(obj: Any) -> dict[str, Any]:
    """Return a shallow dict view of ``obj`` for field inspection."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def get_any(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    """Return the first matching mapping key or attribute, else ``default``."""
    mapping = as_mapping(obj)
    for name in names:
        if name in mapping:
            return mapping[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def require_any(obj: Any, names: Sequence[str], label: str) -> Any:
    """Return the first matching field or raise ``AssertionError``."""
    value = get_any(obj, names, default=None)
    if value is None:
        available = sorted(as_mapping(obj).keys())
        raise AssertionError(
            f"Missing {label}. Tried {names}. Available fields: {available}"
        )
    return value


def check_true(name: str, condition: bool) -> None:
    """Print PASS/FAIL and raise ``AssertionError`` when ``condition`` is false."""
    print(f"{name}: {'PASS' if condition else 'FAIL'}")
    if not condition:
        raise AssertionError(name)


def check_close(
    name: str,
    actual: Any,
    expected: Any,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """Print PASS/FAIL for ``np.allclose`` and raise on mismatch."""
    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    ok = np.allclose(actual_arr, expected_arr, rtol=rtol, atol=atol)
    max_abs = float(np.max(np.abs(actual_arr - expected_arr))) if actual_arr.size else 0.0

    print(f"{name}: {'PASS' if ok else 'FAIL'} | max_abs={max_abs:.6g}")

    if not ok:
        print("actual =", actual)
        print("expected =", expected)
        raise AssertionError(name)


def summarize_array(name: str, arr: Any) -> None:
    """Print shape, sum, min, max, and nonzero count for one array."""
    arr = np.asarray(arr)
    if arr.size == 0:
        print(f"{name}: shape={arr.shape}, empty")
        return

    print(
        f"{name}: shape={arr.shape}, "
        f"sum={arr.sum():.12g}, "
        f"min={arr.min():.12g}, "
        f"max={arr.max():.12g}, "
        f"nnz={np.count_nonzero(arr)}"
    )


def print_object_inventory(name: str, obj: Any) -> None:
    """Print type and sorted field names (or ``dir`` excerpt) for debugging."""
    print(name)
    print("=" * len(name))
    print("type:", type(obj))
    mapping = as_mapping(obj)
    if mapping:
        print("fields:")
        for key in sorted(mapping):
            value = mapping[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: ndarray shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
    else:
        print("dir excerpt:")
        print([x for x in dir(obj) if not x.startswith("_")])


def get_field(obj: Any, names: Sequence[str], label: str) -> Any:
    """Return the first matching dict key or attribute.

    Args:
        obj: Mapping, dataclass, or object with attributes.
        names: Candidate field names in priority order.
        label: Human label used in ``AttributeError`` messages.

    Returns:
        Field value.

    Raises:
        AttributeError: When no candidate name resolves.
    """
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]

        if hasattr(obj, name):
            return getattr(obj, name)

    raise AttributeError(
        f"Could not find {label}. Tried names {list(names)} "
        f"on object of type {type(obj)!r}."
    )


def pair_path_id(pair: Any) -> int:
    """Return transport path id from a pair record (several alias names)."""
    return int(
        get_field(
            pair,
            ["path_id", "ray_id", "transport_path_id"],
            "pair path_id / ray_id / transport_path_id",
        )
    )


def event_segment_index(event: Any) -> int:
    """Return ``segment_index`` from an intersection or scatter event."""
    return int(
        get_field(
            event,
            ["segment_index"],
            "event.segment_index",
        )
    )


def event_entry_t(event: Any) -> float:
    """Return segment parameter ``entry_t`` from an intersection event."""
    return float(
        get_field(
            event,
            ["entry_t"],
            "event.entry_t",
        )
    )


def event_exit_t(event: Any) -> float:
    """Return segment parameter ``exit_t`` from an intersection event."""
    return float(
        get_field(
            event,
            ["exit_t"],
            "event.exit_t",
        )
    )
