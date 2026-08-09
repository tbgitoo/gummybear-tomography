"""Normalize transport/particle records and raise readable validation failures."""

import dataclasses

import numpy as np

from typing import Any, Sequence


def as_mapping(obj):
    """Convert an object to a plain dict for field lookup.

    Accepts ``None`` (empty dict), mappings, dataclasses, or any object with
    ``__dict__``. Other types yield an empty dict.

    Args:
        obj: Record to normalize.

    Returns:
        dict: Field name → value mapping.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def get_any(obj, names, default=None):
    """Read the first matching field from a mapping or object attributes.

    Tries each name in ``names`` against :func:`as_mapping`, then against
    ``getattr``. Returns ``default`` when nothing matches.

    Args:
        obj: Mapping, dataclass, or attribute-bearing object.
        names: Candidate field names, tried in order.
        default: Value returned when no name matches.

    Returns:
        First non-missing value, or ``default``.
    """
    mapping = as_mapping(obj)
    for name in names:
        if name in mapping:
            return mapping[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def require_any(obj, names, label):
    """Like :func:`get_any`, but raise when the resolved value is ``None``.

    **Pass:** a non-``None`` value is found under one of ``names``.

    Args:
        obj: Mapping, dataclass, or attribute-bearing object.
        names: Candidate field names.
        label: Human-readable name used in the ``AssertionError`` message.

    Returns:
        The first non-``None`` matched value.

    Raises:
        AssertionError: When every candidate is missing or ``None``.
    """
    value = get_any(obj, names, default=None)
    if value is None:
        available = sorted(as_mapping(obj).keys())
        raise AssertionError(
            f"Missing {label}. Tried {names}. Available fields: {available}"
        )
    return value


def check_true(name, condition):
    """Print PASS/FAIL and raise when ``condition`` is false.

    **Pass:** ``condition`` is truthy; prints ``{name}: PASS``.

    Args:
        name: Label printed and reused in ``AssertionError``.
        condition: Boolean predicate to assert.

    Raises:
        AssertionError: When ``condition`` is false.
    """
    print(f"{name}: {'PASS' if condition else 'FAIL'}")
    if not condition:
        raise AssertionError(name)


def check_close(name, actual, expected, rtol=1e-9, atol=1e-12):
    """Print PASS/FAIL for an elementwise ``numpy.allclose`` comparison.

    **Pass:** all elements match within ``rtol`` / ``atol``; prints max absolute
    error. On failure, prints ``actual`` and ``expected`` before raising.

    Args:
        name: Label printed and reused in ``AssertionError``.
        actual: Observed array-like values.
        expected: Reference array-like values.
        rtol: Relative tolerance passed to ``numpy.allclose``.
        atol: Absolute tolerance passed to ``numpy.allclose``.

    Raises:
        AssertionError: When arrays differ beyond tolerance.
    """
    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    ok = np.allclose(actual_arr, expected_arr, rtol=rtol, atol=atol)
    max_abs = float(np.max(np.abs(actual_arr - expected_arr))) if actual_arr.size else 0.0

    print(f"{name}: {'PASS' if ok else 'FAIL'} | max_abs={max_abs:.6g}")

    if not ok:
        print("actual =", actual)
        print("expected =", expected)
        raise AssertionError(name)


def summarize_array(name, arr):
    """Print shape, sum, min, max, and nonzero count for one array.

    Empty arrays print shape only. This is diagnostic output only; it never
    raises.

    Args:
        name: Label prefix in the printed line.
        arr: Array-like value to summarize.
    """
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


def print_object_inventory(name, obj):
    """Print type information and field names for debugging unknown records.

    For mapping-like objects, lists each key with ndarray shape/dtype or value
    type. Otherwise prints a short ``dir`` excerpt.

    Args:
        name: Section header printed above the inventory.
        obj: Object to inspect.
    """
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
    """Return the first present field, searching dict keys then attributes.

    Unlike :func:`require_any`, missing fields raise ``AttributeError`` and
    ``None`` is a valid return value.

    Args:
        obj: Mapping or attribute-bearing object.
        names: Candidate field names, tried in order.
        label: Human-readable name used in the error message.

    Returns:
        First matched field value.

    Raises:
        AttributeError: When no candidate name is found.
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
    """Read the transport path id from an affected-pair record.

    Accepts ``path_id``, ``ray_id``, or ``transport_path_id`` as dict keys or
    attributes.

    Args:
        pair: Affected transport pair object or mapping.

    Returns:
        int: Integer path / ray identifier.
    """
    return int(
        get_field(
            pair,
            ["path_id", "ray_id", "transport_path_id"],
            "pair path_id / ray_id / transport_path_id",
        )
    )


def event_segment_index(event: Any) -> int:
    """Read ``segment_index`` from a particle intersection event.

    Args:
        event: Intersection event object or mapping.

    Returns:
        int: Index into the segment bundle that was hit.
    """
    return int(
        get_field(
            event,
            ["segment_index"],
            "event.segment_index",
        )
    )


def event_entry_t(event: Any) -> float:
    """Read normalized entry parameter ``entry_t`` along the hit segment.

    ``entry_t`` is in ``[0, 1]`` relative to segment start → end.

    Args:
        event: Intersection event object or mapping.

    Returns:
        float: Entry location along the segment.
    """
    return float(
        get_field(
            event,
            ["entry_t"],
            "event.entry_t",
        )
    )


def event_exit_t(event: Any) -> float:
    """Read normalized exit parameter ``exit_t`` along the hit segment.

    ``exit_t`` is in ``[0, 1]`` relative to segment start → end.

    Args:
        event: Intersection event object or mapping.

    Returns:
        float: Exit location along the segment.
    """
    return float(
        get_field(
            event,
            ["exit_t"],
            "event.exit_t",
        )
    )
