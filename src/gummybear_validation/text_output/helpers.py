"""String formatters for quick numeric diagnostics in notebooks and tests."""

import numpy as np


def array_mini_summary(name: str, arr) -> str:
    """Build a single-line summary string for one array.

    Includes shape, dtype, min/max, nonzero count, and the first twenty
    flattened elements. Empty arrays omit min/max/nnz/first20.

    Args:
        name: Label prefix (for example ``"E_clean_elem"``).
        arr: Array-like value to summarize.

    Returns:
        str: Human-readable one-line summary suitable for logging or asserts.
    """
    arr = np.asarray(arr)

    if arr.size == 0:
        return f"{name}: shape={arr.shape}, dtype={arr.dtype}, empty"

    return (
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr.min():.12g}, max={arr.max():.12g}, "
        f"nnz={np.count_nonzero(arr)}, first20={arr[:20]}"
    )
