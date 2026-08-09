"""Shared accessors and notebook-style assertion helpers for validation code.

These utilities normalize dataclass / dict / attribute objects and print
PASS/FAIL diagnostics used across particle-transport and sequence checks.
"""

from .access_helpers import (
    as_mapping,
    check_close,
    check_true,
    event_entry_t,
    event_exit_t,
    event_segment_index,
    get_any,
    get_field,
    pair_path_id,
    print_object_inventory,
    require_any,
    summarize_array,
)

__all__ = [
    "as_mapping",
    "get_any",
    "require_any",
    "check_true",
    "check_close",
    "summarize_array",
    "print_object_inventory",
    "get_field",
    "pair_path_id",
    "event_segment_index",
    "event_exit_t",
    "event_entry_t",
]
