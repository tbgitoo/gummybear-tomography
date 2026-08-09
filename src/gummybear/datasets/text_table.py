"""Small text-table helpers for readable dataclass ``__repr__`` output."""

from __future__ import annotations


def short_hash(value: str, *, width: int = 12) -> str:
    """Return a truncated digest prefix for readable repr tables.

    Args:
        value: Full hash or identifier string.
        width: Maximum characters to retain.

    Returns:
        ``value`` unchanged when shorter than ``width``, else the prefix.
    """
    text = str(value)
    if len(text) <= width:
        return text
    return text[:width]


def format_text_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> str:
    """Return a simple aligned monospace table.

    Args:
        headers: Column header strings.
        rows: Row tuples with one cell per header.

    Returns:
        Multi-line table string with padded columns.
    """
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def _fmt(cells: tuple[str, ...]) -> str:
        return "  ".join(
            cell.ljust(widths[index]) for index, cell in enumerate(cells)
        )

    rule = "  ".join("-" * width for width in widths)
    lines = [_fmt(headers), rule]
    lines.extend(_fmt(row) for row in rows)
    return "\n".join(lines)
