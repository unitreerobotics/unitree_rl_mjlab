"""Flat dict diffs used by the detail view."""

from __future__ import annotations

from typing import Any, Iterable

_MISSING = object()


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten ``d`` into dot / bracket paths. Leaves are scalars, None, or tuples."""
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        if not d:
            out[prefix or ""] = {}
            return out
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, key))
    elif isinstance(d, list):
        if not d:
            out[prefix] = []
            return out
        for i, v in enumerate(d):
            key = f"{prefix}[{i}]"
            out.update(flatten(v, key))
    else:
        out[prefix] = d
    return out


def dict_diff(a: dict[str, Any], b: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Return a list of (path, a_value, b_value) where the flattened values differ.

    Keys missing on either side are reported with the string ``"<missing>"``.
    Results are sorted by path for stable display.
    """
    fa = flatten(a)
    fb = flatten(b)
    keys = sorted(set(fa) | set(fb))
    rows: list[tuple[str, Any, Any]] = []
    for k in keys:
        av = fa.get(k, _MISSING)
        bv = fb.get(k, _MISSING)
        if av is _MISSING or bv is _MISSING or av != bv:
            rows.append(
                (
                    k,
                    "<missing>" if av is _MISSING else av,
                    "<missing>" if bv is _MISSING else bv,
                )
            )
    return rows


def dot_get(d: Any, path: str) -> Any:
    """Walk ``path`` (dot-separated, with optional ``[i]`` indexing) into ``d``.

    Returns ``None`` if any intermediate key/index is missing.
    """
    if not path:
        return d
    cursor: Any = d
    for part in _split_path(path):
        if isinstance(part, int):
            if not isinstance(cursor, list) or part >= len(cursor) or part < -len(cursor):
                return None
            cursor = cursor[part]
        else:
            if not isinstance(cursor, dict) or part not in cursor:
                return None
            cursor = cursor[part]
    return cursor


def dot_has(d: Any, path: str) -> bool:
    """True iff ``path`` exists in ``d``."""
    if not path:
        return True
    cursor: Any = d
    for part in _split_path(path):
        if isinstance(part, int):
            if not isinstance(cursor, list) or part >= len(cursor) or part < -len(cursor):
                return False
            cursor = cursor[part]
        else:
            if not isinstance(cursor, dict) or part not in cursor:
                return False
            cursor = cursor[part]
    return True


def _split_path(path: str) -> Iterable[str | int]:
    """Split "a.b[0].c" into ['a', 'b', 0, 'c']."""
    for chunk in path.split("."):
        # Handle optional [i] suffixes on each chunk.
        while "[" in chunk:
            head, rest = chunk.split("[", 1)
            idx_str, after = rest.split("]", 1)
            if head:
                yield head
                head = ""
            yield int(idx_str)
            chunk = after
        if chunk:
            yield chunk
