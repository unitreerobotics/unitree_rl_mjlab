"""Column definitions and value extraction for the runs table.

A column is a plain dict:

    {"name": str, "source": str, "kind": str, "path": str}
    {"name": str, "source": "git_diff", "kind": "regex", "pattern": str}

Valid ``source`` values: ``env``, ``agent``, ``run``, ``git_diff``, ``builtin``.
Valid ``kind`` values: ``exists``, ``value``, ``regex``, ``builtin``.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

from diff import dot_get, dot_has


SOURCES = ("env", "agent", "run", "git_diff", "builtin")
KINDS = ("exists", "value", "regex", "builtin")

BUILTIN_FIELDS = ("run_id", "experiment", "task_id", "max_iter", "max_mean_reward")


def discover_group_keys(
    runs: list[Any],
    source: str,
    group_path: str,
    *,
    require_key: str | None = None,
) -> list[str]:
    """Union of immediate child keys under ``group_path`` across every run.

    ``source`` is ``"env"`` or ``"agent"``. Returns child keys in first-seen
    order (scanning runs newest-first, matching :func:`scanner.scan_runs`).

    If ``require_key`` is given, only include children whose value is a dict
    containing that key — this filters out ObsGroup config flags like
    ``enable_corruption`` and keeps only real term definitions (which carry
    ``func``).
    """
    seen: dict[str, None] = {}
    for run in runs:
        root = run.env if source == "env" else run.agent
        group = dot_get(root, group_path)
        if not isinstance(group, dict):
            continue
        for k, v in group.items():
            if require_key is not None:
                if not (isinstance(v, dict) and require_key in v):
                    continue
            seen.setdefault(k, None)
    return list(seen.keys())


def expand_group(
    runs: list[Any],
    source: str,
    group_path: str,
    kind: str,
    *,
    name_prefix: str = "",
    child_suffix: str = "",
    require_key: str | None = None,
) -> list[dict[str, Any]]:
    """Build one column per child key found under ``group_path``.

    - ``kind="value"`` + ``child_suffix=".weight"`` → per-reward weight columns.
    - ``kind="exists"`` + ``child_suffix=""`` → per-observation presence columns.
    """
    out: list[dict[str, Any]] = []
    for key in discover_group_keys(runs, source, group_path, require_key=require_key):
        path = f"{group_path}.{key}{child_suffix}" if child_suffix else f"{group_path}.{key}"
        out.append({
            "name": f"{name_prefix}{key}",
            "source": source,
            "kind": kind,
            "path": path,
        })
    return out


def default_columns() -> list[dict[str, Any]]:
    return [
        {"name": "run_id",          "source": "builtin", "kind": "builtin", "path": "run_id", "protected": True},
        {"name": "task_id",         "source": "builtin", "kind": "builtin", "path": "task_id", "protected": True},
        {"name": "max_iter",        "source": "builtin", "kind": "builtin", "path": "max_iter", "protected": True},
        {"name": "max_mean_reward", "source": "builtin", "kind": "builtin", "path": "max_mean_reward", "protected": True},
        {"name": "num_envs",        "source": "env",     "kind": "value",   "path": "scene.num_envs"},
        {"name": "seed",            "source": "agent",   "kind": "value",   "path": "seed"},
        {"name": "learning_rate",   "source": "agent",   "kind": "value",   "path": "algorithm.learning_rate"},
        {"name": "rewards_patched", "source": "git_diff","kind": "regex",   "pattern": r"mjlab"},
    ]


def extract(run: Any, col: dict[str, Any]) -> Any:
    """Return the cell value for ``col`` given a ``Run`` dataclass instance.

    Unknown/missing paths return ``None`` (rendered as blank in the table).
    """
    source = col.get("source")
    kind = col.get("kind")

    if source == "builtin" or kind == "builtin":
        field = col.get("path") or col.get("name")
        return getattr(run, field, None)

    if source == "env":
        root = run.env
    elif source == "agent":
        root = run.agent
    elif source == "run":
        root = run.run
    elif source == "git_diff":
        root = run.git_diff
    else:
        return None

    if kind == "exists":
        return dot_has(root, col.get("path", ""))
    if kind == "value":
        val = dot_get(root, col.get("path", ""))
        # Tuples from !!python/tuple are unfriendly in Streamlit tables — coerce.
        if isinstance(val, tuple):
            return list(val)
        return val
    if kind == "regex":
        pattern = col.get("pattern", "")
        if not pattern or not isinstance(root, str):
            return False
        try:
            return bool(re.search(pattern, root))
        except re.error:
            return False
    return None


def encode(columns: list[dict[str, Any]]) -> str:
    """Compact-JSON + urlsafe base64 encoding for the ?cols= query param."""
    raw = json.dumps(columns, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode(token: str) -> list[dict[str, Any]] | None:
    """Inverse of :func:`encode`. Returns ``None`` on any error."""
    if not token:
        return None
    try:
        pad = "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(token + pad)
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None
    cleaned: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "name" not in item or "kind" not in item:
            continue
        cleaned.append(item)
    return cleaned
