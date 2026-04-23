"""Discover Isaac Lab rsl_rl run directories and load their metadata."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_MODEL_ITER_RE = re.compile(r"^model_(\d+)\.pt$")
_MEAN_REWARD_TAG = "Train/mean_reward"
_TB_CACHE_PATH = Path(__file__).resolve().parent / ".tb_cache.json"


@dataclass
class Run:
    run_id: str
    experiment: str
    path: Path
    agent: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    git_diff: str = ""
    max_iter: int | None = None
    tfevents: Path | None = None
    max_mean_reward: float | None = None


def _load_yaml(p: Path) -> dict[str, Any]:
    if not p.is_file():
        return {}
    with p.open("r") as f:
        # Isaac Lab dumps `!!python/tuple` etc. — unsafe_load is required.
        # Source is our own training logs, not untrusted input.
        data = yaml.unsafe_load(f)
    return data if isinstance(data, dict) else {}


def _max_model_iter(run_dir: Path) -> int | None:
    best = -1
    for entry in run_dir.iterdir():
        m = _MODEL_ITER_RE.match(entry.name)
        if m:
            best = max(best, int(m.group(1)))
    return best if best >= 0 else None


def _find_tfevents(run_dir: Path) -> Path | None:
    for entry in run_dir.iterdir():
        if entry.name.startswith("events.out.tfevents."):
            return entry
    return None


def _load_tb_cache() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(_TB_CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_tb_cache(cache: dict[str, dict[str, Any]]) -> None:
    try:
        _TB_CACHE_PATH.write_text(json.dumps(cache))
    except OSError:
        pass


def _compute_max_mean_reward(tfevents: Path) -> float | None:
    # Imported lazily — tensorboard is heavy and may not be installed in every
    # environment that scans runs (e.g. test harness).
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    try:
        ea = EventAccumulator(str(tfevents), size_guidance={"scalars": 0})
        ea.Reload()
        if _MEAN_REWARD_TAG not in ea.Tags().get("scalars", []):
            return None
        events = ea.Scalars(_MEAN_REWARD_TAG)
    except Exception:
        return None
    if not events:
        return None
    return max(e.value for e in events)


def _max_mean_reward(
    tfevents: Path | None, cache: dict[str, dict[str, Any]]
) -> float | None:
    """Read max of ``Train/mean_reward`` from the tfevents file.

    Cached on disk keyed by (path, mtime, size) — a finished run never needs
    to be re-read, and an in-progress run only re-parses when its event file
    has actually grown.
    """
    if tfevents is None:
        return None
    key = str(tfevents)
    try:
        stat = tfevents.stat()
    except OSError:
        return None
    entry = cache.get(key)
    if (
        isinstance(entry, dict)
        and entry.get("mtime") == stat.st_mtime
        and entry.get("size") == stat.st_size
    ):
        return entry.get("value")
    value = _compute_max_mean_reward(tfevents)
    cache[key] = {"mtime": stat.st_mtime, "size": stat.st_size, "value": value}
    return value


def _load_run(
    run_dir: Path, experiment: str, tb_cache: dict[str, dict[str, Any]]
) -> Run:
    git_diff_path = run_dir / "git" / f"{experiment}.diff"
    if not git_diff_path.is_file():
        # Fall back to any *.diff file under git/.
        git_dir = run_dir / "git"
        if git_dir.is_dir():
            diffs = list(git_dir.glob("*.diff"))
            git_diff_path = diffs[0] if diffs else git_diff_path
    git_diff = git_diff_path.read_text(errors="replace") if git_diff_path.is_file() else ""

    tfevents = _find_tfevents(run_dir)
    return Run(
        run_id=run_dir.name,
        experiment=experiment,
        path=run_dir,
        agent=_load_yaml(run_dir / "params" / "agent.yaml"),
        env=_load_yaml(run_dir / "params" / "env.yaml"),
        git_diff=git_diff,
        max_iter=_max_model_iter(run_dir),
        tfevents=tfevents,
        max_mean_reward=_max_mean_reward(tfevents, tb_cache),
    )


def scan_runs(logs_root: Path, experiments: list[str] | None = None) -> list[Run]:
    """Scan ``logs_root`` for rsl_rl runs and return them newest-first.

    Directory layout expected:
        logs_root / <experiment> / <run_id> / {params, git, events..., model_*.pt}
    """
    logs_root = Path(logs_root)
    if not logs_root.is_dir():
        return []

    tb_cache = _load_tb_cache()
    runs: list[Run] = []
    for exp_dir in sorted(logs_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        if experiments is not None and exp_dir.name not in experiments:
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            # A valid run has at least a params dir.
            if not (run_dir / "params").is_dir():
                continue
            runs.append(_load_run(run_dir, exp_dir.name, tb_cache))

    # Drop cache entries for files that no longer exist — otherwise the cache
    # grows unbounded across run-dir deletions.
    alive = {str(r.tfevents) for r in runs if r.tfevents is not None}
    tb_cache = {k: v for k, v in tb_cache.items() if k in alive}
    _save_tb_cache(tb_cache)

    # Newest first — run_id is a sortable timestamp.
    runs.sort(key=lambda r: (r.experiment, r.run_id), reverse=True)
    return runs
