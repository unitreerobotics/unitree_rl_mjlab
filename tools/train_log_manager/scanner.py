"""Discover Isaac Lab rsl_rl run directories and load their metadata."""

from __future__ import annotations

import json
import re
import threading
import time
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
    run: dict[str, Any] = field(default_factory=dict)
    task_id: str | None = None
    git_diff: str = ""
    max_iter: int | None = None
    checkpoints: list[Path] = field(default_factory=list)
    tfevents: Path | None = None
    max_mean_reward: float | None = None


def _load_yaml(p: Path) -> dict[str, Any]:
    if not p.is_file():
        return {}
    try:
        with p.open("r") as f:
            # Isaac Lab dumps `!!python/tuple` etc. — unsafe_load is required.
            # Source is our own training logs, not untrusted input.
            data = yaml.unsafe_load(f)
    except Exception:
        # Older logs may reference Python classes that moved or no longer exist.
        return {}
    return data if isinstance(data, dict) else {}


def _checkpoints(run_dir: Path) -> list[Path]:
    found: list[tuple[int, Path]] = []
    for entry in run_dir.iterdir():
        m = _MODEL_ITER_RE.match(entry.name)
        if m:
            found.append((int(m.group(1)), entry))
    found.sort(key=lambda item: item[0])
    return [path for _, path in found]


def checkpoint_iter(path: Path) -> int | None:
    match = _MODEL_ITER_RE.match(path.name)
    return int(match.group(1)) if match else None


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


_REWARD_SYNC_BUDGET_S = 6.0  # max wall-clock spent reading tfevents on the render path.
_backfill_lock = threading.Lock()
_backfill_threads: dict[str, threading.Thread] = {}


def _reward_from_cache(
    tfevents: Path | None, cache: dict[str, dict[str, Any]]
) -> tuple[float | None, bool]:
    """Return ``(value, is_cached)`` without ever reading the event file."""
    if tfevents is None:
        return None, True
    try:
        stat = tfevents.stat()
    except OSError:
        return None, True
    entry = cache.get(str(tfevents))
    if (
        isinstance(entry, dict)
        and entry.get("mtime") == stat.st_mtime
        and entry.get("size") == stat.st_size
    ):
        return entry.get("value"), True
    return None, False


def _max_mean_reward(
    tfevents: Path | None,
    cache: dict[str, dict[str, Any]],
    *,
    compute: bool = True,
) -> float | None:
    """Max of ``Train/mean_reward``, disk-cached by (path, mtime, size).

    Reading a tfevents file with :class:`EventAccumulator` costs ~8s per 30 MB,
    so a cold scan of many large runs would otherwise block the UI for minutes
    (and Ctrl-C can't interrupt it — the shutdown joins the scan thread). With
    ``compute=False`` this consults only the cache and never touches the event
    file; missing values are filled later by :func:`_ensure_reward_backfill`.
    """
    value, is_cached = _reward_from_cache(tfevents, cache)
    if is_cached or not compute or tfevents is None:
        return value
    try:
        stat = tfevents.stat()
    except OSError:
        return None
    value = _compute_max_mean_reward(tfevents)
    cache[str(tfevents)] = {"mtime": stat.st_mtime, "size": stat.st_size, "value": value}
    return value


def _load_run(
    run_dir: Path,
    experiment: str,
    tb_cache: dict[str, dict[str, Any]],
    *,
    compute_reward: bool = True,
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
    checkpoints = _checkpoints(run_dir)
    max_iter = checkpoint_iter(checkpoints[-1]) if checkpoints else None
    run_meta = _load_yaml(run_dir / "params" / "run.yaml")
    task_id = run_meta.get("task_id")
    if not isinstance(task_id, str):
        task_id = None
    return Run(
        run_id=run_dir.name,
        experiment=experiment,
        path=run_dir,
        agent=_load_yaml(run_dir / "params" / "agent.yaml"),
        env=_load_yaml(run_dir / "params" / "env.yaml"),
        run=run_meta,
        task_id=task_id,
        git_diff=git_diff,
        max_iter=max_iter,
        checkpoints=checkpoints,
        tfevents=tfevents,
        max_mean_reward=_max_mean_reward(tfevents, tb_cache, compute=compute_reward),
    )


def _iter_run_dirs(
    logs_root: Path, experiments: list[str] | None
) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for exp_dir in sorted(logs_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        if experiments is not None and exp_dir.name not in experiments:
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "params").is_dir():
                continue
            pairs.append((exp_dir.name, run_dir))
    # Newest first (run_id is a sortable timestamp).
    pairs.sort(key=lambda pr: (pr[0], pr[1].name), reverse=True)
    return pairs


def _backfill_rewards(logs_root: Path, experiments: list[str] | None) -> None:
    """Compute every still-missing ``max_mean_reward`` and persist it.

    Runs in a daemon thread off the render path. Reloads + re-saves the cache
    per file so progress survives and merges with concurrent writers.
    """
    for _experiment, run_dir in _iter_run_dirs(logs_root, experiments):
        tfevents = _find_tfevents(run_dir)
        if tfevents is None:
            continue
        cache = _load_tb_cache()
        _value, is_cached = _reward_from_cache(tfevents, cache)
        if is_cached:
            continue
        try:
            stat = tfevents.stat()
        except OSError:
            continue
        value = _compute_max_mean_reward(tfevents)
        cache = _load_tb_cache()  # reload to merge any concurrent updates
        cache[str(tfevents)] = {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "value": value,
        }
        _save_tb_cache(cache)


def _ensure_reward_backfill(logs_root: Path, experiments: list[str] | None) -> None:
    key = str(logs_root)
    with _backfill_lock:
        existing = _backfill_threads.get(key)
        if existing is not None and existing.is_alive():
            return
        thread = threading.Thread(
            target=_backfill_rewards,
            args=(logs_root, experiments),
            name="tb-reward-backfill",
            daemon=True,
        )
        _backfill_threads[key] = thread
        thread.start()


def rewards_pending(logs_root: Path | str) -> bool:
    """True while a background reward backfill is still running for ``logs_root``."""
    thread = _backfill_threads.get(str(Path(logs_root)))
    return thread is not None and thread.is_alive()


def scan_runs(logs_root: Path, experiments: list[str] | None = None) -> list[Run]:
    """Scan ``logs_root`` for rsl_rl runs and return them newest-first.

    The reward column is read from tfevents files, which is slow; to keep the UI
    responsive only a small wall-clock budget (newest-first) is spent computing
    it synchronously, and the rest is filled by a background daemon thread into
    the on-disk cache (picked up on the next rescan).

    Directory layout expected:
        logs_root / <experiment> / <run_id> / {params, git, events..., model_*.pt}
    """
    logs_root = Path(logs_root)
    if not logs_root.is_dir():
        return []

    tb_cache = _load_tb_cache()
    pairs = _iter_run_dirs(logs_root, experiments)

    deadline = time.monotonic() + _REWARD_SYNC_BUDGET_S
    runs: list[Run] = []
    for experiment, run_dir in pairs:
        compute = time.monotonic() < deadline
        runs.append(_load_run(run_dir, experiment, tb_cache, compute_reward=compute))

    # Drop cache entries for files that no longer exist, then persist whatever
    # was computed synchronously this pass.
    alive = {str(r.tfevents) for r in runs if r.tfevents is not None}
    tb_cache = {k: v for k, v in tb_cache.items() if k in alive}
    _save_tb_cache(tb_cache)

    # Anything still missing: compute in the background for the next rescan.
    if any(r.max_mean_reward is None and r.tfevents is not None for r in runs):
        _ensure_reward_backfill(logs_root, experiments)

    return runs  # already newest-first
