"""Backfill run metadata files for existing training logs."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

KNOWN_GO2_TASKS = {
  "Unitree-Go2-Rough",
  "Unitree-Go2-Rough-No-Height-Scan",
  "Unitree-Go2-Flat",
  "Unitree-Go2-Flat-Scan",
  "Unitree-Go2-No-Phase-Rough",
  "Unitree-Go2-Test",
  "Unitree-Go2-Test-Train",
}


def registered_task_ids() -> set[str]:
  try:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks
  except Exception:
    return set(KNOWN_GO2_TASKS)
  return set(list_tasks()) | KNOWN_GO2_TASKS


def parse_scalar(text: str, key: str) -> str | None:
  match = re.search(rf"^\s*{re.escape(key)}:\s*(.*)$", text, re.MULTILINE)
  if not match:
    return None
  value = match.group(1).strip()
  if value in {"", "null", "None", "~"}:
    return None
  if (value.startswith("'") and value.endswith("'")) or (
    value.startswith('"') and value.endswith('"')
  ):
    value = value[1:-1]
  return value


def parse_int(text: str, key: str) -> int | None:
  value = parse_scalar(text, key)
  if value is None:
    return None
  try:
    return int(value)
  except ValueError:
    return None


def parse_bool(text: str, key: str) -> bool | None:
  value = parse_scalar(text, key)
  if value is None:
    return None
  lowered = value.lower()
  if lowered == "true":
    return True
  if lowered == "false":
    return False
  return None


def parse_clip_actions(text: str) -> float | None:
  value = parse_scalar(text, "clip_actions")
  if value is None:
    return None
  try:
    return float(value)
  except ValueError:
    return None


def task_from_run_name(run_dir: Path, task_ids: set[str]) -> str | None:
  name = run_dir.name
  timestamp_len = len("YYYY-MM-DD_HH-MM-SS")
  if len(name) <= timestamp_len or name[timestamp_len] != "_":
    return None
  suffix = name[timestamp_len + 1 :]
  return suffix if suffix in task_ids else None


def first_num_rows_cols(env_text: str) -> tuple[int | None, int | None]:
  return parse_int(env_text, "num_rows"), parse_int(env_text, "num_cols")


def infer_go2_task(env_text: str) -> tuple[str | None, str, str | None]:
  terrain_type = parse_scalar(env_text, "terrain_type")
  has_height_scan = bool(re.search(r"^\s+height_scan:\s*$", env_text, re.MULTILINE))
  rows, cols = first_num_rows_cols(env_text)
  has_test_markers = all(
    marker in env_text
    for marker in ("huge_step:", "steep_hill:", "cliff:", "balance_beam:")
  )

  if terrain_type == "plane" and not has_height_scan:
    return "Unitree-Go2-Flat", "inferred_env_config", None
  if terrain_type == "generator" and rows == 5 and cols == 5 and has_height_scan:
    return "Unitree-Go2-Flat-Scan", "inferred_env_config", None
  if terrain_type == "generator" and rows == 5 and cols == 4 and has_test_markers:
    return "Unitree-Go2-Test-Train", "inferred_env_config", None
  if terrain_type == "generator" and rows == 10 and cols == 20:
    if has_height_scan:
      return "Unitree-Go2-Rough", "inferred_env_config", None
    return "Unitree-Go2-Rough-No-Height-Scan", "inferred_env_config", None

  note = (
    "could not infer task from env.yaml "
    f"(terrain_type={terrain_type}, num_rows={rows}, num_cols={cols}, "
    f"height_scan={has_height_scan})"
  )
  return None, "unknown", note


def infer_task(
  run_dir: Path,
  env_text: str | None,
  task_ids: set[str],
) -> tuple[str | None, str, str | None]:
  task_id = task_from_run_name(run_dir, task_ids)
  if task_id is not None:
    return task_id, "run_dir", None
  if env_text is None:
    return None, "unknown", "params/env.yaml not found"
  if run_dir.parent.name == "go2_velocity":
    return infer_go2_task(env_text)
  return None, "unknown", "task inference is only implemented for go2_velocity runs"


def build_metadata(run_dir: Path, task_ids: set[str]) -> dict[str, Any]:
  env_path = run_dir / "params" / "env.yaml"
  agent_path = run_dir / "params" / "agent.yaml"
  env_text = env_path.read_text() if env_path.exists() else None
  agent_text = agent_path.read_text() if agent_path.exists() else ""

  task_id, task_source, note = infer_task(run_dir, env_text, task_ids)
  timestamp_source = env_path if env_path.exists() else run_dir
  created_at = datetime.fromtimestamp(timestamp_source.stat().st_mtime).isoformat(
    timespec="seconds"
  )

  metadata: dict[str, Any] = {
    "metadata_version": 1,
    "task_id": task_id,
    "task_id_source": task_source,
    "launcher": "unknown",
    "command": None,
    "launcher_command": None,
    "log_dir": str(run_dir),
    "created_at": created_at,
    "cwd": None,
    "python_executable": None,
    "experiment_name": parse_scalar(agent_text, "experiment_name")
    or run_dir.parent.name,
    "run_name": parse_scalar(agent_text, "run_name"),
    "resume": parse_bool(agent_text, "resume"),
    "load_run": parse_scalar(agent_text, "load_run"),
    "load_checkpoint": parse_scalar(agent_text, "load_checkpoint"),
    "num_envs": parse_int(env_text or "", "num_envs"),
    "max_iterations": parse_int(agent_text, "max_iterations"),
    "clip_actions": parse_clip_actions(agent_text),
    "cuda_visible_devices": None,
    "selected_gpus": None,
    "num_gpus": None,
    "world_size": None,
    "backfilled_at": datetime.now().isoformat(timespec="seconds"),
  }
  if note is not None:
    metadata["inference_note"] = note
  return metadata


def iter_run_dirs(logs_root: Path) -> list[Path]:
  return sorted(
    run_dir
    for experiment_dir in logs_root.iterdir()
    if experiment_dir.is_dir()
    for run_dir in experiment_dir.iterdir()
    if run_dir.is_dir() and run_dir.name != "wandb_checkpoints"
  )


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--logs-root", type=Path, default=Path("logs/rsl_rl"))
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--overwrite", action="store_true")
  args = parser.parse_args()

  if not args.logs_root.exists():
    parser.error(f"logs root does not exist: {args.logs_root}")

  task_ids = registered_task_ids()
  wrote = 0
  skipped = 0
  for run_dir in iter_run_dirs(args.logs_root):
    out_path = run_dir / "params" / "run.yaml"
    if out_path.exists() and not args.overwrite:
      print(f"SKIP existing {out_path}")
      skipped += 1
      continue
    metadata = build_metadata(run_dir, task_ids)
    label = metadata["task_id"] or "unknown"
    action = "WOULD WRITE" if args.dry_run else "WRITE"
    print(f"{action} {out_path} task_id={label} source={metadata['task_id_source']}")
    if not args.dry_run:
      out_path.parent.mkdir(parents=True, exist_ok=True)
      with out_path.open("w") as f:
        yaml.dump(metadata, f, sort_keys=False)
      wrote += 1
  print(f"done: wrote={wrote} skipped={skipped}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
