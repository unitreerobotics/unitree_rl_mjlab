from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def tensor_to_numpy(value: Any, env_idx: int = 0) -> np.ndarray | None:
  if value is None:
    return None
  if isinstance(value, np.ndarray):
    return value
  if isinstance(value, torch.Tensor):
    if value.ndim > 0 and value.shape[0] > env_idx:
      value = value[env_idx]
    return value.detach().cpu().numpy()
  return np.asarray(value)


class EvaluationRunLogger:
  def __init__(self, run_dir: Path, *, run_id: int, seed: int, checkpoint: Path):
    self.run_dir = run_dir
    self.run_dir.mkdir(parents=True, exist_ok=True)
    self.run_id = run_id
    self.seed = seed
    self.checkpoint = checkpoint
    self.rows: list[dict[str, Any]] = []
    self.arrays: dict[str, list[np.ndarray]] = {}
    self.events: dict[str, Any] = {
      "run_id": run_id,
      "seed": seed,
      "checkpoint": str(checkpoint),
      "patch_events": [],
    }

  def add_array(self, name: str, value: Any) -> None:
    arr = tensor_to_numpy(value)
    if arr is None:
      return
    self.arrays.setdefault(name, []).append(np.asarray(arr))

  def log_step(self, row: dict[str, Any], arrays: dict[str, Any]) -> None:
    self.rows.append(row)
    for key, value in arrays.items():
      self.add_array(key, value)

  def write(self) -> None:
    if self.rows:
      with (self.run_dir / "raw.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
        writer.writeheader()
        writer.writerows(self.rows)
    npz_data = {}
    for key, values in self.arrays.items():
      try:
        npz_data[key] = np.stack(values)
      except ValueError:
        npz_data[key] = np.asarray(values, dtype=object)
    if npz_data:
      np.savez_compressed(self.run_dir / "raw.npz", **npz_data)
    (self.run_dir / "events.json").write_text(json.dumps(self.events, indent=2))


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    return
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
