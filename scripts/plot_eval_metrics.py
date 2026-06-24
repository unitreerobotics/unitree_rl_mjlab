"""Plot per-encoder evaluation metrics from a combined_summary.csv.

Reads ``<base>/combined_summary.csv`` (the aggregate produced by
``tmp/aggregate_eval.py``) and writes a set of comparison plots into the same
directory, one bar per checkpoint/encoder, grouped by terrain. Works for both
single-terrain runs (e.g. stairs_corridor) and multi-terrain runs.

Usage:
    python scripts/plot_eval_metrics.py --base logs/data/eval/20260623_stairs_100runs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Mirrors scripts/make_eval_grid.py so encoder labels/order stay consistent.
PREFERRED_ENCODER_ORDER = [
  "conv1d",
  "conv1d_state",
  "conv2d",
  "conv2d_state",
  "mlp",
  "mlp_state",
  "raw",
  "ae",
]

# Bar metrics: column -> (plot filename, axis label, title).
BAR_METRICS = {
  "traversal_rate_mean": (
    "metric_traversal_rate.png",
    "Traversal rate",
    "Traversal rate (fraction of corridor)",
  ),
  "mean_speed_mean": (
    "metric_mean_speed.png",
    "Mean speed (m/s)",
    "Mean forward speed",
  ),
  "mean_path_lateral_error_mean": (
    "metric_path_lateral_error.png",
    "Lateral error (m)",
    "Mean path lateral error",
  ),
  "mean_velocity_tracking_error_mean": (
    "metric_velocity_tracking_error.png",
    "Velocity tracking error (m/s)",
    "Mean velocity tracking error",
  ),
}

FAILURE_COLORS = {
  "fall_rate": "#c82828",
  "stuck_rate": "#d2821e",
  "other": "#787878",
  "success_rate": "#22a03c",
}


def _encoder_label(checkpoint_label: str) -> str:
  match = re.match(
    r"^go2_velocity_encoder_(.+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$",
    checkpoint_label,
  )
  return match.group(1) if match else checkpoint_label


def _encoder_sort_key(label: str) -> tuple[int, str]:
  try:
    return (PREFERRED_ENCODER_ORDER.index(label), label)
  except ValueError:
    return (len(PREFERRED_ENCODER_ORDER), label)


def _terrain_label(terrain: str) -> str:
  return terrain.removesuffix("_corridor")


def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
  df = df.copy()
  df["encoder"] = df["checkpoint_label"].map(_encoder_label)
  encoders = sorted(df["encoder"].unique(), key=_encoder_sort_key)
  terrains = sorted(df["terrain"].unique())
  return df, encoders, terrains


def _grouped_bar(
  ax: plt.Axes,
  df: pd.DataFrame,
  encoders: list[str],
  terrains: list[str],
  value_col: str,
  err_col: str | None = None,
) -> None:
  x = np.arange(len(encoders))
  n = len(terrains)
  width = 0.8 / max(n, 1)
  for i, terrain in enumerate(terrains):
    sub = df[df["terrain"] == terrain].set_index("encoder")
    vals = [float(sub.loc[e, value_col]) if e in sub.index else 0.0 for e in encoders]
    errs = None
    if err_col is not None and err_col in df.columns:
      errs = [float(sub.loc[e, err_col]) if e in sub.index else 0.0 for e in encoders]
    offset = (i - (n - 1) / 2) * width
    ax.bar(
      x + offset,
      vals,
      width,
      yerr=errs,
      capsize=3,
      label=_terrain_label(terrain),
    )
  ax.set_xticks(x)
  ax.set_xticklabels(encoders, rotation=30, ha="right")
  ax.grid(axis="y", alpha=0.3)
  if n > 1:
    ax.legend(fontsize=8)


def _plot_bar_metric(
  df: pd.DataFrame,
  encoders: list[str],
  terrains: list[str],
  value_col: str,
  err_col: str | None,
  ylabel: str,
  title: str,
  out_path: Path,
) -> None:
  fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(encoders)), 4.5))
  _grouped_bar(ax, df, encoders, terrains, value_col, err_col)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
  plt.close(fig)
  print("wrote", out_path)


def _plot_success_rate(
  df: pd.DataFrame, encoders: list[str], terrains: list[str], out_path: Path
) -> None:
  fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(encoders)), 4.5))
  _grouped_bar(ax, df, encoders, terrains, "success_rate")
  ax.set_ylabel("Success rate")
  ax.set_ylim(0, 1)
  ax.set_title("Success rate by terrain")
  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
  plt.close(fig)
  print("wrote", out_path)


def _failure_components(sub: pd.DataFrame, encoders: list[str]) -> dict[str, list[float]]:
  s = sub.set_index("encoder")
  comp: dict[str, list[float]] = {k: [] for k in ("success_rate", "fall_rate", "stuck_rate", "other")}
  for e in encoders:
    if e in s.index:
      success = float(s.loc[e, "success_rate"])
      fall = float(s.loc[e, "fall_rate"])
      stuck = float(s.loc[e, "stuck_rate"])
    else:
      success = fall = stuck = 0.0
    other = max(0.0, 1.0 - success - fall - stuck)
    comp["success_rate"].append(success)
    comp["fall_rate"].append(fall)
    comp["stuck_rate"].append(stuck)
    comp["other"].append(other)
  return comp


def _stacked_failures(
  ax: plt.Axes, comp: dict[str, list[float]], encoders: list[str]
) -> None:
  x = np.arange(len(encoders))
  bottom = np.zeros(len(encoders))
  order = ["success_rate", "fall_rate", "stuck_rate", "other"]
  labels = {
    "success_rate": "success",
    "fall_rate": "fell over",
    "stuck_rate": "stuck",
    "other": "other / max steps",
  }
  for key in order:
    vals = np.array(comp[key])
    ax.bar(x, vals, bottom=bottom, color=FAILURE_COLORS[key], label=labels[key])
    bottom += vals
  ax.set_xticks(x)
  ax.set_xticklabels(encoders, rotation=30, ha="right")
  ax.set_ylim(0, 1)
  ax.grid(axis="y", alpha=0.3)


def _plot_failure_breakdown(
  df: pd.DataFrame, encoders: list[str], terrains: list[str], out_path: Path
) -> None:
  fig, axes = plt.subplots(
    1, len(terrains), figsize=(max(7, 1.2 * len(encoders) * len(terrains)), 4.8), squeeze=False
  )
  for ax, terrain in zip(axes[0], terrains, strict=True):
    comp = _failure_components(df[df["terrain"] == terrain], encoders)
    _stacked_failures(ax, comp, encoders)
    ax.set_title(_terrain_label(terrain))
  axes[0][0].set_ylabel("Episode outcome fraction")
  axes[0][-1].legend(fontsize=8, loc="upper right")
  fig.suptitle("Outcome breakdown by encoder")
  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
  plt.close(fig)
  print("wrote", out_path)


def _plot_comparison_grid(
  df: pd.DataFrame, encoders: list[str], terrains: list[str], out_path: Path
) -> None:
  panels = [
    ("success_rate", None, "Success rate"),
    ("traversal_rate_mean", "traversal_rate_std", "Traversal rate"),
    ("mean_speed_mean", None, "Mean speed (m/s)"),
    ("mean_velocity_tracking_error_mean", None, "Vel. tracking err (m/s)"),
    ("mean_path_lateral_error_mean", None, "Lateral err (m)"),
  ]
  fig, axes = plt.subplots(2, 3, figsize=(16, 9))
  flat = axes.flatten()
  for ax, (col, err, title) in zip(flat, panels, strict=False):
    _grouped_bar(ax, df, encoders, terrains, col, err)
    ax.set_title(title)
  # Last panel: stacked outcome breakdown (first terrain).
  ax = flat[len(panels)]
  comp = _failure_components(df[df["terrain"] == terrains[0]], encoders)
  _stacked_failures(ax, comp, encoders)
  ax.set_title(f"Outcomes ({_terrain_label(terrains[0])})")
  ax.legend(fontsize=7)
  fig.suptitle("Evaluation metric comparison", fontsize=14)
  fig.tight_layout(rect=(0, 0, 1, 0.97))
  fig.savefig(out_path, dpi=150)
  plt.close(fig)
  print("wrote", out_path)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--base", required=True, help="Eval dir with combined_summary.csv.")
  parser.add_argument("--out-dir", default=None, help="Output dir (default: --base).")
  args = parser.parse_args()

  base = Path(args.base).expanduser().resolve()
  summary = base / "combined_summary.csv"
  if not summary.is_file():
    raise FileNotFoundError(f"combined_summary.csv not found under {base}")
  out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else base
  out_dir.mkdir(parents=True, exist_ok=True)

  df, encoders, terrains = _prepare(pd.read_csv(summary))
  print(f"encoders={encoders}")
  print(f"terrains={terrains}")

  _plot_success_rate(df, encoders, terrains, out_dir / "success_rate_by_terrain.png")
  for col, (fname, ylabel, title) in BAR_METRICS.items():
    err = "traversal_rate_std" if col == "traversal_rate_mean" else None
    _plot_bar_metric(df, encoders, terrains, col, err, ylabel, title, out_dir / fname)
  _plot_failure_breakdown(df, encoders, terrains, out_dir / "hard_terrain_failure_breakdown.png")
  _plot_comparison_grid(df, encoders, terrains, out_dir / "metric_comparison_grid.png")


if __name__ == "__main__":
  main()
