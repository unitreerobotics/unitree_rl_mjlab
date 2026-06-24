"""Shared engine for evaluating/visualizing the traversability estimator.

Used by both the ``tools/eval_traversability.py`` CLI and the "Evaluator" tab in
``tools/train_log_manager/app.py``. The plotting builders return
``matplotlib.figure.Figure`` objects (plus small result dicts) so the CLI can
``fig.savefig(...)`` while Streamlit does ``st.pyplot(fig)`` — one engine, no
duplicated plotting.

Reuses, rather than reimplements, the pipeline's own functions:

* ``parse_layout`` / ``compute_episode_geometry`` / ``build_spatial_labels`` from
  ``tools/build_traversability_labels.py`` (raw-rollout layout + labels).
* ``_binary_metrics`` from ``tools/train_traversability.py`` (AUC/AP/acc/Brier).
* ``load_traversability_estimator`` from ``src/rl_models/traversability.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless/threaded-safe; figure builders import pyplot.

import numpy as np
import torch

# Make sibling tools importable regardless of CWD (``streamlit run`` included).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

import build_traversability_labels as btl  # noqa: E402
import train_traversability as tt  # noqa: E402

from src.rl_models.traversability import load_traversability_estimator  # noqa: E402

# Re-export the pipeline functions we reuse so callers import them from one place.
parse_layout = btl.parse_layout
compute_episode_geometry = btl.compute_episode_geometry
build_spatial_labels = btl.build_spatial_labels
quat_to_yaw = btl.quat_to_yaw
binary_metrics = tt._binary_metrics

__all__ = [
  "load_traversability_estimator",
  "parse_layout",
  "compute_episode_geometry",
  "build_spatial_labels",
  "quat_to_yaw",
  "binary_metrics",
  "split_actor_obs",
  "infer_scalar",
  "infer_spatial",
  "roc_curve",
  "pr_curve",
  "calibration_bins",
  "threshold_sweep",
]


def split_actor_obs(
  actor_obs: np.ndarray, layout: list[tuple[str, slice]], keys: list[str]
) -> dict[str, np.ndarray]:
  """Split the concatenated actor obs into the named groups the model expects.

  ``actor_obs`` has shape ``[..., A]``; the returned arrays keep the leading
  dims and carry the per-group flat dimension as the last axis.
  """
  by_name = dict(layout)
  missing = [k for k in keys if k not in by_name]
  if missing:
    raise KeyError(
      f"Model input keys {missing} not in rollout layout {list(by_name)}."
    )
  return {k: actor_obs[..., by_name[k]] for k in keys}


@torch.no_grad()
def infer_scalar(
  model, obs_groups: dict[str, np.ndarray], device, batch: int = 65536
) -> np.ndarray:
  """Run the scalar head over flattened samples; returns ``P(failure)`` ``[M]``."""
  keys = list(obs_groups.keys())
  m = int(next(iter(obs_groups.values())).shape[0])
  out = np.empty(m, dtype=np.float32)
  for s in range(0, m, batch):
    e = min(s + batch, m)
    obs = {
      k: torch.from_numpy(np.ascontiguousarray(obs_groups[k][s:e])).to(device)
      for k in keys
    }
    out[s:e] = model.predict_proba(obs).float().cpu().numpy()
  return out


@torch.no_grad()
def infer_spatial(
  model, obs_groups: dict[str, np.ndarray], device, batch: int = 16384
) -> np.ndarray:
  """Run the spatial head over flattened samples; returns ``[M, NW, NH]``."""
  keys = list(obs_groups.keys())
  m = int(next(iter(obs_groups.values())).shape[0])
  nw, nh = model.spatial_grid
  out = np.empty((m, nw, nh), dtype=np.float32)
  for s in range(0, m, batch):
    e = min(s + batch, m)
    obs = {
      k: torch.from_numpy(np.ascontiguousarray(obs_groups[k][s:e])).to(device)
      for k in keys
    }
    out[s:e] = model.predict_spatial_proba(obs).float().cpu().numpy()
  return out


def roc_curve(scores: np.ndarray, labels: np.ndarray):
  """(fpr, tpr) points. Mirrors the cumulative logic in ``_binary_metrics``."""
  labels = labels.astype(np.int64)
  desc = np.argsort(-scores, kind="mergesort")
  y = labels[desc]
  tp = np.cumsum(y)
  fp = np.cumsum(1 - y)
  n_pos = max(int(tp[-1]), 1)
  n_neg = max(int(fp[-1]), 1)
  tpr = np.concatenate([[0.0], tp / n_pos])
  fpr = np.concatenate([[0.0], fp / n_neg])
  return fpr, tpr


def pr_curve(scores: np.ndarray, labels: np.ndarray):
  """(recall, precision) points."""
  labels = labels.astype(np.int64)
  desc = np.argsort(-scores, kind="mergesort")
  y = labels[desc]
  tp = np.cumsum(y)
  n_pos = max(int(tp[-1]), 1)
  precision = tp / np.arange(1, len(y) + 1)
  recall = tp / n_pos
  return recall, precision


def calibration_bins(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10):
  """Reliability-diagram bins: (mean predicted prob, empirical rate, count)."""
  edges = np.linspace(0.0, 1.0, n_bins + 1)
  idx = np.clip(np.digitize(scores, edges[1:-1]), 0, n_bins - 1)
  mean_pred = np.full(n_bins, np.nan)
  emp_rate = np.full(n_bins, np.nan)
  counts = np.zeros(n_bins, dtype=np.int64)
  for b in range(n_bins):
    sel = idx == b
    counts[b] = int(sel.sum())
    if counts[b]:
      mean_pred[b] = float(scores[sel].mean())
      emp_rate[b] = float(labels[sel].mean())
  return mean_pred, emp_rate, counts


def threshold_sweep(scores: np.ndarray, labels: np.ndarray, thresholds):
  """Per-threshold precision/recall/F1/FPR + counts. Returns a list of dicts."""
  labels = labels.astype(np.int64)
  n_pos = int(labels.sum())
  n_neg = labels.shape[0] - n_pos
  rows = []
  for thr in thresholds:
    pred = scores >= thr
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = n_pos - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_pos, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fpr = fp / max(n_neg, 1)
    rows.append(
      {
        "threshold": float(thr),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
      }
    )
  return rows


# --------------------------------------------------------------------------- #
# Data loading + scoring (the heavy step the UI caches).
# --------------------------------------------------------------------------- #
def score_labels_file(
  labels_path,
  checkpoint,
  *,
  device: str = "cpu",
  val_frac: float = 0.15,
  seed: int = 0,
  split: str = "val",
):
  """Load ``labels.npz``, run the scalar head, return ``(scores, labels, info)``.

  Reproduces the exact held-out split of ``train_traversability.py`` (same
  ``random_split`` seed + ``val_frac``) so the ``report`` numbers line up with
  ``--eval-only``. ``split`` is ``"val"`` | ``"train"`` | ``"all"``.
  """
  from types import SimpleNamespace

  from torch.utils.data import DataLoader, random_split

  model = load_traversability_estimator(checkpoint, map_location=device).to(device)
  args = SimpleNamespace(
    input=Path(labels_path),
    input_keys=list(model.encoder_input_keys),
    spatial_weight=0.0,
    input_hw=list(model.input_hw),
    pos_weight=None,
  )
  dataset, meta = tt._load_dataset(args)
  n_val = int(len(dataset) * val_frac)
  n_train = len(dataset) - n_val
  gen = torch.Generator().manual_seed(seed)
  train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)
  chosen = {"val": val_set, "train": train_set, "all": dataset}[split]
  loader = DataLoader(chosen, batch_size=8192, shuffle=False)

  scores, labels = [], []
  model.eval()
  with torch.no_grad():
    for batch in loader:
      obs, y_scalar, _, _ = tt._split_batch(tt._to_device(list(batch), device), meta)
      scores.append(torch.sigmoid(model(obs)["scalar_logit"]).float().cpu().numpy())
      labels.append(y_scalar.cpu().numpy())
  info = {"keys": meta["keys"], "n_samples": len(chosen), "split": split}
  return np.concatenate(scores), np.concatenate(labels), info


def score_rollouts_file(rollouts_path, checkpoint, *, device: str = "cpu", spatial: bool = False, keep_heavy: bool = True):
  """Load ``raw_rollouts.npz``, score it, and keep the ``[T,N]`` time structure.

  Returns a dict with ``risk[T,N]`` (+ ``risk_map[T,N,NW,NH]`` when ``spatial``),
  the episode geometry from ``compute_episode_geometry``, and the raw arrays the
  spatial visualization needs.
  """
  data = np.load(rollouts_path, allow_pickle=True)
  actor_obs = data["actor_obs"]  # [T, N, A]
  done = data["done"].astype(bool)  # [T, N]
  failure = data["failure"].astype(bool)  # [T, N]
  layout = parse_layout(data["actor_layout"])
  horizon = int(data["horizon"])
  step_dt = float(data["step_dt"])
  T, N = done.shape

  model = load_traversability_estimator(checkpoint, map_location=device).to(device)
  keys = list(model.encoder_input_keys)
  groups = split_actor_obs(actor_obs, layout, keys)  # each [T, N, dim]
  flat = {k: np.ascontiguousarray(v).reshape(T * N, -1) for k, v in groups.items()}

  risk = infer_scalar(model, flat, device).reshape(T, N)
  next_done_idx, scalar_label, valid = compute_episode_geometry(done, failure, horizon)

  out = {
    "risk": risk,
    "done": done,
    "failure": failure,
    "scalar_label": scalar_label,
    "valid": valid,
    "next_done_idx": next_done_idx,
    "horizon": horizon,
    "step_dt": step_dt,
    "T": T,
    "N": N,
    "model": model,
    "layout": layout,
    "actor_obs": actor_obs,
    "root_pos_w": data["root_pos_w"],
    "root_quat_w": data["root_quat_w"],
  }
  if not keep_heavy:
    # The timeline / lead-time UI only needs the light [T,N] arrays; releasing
    # the model + raw obs/pose lets st.cache_data hold the result cheaply.
    for _k in ("model", "actor_obs", "root_pos_w", "root_quat_w"):
      out.pop(_k, None)
  if spatial:
    out["risk_map"] = infer_spatial(model, flat, device).reshape(
      T, N, model.spatial_grid[0], model.spatial_grid[1]
    )
  return out


# --------------------------------------------------------------------------- #
# Episode helpers for the timeline / lead-time analysis.
# --------------------------------------------------------------------------- #
def episode_bounds(done: np.ndarray, n: int):
  """Yield ``(start, end_inclusive)`` step ranges for env ``n`` (resets at done)."""
  T = done.shape[0]
  start = 0
  for t in range(T):
    if done[t, n]:
      yield start, t
      start = t + 1
  if start <= T - 1:
    yield start, T - 1


def lead_time_stats(scored: dict, threshold: float):
  """Per-failure warning lead time + dataset-level false-alarm rate at ``threshold``.

  Lead time = steps between the first in-window alarm (risk >= threshold, same
  episode, within the H-step pre-failure window) and the failure step. Failures
  with no alarm in the window are 'missed'.
  """
  risk = scored["risk"]
  failure = scored["failure"]
  done = scored["done"]
  H = scored["horizon"]
  step_dt = scored["step_dt"]
  T, N = risk.shape

  # Episode start index per (t, n): one before the previous done.
  ep_start = np.zeros((T, N), dtype=np.int64)
  for n in range(N):
    s = 0
    for t in range(T):
      ep_start[t, n] = s
      if done[t, n]:
        s = t + 1

  leads, missed = [], 0
  fail_t, fail_n = np.nonzero(failure)
  for tf, nf in zip(fail_t.tolist(), fail_n.tolist()):
    lo = max(int(ep_start[tf, nf]), tf - H)
    window = risk[lo : tf + 1, nf]
    hit = np.nonzero(window >= threshold)[0]
    if hit.size:
      leads.append(tf - (lo + int(hit[0])))
    else:
      missed += 1

  # False alarms: safe, valid steps that fire.
  safe = (scored["scalar_label"] < 0.5) & scored["valid"]
  n_safe = int(safe.sum())
  n_safe_alarm = int(((risk >= threshold) & safe).sum())
  far = n_safe_alarm / max(n_safe, 1)
  leads_arr = np.asarray(leads, dtype=np.float64)
  return {
    "threshold": float(threshold),
    "n_failures": int(fail_t.size),
    "n_detected": int(leads_arr.size),
    "n_missed": int(missed),
    "detection_rate": float(leads_arr.size / max(fail_t.size, 1)),
    "lead_median_steps": float(np.median(leads_arr)) if leads_arr.size else float("nan"),
    "lead_iqr_steps": (
      float(np.percentile(leads_arr, 75) - np.percentile(leads_arr, 25))
      if leads_arr.size
      else float("nan")
    ),
    "lead_median_s": float(np.median(leads_arr) * step_dt) if leads_arr.size else float("nan"),
    "false_alarm_rate": float(far),
    "false_alarms_per_min": float(far * 60.0 / max(step_dt, 1e-9)),
    "leads_steps": leads_arr,
  }


# --------------------------------------------------------------------------- #
# Figure builders — return matplotlib Figures (CLI saves; Streamlit st.pyplot).
# --------------------------------------------------------------------------- #
def build_report_figure(scores, labels, *, n_bins: int = 10, thresholds=None):
  """ROC + PR + calibration + score-histogram panel. Returns ``(fig, result)``."""
  import matplotlib.pyplot as plt

  if thresholds is None:
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
  metrics = binary_metrics(scores, labels)
  pos_rate = float(np.mean(labels))
  fpr, tpr = roc_curve(scores, labels)
  rec, prec = pr_curve(scores, labels)
  mean_pred, emp_rate, counts = calibration_bins(scores, labels, n_bins)
  sweep = threshold_sweep(scores, labels, thresholds)
  best = max(sweep, key=lambda r: r["f1"]) if sweep else None

  fig, axes = plt.subplots(2, 2, figsize=(11, 9))
  ax = axes[0, 0]
  ax.plot(fpr, tpr, color="C0")
  ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
  ax.set(title=f"ROC (AUC={metrics['auc']:.4f})", xlabel="FPR", ylabel="TPR")

  ax = axes[0, 1]
  ax.plot(rec, prec, color="C1")
  ax.axhline(pos_rate, ls="--", color="gray", lw=0.8, label=f"baseline={pos_rate:.4f}")
  ax.set(title=f"Precision-Recall (AP={metrics['ap']:.4f})", xlabel="Recall", ylabel="Precision")
  ax.legend(loc="upper right", fontsize=8)

  ax = axes[1, 0]
  ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
  ax.plot(mean_pred, emp_rate, "o-", color="C2")
  ax.set(title=f"Calibration (Brier={metrics['brier']:.4f})",
         xlabel="Predicted P(failure)", ylabel="Empirical failure rate",
         xlim=(0, 1), ylim=(0, 1))

  ax = axes[1, 1]
  pos = scores[labels.astype(bool)]
  neg = scores[~labels.astype(bool)]
  bins = np.linspace(0, 1, 41)
  ax.hist(neg, bins=bins, color="C0", alpha=0.6, label=f"safe (n={neg.size})")
  ax.hist(pos, bins=bins, color="C3", alpha=0.6, label=f"failure-soon (n={pos.size})")
  ax.set(title="Score histogram", xlabel="P(failure)", ylabel="count")
  ax.set_yscale("log")
  ax.legend(fontsize=8)

  fig.tight_layout()
  result = {"metrics": metrics, "pos_rate": pos_rate, "sweep": sweep, "best_f1": best}
  return fig, result


def build_timeline_figure(scored: dict, *, threshold: float = 0.5, max_fail: int = 6, n_safe: int = 4, seed: int = 0):
  """Risk(t) over a sample of failing + safe episodes. Returns ``(fig, info)``."""
  import matplotlib.pyplot as plt

  rng = np.random.default_rng(seed)
  risk = scored["risk"]
  failure = scored["failure"]
  done = scored["done"]
  H = scored["horizon"]
  step_dt = scored["step_dt"]

  fail_t, fail_n = np.nonzero(failure)
  order = rng.permutation(fail_t.size)[:max_fail]
  fail_events = [(int(fail_t[i]), int(fail_n[i])) for i in order]

  fail_envs = set(fail_n.tolist())
  safe_envs = [n for n in range(scored["N"]) if n not in fail_envs]
  rng.shuffle(safe_envs)
  safe_envs = safe_envs[:n_safe]

  n_panels = len(fail_events) + len(safe_envs)
  n_panels = max(n_panels, 1)
  ncol = 2
  nrow = (n_panels + ncol - 1) // ncol
  fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.6 * nrow), squeeze=False)
  flat_axes = axes.ravel()

  panel = 0
  for tf, nf in fail_events:
    s, e = next(((a, b) for a, b in episode_bounds(done, nf) if a <= tf <= b), (tf, tf))
    ax = flat_axes[panel]; panel += 1
    ts = np.arange(s, e + 1) * step_dt
    ax.plot(ts, risk[s : e + 1, nf], color="C3")
    ax.axhline(threshold, ls="--", color="gray", lw=0.8)
    ax.axvspan(max(s, tf - H) * step_dt, tf * step_dt, color="C3", alpha=0.12)
    ax.axvline(tf * step_dt, color="k", lw=1.0)
    ax.set(title=f"FAIL env {nf} @ t={tf}", ylim=(-0.02, 1.02), xlabel="s", ylabel="risk")

  for nf in safe_envs:
    s, e = next(episode_bounds(done, nf), (0, scored["T"] - 1))
    ax = flat_axes[panel]; panel += 1
    ts = np.arange(s, e + 1) * step_dt
    ax.plot(ts, risk[s : e + 1, nf], color="C0")
    ax.axhline(threshold, ls="--", color="gray", lw=0.8)
    ax.set(title=f"safe env {nf}", ylim=(-0.02, 1.02), xlabel="s", ylabel="risk")

  for k in range(panel, len(flat_axes)):
    flat_axes[k].axis("off")
  fig.tight_layout()
  return fig, lead_time_stats(scored, threshold)


def build_leadtime_figure(scored: dict, *, threshold: float = 0.5):
  """Histogram of detection lead times (seconds). Returns ``(fig, stats)``."""
  import matplotlib.pyplot as plt

  stats = lead_time_stats(scored, threshold)
  leads_s = stats["leads_steps"] * scored["step_dt"]
  fig, ax = plt.subplots(figsize=(7, 4))
  if leads_s.size:
    ax.hist(leads_s, bins=20, color="C2", alpha=0.8)
    ax.axvline(stats["lead_median_s"], color="k", ls="--",
               label=f"median={stats['lead_median_s']:.2f}s")
    ax.legend()
  ax.set(title=(f"Warning lead time @ thr={threshold:.2f}  "
                f"(detected {stats['n_detected']}/{stats['n_failures']}, "
                f"FAR {stats['false_alarms_per_min']:.2f}/min)"),
         xlabel="lead time before failure (s)", ylabel="count")
  fig.tight_layout()
  return fig, stats


def build_spatial_figure(scored: dict, *, num_samples: int = 6, seed: int = 0):
  """Predicted risk map | GT label | path mask | height-scan, for sampled steps.

  Samples bias toward steps shortly before a failure (where the map should fire).
  Returns ``(fig, info)``.
  """
  import matplotlib.pyplot as plt

  model = scored["model"]
  if "risk_map" not in scored:
    raise ValueError("score_rollouts_file(..., spatial=True) is required for spatial plots.")
  risk_map = scored["risk_map"]  # [T, N, NW, NH]
  scalar_label = scored["scalar_label"]
  valid = scored["valid"]
  rng = np.random.default_rng(seed)

  # Prefer pre-failure positives; fall back to any valid step.
  cand_t, cand_n = np.nonzero((scalar_label > 0.5) & valid)
  if cand_t.size < num_samples:
    vt, vn = np.nonzero(valid)
    cand_t = np.concatenate([cand_t, vt])
    cand_n = np.concatenate([cand_n, vn])
  pick = rng.permutation(cand_t.size)[:num_samples]
  samples = [(int(cand_t[i]), int(cand_n[i])) for i in pick]

  # GT spatial labels for exactly these samples, aligned to the model grid.
  yaw_tn = quat_to_yaw(scored["root_quat_w"])
  st = np.array([t for t, _ in samples], dtype=np.int64)
  sn = np.array([n for _, n in samples], dtype=np.int64)
  gt_label, gt_mask = build_spatial_labels(
    sample_t=st,
    sample_n=sn,
    next_done_idx=scored["next_done_idx"],
    scalar_label=scalar_label,
    root_pos_w=scored["root_pos_w"],
    yaw_tn=yaw_tn,
    horizon=scored["horizon"],
    size_m=model.spatial_size_m,
    grid=model.spatial_grid,
    fail_radius=0.3,
    visit_radius=0.25,
    future_samples=8,
  )

  hs_key = model.height_scan_key
  hs_groups = split_actor_obs(scored["actor_obs"], scored["layout"], [hs_key])[hs_key]
  hw = model.input_hw

  nrow = len(samples)
  fig, axes = plt.subplots(nrow, 4, figsize=(12, 2.6 * nrow), squeeze=False)
  titles = ["pred risk map", "GT label", "path mask", "height-scan"]
  for r, (t, n) in enumerate(samples):
    panels = [
      risk_map[t, n].T,
      gt_label[r].T,
      gt_mask[r].T,
      hs_groups[t, n].reshape(hw[0], hw[1]).T,
    ]
    for c, (img, title) in enumerate(zip(panels, titles)):
      ax = axes[r, c]
      cmap = "viridis" if c < 3 else "gray"
      vmax = 1.0 if c < 3 else None
      ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, vmin=0 if c < 3 else None, vmax=vmax)
      ax.set_xticks([]); ax.set_yticks([])
      if r == 0:
        ax.set_title(title, fontsize=9)
      if c == 0:
        ax.set_ylabel(f"env {n}\n t={t}", fontsize=8)
  fig.suptitle("Spatial head: forward = up, lateral = across", fontsize=10)
  fig.tight_layout()
  return fig, {"samples": samples}
