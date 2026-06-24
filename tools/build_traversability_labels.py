"""Build traversability labels from collected policy rollouts.

Consumes the ``.npz`` written by ``tools/collect_traversability.py`` and produces
a labeled ``.npz`` for ``tools/train_traversability.py``.

Two labels per timestep, both derived from the policy's own future:

* **scalar** (core) -- short-horizon failure risk:
  ``label[t] = 1`` iff the policy fails (``fell_over`` / ``illegal_contact``)
  within the next ``H`` steps *of the same episode*, else 0. The lookahead window
  is clipped at the episode boundary, so labels never leak across the auto-reset.

* **spatial** (extension, ``--spatial``) -- per-cell failure map on a CONFIGURABLE
  robot-frame grid (``--spatial-size-m W H`` metres, ``--spatial-grid NW NH`` cells,
  independent of the 17x11 height-scan input). Each cell is back-projected to a
  world location; a cell is labelled 1 if the upcoming failure happens near it, and
  ``mask=1`` only for cells the robot actually traverses within the window (masked
  loss). This yields a forward-looking, path-aligned risk map.

Example:
    python tools/build_traversability_labels.py \
        --input logs/traversability/raw_rollouts.npz \
        --horizon 75 --spatial --spatial-size-m 2.0 1.0 --spatial-grid 20 10 \
        --output logs/traversability/labels.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--input", required=True, type=Path)
  p.add_argument("--output", required=True, type=Path)
  p.add_argument(
    "--horizon",
    type=int,
    default=None,
    help="Failure lookahead H (steps). Defaults to the value stored by the collector.",
  )
  p.add_argument("--spatial", action="store_true", help="Also build spatial per-cell labels.")
  p.add_argument(
    "--spatial-size-m",
    type=float,
    nargs=2,
    metavar=("W", "H"),
    default=[2.0, 1.0],
    help="Real-world extent of the spatial map (forward, lateral) in metres.",
  )
  p.add_argument(
    "--spatial-grid",
    type=int,
    nargs=2,
    metavar=("NW", "NH"),
    default=[20, 10],
    help="Spatial map resolution (forward cells, lateral cells).",
  )
  p.add_argument(
    "--fail-radius",
    type=float,
    default=0.3,
    help="Cells within this radius (m) of the failure location are labelled 1.",
  )
  p.add_argument(
    "--visit-radius",
    type=float,
    default=0.25,
    help="Cells within this radius (m) of the future path get mask=1.",
  )
  p.add_argument(
    "--spatial-future-samples",
    type=int,
    default=8,
    help="Number of future poses sampled per step to build the path mask.",
  )
  p.add_argument(
    "--balance",
    choices=["none", "subsample"],
    default="none",
    help="'subsample' caps negatives at --neg-per-pos x positives.",
  )
  p.add_argument("--neg-per-pos", type=float, default=20.0)
  p.add_argument("--seed", type=int, default=0)
  return p.parse_args()


def quat_to_yaw(q: np.ndarray) -> np.ndarray:
  """Yaw (rad) from wxyz quaternions; ``q`` has shape ``[..., 4]``."""
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def parse_layout(layout_json: str) -> list[tuple[str, slice]]:
  layout = json.loads(str(layout_json))
  names = layout["term_names"]
  dims = layout["term_dims"]
  slices: list[tuple[str, slice]] = []
  offset = 0
  for name, dim in zip(names, dims):
    slices.append((name, slice(offset, offset + int(dim))))
    offset += int(dim)
  return slices


def compute_episode_geometry(done: np.ndarray, failure: np.ndarray, horizon: int):
  """Per-timestep episode end + scalar label + validity, vectorised over envs.

  Returns ``(next_done_idx, scalar_label, valid)`` each of shape ``[T, N]``.
  ``next_done_idx[t] = -1`` marks the trailing open episode (no terminating done).
  """
  T, N = done.shape
  next_done_idx = np.full((T, N), -1, dtype=np.int64)
  nd = np.full(N, -1, dtype=np.int64)
  for t in range(T - 1, -1, -1):
    nd = np.where(done[t], t, nd)
    next_done_idx[t] = nd

  t_grid = np.arange(T, dtype=np.int64)[:, None]
  has_end = next_done_idx >= 0
  gap = np.where(has_end, next_done_idx - t_grid, 0)

  # failure flag at the (clipped) episode end.
  env_idx = np.arange(N, dtype=np.int64)[None, :].repeat(T, axis=0)
  end_for_gather = np.where(has_end, next_done_idx, 0)
  fail_at_end = failure[end_for_gather, env_idx] & has_end

  scalar_label = (fail_at_end & (gap < horizon)).astype(np.float32)

  # Closed episodes: outcome within the horizon is fully determined.
  # Open trailing episode: only valid if the whole H-step window is recorded.
  fully_observed_open = (~has_end) & ((t_grid + horizon) <= T)
  valid = has_end | fully_observed_open
  return next_done_idx, scalar_label, valid


def build_spatial_labels(
  *,
  sample_t: np.ndarray,
  sample_n: np.ndarray,
  next_done_idx: np.ndarray,
  scalar_label: np.ndarray,
  root_pos_w: np.ndarray,
  yaw_tn: np.ndarray,
  horizon: int,
  size_m: tuple[float, float],
  grid: tuple[int, int],
  fail_radius: float,
  visit_radius: float,
  future_samples: int,
):
  """Per-cell failure labels + path mask for each emitted sample.

  Returns ``(label_spatial, mask_spatial)`` of shape ``[M, NW, NH]``.
  """
  M = sample_t.shape[0]
  T = root_pos_w.shape[0]
  W, Hm = size_m
  NW, NH = grid
  P = NW * NH

  # Fixed robot-frame cell centres: x forward (NW cells over W), y lateral (NH over Hm).
  res_w, res_h = W / NW, Hm / NH
  xs = (np.arange(NW) + 0.5) * res_w - W / 2.0
  ys = (np.arange(NH) + 0.5) * res_h - Hm / 2.0
  cx, cy = np.meshgrid(xs, ys, indexing="ij")  # [NW, NH]
  cells_b = np.stack([cx.reshape(-1), cy.reshape(-1)], axis=1)  # [P, 2]

  pos2d = root_pos_w[sample_t, sample_n, :2]  # [M, 2]
  yaw = yaw_tn[sample_t, sample_n]  # [M]
  end = np.where(next_done_idx[sample_t, sample_n] >= 0,
                 next_done_idx[sample_t, sample_n], T - 1)  # [M]
  fail_in_window = scalar_label[sample_t, sample_n] > 0.5  # [M]
  fail_pos = root_pos_w[end, sample_n, :2]  # [M, 2]

  # Future pose samples along (t, min(t+H-1, end)] for the path mask.
  win_end = np.minimum(sample_t + horizon - 1, end)  # [M]
  S = max(1, future_samples)
  frac = np.linspace(0.0, 1.0, S)[None, :]  # [1, S]
  start = sample_t + 1
  span = np.maximum(win_end - start, 0)[:, None]  # [M, 1]
  fut_idx = np.clip(np.round(start[:, None] + frac * span).astype(np.int64), 0, T - 1)
  empty = (win_end < start)[:, None]
  fut_idx = np.where(empty, sample_t[:, None], fut_idx)  # degenerate -> current pose
  fut_pos = root_pos_w[fut_idx, sample_n[:, None], :2]  # [M, S, 2]

  label = np.zeros((M, P), dtype=np.float32)
  mask = np.zeros((M, P), dtype=np.float32)
  fail_r2, visit_r2 = fail_radius**2, visit_radius**2

  chunk = 4096
  for s in range(0, M, chunk):
    e = min(s + chunk, M)
    c = e - s
    cos, sin = np.cos(yaw[s:e]), np.sin(yaw[s:e])
    rot = np.stack([np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], axis=1)  # [c,2,2]
    cells_world = pos2d[s:e, None, :] + np.einsum("cij,pj->cpi", rot, cells_b)  # [c,P,2]

    d_path2 = ((cells_world[:, :, None, :] - fut_pos[s:e, None, :, :]) ** 2).sum(-1)  # [c,P,S]
    mask[s:e] = (d_path2.min(2) < visit_r2).astype(np.float32)

    d_fail2 = ((cells_world - fail_pos[s:e, None, :]) ** 2).sum(-1)  # [c,P]
    lab = fail_in_window[s:e, None] & (d_fail2 < fail_r2)
    label[s:e] = (lab & (mask[s:e] > 0.5)).astype(np.float32)

  return label.reshape(M, NW, NH), mask.reshape(M, NW, NH)


def main() -> None:
  args = parse_args()
  rng = np.random.default_rng(args.seed)

  data = np.load(args.input, allow_pickle=True)
  actor_obs = data["actor_obs"]  # [T, N, A]
  root_pos_w = data["root_pos_w"]  # [T, N, 3]
  root_quat_w = data["root_quat_w"]  # [T, N, 4]
  done = data["done"].astype(bool)  # [T, N]
  failure = data["failure"].astype(bool)  # [T, N]
  layout = parse_layout(data["actor_layout"])
  horizon = int(args.horizon if args.horizon is not None else data["horizon"])
  T, N, _ = actor_obs.shape
  print(f"[LABEL] loaded T={T} N={N} horizon={horizon}")

  next_done_idx, scalar_label, valid = compute_episode_geometry(done, failure, horizon)

  t_idx, n_idx = np.nonzero(valid)
  labels = scalar_label[t_idx, n_idx]

  if args.balance == "subsample":
    pos = labels > 0.5
    n_pos = int(pos.sum())
    keep_neg = int(min((~pos).sum(), round(args.neg_per_pos * max(n_pos, 1))))
    neg_all = np.nonzero(~pos)[0]
    neg_keep = rng.choice(neg_all, size=keep_neg, replace=False)
    sel = np.sort(np.concatenate([np.nonzero(pos)[0], neg_keep]))
    t_idx, n_idx, labels = t_idx[sel], n_idx[sel], labels[sel]

  M = t_idx.shape[0]
  n_pos = int((labels > 0.5).sum())
  n_neg = M - n_pos
  pos_weight = float(n_neg / max(n_pos, 1))
  print(
    f"[LABEL] samples={M} pos={n_pos} neg={n_neg} "
    f"pos_frac={n_pos / max(M, 1):.4f} pos_weight={pos_weight:.2f}"
  )
  if n_pos == 0:
    print("[LABEL][WARN] no positive samples -- collect more/harder rollouts.")

  out: dict[str, np.ndarray] = {"label_scalar": labels.astype(np.float32)}
  for name, sl in layout:
    out[name] = actor_obs[t_idx, n_idx, sl].astype(np.float32)

  attrs = {
    "pos_weight": np.float32(pos_weight),
    "horizon": np.int64(horizon),
    "input_keys": json.dumps([name for name, _ in layout]),
    "checkpoint": str(data["checkpoint"]) if "checkpoint" in data else "",
    "task": str(data["task"]) if "task" in data else "",
  }

  if args.spatial:
    yaw_tn = quat_to_yaw(root_quat_w)  # [T, N]
    label_spatial, mask_spatial = build_spatial_labels(
      sample_t=t_idx,
      sample_n=n_idx,
      next_done_idx=next_done_idx,
      scalar_label=scalar_label,
      root_pos_w=root_pos_w,
      yaw_tn=yaw_tn,
      horizon=horizon,
      size_m=tuple(args.spatial_size_m),
      grid=tuple(args.spatial_grid),
      fail_radius=args.fail_radius,
      visit_radius=args.visit_radius,
      future_samples=args.spatial_future_samples,
    )
    out["label_spatial"] = label_spatial
    out["mask_spatial"] = mask_spatial
    attrs["spatial_size_m"] = np.asarray(args.spatial_size_m, dtype=np.float32)
    attrs["spatial_grid"] = np.asarray(args.spatial_grid, dtype=np.int64)
    cells_supervised = float(mask_spatial.mean())
    cells_positive = float(label_spatial.mean())
    print(
      f"[LABEL] spatial grid={tuple(args.spatial_grid)} size_m={tuple(args.spatial_size_m)} "
      f"mask_frac={cells_supervised:.4f} pos_cell_frac={cells_positive:.5f}"
    )

  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(args.output, **out, **attrs)
  print(f"[LABEL] saved {args.output}")


if __name__ == "__main__":
  main()
