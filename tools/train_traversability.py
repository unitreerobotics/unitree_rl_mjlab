"""Train a policy-conditioned traversability estimator.

Consumes the ``labels.npz`` from ``tools/build_traversability_labels.py`` and
trains :class:`~src.rl_models.traversability.TraversabilityEstimator`.

The scalar head (``P(failure soon)``) is the core; the spatial head is trained
only when ``--spatial-weight > 0`` (and the labels contain spatial targets). The
input feature set is configurable via ``--input-keys`` (a subset of the groups
stored in the labels file).

Examples:
    # scalar head only (default input set = whatever the labeler stored)
    python tools/train_traversability.py \
        --input logs/traversability/labels.npz \
        --output logs/traversability/traversability.pt --epochs 50

    # enable the spatial head
    python tools/train_traversability.py --input logs/traversability/labels.npz \
        --output logs/traversability/traversability_spatial.pt --spatial-weight 1.0

    # evaluate a trained checkpoint on a held-out split
    python tools/train_traversability.py --input logs/traversability/labels.npz \
        --eval-only --checkpoint logs/traversability/traversability.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.rl_models.traversability import (
  TraversabilityEstimator,
  load_traversability_estimator,
)


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--input", required=True, type=Path)
  p.add_argument("--output", type=Path, default=None, help="Checkpoint path to write.")
  p.add_argument(
    "--input-keys",
    nargs="*",
    default=None,
    help="Observation groups to feed the estimator. Default: all groups in the labels file.",
  )
  p.add_argument("--latent-dim", type=int, default=64)
  p.add_argument("--scalar-hidden", type=int, nargs="*", default=[128, 64])
  p.add_argument("--spatial-hidden", type=int, nargs="*", default=[128])
  p.add_argument("--activation", default="elu")
  p.add_argument("--input-hw", type=int, nargs=2, default=[17, 11])
  p.add_argument("--spatial-weight", type=float, default=0.0)
  p.add_argument("--pos-weight", type=float, default=None, help="Default: from labels attr.")
  p.add_argument("--batch-size", type=int, default=512)
  p.add_argument("--epochs", type=int, default=50)
  p.add_argument("--lr", type=float, default=1.0e-3)
  p.add_argument("--val-frac", type=float, default=0.15)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  p.add_argument("--eval-only", action="store_true")
  p.add_argument("--checkpoint", type=Path, default=None, help="Model to load for --eval-only.")
  return p.parse_args()


def _binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
  """ROC-AUC, PR-AUC (average precision), accuracy@0.5 and Brier score."""
  labels = labels.astype(np.int64)
  n_pos = int(labels.sum())
  n_neg = labels.shape[0] - n_pos
  acc = float(((scores >= 0.5).astype(np.int64) == labels).mean())
  brier = float(np.mean((scores - labels) ** 2))
  if n_pos == 0 or n_neg == 0:
    return {"auc": float("nan"), "ap": float("nan"), "acc": acc, "brier": brier}

  order = np.argsort(scores, kind="mergesort")
  ranks = np.empty_like(order, dtype=np.float64)
  ranks[order] = np.arange(1, len(scores) + 1)
  auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

  desc = np.argsort(-scores, kind="mergesort")
  y = labels[desc]
  tp = np.cumsum(y)
  precision = tp / np.arange(1, len(y) + 1)
  recall = tp / n_pos
  rec_prev = np.concatenate([[0.0], recall[:-1]])
  ap = float(np.sum((recall - rec_prev) * precision))
  return {"auc": float(auc), "ap": ap, "acc": acc, "brier": brier}


def _load_dataset(args):
  data = np.load(args.input, allow_pickle=True)
  stored_keys = json.loads(str(data["input_keys"])) if "input_keys" in data else None
  keys = list(args.input_keys) if args.input_keys else stored_keys
  if keys is None:
    raise ValueError("--input-keys not given and labels file has no 'input_keys' attr.")
  missing = [k for k in keys if k not in data]
  if missing:
    raise KeyError(f"Requested input keys {missing} not in labels file {list(data.keys())}.")

  inputs = [torch.from_numpy(data[k].astype(np.float32)) for k in keys]
  obs_shapes = {k: (int(t.shape[1]),) for k, t in zip(keys, inputs)}
  y_scalar = torch.from_numpy(data["label_scalar"].astype(np.float32))

  has_spatial = "label_spatial" in data and args.spatial_weight > 0.0
  spatial_grid = (
    tuple(int(x) for x in data["spatial_grid"])
    if "spatial_grid" in data
    else (args.input_hw[0], args.input_hw[1])
  )
  spatial_size_m = (
    tuple(float(x) for x in data["spatial_size_m"])
    if "spatial_size_m" in data
    else (2.0, 1.0)
  )

  tensors = [*inputs, y_scalar]
  if has_spatial:
    tensors.append(torch.from_numpy(data["label_spatial"].astype(np.float32)))
    tensors.append(torch.from_numpy(data["mask_spatial"].astype(np.float32)))

  pos_weight = (
    args.pos_weight
    if args.pos_weight is not None
    else (float(data["pos_weight"]) if "pos_weight" in data else 1.0)
  )
  meta = {
    "keys": keys,
    "obs_shapes": obs_shapes,
    "n_inputs": len(inputs),
    "has_spatial": has_spatial,
    "spatial_grid": spatial_grid,
    "spatial_size_m": spatial_size_m,
    "pos_weight": pos_weight,
  }
  return TensorDataset(*tensors), meta


def _split_batch(batch, meta):
  n = meta["n_inputs"]
  obs = {k: batch[i] for i, k in enumerate(meta["keys"])}
  y_scalar = batch[n]
  if meta["has_spatial"]:
    return obs, y_scalar, batch[n + 1], batch[n + 2]
  return obs, y_scalar, None, None


def _to_device(batch, device):
  return [t.to(device) for t in batch]


@torch.no_grad()
def evaluate(model, loader, meta, device) -> dict[str, float]:
  model.eval()
  scores, labels = [], []
  for batch in loader:
    obs, y_scalar, _, _ = _split_batch(_to_device(batch, device), meta)
    scores.append(torch.sigmoid(model(obs)["scalar_logit"]).cpu().numpy())
    labels.append(y_scalar.cpu().numpy())
  return _binary_metrics(np.concatenate(scores), np.concatenate(labels))


def main() -> None:
  args = parse_args()
  torch.manual_seed(args.seed)
  device = torch.device(args.device)

  dataset, meta = _load_dataset(args)
  n_val = int(len(dataset) * args.val_frac)
  n_train = len(dataset) - n_val
  gen = torch.Generator().manual_seed(args.seed)
  train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)
  train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
  val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
  print(
    f"[TRAIN] samples={len(dataset)} train={n_train} val={n_val} "
    f"keys={meta['keys']} pos_weight={meta['pos_weight']:.2f} spatial={meta['has_spatial']}"
  )

  if args.eval_only:
    if args.checkpoint is None:
      raise ValueError("--eval-only requires --checkpoint.")
    model = load_traversability_estimator(args.checkpoint, map_location=device).to(device)
    metrics = evaluate(model, val_loader, meta, device)
    print(f"[EVAL] val {metrics}")
    return

  model = TraversabilityEstimator(
    obs_shapes=meta["obs_shapes"],
    encoder_input_keys=meta["keys"],
    input_hw=tuple(args.input_hw),
    spatial_grid=meta["spatial_grid"],
    spatial_size_m=meta["spatial_size_m"],
    latent_dim=args.latent_dim,
    scalar_hidden=args.scalar_hidden,
    spatial_hidden=args.spatial_hidden,
    activation=args.activation,
  ).to(device)
  optim = torch.optim.Adam(model.parameters(), lr=args.lr)
  pos_weight = torch.tensor(meta["pos_weight"], device=device)

  for epoch in range(1, args.epochs + 1):
    model.train()
    total, count = 0.0, 0
    for batch in train_loader:
      obs, y_scalar, y_sp, m_sp = _split_batch(_to_device(batch, device), meta)
      out = model(obs)
      loss = F.binary_cross_entropy_with_logits(
        out["scalar_logit"], y_scalar, pos_weight=pos_weight
      )
      if meta["has_spatial"]:
        per_cell = F.binary_cross_entropy_with_logits(
          out["spatial_logit"], y_sp, reduction="none"
        )
        denom = m_sp.sum().clamp_min(1.0)
        loss = loss + args.spatial_weight * (per_cell * m_sp).sum() / denom
      optim.zero_grad()
      loss.backward()
      optim.step()
      bs = y_scalar.shape[0]
      total += float(loss.detach()) * bs
      count += bs
    metrics = evaluate(model, val_loader, meta, device)
    print(
      f"[TRAIN] epoch {epoch:04d} loss {total / max(count, 1):.4f}  "
      f"val auc {metrics['auc']:.4f} ap {metrics['ap']:.4f} "
      f"acc {metrics['acc']:.4f} brier {metrics['brier']:.4f}"
    )

  if args.output is not None:
    model.save(args.output)
    print(f"[TRAIN] saved {args.output}")


if __name__ == "__main__":
  main()
