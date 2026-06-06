"""Train a small height-scan autoencoder checkpoint.

The output checkpoint can be used by the ``pretrained_ae`` observation encoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.rl_models.autoencoder import HeightScanAutoEncoder


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", required=True, help="Input .pt/.pth tensor or .npy/.npz array.")
  parser.add_argument("--output", required=True, help="Checkpoint path to write.")
  parser.add_argument("--npz-key", default=None, help="Array key when --input is .npz.")
  parser.add_argument("--input-dim", type=int, default=187)
  parser.add_argument("--latent-dim", type=int, default=32)
  parser.add_argument("--hidden-dims", type=int, nargs="*", default=[256, 128])
  parser.add_argument("--decoder-hidden-dims", type=int, nargs="*", default=None)
  parser.add_argument("--activation", default="elu")
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--lr", type=float, default=1.0e-3)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return parser.parse_args()


def load_samples(path: str | Path, npz_key: str | None) -> torch.Tensor:
  path = Path(path)
  suffix = path.suffix.lower()
  if suffix in {".pt", ".pth"}:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
      for key in (npz_key, "height_scan", "samples", "data"):
        if key is not None and key in data:
          data = data[key]
          break
    samples = torch.as_tensor(data, dtype=torch.float32)
  elif suffix == ".npy":
    samples = torch.as_tensor(np.load(path), dtype=torch.float32)
  elif suffix == ".npz":
    archive = np.load(path)
    key = npz_key or ("height_scan" if "height_scan" in archive else archive.files[0])
    samples = torch.as_tensor(archive[key], dtype=torch.float32)
  else:
    raise ValueError(f"Unsupported input suffix '{suffix}'. Use .pt, .pth, .npy, or .npz.")

  if samples.ndim < 2:
    raise ValueError(f"Expected samples with shape [N, ...]; got {tuple(samples.shape)}.")
  return samples.reshape(samples.shape[0], -1)


def main() -> None:
  args = parse_args()
  torch.manual_seed(args.seed)

  samples = load_samples(args.input, args.npz_key)
  if samples.shape[1] != args.input_dim:
    raise ValueError(
      f"Loaded flattened dim {samples.shape[1]}, but --input-dim is {args.input_dim}."
    )

  device = torch.device(args.device)
  model = HeightScanAutoEncoder(
    input_dim=args.input_dim,
    latent_dim=args.latent_dim,
    hidden_dims=args.hidden_dims,
    decoder_hidden_dims=args.decoder_hidden_dims,
    activation=args.activation,
  ).to(device)
  loader = DataLoader(
    TensorDataset(samples),
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
  )
  optim = torch.optim.Adam(model.parameters(), lr=args.lr)

  model.train()
  for epoch in range(1, args.epochs + 1):
    total_loss = 0.0
    total_count = 0
    for (batch,) in loader:
      batch = batch.to(device)
      recon = model(batch)
      loss = F.mse_loss(recon, batch)
      optim.zero_grad()
      loss.backward()
      optim.step()
      total_loss += float(loss.detach().cpu()) * batch.shape[0]
      total_count += batch.shape[0]
    print(f"epoch {epoch:04d} mse {total_loss / max(total_count, 1):.8f}")

  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      "state_dict": model.cpu().state_dict(),
      "model_class": "src.rl_models.autoencoder:HeightScanAutoEncoder",
      "model_kwargs": {
        "input_dim": args.input_dim,
        "latent_dim": args.latent_dim,
        "hidden_dims": list(args.hidden_dims),
        "decoder_hidden_dims": args.decoder_hidden_dims,
        "activation": args.activation,
      },
    },
    output,
  )
  print(f"saved {output}")


if __name__ == "__main__":
  main()
