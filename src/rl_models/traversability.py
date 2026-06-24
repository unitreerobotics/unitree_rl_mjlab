"""Policy-conditioned traversability estimator.

A small network that predicts, for a *fixed chosen locomotion policy*, how likely
that policy is to fail soon given the current (deployable) observation. It reuses
the project's observation-encoder framework (``build_observation_encoder``) so the
height-scan branch and the proprioception fusion match the encoders used by the
policies themselves.

Two heads on a shared encoder:

* ``scalar_logit`` -> ``P(failure within H steps)`` (core).
* ``spatial_logit`` -> per-cell failure map over a CONFIGURABLE robot-frame grid
  (``spatial_grid = (NW, NH)`` cells covering ``spatial_size_m = (W, H)`` metres),
  independent of the fixed height-scan input grid.

The input feature set is configurable: ``encoder_input_keys`` selects which
observation groups feed the estimator (e.g. ``height_scan`` + a subset of
proprioception). The checkpoint stores everything needed to rebuild the model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.rl_models.encoders.base import build_mlp
from src.rl_models.encoders.builder import build_observation_encoder

HEIGHT_SCAN_KEY = "height_scan"


def _default_encoder_cfg(
  encoder_input_keys: Sequence[str],
  height_scan_key: str,
  input_hw: Sequence[int],
  latent_dim: int,
  activation: str,
) -> dict[str, Any]:
  """Conv2d over the height-scan grid, fusing the remaining groups as context.

  Falls back to a plain MLP encoder when no height-scan group is selected.
  """
  keys = list(encoder_input_keys)
  if height_scan_key in keys:
    context_keys = [k for k in keys if k != height_scan_key]
    return {
      "type": "conv2d",
      "primary_key": height_scan_key,
      "latent_dim": latent_dim,
      "context_keys": context_keys,
      "input_hw": list(input_hw),
      "activation": activation,
    }
  return {"type": "mlp", "latent_dim": latent_dim, "activation": activation}


class TraversabilityEstimator(nn.Module):
  def __init__(
    self,
    obs_shapes: Mapping[str, tuple[int, ...]],
    encoder_input_keys: Sequence[str],
    *,
    height_scan_key: str = HEIGHT_SCAN_KEY,
    input_hw: Sequence[int] = (17, 11),
    spatial_grid: Sequence[int] = (20, 10),
    spatial_size_m: Sequence[float] = (2.0, 1.0),
    encoder_cfg: Mapping[str, Any] | None = None,
    latent_dim: int = 64,
    scalar_hidden: Sequence[int] = (128, 64),
    spatial_hidden: Sequence[int] = (128,),
    activation: str = "elu",
  ) -> None:
    super().__init__()
    self.encoder_input_keys = list(encoder_input_keys)
    self.obs_shapes = {k: tuple(v) for k, v in obs_shapes.items()}
    self.height_scan_key = height_scan_key
    self.input_hw = tuple(input_hw)
    self.spatial_grid = tuple(int(x) for x in spatial_grid)
    self.spatial_size_m = tuple(float(x) for x in spatial_size_m)
    self.latent_dim = int(latent_dim)
    self.scalar_hidden = list(scalar_hidden)
    self.spatial_hidden = list(spatial_hidden)
    self.activation = activation

    cfg = (
      dict(encoder_cfg)
      if encoder_cfg is not None
      else _default_encoder_cfg(
        self.encoder_input_keys, height_scan_key, self.input_hw, latent_dim, activation
      )
    )
    self._encoder_cfg = cfg
    self.encoder = build_observation_encoder(cfg, self.obs_shapes, self.encoder_input_keys)

    out_dim = self.encoder.output_dim
    n_cells = self.spatial_grid[0] * self.spatial_grid[1]
    self.scalar_head = build_mlp(out_dim, 1, self.scalar_hidden, activation)
    self.spatial_head = build_mlp(out_dim, n_cells, self.spatial_hidden, activation)

  def forward(self, obs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    z = self.encoder(obs)
    nw, nh = self.spatial_grid
    return {
      "scalar_logit": self.scalar_head(z).squeeze(-1),
      "spatial_logit": self.spatial_head(z).reshape(-1, nw, nh),
    }

  @torch.no_grad()
  def predict_proba(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return ``P(failure soon)`` in ``[0, 1]`` for the scalar head."""
    return torch.sigmoid(self.forward(obs)["scalar_logit"])

  @torch.no_grad()
  def predict_spatial_proba(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return the per-cell failure probabilities, shape ``[B, NW, NH]``."""
    return torch.sigmoid(self.forward(obs)["spatial_logit"])

  def model_kwargs(self) -> dict[str, Any]:
    return {
      "obs_shapes": self.obs_shapes,
      "encoder_input_keys": self.encoder_input_keys,
      "height_scan_key": self.height_scan_key,
      "input_hw": list(self.input_hw),
      "spatial_grid": list(self.spatial_grid),
      "spatial_size_m": list(self.spatial_size_m),
      "encoder_cfg": self._encoder_cfg,
      "latent_dim": self.latent_dim,
      "scalar_hidden": self.scalar_hidden,
      "spatial_hidden": self.spatial_hidden,
      "activation": self.activation,
    }

  def save(self, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
      {
        "state_dict": self.state_dict(),
        "model_class": "src.rl_models.traversability:TraversabilityEstimator",
        "model_kwargs": self.model_kwargs(),
      },
      path,
    )


def load_traversability_estimator(
  path: str | Path, map_location: Any = "cpu"
) -> TraversabilityEstimator:
  """Rebuild a :class:`TraversabilityEstimator` from a saved checkpoint."""
  ckpt = torch.load(path, map_location=map_location, weights_only=False)
  model = TraversabilityEstimator(**ckpt["model_kwargs"])
  model.load_state_dict(ckpt["state_dict"])
  model.eval()
  return model
