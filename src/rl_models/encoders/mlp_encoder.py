"""MLP feature encoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from rsl_rl.modules import MLP

from src.rl_models.encoders.base import BaseObservationEncoder, ObsInput, flatten_concat


class MLPObservationEncoder(BaseObservationEncoder):
  """Flatten + concatenate selected groups, then encode with an MLP.

  Also supports state-conditioned terrain encoding: list robot-state groups
  (command, projected_gravity, ...) alongside ``height_scan`` in
  ``encoder_input_keys`` and they are concatenated before the MLP.
  """

  def __init__(
    self,
    encoder_input_keys: Sequence[str],
    input_dim: int,
    latent_dim: int,
    hidden_dims: Sequence[int] = (256, 128),
    activation: str = "elu",
    layer_norm: bool = False,
  ) -> None:
    super().__init__()
    self.encoder_input_keys = list(encoder_input_keys)
    self.mlp = MLP(input_dim, latent_dim, list(hidden_dims), activation)
    self.norm = nn.LayerNorm(latent_dim) if layer_norm else nn.Identity()
    self.output_dim = int(latent_dim)

  def forward(self, obs: ObsInput) -> torch.Tensor:
    x = flatten_concat(obs, self.encoder_input_keys)
    return self.norm(self.mlp(x))
