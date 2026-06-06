"""Small autoencoder modules used by pretrained observation encoders."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from rsl_rl.modules import MLP


class HeightScanAutoEncoder(nn.Module):
  """MLP autoencoder for flat height-scan observations."""

  def __init__(
    self,
    input_dim: int = 187,
    latent_dim: int = 32,
    hidden_dims: Sequence[int] = (256, 128),
    decoder_hidden_dims: Sequence[int] | None = None,
    activation: str = "elu",
  ) -> None:
    super().__init__()
    self.input_dim = int(input_dim)
    self.latent_dim = int(latent_dim)
    enc_hidden = list(hidden_dims)
    dec_hidden = (
      list(decoder_hidden_dims)
      if decoder_hidden_dims is not None
      else list(reversed(enc_hidden))
    )
    self.encoder = MLP(self.input_dim, self.latent_dim, enc_hidden, activation)
    self.decoder = MLP(self.latent_dim, self.input_dim, dec_hidden, activation)

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """Return the latent representation for ``x``."""
    return self.encoder(x.reshape(x.shape[0], -1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.decoder(self.encode(x))
