"""Identity observation encoder (raw baseline)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.rl_models.encoders.base import BaseObservationEncoder, ObsInput, flatten_concat


class IdentityObservationEncoder(BaseObservationEncoder):
  """Flatten and concatenate the selected groups, returning the raw vector.

  Equivalent to "raw observation + MLP": there are no learnable parameters, so
  ``output_dim`` equals the total flattened input dimension.
  """

  def __init__(
    self,
    encoder_input_keys: Sequence[str],
    input_dim: int,
    flatten: bool = True,
  ) -> None:
    super().__init__()
    if not flatten:
      raise ValueError("IdentityObservationEncoder only supports flatten=True.")
    self.encoder_input_keys = list(encoder_input_keys)
    self.output_dim = int(input_dim)

  def forward(self, obs: ObsInput) -> torch.Tensor:
    return flatten_concat(obs, self.encoder_input_keys)
