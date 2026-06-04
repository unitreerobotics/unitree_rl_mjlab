"""Base class and shared helpers for observation encoders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
from rsl_rl.modules import MLP

ObsInput = Mapping[str, torch.Tensor] | torch.Tensor


def build_mlp(
  in_dim: int, out_dim: int, hidden_dims: Sequence[int], activation: str
) -> nn.Module:
  """MLP mapping ``in_dim`` -> ``out_dim``; a single ``Linear`` if no hidden dims."""
  if hidden_dims:
    return MLP(in_dim, out_dim, list(hidden_dims), activation)
  return nn.Linear(in_dim, out_dim)


class BaseObservationEncoder(nn.Module):
  """Common interface for observation encoders.

  Subclasses must set ``self.output_dim`` (an ``int``) in ``__init__`` and
  return a tensor of shape ``[batch, output_dim]`` from ``forward``.

  The model wrapper only relies on ``output_dim`` and ``forward`` -- it does not
  care which encoder type is used.
  """

  output_dim: int

  def forward(self, obs: ObsInput) -> torch.Tensor:  # pragma: no cover - abstract
    raise NotImplementedError


def flatten_concat(obs: ObsInput, keys: Sequence[str]) -> torch.Tensor:
  """Flatten each selected group to ``[B, -1]`` and concatenate in ``keys`` order.

  Accepts either a mapping of group name -> tensor or, when ``keys`` has a single
  entry, a bare tensor.
  """
  tensors = _gather(obs, keys)
  flat = [t.reshape(t.shape[0], -1) for t in tensors]
  return torch.cat(flat, dim=-1)


def concat_dim(obs_shapes: Mapping[str, tuple[int, ...]], keys: Sequence[str]) -> int:
  """Total flattened feature dimension of ``keys`` given per-group shapes."""
  total = 0
  for key in keys:
    shape = obs_shapes[key]
    dim = 1
    for s in shape:
      dim *= s
    total += dim
  return total


def _gather(obs: ObsInput, keys: Sequence[str]) -> list[torch.Tensor]:
  if isinstance(obs, torch.Tensor):
    if len(keys) != 1:
      raise ValueError(
        f"Encoder received a bare tensor but expected {len(keys)} groups {list(keys)}."
      )
    return [obs]
  missing = [k for k in keys if k not in obs]
  if missing:
    raise KeyError(f"Encoder inputs missing groups {missing}; available: {list(obs)}.")
  return [obs[k] for k in keys]


def get_group(obs: ObsInput, key: str) -> torch.Tensor:
  """Fetch a single group tensor from a mapping or bare tensor."""
  if isinstance(obs, torch.Tensor):
    return obs
  if key not in obs:
    raise KeyError(f"Encoder input missing group '{key}'; available: {list(obs)}.")
  return obs[key]
