"""1D convolutional encoder for ordered height-scan vectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
from rsl_rl.utils import resolve_nn_activation

from src.rl_models.encoders.base import (
  BaseObservationEncoder,
  ObsInput,
  build_mlp,
  flatten_concat,
  get_group,
)


class Conv1dObservationEncoder(BaseObservationEncoder):
  """Conv1d over a primary 1D group, optionally fused with context groups.

  The primary group (e.g. ``height_scan``) is processed with a Conv1d stack and
  globally pooled to a fixed vector. Optional context groups (e.g. ``command``,
  ``projected_gravity``) go through a small MLP. Both latents are concatenated
  and projected to ``latent_dim``.
  """

  def __init__(
    self,
    encoder_input_keys: Sequence[str],
    obs_shapes: Mapping[str, tuple[int, ...]],
    primary_key: str,
    latent_dim: int,
    context_keys: Sequence[str] | None = None,
    input_channels: int = 1,
    channels: Sequence[int] = (16, 32, 64),
    kernel_sizes: Sequence[int] = (5, 3, 3),
    strides: Sequence[int] = (2, 2, 1),
    activation: str = "elu",
    global_pool: str = "avg",
    context_hidden_dims: Sequence[int] = (64,),
  ) -> None:
    super().__init__()
    if not (len(channels) == len(kernel_sizes) == len(strides)):
      raise ValueError(
        "channels, kernel_sizes and strides must have equal length; got "
        f"{len(channels)}, {len(kernel_sizes)}, {len(strides)}."
      )
    self.primary_key = primary_key
    self.context_keys = list(context_keys or [])

    c_in = _primary_channels(obs_shapes[primary_key], input_channels, primary_key)
    self._primary_ndim = len(obs_shapes[primary_key])

    act = resolve_nn_activation(activation)
    layers: list[nn.Module] = []
    prev = c_in
    for out_ch, k, s in zip(channels, kernel_sizes, strides):
      layers += [nn.Conv1d(prev, out_ch, kernel_size=k, stride=s), act]
      prev = out_ch
    self.conv = nn.Sequential(*layers)
    self.pool = _global_pool_1d(global_pool)
    conv_dim = int(channels[-1])

    ctx_dim = 0
    self.context_mlp: nn.Module | None = None
    if self.context_keys:
      ctx_in = sum(_flat_dim(obs_shapes[k]) for k in self.context_keys)
      ctx_dim = int(context_hidden_dims[-1])
      self.context_mlp = build_mlp(
        ctx_in, ctx_dim, list(context_hidden_dims[:-1]), activation
      )

    self.proj = nn.Linear(conv_dim + ctx_dim, latent_dim)
    self.output_dim = int(latent_dim)

  def forward(self, obs: ObsInput) -> torch.Tensor:
    x = get_group(obs, self.primary_key)
    if x.dim() == 2:  # [B, L] -> [B, 1, L]
      x = x.unsqueeze(1)
    elif x.dim() != 3:  # expect [B, C, L]
      raise ValueError(
        f"Conv1d primary '{self.primary_key}' must be [B, L] or [B, C, L]; got "
        f"{tuple(x.shape)}."
      )
    x = self.pool(self.conv(x)).flatten(1)
    if self.context_mlp is not None:
      c = self.context_mlp(flatten_concat(obs, self.context_keys))
      x = torch.cat([x, c], dim=-1)
    return self.proj(x)


def _primary_channels(shape: tuple[int, ...], input_channels: int, key: str) -> int:
  if len(shape) == 1:
    return 1
  if len(shape) == 2:
    if input_channels not in (1, shape[0]):
      raise ValueError(
        f"input_channels={input_channels} does not match primary '{key}' "
        f"channel dim {shape[0]}."
      )
    return shape[0]
  raise ValueError(
    f"Conv1d primary '{key}' shape {shape} unsupported (expected (L,) or (C, L))."
  )


def _global_pool_1d(kind: str) -> nn.Module:
  if kind == "avg":
    return nn.AdaptiveAvgPool1d(1)
  if kind == "max":
    return nn.AdaptiveMaxPool1d(1)
  raise ValueError(f"global_pool must be 'avg' or 'max'; got '{kind}'.")


def _flat_dim(shape: tuple[int, ...]) -> int:
  dim = 1
  for s in shape:
    dim *= s
  return dim
