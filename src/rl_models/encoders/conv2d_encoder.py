"""2D CNN encoder for grid-like height maps."""

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
from src.rl_models.encoders.conv1d_encoder import _flat_dim


class Conv2dObservationEncoder(BaseObservationEncoder):
  """Conv2d over a primary grid group, optionally fused with context groups.

  If the primary group is stored flat (e.g. Go2 ``height_scan`` is ``[B, 187]``
  for a 17x11 grid), set ``input_hw=[H, W]`` to reshape it before convolving.
  """

  def __init__(
    self,
    encoder_input_keys: Sequence[str],
    obs_shapes: Mapping[str, tuple[int, ...]],
    primary_key: str,
    latent_dim: int,
    context_keys: Sequence[str] | None = None,
    input_channels: int = 1,
    input_hw: Sequence[int] | None = None,
    channels: Sequence[int] = (16, 32, 64),
    kernel_sizes: Sequence[int] = (3, 3, 3),
    strides: Sequence[int] = (1, 2, 2),
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
    self.input_hw = tuple(input_hw) if input_hw is not None else None

    c_in = _resolve_2d_channels(
      obs_shapes[primary_key], input_channels, self.input_hw, primary_key
    )

    act = resolve_nn_activation(activation)
    layers: list[nn.Module] = []
    prev = c_in
    for out_ch, k, s in zip(channels, kernel_sizes, strides):
      layers += [nn.Conv2d(prev, out_ch, kernel_size=k, stride=s, padding=k // 2), act]
      prev = out_ch
    self.conv = nn.Sequential(*layers)
    self.pool = _global_pool_2d(global_pool)
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
    if x.dim() == 2:  # [B, H*W] -> [B, 1, H, W] (needs input_hw)
      if self.input_hw is None:
        raise ValueError(
          f"Conv2d primary '{self.primary_key}' is flat {tuple(x.shape)}; set "
          "input_hw=[H, W] to reshape it into a grid."
        )
      h, w = self.input_hw
      x = x.reshape(x.shape[0], 1, h, w)
    elif x.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
      x = x.unsqueeze(1)
    elif x.dim() != 4:  # expect [B, C, H, W]
      raise ValueError(
        f"Conv2d primary '{self.primary_key}' must be [B, H, W] or [B, C, H, W] "
        f"(or flat [B, H*W] with input_hw); got {tuple(x.shape)}."
      )
    x = self.pool(self.conv(x)).flatten(1)
    if self.context_mlp is not None:
      c = self.context_mlp(flatten_concat(obs, self.context_keys))
      x = torch.cat([x, c], dim=-1)
    return self.proj(x)


def _resolve_2d_channels(
  shape: tuple[int, ...],
  input_channels: int,
  input_hw: tuple[int, ...] | None,
  key: str,
) -> int:
  if len(shape) == 1:  # flat, reshaped at runtime via input_hw
    if input_hw is None:
      raise ValueError(
        f"Conv2d primary '{key}' is flat {shape}; provide input_hw=[H, W]."
      )
    if input_hw[0] * input_hw[1] != shape[0]:
      raise ValueError(
        f"input_hw={tuple(input_hw)} does not match flat primary '{key}' dim "
        f"{shape[0]}."
      )
    return 1
  if len(shape) == 2:  # (H, W)
    return 1
  if len(shape) == 3:  # (C, H, W)
    if input_channels not in (1, shape[0]):
      raise ValueError(
        f"input_channels={input_channels} does not match primary '{key}' "
        f"channel dim {shape[0]}."
      )
    return shape[0]
  raise ValueError(
    f"Conv2d primary '{key}' shape {shape} unsupported (expected (H,W), (C,H,W), "
    "or flat (H*W,) with input_hw)."
  )


def _global_pool_2d(kind: str) -> nn.Module:
  if kind == "avg":
    return nn.AdaptiveAvgPool2d(1)
  if kind == "max":
    return nn.AdaptiveMaxPool2d(1)
  raise ValueError(f"global_pool must be 'avg' or 'max'; got '{kind}'.")
