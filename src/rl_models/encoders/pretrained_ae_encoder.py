"""Pretrained autoencoder observation encoder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.utils import resolve_callable

from src.rl_models.encoders.base import (
  BaseObservationEncoder,
  ObsInput,
  build_mlp,
  flatten_concat,
  get_group,
)
from src.rl_models.encoders.conv1d_encoder import _flat_dim


class PretrainedAEObservationEncoder(BaseObservationEncoder):
  """Use a pretrained autoencoder encoder as an observation feature extractor."""

  def __init__(
    self,
    encoder_input_keys: Sequence[str],
    obs_shapes: Mapping[str, tuple[int, ...]],
    checkpoint_path: str,
    encoder_class: str | type[nn.Module] = "src.rl_models.autoencoder:HeightScanAutoEncoder",
    latent_dim: int | None = None,
    freeze: bool = True,
    strict: bool = False,
    primary_key: str | None = None,
    context_keys: Sequence[str] | None = None,
    context_hidden_dims: Sequence[int] = (64,),
    encoder_kwargs: Mapping[str, Any] | None = None,
    activation: str = "elu",
  ) -> None:
    super().__init__()
    self.primary_key = primary_key or _single_primary_key(encoder_input_keys)
    self.context_keys = list(context_keys or [])
    self.freeze_pretrained = bool(freeze)

    unknown_context = [k for k in self.context_keys if k not in encoder_input_keys]
    if unknown_context:
      raise ValueError(
        f"context_keys {unknown_context} must also be in encoder_input_keys "
        f"{list(encoder_input_keys)}."
      )

    ae = self._load_autoencoder(
      encoder_class,
      checkpoint_path,
      obs_shapes[self.primary_key],
      latent_dim,
      strict,
      encoder_kwargs,
    )
    self.pretrained_encoder, self._encode_with_method = _extract_encoder(ae)
    ae_dim = _infer_latent_dim(ae, latent_dim)

    ctx_dim = 0
    self.context_mlp: nn.Module | None = None
    if self.context_keys:
      ctx_in = sum(_flat_dim(obs_shapes[k]) for k in self.context_keys)
      ctx_dim = int(context_hidden_dims[-1])
      self.context_mlp = build_mlp(
        ctx_in, ctx_dim, list(context_hidden_dims[:-1]), activation
      )

    out_dim = int(latent_dim) if latent_dim is not None else ae_dim
    self.proj: nn.Module
    if self.context_mlp is not None or ae_dim != out_dim:
      self.proj = nn.Linear(ae_dim + ctx_dim, out_dim)
    else:
      self.proj = nn.Identity()
    self.output_dim = out_dim

    if self.freeze_pretrained:
      self._freeze_pretrained()

  def forward(self, obs: ObsInput) -> torch.Tensor:
    x = get_group(obs, self.primary_key)
    z = self._encode_primary(x)
    if self.context_mlp is not None:
      c = self.context_mlp(flatten_concat(obs, self.context_keys))
      z = torch.cat([z, c], dim=-1)
    return self.proj(z)

  def train(self, mode: bool = True) -> PretrainedAEObservationEncoder:
    super().train(mode)
    if self.freeze_pretrained:
      self._freeze_pretrained()
    return self

  def _encode_primary(self, x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(x.shape[0], -1)
    if self._encode_with_method:
      return self.pretrained_encoder.encode(x)
    return self.pretrained_encoder(x)

  def _freeze_pretrained(self) -> None:
    self.pretrained_encoder.requires_grad_(False)
    self.pretrained_encoder.eval()

  def _load_autoencoder(
    self,
    encoder_class: str | type[nn.Module],
    checkpoint_path: str,
    primary_shape: tuple[int, ...],
    latent_dim: int | None,
    strict: bool,
    encoder_kwargs: Mapping[str, Any] | None,
  ) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cls = resolve_callable(encoder_class)
    checkpoint_kwargs = _checkpoint_model_kwargs(checkpoint)
    kwargs = {**checkpoint_kwargs, **dict(encoder_kwargs or {})}
    kwargs.setdefault("input_dim", _flat_dim(primary_shape))
    if latent_dim is not None:
      kwargs.setdefault("latent_dim", int(latent_dim))
    ae = cls(**kwargs)
    if not isinstance(ae, nn.Module):
      raise TypeError(
        f"encoder_class '{encoder_class}' must instantiate an nn.Module; got "
        f"{type(ae).__name__}."
      )
    state_dict = _checkpoint_state_dict(checkpoint)
    ae.load_state_dict(state_dict, strict=bool(strict))
    return ae


def _single_primary_key(encoder_input_keys: Sequence[str]) -> str:
  if len(encoder_input_keys) != 1:
    raise ValueError(
      "pretrained_ae requires primary_key when encoder_input_keys contains "
      f"multiple groups; got {list(encoder_input_keys)}."
    )
  return str(encoder_input_keys[0])


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
  if isinstance(checkpoint, Mapping):
    for key in ("state_dict", "model_state_dict", "autoencoder_state_dict"):
      value = checkpoint.get(key)
      if isinstance(value, Mapping):
        return value
    if all(isinstance(k, str) for k in checkpoint.keys()):
      return checkpoint
  raise ValueError(
    "Autoencoder checkpoint must be a state_dict or contain one of "
    "'state_dict', 'model_state_dict', or 'autoencoder_state_dict'."
  )


def _checkpoint_model_kwargs(checkpoint: Any) -> dict[str, Any]:
  if isinstance(checkpoint, Mapping):
    for key in ("model_kwargs", "encoder_kwargs"):
      value = checkpoint.get(key)
      if isinstance(value, Mapping):
        return dict(value)
  return {}


def _extract_encoder(ae: nn.Module) -> tuple[nn.Module, bool]:
  encoder = getattr(ae, "encoder", None)
  if isinstance(encoder, nn.Module):
    return encoder, False
  if callable(getattr(ae, "encode", None)):
    return ae, True
  raise AttributeError(
    "Autoencoder module must expose either an nn.Module 'encoder' attribute or "
    "an encode(x) method."
  )


def _infer_latent_dim(ae: nn.Module, fallback_latent_dim: int | None) -> int:
  latent_dim = getattr(ae, "latent_dim", None)
  if latent_dim is not None:
    return int(latent_dim)
  if fallback_latent_dim is not None:
    return int(fallback_latent_dim)
  raise ValueError(
    "latent_dim must be configured when encoder_class does not expose a "
    "'latent_dim' attribute."
  )
