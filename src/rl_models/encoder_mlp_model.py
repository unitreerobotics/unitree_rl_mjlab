"""Configurable encoder + MLP actor/critic model for rsl_rl.

``EncoderMLPModel`` subclasses rsl_rl's :class:`~rsl_rl.models.MLPModel`. When
``observation_encoder_cfg`` is provided it routes a configurable subset of
observation groups through a pluggable observation encoder, concatenates the
resulting latent with the remaining (passthrough) groups, and feeds the result
to the MLP head. When ``observation_encoder_cfg`` is ``None`` it is a literal
drop-in for ``MLPModel`` (identical behavior).

The design mirrors rsl_rl's ``CNNModel``: the encoder is built before
``super().__init__`` (so ``_get_latent_dim`` can size the MLP) and registered as
a submodule afterwards, which makes its parameters part of the model and thus
optimized by PPO automatically.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict

from src.rl_models.encoders import build_observation_encoder

_EXPORT_MSG = (
  "{op} export of EncoderMLPModel is out of scope for the encoder-ablation "
  "phase. Use a runner that does not export (e.g. MjlabOnPolicyRunner) or set "
  "observation_encoder_cfg=None to fall back to MLPModel."
)


class EncoderMLPModel(MLPModel):
  """MLP model with an optional, configurable observation encoder."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: Sequence[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    observation_encoder_cfg: dict[str, Any] | None = None,
  ) -> None:
    self._oe_cfg = observation_encoder_cfg

    if observation_encoder_cfg is None:
      # Drop-in: behave exactly like MLPModel.
      super().__init__(
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims,
        activation,
        obs_normalization,
        distribution_cfg,
      )
      return

    # Resolve encoder/passthrough groups, then build the encoder up front so
    # _get_latent_dim (called inside super().__init__) can size the MLP head.
    self._get_obs_dim(obs, obs_groups, obs_set)
    obs_shapes = {k: tuple(obs[k].shape[1:]) for k in self._encoder_input_keys}
    encoder = build_observation_encoder(
      observation_encoder_cfg, obs_shapes, self._encoder_input_keys
    )
    self._encoder_output_dim = int(encoder.output_dim)

    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims,
      activation,
      obs_normalization,
      distribution_cfg,
    )

    # Register after nn.Module.__init__ so params are tracked (and optimized).
    self.observation_encoder = encoder

  # -- Latent construction -------------------------------------------------

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    if self._oe_cfg is None:
      return super().get_latent(obs, masks, hidden_state)
    enc_in = {k: obs[k] for k in self._encoder_input_keys}
    z = self.observation_encoder(enc_in)
    if self.obs_groups:  # passthrough groups present -> normalize + concat
      passthrough = super().get_latent(obs, masks, hidden_state)
      return torch.cat([z, passthrough], dim=-1)
    return z

  def update_normalization(self, obs: TensorDict) -> None:
    # Guard the empty-passthrough case (super() would cat an empty list).
    if self._oe_cfg is not None and not self.obs_groups:
      return
    super().update_normalization(obs)

  # -- Dimension hooks (mirror CNNModel) -----------------------------------

  def _get_obs_dim(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
  ) -> tuple[list[str], int]:
    if self._oe_cfg is None:
      return super()._get_obs_dim(obs, obs_groups, obs_set)

    active = list(obs_groups[obs_set])
    enc_keys = list(self._oe_cfg["encoder_input_keys"])
    missing = [k for k in enc_keys if k not in active]
    if missing:
      raise ValueError(
        f"encoder_input_keys {missing} are not in obs_groups['{obs_set}'] = "
        f"{active}. Expose them as observation groups or fix the config."
      )

    pass_keys = self._oe_cfg.get("passthrough_keys")
    if pass_keys is None:
      pass_keys = [g for g in active if g not in enc_keys]
    else:
      pass_keys = list(pass_keys)
      missing_p = [k for k in pass_keys if k not in active]
      if missing_p:
        raise ValueError(
          f"passthrough_keys {missing_p} are not in obs_groups['{obs_set}'] = "
          f"{active}."
        )

    self._encoder_input_keys = enc_keys

    pass_dim = 0
    for g in pass_keys:
      if len(obs[g].shape) != 2:
        raise ValueError(
          f"Passthrough group '{g}' must be 1D (shape [B, D]); got "
          f"{tuple(obs[g].shape)}. Route multi-dim groups through the encoder."
        )
      pass_dim += obs[g].shape[-1]
    return pass_keys, pass_dim

  def _get_latent_dim(self) -> int:
    if self._oe_cfg is None:
      return super()._get_latent_dim()
    return self._encoder_output_dim + self.obs_dim

  # -- Export (unsupported with an active encoder) -------------------------

  def as_jit(self) -> nn.Module:
    if self._oe_cfg is None:
      return super().as_jit()
    raise NotImplementedError(_EXPORT_MSG.format(op="JIT"))

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    if self._oe_cfg is None:
      return super().as_onnx(verbose)
    raise NotImplementedError(_EXPORT_MSG.format(op="ONNX"))
