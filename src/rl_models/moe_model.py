"""Mixture-of-Experts MLP actor for Go2 locomotion policies."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization, MLP
from tensordict import TensorDict


class _MoEHead(nn.Module):
  """Action-mean mixture over frozen or trainable expert MLPs."""

  def __init__(self, obs_dim: int, action_dim: int, moe_cfg: dict[str, Any]) -> None:
    super().__init__()
    experts_cfg = list(moe_cfg.get("experts", ()))
    if not experts_cfg:
      raise ValueError("moe_cfg['experts'] must contain at least one expert.")

    self.expert_names = [str(e.get("name", f"expert_{i}")) for i, e in enumerate(experts_cfg)]
    self.gate_temperature = float(moe_cfg.get("gate_temperature", 1.0))
    if self.gate_temperature <= 0.0:
      raise ValueError("moe_cfg['gate_temperature'] must be > 0.")

    expert_hidden_dims = tuple(moe_cfg.get("expert_hidden_dims", (512, 256, 128)))
    expert_activation = str(moe_cfg.get("expert_activation", "elu"))
    gate_hidden_dims = tuple(moe_cfg.get("gate_hidden_dims", (256, 128)))
    gate_activation = str(moe_cfg.get("gate_activation", "elu"))
    expert_obs_norm = bool(moe_cfg.get("expert_obs_normalization", True))
    gate_obs_norm = bool(moe_cfg.get("gate_obs_normalization", True))

    normalizer_cls = EmpiricalNormalization if expert_obs_norm else nn.Identity
    self.expert_normalizers = nn.ModuleList(
      [normalizer_cls(obs_dim) for _ in experts_cfg]
    )
    self.experts = nn.ModuleList(
      [MLP(obs_dim, action_dim, expert_hidden_dims, expert_activation) for _ in experts_cfg]
    )
    self.gate_normalizer = EmpiricalNormalization(obs_dim) if gate_obs_norm else nn.Identity()
    self.gate = MLP(obs_dim, len(experts_cfg), gate_hidden_dims, gate_activation)

    allow_missing = bool(moe_cfg.get("allow_missing_checkpoints", False))
    for i, expert_cfg in enumerate(experts_cfg):
      self._load_expert_checkpoint(i, str(expert_cfg.get("checkpoint", "")), allow_missing)

    if bool(moe_cfg.get("freeze_experts", True)):
      for module in (*self.expert_normalizers, *self.experts):
        module.requires_grad_(False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    weights = self.gate_weights(x)
    expert_means = torch.stack(
      [
        expert(normalizer(x))
        for normalizer, expert in zip(self.expert_normalizers, self.experts, strict=True)
      ],
      dim=-2,
    )
    return torch.sum(expert_means * weights.unsqueeze(-1), dim=-2)

  def gate_weights(self, x: torch.Tensor) -> torch.Tensor:
    logits = self.gate(self.gate_normalizer(x)) / self.gate_temperature
    return torch.softmax(logits, dim=-1)

  @torch.no_grad()
  def update_gate_normalization(self, x: torch.Tensor) -> None:
    if isinstance(self.gate_normalizer, EmpiricalNormalization):
      self.gate_normalizer.update(x)

  def _load_expert_checkpoint(
    self, expert_idx: int, checkpoint_path: str, allow_missing: bool
  ) -> None:
    name = self.expert_names[expert_idx]
    if not checkpoint_path or not Path(checkpoint_path).is_file():
      msg = (
        f"MoE expert '{name}' checkpoint is missing: {checkpoint_path!r}. "
        "Keeping random initialization."
      )
      if allow_missing:
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return
      raise FileNotFoundError(msg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in checkpoint:
      raise KeyError(
        f"MoE expert '{name}' checkpoint {checkpoint_path!r} has no actor_state_dict."
      )

    actor_sd = checkpoint["actor_state_dict"]
    expert_sd = {
      key.removeprefix("mlp."): value
      for key, value in actor_sd.items()
      if key.startswith("mlp.")
    }
    normalizer_sd = {
      key.removeprefix("obs_normalizer."): value
      for key, value in actor_sd.items()
      if key.startswith("obs_normalizer.")
    }

    try:
      self.experts[expert_idx].load_state_dict(expert_sd, strict=True)
      normalizer = self.expert_normalizers[expert_idx]
      if isinstance(normalizer, EmpiricalNormalization):
        normalizer.load_state_dict(normalizer_sd, strict=True)
    except RuntimeError as exc:
      raise RuntimeError(
        f"Failed to load MoE expert '{name}' from {checkpoint_path!r}: {exc}"
      ) from exc


class MoEMLPModel(MLPModel):
  """MLPModel-compatible action-level Mixture-of-Experts actor."""

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
    moe_cfg: dict[str, Any] | None = None,
  ) -> None:
    if moe_cfg is None:
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

    if obs_normalization:
      raise ValueError(
        "MoEMLPModel owns per-expert/gate normalizers inside moe_cfg; "
        "set obs_normalization=False on the actor."
      )

    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims,
      activation,
      obs_normalization=False,
      distribution_cfg=distribution_cfg,
    )
    self.mlp = _MoEHead(self._get_latent_dim(), output_dim, moe_cfg)
    self._moe_active = True

  def update_normalization(self, obs: TensorDict) -> None:
    if not getattr(self, "_moe_active", False):
      return super().update_normalization(obs)
    obs_list = [obs[obs_group] for obs_group in self.obs_groups]
    flat_obs = torch.cat(obs_list, dim=-1)
    self.mlp.update_gate_normalization(flat_obs)

  def gate_weights(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
    if not getattr(self, "_moe_active", False):
      raise RuntimeError("gate_weights is only available when moe_cfg is active.")
    if isinstance(obs, TensorDict):
      flat_obs = self.get_latent(obs)
    else:
      flat_obs = obs
    return self.mlp.gate_weights(flat_obs)
