from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def flip_phase(env: ManagerBasedRlEnv, duration_s: float) -> torch.Tensor:
  """Return normalized episode phase and its circular encoding."""
  phase = torch.clamp(env.episode_length_buf * env.step_dt / duration_s, 0.0, 1.0)
  return torch.stack(
    (
      phase,
      torch.sin(phase * torch.pi * 2.0),
      torch.cos(phase * torch.pi * 2.0),
    ),
    dim=1,
  )


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float()


def base_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2].unsqueeze(1)
