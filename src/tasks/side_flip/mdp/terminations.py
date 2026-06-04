from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from src.tasks.back_flip.mdp.terminations import (  # noqa: F401
  base_height_below,
  illegal_contact_after_time,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class one_flip_success:
  """Timeout-style success when one sideflip is landed."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self.progress = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.progress[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    start_s: float,
    min_progress: float,
    max_tilt_xy: float,
    min_height: float,
    max_height: float,
    max_ang_vel: float,
    min_contacts: int = 2,
    direction: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    sign = 1.0 if direction >= 0.0 else -1.0
    side_roll_rate = torch.clamp(sign * asset.data.root_link_ang_vel_b[:, 0], min=0.0)
    self.progress = torch.clamp(
      self.progress + side_roll_rate * env.step_dt / (2.0 * math.pi),
      min=0.0,
      max=2.0,
    )

    active = env.episode_length_buf * env.step_dt >= start_s
    gravity = asset.data.projected_gravity_b
    upright = (torch.sum(torch.square(gravity[:, :2]), dim=1) < max_tilt_xy**2) & (
      gravity[:, 2] < 0.0
    )
    height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    height_ok = (height >= min_height) & (height <= max_height)
    ang_vel_ok = torch.norm(asset.data.root_link_ang_vel_b, dim=1) <= max_ang_vel

    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    landed = (sensor.data.found > 0).float().sum(dim=1) >= float(min_contacts)
    progress_ok = self.progress >= min_progress
    return active & progress_ok & upright & height_ok & ang_vel_ok & landed
