from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact_after_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float,
  grace_s: float,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  active = env.episode_length_buf * env.step_dt > grace_s
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    illegal = (force_mag > force_threshold).any(dim=-1).any(dim=-1)
  else:
    assert data.found is not None
    illegal = torch.any(data.found, dim=-1)
  return active & illegal


def base_height_below(
  env: ManagerBasedRlEnv,
  min_height: float,
  grace_s: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  active = env.episode_length_buf * env.step_dt > grace_s
  return active & (asset.data.root_link_pos_w[:, 2] < min_height)


def one_flip_success(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  start_s: float,
  max_tilt_xy: float,
  min_height: float,
  max_height: float,
  max_ang_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  active = env.episode_length_buf * env.step_dt >= start_s
  gravity = asset.data.projected_gravity_b
  upright = (torch.sum(torch.square(gravity[:, :2]), dim=1) < max_tilt_xy**2) & (
    gravity[:, 2] < 0.0
  )
  height = asset.data.root_link_pos_w[:, 2]
  height_ok = (height >= min_height) & (height <= max_height)
  ang_vel_ok = torch.norm(asset.data.root_link_ang_vel_b, dim=1) <= max_ang_vel

  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  landed = (sensor.data.found > 0).float().sum(dim=1) >= 2.0
  return active & upright & height_ok & ang_vel_ok & landed

