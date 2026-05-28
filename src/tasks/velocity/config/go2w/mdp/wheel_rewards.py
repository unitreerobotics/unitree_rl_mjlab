from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _total_command_magnitude(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  return linear_norm + angular_norm


def wheel_roll_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  wheel_radius: float,
  wheel_track: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward wheel angular velocity matching the commanded planar motion."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  side_signs = torch.tensor(
    [1.0, -1.0, 1.0, -1.0],
    device=env.device,
    dtype=wheel_vel.dtype,
  ).unsqueeze(0)

  lin_vel_x = command[:, 0:1]
  ang_vel_z = command[:, 2:3]
  target_wheel_vel = (
    lin_vel_x + 0.5 * wheel_track * side_signs * ang_vel_z
  ) / wheel_radius

  err = torch.mean(torch.square(wheel_vel - target_wheel_vel), dim=1)
  reward = torch.exp(-err / (std**2))
  env.extras["log"]["Metrics/wheel_roll_error_mean"] = torch.mean(torch.sqrt(err))
  return reward


def wheel_speed_limit_penalty(
  env: ManagerBasedRlEnv,
  max_abs_speed: float,
  command_name: str | None = None,
  command_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize wheel speeds above the configured limit."""
  asset: Entity = env.scene[asset_cfg.name]
  wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  excess = torch.clamp(torch.abs(wheel_vel) - max_abs_speed, min=0.0)
  cost = torch.mean(torch.square(excess), dim=1)

  if command_name is not None:
    active = (_total_command_magnitude(env, command_name) > command_threshold).float()
    cost = cost * active

  env.extras["log"]["Metrics/wheel_speed_excess_mean"] = torch.mean(
    torch.mean(excess, dim=1)
  )
  return cost


def leg_motion_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize leg motion while the robot is commanded to move."""
  asset: Entity = env.scene[asset_cfg.name]
  leg_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  cost = torch.mean(torch.square(leg_vel), dim=1)
  active = (_total_command_magnitude(env, command_name) > command_threshold).float()
  return cost * active


def adaptive_leg_motion_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  command_threshold: float = 0.05,
  tilt_relax_start: float = 0.08,
  tilt_relax_end: float = 0.30,
  contact_target: float = 0.85,
  min_penalty_scale: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize leg motion less when tilt or wheel contact suggests recovery."""
  asset: Entity = env.scene[asset_cfg.name]
  leg_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  base_cost = torch.mean(torch.square(leg_vel), dim=1)
  active = (_total_command_magnitude(env, command_name) > command_threshold).float()

  tilt_xy = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
  tilt_relax = torch.clamp(
    (tilt_xy - tilt_relax_start) / max(tilt_relax_end - tilt_relax_start, 1.0e-6),
    min=0.0,
    max=1.0,
  )

  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()
  contact_fraction = torch.mean(in_contact, dim=1)
  if contact_fraction.ndim > 1:
    contact_fraction = torch.mean(contact_fraction, dim=1)
  contact_relax = torch.clamp(
    (contact_target - contact_fraction) / max(contact_target, 1.0e-6),
    min=0.0,
    max=1.0,
  )

  relax = torch.maximum(tilt_relax, contact_relax)
  penalty_scale = 1.0 - (1.0 - min_penalty_scale) * relax
  env.extras["log"]["Metrics/leg_penalty_scale_mean"] = torch.mean(penalty_scale)
  return base_cost * active * penalty_scale


def contact_fraction_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Reward persistent wheel-ground contact."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()
  return torch.mean(in_contact, dim=1)
