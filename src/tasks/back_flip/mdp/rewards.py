from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _time(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf * env.step_dt


def _phase(env: ManagerBasedRlEnv, duration_s: float) -> torch.Tensor:
  return torch.clamp(_time(env) / duration_s, 0.0, 1.0)


def _window(env: ManagerBasedRlEnv, start_s: float, end_s: float) -> torch.Tensor:
  t = _time(env)
  return ((t >= start_s) & (t <= end_s)).float()


def _foot_contact_count(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float().sum(dim=1)


def _backward_pitch_gravity_target(
  env: ManagerBasedRlEnv,
  takeoff_s: float,
  landing_s: float,
) -> torch.Tensor:
  target_progress = torch.clamp(
    (_time(env) - takeoff_s) / (landing_s - takeoff_s), 0.0, 1.0
  )
  target_angle = target_progress * (2.0 * math.pi)
  return torch.stack(
    (
      -torch.sin(target_angle),
      torch.zeros_like(target_angle),
      -torch.cos(target_angle),
    ),
    dim=1,
  )


class backflip_progress:
  """Reward tracking a one-revolution backward pitch progress schedule."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.progress = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.progress[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    duration_s: float,
    takeoff_s: float,
    landing_s: float,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    backward_pitch_rate = torch.clamp(-asset.data.root_link_ang_vel_b[:, 1], min=0.0)
    self.progress = torch.clamp(
      self.progress + backward_pitch_rate * env.step_dt / (2.0 * math.pi),
      min=0.0,
      max=1.25,
    )

    target = torch.clamp((_time(env) - takeoff_s) / (landing_s - takeoff_s), 0.0, 1.0)
    error = torch.square(self.progress - target)
    reward = torch.exp(-error / std**2)
    env.extras["log"]["Metrics/backflip_progress_mean"] = torch.mean(self.progress)
    env.extras["log"]["Metrics/backflip_target_mean"] = torch.mean(target)
    return reward * (_phase(env, duration_s) < 1.0).float()


def backward_pitch_rate(
  env: ManagerBasedRlEnv,
  start_s: float,
  end_s: float,
  target_rate: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  rate = torch.clamp(-asset.data.root_link_ang_vel_b[:, 1], min=0.0)
  reward = torch.exp(-torch.square(rate - target_rate) / std**2)
  return reward * _window(env, start_s, end_s)


def backward_pitch_orientation(
  env: ManagerBasedRlEnv,
  takeoff_s: float,
  landing_s: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track the gravity vector expected during a single backward pitch rotation."""
  asset: Entity = env.scene[asset_cfg.name]
  target_gravity_b = _backward_pitch_gravity_target(env, takeoff_s, landing_s)
  error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target_gravity_b), dim=1
  )
  active = ((_time(env) >= takeoff_s) & (_time(env) <= landing_s)).float()
  return active * torch.exp(-error / std**2)


def vertical_midflip_orientation(
  env: ManagerBasedRlEnv,
  center_s: float,
  width_s: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward a vertical, pitch-axis posture halfway through the flip."""
  asset: Entity = env.scene[asset_cfg.name]
  t = _time(env)
  active = (torch.abs(t - center_s) <= width_s).float()
  projected_gravity = asset.data.projected_gravity_b
  vertical_error = torch.square(torch.abs(projected_gravity[:, 0]) - 1.0)
  lateral_error = torch.square(projected_gravity[:, 1])
  upright_error = torch.square(projected_gravity[:, 2])
  error = vertical_error + lateral_error + upright_error
  return active * torch.exp(-error / std**2)


def off_axis_ang_vel_l2(
  env: ManagerBasedRlEnv,
  start_s: float,
  end_s: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize roll/yaw angular velocity so rotation stays on the pitch axis."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.root_link_ang_vel_b
  return _window(env, start_s, end_s) * (
    torch.square(ang_vel[:, 0]) + torch.square(ang_vel[:, 2])
  )


class excess_backflip_rotation:
  """Penalize progress beyond one backward revolution."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.progress = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.progress[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    start_s: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    backward_pitch_rate = torch.clamp(-asset.data.root_link_ang_vel_b[:, 1], min=0.0)
    self.progress = torch.clamp(
      self.progress + backward_pitch_rate * env.step_dt / (2.0 * math.pi),
      min=0.0,
      max=2.0,
    )
    excess_progress = torch.square(torch.clamp(self.progress - 1.0, min=0.0))
    post_flip_spin = (_time(env) >= start_s).float() * torch.square(backward_pitch_rate)
    env.extras["log"]["Metrics/backflip_excess_progress_mean"] = torch.mean(
      torch.clamp(self.progress - 1.0, min=0.0)
    )
    return excess_progress + 0.02 * post_flip_spin


def base_height_schedule(
  env: ManagerBasedRlEnv,
  duration_s: float,
  crouch_height: float,
  air_height: float,
  landing_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  phase = _phase(env, duration_s)
  height = asset.data.root_link_pos_w[:, 2]
  target = torch.where(
    phase < 0.15,
    torch.full_like(phase, crouch_height),
    torch.where(
      phase < 0.72,
      torch.full_like(phase, air_height),
      torch.full_like(phase, landing_height),
    ),
  )
  return torch.exp(-torch.square(height - target) / std**2)


def foot_contact_schedule(
  env: ManagerBasedRlEnv,
  duration_s: float,
  sensor_name: str,
) -> torch.Tensor:
  phase = _phase(env, duration_s)
  count = _foot_contact_count(env, sensor_name)
  early_contact = (count >= 3.0).float()
  airborne = (count == 0.0).float()
  landing_contact = (count >= 2.0).float()
  return torch.where(
    phase < 0.18,
    early_contact,
    torch.where(phase < 0.72, airborne, landing_contact),
  )


def upright_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  start_s: float,
  height_target: float,
  height_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  active = (_time(env) >= start_s).float()
  gravity_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  upright = torch.exp(-gravity_error / 0.25**2)
  height = torch.exp(
    -torch.square(asset.data.root_link_pos_w[:, 2] - height_target) / height_std**2
  )
  contacts = (_foot_contact_count(env, sensor_name) >= 2.0).float()

  return active * upright * height * contacts


def rearward_displacement(
  env: ManagerBasedRlEnv,
  duration_s: float,
  target_x: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  active = (_phase(env, duration_s) > 0.70).float()
  x_from_origin = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  return active * torch.exp(-torch.square(x_from_origin - target_x) / std**2)


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize high impact forces at landing."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None
  forces = contact_sensor.data.force
  force_magnitude = torch.norm(forces, dim=-1)
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
  landing_impact = force_magnitude * first_contact.float()
  cost = torch.sum(landing_impact, dim=1)
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  return cost

