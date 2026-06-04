from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _time(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf * env.step_dt


def _window(env: ManagerBasedRlEnv, start_s: float, end_s: float) -> torch.Tensor:
  t = _time(env)
  return ((t >= start_s) & (t <= end_s)).float()


def _foot_contact_count(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float().sum(dim=1)


def _site_pos_b(asset: Entity, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  site_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  root_pos_w = asset.data.root_link_pos_w[:, None, :]
  rel_pos_w = site_pos_w - root_pos_w
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, rel_pos_w.shape[1], -1
  )
  return quat_apply_inverse(
    root_quat_w.reshape(-1, 4), rel_pos_w.reshape(-1, 3)
  ).reshape_as(rel_pos_w)


def _relative_base_height(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
  return asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def _side_roll_rate(asset: Entity, direction: float) -> torch.Tensor:
  sign = 1.0 if direction >= 0.0 else -1.0
  return torch.clamp(sign * asset.data.root_link_ang_vel_b[:, 0], min=0.0)


def _progress_gravity_target(
  progress: torch.Tensor,
  direction: float,
) -> torch.Tensor:
  sign = 1.0 if direction >= 0.0 else -1.0
  target_angle = torch.clamp(progress, 0.0, 1.0) * (2.0 * math.pi)
  return torch.stack(
    (
      torch.zeros_like(target_angle),
      -sign * torch.sin(target_angle),
      -torch.cos(target_angle),
    ),
    dim=1,
  )


class sideflip_state:
  """Shared state update for reference-free sideflip reward terms."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.progress = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    self.max_height = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    self.airborne_once = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.progress[env_ids] = 0.0
    self.max_height[env_ids] = 0.0
    self.airborne_once[env_ids] = False

  def _update_state(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str | None = None,
    max_rate: float | None = None,
    direction: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    asset: Entity = env.scene[asset_cfg.name]
    rate = _side_roll_rate(asset, direction)
    if max_rate is not None:
      rate = torch.clamp(rate, max=max_rate)

    previous_progress = self.progress.clone()
    self.progress = torch.clamp(
      self.progress + rate * env.step_dt / (2.0 * math.pi),
      min=0.0,
      max=2.0,
    )
    delta_progress = torch.clamp(self.progress - previous_progress, min=0.0)

    height = _relative_base_height(env, asset)
    self.max_height = torch.maximum(self.max_height, height)

    if sensor_name is not None:
      self.airborne_once |= _foot_contact_count(env, sensor_name) == 0.0

    env.extras["log"]["Metrics/sideflip_progress_mean"] = torch.mean(self.progress)
    env.extras["log"]["Metrics/sideflip_max_height_mean"] = torch.mean(
      self.max_height
    )
    env.extras["log"]["Metrics/sideflip_airborne_frac"] = torch.mean(
      self.airborne_once.float()
    )
    return self.progress, delta_progress, self.max_height, self.airborne_once.float()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    self._update_state(env, sensor_name=sensor_name, asset_cfg=asset_cfg)
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def takeoff_vertical_velocity(
  env: ManagerBasedRlEnv,
  start_s: float,
  end_s: float,
  target_vz: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  vz = asset.data.root_link_lin_vel_w[:, 2]
  return _window(env, start_s, end_s) * torch.clamp(vz / target_vz, 0.0, 1.0)


def side_roll_rate_dense(
  env: ManagerBasedRlEnv,
  start_s: float,
  end_s: float,
  target_rate: float,
  max_rate: float,
  direction: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  rate = torch.clamp(_side_roll_rate(asset, direction), min=0.0, max=max_rate)
  return _window(env, start_s, end_s) * torch.clamp(rate / target_rate, 0.0, 1.0)


class sideflip_progress_delta(sideflip_state):
  """Dense reward for discovering sideward rotation, independent of timing."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    start_s: float,
    end_s: float,
    max_rate: float,
    max_delta: float | None = None,
    direction: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    _, delta_progress, _, _ = self._update_state(
      env, max_rate=max_rate, direction=direction, asset_cfg=asset_cfg
    )
    if max_delta is not None:
      delta_progress = torch.clamp(delta_progress, max=max_delta)
    return _window(env, start_s, end_s) * delta_progress


class sideflip_progress_final(sideflip_state):
  """Reward being close to one completed sideflip after rotation emerges."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    start_s: float,
    std: float,
    direction: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    progress, _, _, _ = self._update_state(
      env, direction=direction, asset_cfg=asset_cfg
    )
    active = (_time(env) >= start_s).float()
    return active * torch.exp(-torch.square(progress - 1.0) / std**2)


class progress_based_sideflip_orientation(sideflip_state):
  """Track roll orientation from cumulative progress instead of wall-clock time."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    start_s: float,
    end_s: float,
    std: float,
    max_rate: float | None = None,
    direction: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    progress, _, _, _ = self._update_state(
      env, max_rate=max_rate, direction=direction, asset_cfg=asset_cfg
    )
    target_gravity_b = _progress_gravity_target(progress, direction)
    error = torch.sum(
      torch.square(asset.data.projected_gravity_b - target_gravity_b), dim=1
    )
    active = _window(env, start_s, end_s)
    env.extras["log"]["Metrics/sideflip_orientation_progress_mean"] = torch.mean(
      progress
    )
    return active * torch.exp(-error / std**2)


class apex_height_reward(sideflip_state):
  """Reward reaching a useful apex without prescribing a full height schedule."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    start_s: float,
    target_height: float,
    std: float,
    sensor_name: str | None = None,
    direction: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    _, _, max_height, _ = self._update_state(
      env, sensor_name=sensor_name, direction=direction, asset_cfg=asset_cfg
    )
    active = (_time(env) >= start_s).float()
    return active * torch.exp(-torch.square(max_height - target_height) / std**2)


def off_axis_ang_vel_l2(
  env: ManagerBasedRlEnv,
  start_s: float,
  end_s: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize pitch/yaw angular velocity so rotation stays on the roll axis."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.root_link_ang_vel_b
  return _window(env, start_s, end_s) * (
    torch.square(ang_vel[:, 1]) + torch.square(ang_vel[:, 2])
  )


class excess_sideflip_rotation(sideflip_state):
  """Penalize progress beyond one sideward revolution."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    start_s: float,
    max_rate: float | None = None,
    direction: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    progress, _, _, _ = self._update_state(
      env, max_rate=max_rate, direction=direction, asset_cfg=asset_cfg
    )
    asset: Entity = env.scene[asset_cfg.name]
    side_roll_rate = _side_roll_rate(asset, direction)
    excess_progress = torch.square(torch.clamp(progress - 1.0, min=0.0))
    post_flip_spin = (_time(env) >= start_s).float() * torch.square(side_roll_rate)
    env.extras["log"]["Metrics/sideflip_excess_progress_mean"] = torch.mean(
      torch.clamp(progress - 1.0, min=0.0)
    )
    return excess_progress + 0.02 * post_flip_spin


def feet_contact_before_takeoff(
  env: ManagerBasedRlEnv,
  start_s: float,
  end_s: float,
  sensor_name: str,
  min_contacts: int = 3,
) -> torch.Tensor:
  return _window(env, start_s, end_s) * (
    _foot_contact_count(env, sensor_name) >= float(min_contacts)
  ).float()


def airborne_after_takeoff(
  env: ManagerBasedRlEnv,
  start_s: float,
  end_s: float,
  sensor_name: str,
) -> torch.Tensor:
  return _window(env, start_s, end_s) * (
    _foot_contact_count(env, sensor_name) == 0.0
  ).float()


class landing_success(sideflip_state):
  """Smooth landing reward gated by progress, contacts, posture, and low motion."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    start_s: float,
    min_progress: float,
    min_contacts: int,
    max_tilt_xy: float,
    height_target: float,
    height_std: float,
    ang_vel_std: float,
    lin_vel_xy_std: float,
    direction: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    progress, _, _, _ = self._update_state(
      env, sensor_name=sensor_name, direction=direction, asset_cfg=asset_cfg
    )
    asset: Entity = env.scene[asset_cfg.name]
    active = (_time(env) >= start_s).float()

    progress_reward = torch.exp(-torch.square(progress - 1.0) / 0.18**2) * (
      progress >= min_progress
    ).float()
    gravity_xy_l2 = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    upright_reward = torch.exp(-gravity_xy_l2 / max_tilt_xy**2)
    height = _relative_base_height(env, asset)
    height_reward = torch.exp(-torch.square(height - height_target) / height_std**2)
    contact_indicator = (
      _foot_contact_count(env, sensor_name) >= float(min_contacts)
    ).float()
    low_ang_vel_reward = torch.exp(
      -torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1) / ang_vel_std**2
    )
    low_lin_vel_reward = torch.exp(
      -torch.sum(torch.square(asset.data.root_link_lin_vel_w[:, :2]), dim=1)
      / lin_vel_xy_std**2
    )

    success = (progress >= min_progress) & (gravity_xy_l2 <= max_tilt_xy**2) & (
      contact_indicator > 0.0
    )
    env.extras["log"]["Metrics/sideflip_landing_success_frac"] = torch.mean(
      success.float()
    )
    return (
      active
      * progress_reward
      * upright_reward
      * height_reward
      * contact_indicator
      * low_ang_vel_reward
      * low_lin_vel_reward
    )


class landing_position(sideflip_state):
  """Reward landing close to the reset XY position after a plausible flip."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    start_s: float,
    min_progress: float,
    min_contacts: int,
    xy_std: float,
    direction: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    progress, _, _, _ = self._update_state(
      env, sensor_name=sensor_name, direction=direction, asset_cfg=asset_cfg
    )
    asset: Entity = env.scene[asset_cfg.name]
    active = (_time(env) >= start_s).float()
    contact_indicator = (
      _foot_contact_count(env, sensor_name) >= float(min_contacts)
    ).float()
    progress_indicator = (progress >= min_progress).float()

    xy_from_origin = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    xy_error = torch.sum(torch.square(xy_from_origin), dim=1)
    displacement = torch.sqrt(xy_error)
    env.extras["log"]["Metrics/sideflip_landing_xy_displacement_mean"] = torch.mean(
      displacement
    )
    return (
      active
      * progress_indicator
      * contact_indicator
      * torch.exp(-xy_error / xy_std**2)
    )


class landing_joint_posture(sideflip_state):
  """Reward returning near the default joint pose after a plausible landing."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    start_s: float,
    min_progress: float,
    min_contacts: int,
    std: float,
    direction: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    progress, _, _, _ = self._update_state(
      env, sensor_name=sensor_name, direction=direction, asset_cfg=asset_cfg
    )
    asset: Entity = env.scene[asset_cfg.name]
    active = (_time(env) >= start_s).float()
    contact_indicator = (
      _foot_contact_count(env, sensor_name) >= float(min_contacts)
    ).float()
    progress_indicator = (progress >= min_progress).float()

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    normalized_error = torch.square((joint_pos - default_joint_pos) / std)
    reward = torch.exp(-torch.mean(normalized_error, dim=1))
    env.extras["log"]["Metrics/sideflip_landing_joint_error_mean"] = torch.mean(
      torch.sqrt(torch.mean(torch.square(joint_pos - default_joint_pos), dim=1))
    )
    return active * progress_indicator * contact_indicator * reward


class landing_foot_stance(sideflip_state):
  """Reward uncrossed left/right foot placement at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    site_names = tuple(cfg.params["asset_cfg"].site_names)
    self.fr_idx = site_names.index("FR")
    self.fl_idx = site_names.index("FL")
    self.rr_idx = site_names.index("RR")
    self.rl_idx = site_names.index("RL")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    start_s: float,
    min_progress: float,
    min_contacts: int,
    min_side_y: float,
    side_std: float,
    target_width: float,
    width_std: float,
    direction: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    progress, _, _, _ = self._update_state(
      env, sensor_name=sensor_name, direction=direction, asset_cfg=asset_cfg
    )
    asset: Entity = env.scene[asset_cfg.name]
    active = (_time(env) >= start_s).float()
    contact_indicator = (
      _foot_contact_count(env, sensor_name) >= float(min_contacts)
    ).float()
    progress_indicator = (progress >= min_progress).float()

    foot_pos_b = _site_pos_b(asset, asset_cfg)
    fr_y = foot_pos_b[:, self.fr_idx, 1]
    fl_y = foot_pos_b[:, self.fl_idx, 1]
    rr_y = foot_pos_b[:, self.rr_idx, 1]
    rl_y = foot_pos_b[:, self.rl_idx, 1]

    side_error = (
      torch.square(torch.clamp(min_side_y - fl_y, min=0.0))
      + torch.square(torch.clamp(min_side_y - rl_y, min=0.0))
      + torch.square(torch.clamp(fr_y + min_side_y, min=0.0))
      + torch.square(torch.clamp(rr_y + min_side_y, min=0.0))
    )
    front_width = fl_y - fr_y
    rear_width = rl_y - rr_y
    width_error = torch.square(front_width - target_width) + torch.square(
      rear_width - target_width
    )

    side_reward = torch.exp(-side_error / side_std**2)
    width_reward = torch.exp(-width_error / width_std**2)
    crossed = (fl_y <= 0.0) | (rl_y <= 0.0) | (fr_y >= 0.0) | (rr_y >= 0.0)
    env.extras["log"]["Metrics/sideflip_landing_crossed_feet_frac"] = torch.mean(
      crossed.float()
    )
    env.extras["log"]["Metrics/sideflip_landing_front_width_mean"] = torch.mean(
      front_width
    )
    env.extras["log"]["Metrics/sideflip_landing_rear_width_mean"] = torch.mean(
      rear_width
    )
    return active * progress_indicator * contact_indicator * side_reward * width_reward


class soft_landing_gated(sideflip_state):
  """Penalize first-contact impact only after a plausible flip landing."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    start_s: float,
    min_progress: float,
    force_scale: float,
    max_penalty: float,
    direction: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    progress, _, _, _ = self._update_state(
      env, sensor_name=sensor_name, direction=direction, asset_cfg=asset_cfg
    )
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.force is not None
    force_magnitude = torch.norm(contact_sensor.data.force, dim=-1)
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt).float()
    plausible_landing = ((_time(env) >= start_s) & (progress >= min_progress)).float()
    landing_impact = force_magnitude * first_contact
    raw_cost = torch.sum(landing_impact, dim=1) / force_scale
    cost = plausible_landing * torch.clamp(raw_cost, max=max_penalty)
    num_landings = torch.sum(first_contact)
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/sideflip_landing_force_mean"] = mean_landing_force
    return cost
