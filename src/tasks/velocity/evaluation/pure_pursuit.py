from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi


@dataclass
class PurePursuitResult:
  command_b: np.ndarray
  target: np.ndarray
  progress: float
  lateral_error: float
  heading_error: float
  reached_goal: bool


@dataclass
class PurePursuitController:
  waypoints: np.ndarray
  lookahead_distance: float = 1.0
  target_speed: float = 0.8
  max_linear_velocity: float = 1.5
  max_yaw_rate: float = 1.5
  goal_tolerance: float = 0.5

  def __post_init__(self) -> None:
    if self.waypoints.ndim != 2 or self.waypoints.shape[1] < 2:
      raise ValueError("waypoints must have shape (N, >=2)")
    diffs = np.diff(self.waypoints[:, :2], axis=0)
    self.segment_lengths = np.linalg.norm(diffs, axis=1)
    self.cumulative = np.concatenate([[0.0], np.cumsum(self.segment_lengths)])
    self.total_length = float(self.cumulative[-1])

  def compute(self, pos_xy: np.ndarray, yaw: float) -> PurePursuitResult:
    progress, lateral_error = self.project_progress(pos_xy)
    target_progress = min(progress + self.lookahead_distance, self.total_length)
    target = self.point_at_progress(target_progress)
    to_target_w = target[:2] - pos_xy
    reached_goal = self.total_length - progress <= self.goal_tolerance
    speed = 0.0 if reached_goal else min(self.target_speed, self.max_linear_velocity)
    heading_to_target = float(np.arctan2(to_target_w[1], to_target_w[0]))
    heading_error = float((heading_to_target - yaw + np.pi) % (2 * np.pi) - np.pi)
    yaw_rate = float(
      np.clip(2.0 * speed * np.sin(heading_error) / max(self.lookahead_distance, 1e-6),
              -self.max_yaw_rate, self.max_yaw_rate)
    )
    command_w = np.array([speed * np.cos(heading_to_target), speed * np.sin(heading_to_target)])
    c, s = np.cos(-yaw), np.sin(-yaw)
    command_b_xy = np.array([c * command_w[0] - s * command_w[1], s * command_w[0] + c * command_w[1]])
    return PurePursuitResult(
      command_b=np.array([command_b_xy[0], command_b_xy[1], yaw_rate], dtype=np.float32),
      target=target,
      progress=float(progress),
      lateral_error=float(lateral_error),
      heading_error=heading_error,
      reached_goal=bool(reached_goal),
    )

  def project_progress(self, pos_xy: np.ndarray) -> tuple[float, float]:
    best_dist = float("inf")
    best_progress = 0.0
    for i, seg_len in enumerate(self.segment_lengths):
      if seg_len <= 1e-9:
        continue
      a = self.waypoints[i, :2]
      b = self.waypoints[i + 1, :2]
      t = float(np.clip(np.dot(pos_xy - a, b - a) / (seg_len * seg_len), 0.0, 1.0))
      closest = a + t * (b - a)
      dist = float(np.linalg.norm(pos_xy - closest))
      if dist < best_dist:
        best_dist = dist
        best_progress = float(self.cumulative[i] + t * seg_len)
    return best_progress, best_dist

  def point_at_progress(self, progress: float) -> np.ndarray:
    progress = float(np.clip(progress, 0.0, self.total_length))
    idx = int(np.searchsorted(self.cumulative, progress, side="right") - 1)
    idx = min(max(idx, 0), len(self.segment_lengths) - 1)
    seg_len = max(float(self.segment_lengths[idx]), 1e-9)
    t = (progress - float(self.cumulative[idx])) / seg_len
    return self.waypoints[idx] + t * (self.waypoints[idx + 1] - self.waypoints[idx])


@dataclass(kw_only=True)
class PurePursuitVelocityCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  waypoints: list[list[float]] = field(default_factory=list)
  lookahead_distance: float = 1.0
  target_speed: float = 0.8
  max_linear_velocity: float = 1.5
  max_yaw_rate: float = 1.5
  goal_tolerance: float = 0.5

  def build(self, env):
    return PurePursuitVelocityCommand(self, env)


class PurePursuitVelocityCommand(CommandTerm):
  cfg: PurePursuitVelocityCommandCfg

  def __init__(self, cfg: PurePursuitVelocityCommandCfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    waypoints = torch.tensor(cfg.waypoints, dtype=torch.float, device=self.device)
    if waypoints.ndim != 2 or waypoints.shape[0] < 2 or waypoints.shape[1] < 2:
      raise ValueError("Pure Pursuit requires at least two x/y waypoints.")
    self.waypoints = waypoints[:, :3] if waypoints.shape[1] >= 3 else torch.nn.functional.pad(waypoints, (0, 1))
    diffs = self.waypoints[1:, :2] - self.waypoints[:-1, :2]
    self.segment_lengths = torch.linalg.norm(diffs, dim=1)
    self.cumulative = torch.cat(
      [torch.zeros(1, device=self.device), torch.cumsum(self.segment_lengths, dim=0)]
    )
    self.total_length = self.cumulative[-1]
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.progress = torch.zeros(self.num_envs, device=self.device)
    self.lateral_error = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.reached_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.metrics["path_lateral_error"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    self.metrics["path_lateral_error"] += self.lateral_error / max(
      self.cfg.resampling_time_range[1] / self._env.step_dt, 1.0
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.progress[env_ids] = 0.0
    self.lateral_error[env_ids] = 0.0
    self.heading_error[env_ids] = 0.0
    self.reached_goal[env_ids] = False
    self._update_command_for_envs(env_ids)

  def _update_command(self) -> None:
    env_ids = torch.arange(self.num_envs, device=self.device)
    self._update_command_for_envs(env_ids)

  def _update_command_for_envs(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    pos_xy = self.robot.data.root_link_pos_w[env_ids, :2]
    heading = self.robot.data.heading_w[env_ids]
    progress, lateral = self._project_progress(pos_xy)
    target_progress = torch.minimum(
      progress + self.cfg.lookahead_distance,
      self.total_length.expand_as(progress),
    )
    target = self._point_at_progress(target_progress)
    delta = target[:, :2] - pos_xy
    target_heading = torch.atan2(delta[:, 1], delta[:, 0])
    heading_error = wrap_to_pi(target_heading - heading)
    reached = (self.total_length - progress) <= self.cfg.goal_tolerance

    speed = torch.full_like(progress, min(self.cfg.target_speed, self.cfg.max_linear_velocity))
    speed = torch.where(reached, torch.zeros_like(speed), speed)
    command_w = torch.stack(
      [speed * torch.cos(target_heading), speed * torch.sin(target_heading), torch.zeros_like(speed)],
      dim=1,
    )
    quat = self.robot.data.root_link_quat_w[env_ids]
    command_b = quat_apply_inverse(quat, command_w)
    yaw_rate = 2.0 * speed * torch.sin(heading_error) / max(
      self.cfg.lookahead_distance, 1e-6
    )
    yaw_rate = torch.clamp(
      yaw_rate,
      min=-self.cfg.max_yaw_rate,
      max=self.cfg.max_yaw_rate,
    )

    self.vel_command_b[env_ids, 0] = torch.clamp(
      command_b[:, 0],
      min=-self.cfg.max_linear_velocity,
      max=self.cfg.max_linear_velocity,
    )
    self.vel_command_b[env_ids, 1] = torch.clamp(
      command_b[:, 1],
      min=-self.cfg.max_linear_velocity,
      max=self.cfg.max_linear_velocity,
    )
    self.vel_command_b[env_ids, 2] = yaw_rate
    self.target_w[env_ids] = target
    self.progress[env_ids] = progress
    self.lateral_error[env_ids] = lateral
    self.heading_error[env_ids] = heading_error
    self.reached_goal[env_ids] = reached

  def _project_progress(self, pos_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    best_dist = torch.full((pos_xy.shape[0],), float("inf"), device=self.device)
    best_progress = torch.zeros_like(best_dist)
    for i, seg_len in enumerate(self.segment_lengths):
      if float(seg_len.item()) <= 1e-9:
        continue
      a = self.waypoints[i, :2]
      b = self.waypoints[i + 1, :2]
      ab = b - a
      t = torch.sum((pos_xy - a) * ab, dim=1) / (seg_len * seg_len)
      t = torch.clamp(t, 0.0, 1.0)
      closest = a + t[:, None] * ab
      dist = torch.linalg.norm(pos_xy - closest, dim=1)
      better = dist < best_dist
      best_dist = torch.where(better, dist, best_dist)
      best_progress = torch.where(better, self.cumulative[i] + t * seg_len, best_progress)
    return best_progress, best_dist

  def _point_at_progress(self, progress: torch.Tensor) -> torch.Tensor:
    idx = torch.searchsorted(self.cumulative, progress, right=True) - 1
    idx = torch.clamp(idx, 0, len(self.segment_lengths) - 1)
    seg_len = torch.clamp(self.segment_lengths[idx], min=1e-9)
    t = (progress - self.cumulative[idx]) / seg_len
    return self.waypoints[idx] + t[:, None] * (self.waypoints[idx + 1] - self.waypoints[idx])

