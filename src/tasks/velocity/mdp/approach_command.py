"""Velocity commands with extra slow straight walks: the operator's command at a pallet.

mjlab's UniformVelocityCommand draws vx uniformly (R1 steps: -0.5 to 1.0 m/s), so few commands
fall where the operator walks the robot onto a pallet, 0.3 to 0.6 m/s straight ahead. There the
perceptive policy v3 hesitated and stopped before the edge (offline: 1 success in 16 at 0.3 to
0.4 m/s). A share of the commands is redrawn in that band.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class ApproachVelocityCommand(UniformVelocityCommand):
  cfg: ApproachVelocityCommandCfg

  def __init__(self, cfg: ApproachVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    # Which envs walk an approach command now; the rewards' heading_frame option reads it.
    self.is_approach_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    if len(env_ids) == 0:
      return
    self.is_approach_env[env_ids] = False
    if self.cfg.approach_prob <= 0.0:
      return
    r = torch.empty(len(env_ids), device=self.device)
    approach = env_ids[r.uniform_(0.0, 1.0) < self.cfg.approach_prob]
    if len(approach) == 0:
      return
    self.is_approach_env[approach] = True
    r = torch.empty(len(approach), device=self.device)
    self.vel_command_b[approach, 0] = r.uniform_(*self.cfg.approach_lin_vel_x)
    self.vel_command_b[approach, 1] = r.uniform_(*self.cfg.approach_lin_vel_y)
    self.is_standing_env[approach] = False   # _update_command zeroes a standing env's command
    if self.cfg.approach_hold_heading and self.cfg.heading_command:
      # Straight on along the heading the robot has now (a little off it at most), as an operator
      # holds it: _update_command turns the heading error into the yaw rate command, so turning
      # away from the line costs reward for as long as it lasts, not only while turning. With a
      # drawn yaw rate instead, checkpoint 15300 learned to ignore small turn commands and to turn
      # 30-40 deg away just before a step (offline, heading steering: 5 of 60 at the EUR pallet).
      self.heading_target[approach] = wrap_to_pi(
        self.robot.data.heading_w[approach] + r.uniform_(*self.cfg.approach_heading_offset))
      self.is_heading_env[approach] = True
    else:
      self.vel_command_b[approach, 2] = r.uniform_(*self.cfg.approach_ang_vel_z)
      self.is_heading_env[approach] = False


@dataclass(kw_only=True)
class ApproachVelocityCommandCfg(UniformVelocityCommandCfg):
  approach_prob: float = 0.3
  """Share of resampled commands redrawn as a slow straight walk."""
  approach_lin_vel_x: tuple[float, float] = (0.3, 0.6)
  approach_lin_vel_y: tuple[float, float] = (-0.1, 0.1)
  approach_hold_heading: bool = True
  """Hold a heading (heading env) instead of a drawn yaw rate."""
  approach_heading_offset: tuple[float, float] = (-0.1, 0.1)
  """Heading to hold relative to the robot's heading at the resample, in rad."""
  approach_ang_vel_z: tuple[float, float] = (-0.2, 0.2)
  """Yaw rate of these walks when approach_hold_heading is False."""

  def build(self, env: ManagerBasedRlEnv) -> ApproachVelocityCommand:
    return ApproachVelocityCommand(self, env)
