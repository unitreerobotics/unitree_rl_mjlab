"""Project velocity command with mjlab v1.6 reset and viewer interfaces."""

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand as MjlabUniformVelocityCommand,
)
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommandCfg as MjlabUniformVelocityCommandCfg,
)

__all__ = ["UniformVelocityCommand", "UniformVelocityCommandCfg"]


class UniformVelocityCommand(MjlabUniformVelocityCommand):
  """Retain the project's small-command deadband on top of the upstream term."""

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    cmd = self.vel_command_b[env_ids]
    total_command = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])
    active = (total_command > 0.1).unsqueeze(1)
    self.vel_command_b[env_ids] *= active
    self.vel_command_w[env_ids] *= active


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(MjlabUniformVelocityCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)
