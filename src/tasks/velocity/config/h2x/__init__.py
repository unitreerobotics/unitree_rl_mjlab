from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_h2x_flat_env_cfg,
  unitree_h2x_rough_env_cfg,
)
from .rl_cfg import unitree_h2x_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-H2X-Rough",
  env_cfg=unitree_h2x_rough_env_cfg(),
  play_env_cfg=unitree_h2x_rough_env_cfg(play=True),
  rl_cfg=unitree_h2x_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-H2X-Flat",
  env_cfg=unitree_h2x_flat_env_cfg(),
  play_env_cfg=unitree_h2x_flat_env_cfg(play=True),
  rl_cfg=unitree_h2x_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
