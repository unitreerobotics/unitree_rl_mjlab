from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import SymmetricVelocityOnPolicyRunner, VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_r1_flat_env_cfg,
  unitree_r1_perceptive_env_cfg,
  unitree_r1_rough_env_cfg,
  unitree_r1_steps_env_cfg,
)
from .rl_cfg import (
  unitree_r1_perceptive_ppo_runner_cfg,
  unitree_r1_ppo_runner_cfg,
  unitree_r1_steps_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Unitree-R1-Rough",
  env_cfg=unitree_r1_rough_env_cfg(),
  play_env_cfg=unitree_r1_rough_env_cfg(play=True),
  rl_cfg=unitree_r1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-R1-Flat",
  env_cfg=unitree_r1_flat_env_cfg(),
  play_env_cfg=unitree_r1_flat_env_cfg(play=True),
  rl_cfg=unitree_r1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-R1-Steps",
  env_cfg=unitree_r1_steps_env_cfg(),
  play_env_cfg=unitree_r1_steps_env_cfg(play=True),
  rl_cfg=unitree_r1_steps_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-R1-Perceptive",
  env_cfg=unitree_r1_perceptive_env_cfg(),
  play_env_cfg=unitree_r1_perceptive_env_cfg(play=True),
  rl_cfg=unitree_r1_perceptive_ppo_runner_cfg(),
  runner_cls=SymmetricVelocityOnPolicyRunner,
)
