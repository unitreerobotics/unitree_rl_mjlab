"""Canonical Unitree Go2-W velocity tasks."""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2w_flat_env_cfg,
  unitree_go2w_flat_legs_only_env_cfg,
  unitree_go2w_flat_legs_only_omni_env_cfg,
  unitree_go2w_rough_env_cfg,
)
from .rl_cfg import (
  unitree_go2w_flat_legs_only_omni_finetune_ppo_runner_cfg,
  unitree_go2w_flat_legs_only_omni_ppo_runner_cfg,
  unitree_go2w_flat_legs_only_ppo_runner_cfg,
  unitree_go2w_ppo_runner_cfg,
  unitree_go2w_rough_finetune_ppo_runner_cfg,
  unitree_go2w_rough_ppo_runner_cfg,
)


register_mjlab_task(
  task_id="go2w-rough",
  env_cfg=unitree_go2w_rough_env_cfg(),
  play_env_cfg=unitree_go2w_rough_env_cfg(play=True),
  rl_cfg=unitree_go2w_rough_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="go2w-rough-finetune",
  env_cfg=unitree_go2w_rough_env_cfg(),
  play_env_cfg=unitree_go2w_rough_env_cfg(play=True),
  rl_cfg=unitree_go2w_rough_finetune_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="go2w-flat",
  env_cfg=unitree_go2w_flat_env_cfg(),
  play_env_cfg=unitree_go2w_flat_env_cfg(play=True),
  rl_cfg=unitree_go2w_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="go2w-flat-legs",
  env_cfg=unitree_go2w_flat_legs_only_env_cfg(),
  play_env_cfg=unitree_go2w_flat_legs_only_env_cfg(play=True),
  rl_cfg=unitree_go2w_flat_legs_only_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="go2w-flat-legs-omni",
  env_cfg=unitree_go2w_flat_legs_only_omni_env_cfg(),
  play_env_cfg=unitree_go2w_flat_legs_only_omni_env_cfg(play=True),
  rl_cfg=unitree_go2w_flat_legs_only_omni_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="go2w-flat-legs-omni-finetune",
  env_cfg=unitree_go2w_flat_legs_only_omni_env_cfg(),
  play_env_cfg=unitree_go2w_flat_legs_only_omni_env_cfg(play=True),
  rl_cfg=unitree_go2w_flat_legs_only_omni_finetune_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
