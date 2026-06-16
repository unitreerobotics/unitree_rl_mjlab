from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2w_flat_env_cfg,
  unitree_go2w_rough_env_cfg,
)
from .moe_env_cfgs import (
  unitree_go2w_expert_climb_env_cfg,
  unitree_go2w_expert_flat_env_cfg,
  unitree_go2w_expert_rough_env_cfg,
  unitree_go2w_expert_stairs_env_cfg,
  unitree_go2w_moe_mixed_env_cfg,
  unitree_go2w_noheight_expert_climb_env_cfg,
  unitree_go2w_noheight_expert_flat_env_cfg,
  unitree_go2w_noheight_expert_rough_env_cfg,
  unitree_go2w_noheight_expert_stairs_env_cfg,
  unitree_go2w_noheight_moe_mixed_env_cfg,
)
from .moe_rl_cfg import (
  unitree_go2w_expert_runner_cfg,
  unitree_go2w_moe_runner_cfg,
  unitree_go2w_noheight_moe_runner_cfg,
)
from .rl_cfg import unitree_go2w_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2W-Rough",
  env_cfg=unitree_go2w_rough_env_cfg(),
  play_env_cfg=unitree_go2w_rough_env_cfg(play=True),
  rl_cfg=unitree_go2w_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2W-Flat",
  env_cfg=unitree_go2w_flat_env_cfg(),
  play_env_cfg=unitree_go2w_flat_env_cfg(play=True),
  rl_cfg=unitree_go2w_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2W-Expert-Flat",
  env_cfg=unitree_go2w_expert_flat_env_cfg(),
  play_env_cfg=unitree_go2w_expert_flat_env_cfg(play=True),
  rl_cfg=unitree_go2w_expert_runner_cfg("go2w_expert_flat"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2W-Expert-Rough",
  env_cfg=unitree_go2w_expert_rough_env_cfg(),
  play_env_cfg=unitree_go2w_expert_rough_env_cfg(play=True),
  rl_cfg=unitree_go2w_expert_runner_cfg("go2w_expert_rough"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2W-Expert-Stairs",
  env_cfg=unitree_go2w_expert_stairs_env_cfg(),
  play_env_cfg=unitree_go2w_expert_stairs_env_cfg(play=True),
  rl_cfg=unitree_go2w_expert_runner_cfg("go2w_expert_stairs"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2W-Expert-Climb",
  env_cfg=unitree_go2w_expert_climb_env_cfg(),
  play_env_cfg=unitree_go2w_expert_climb_env_cfg(play=True),
  rl_cfg=unitree_go2w_expert_runner_cfg("go2w_expert_climb"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2W-MoE-Mixed",
  env_cfg=unitree_go2w_moe_mixed_env_cfg(),
  play_env_cfg=unitree_go2w_moe_mixed_env_cfg(play=True),
  rl_cfg=unitree_go2w_moe_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2W-NoHeight-Expert-Flat",
  env_cfg=unitree_go2w_noheight_expert_flat_env_cfg(),
  play_env_cfg=unitree_go2w_noheight_expert_flat_env_cfg(play=True),
  rl_cfg=unitree_go2w_expert_runner_cfg("go2w_noheight_expert_flat"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2W-NoHeight-Expert-Rough",
  env_cfg=unitree_go2w_noheight_expert_rough_env_cfg(),
  play_env_cfg=unitree_go2w_noheight_expert_rough_env_cfg(play=True),
  rl_cfg=unitree_go2w_expert_runner_cfg("go2w_noheight_expert_rough"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2W-NoHeight-Expert-Stairs",
  env_cfg=unitree_go2w_noheight_expert_stairs_env_cfg(),
  play_env_cfg=unitree_go2w_noheight_expert_stairs_env_cfg(play=True),
  rl_cfg=unitree_go2w_expert_runner_cfg("go2w_noheight_expert_stairs"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2W-NoHeight-Expert-Climb",
  env_cfg=unitree_go2w_noheight_expert_climb_env_cfg(),
  play_env_cfg=unitree_go2w_noheight_expert_climb_env_cfg(play=True),
  rl_cfg=unitree_go2w_expert_runner_cfg("go2w_noheight_expert_climb"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2W-NoHeight-MoE-Mixed",
  env_cfg=unitree_go2w_noheight_moe_mixed_env_cfg(),
  play_env_cfg=unitree_go2w_noheight_moe_mixed_env_cfg(play=True),
  rl_cfg=unitree_go2w_noheight_moe_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
