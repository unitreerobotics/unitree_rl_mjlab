from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_flat_scan_env_cfg,
  unitree_go2_no_phase_env_cfg,
  unitree_go2_rough_env_cfg,
  unitree_go2_rough_no_height_scan_env_cfg,
  unitree_go2_test_env_cfg,
  unitree_go2_test_train_env_cfg,
)
from .rl_cfg import unitree_go2_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2-Rough",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-No-Height-Scan",
  env_cfg=unitree_go2_rough_no_height_scan_env_cfg(),
  play_env_cfg=unitree_go2_rough_no_height_scan_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat-Scan",
  env_cfg=unitree_go2_flat_scan_env_cfg(),
  play_env_cfg=unitree_go2_flat_scan_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-No-Phase-Rough",
  env_cfg=unitree_go2_no_phase_env_cfg(),
  play_env_cfg=unitree_go2_no_phase_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Test",
  env_cfg=unitree_go2_test_env_cfg(),
  play_env_cfg=unitree_go2_test_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Test-Train",
  env_cfg=unitree_go2_test_train_env_cfg(),
  play_env_cfg=unitree_go2_test_train_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
