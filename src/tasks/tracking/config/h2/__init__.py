from mjlab.tasks.registry import register_mjlab_task
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import unitree_h2_flat_tracking_env_cfg
from .rl_cfg import unitree_h2_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-H2-Tracking",
  env_cfg=unitree_h2_flat_tracking_env_cfg(),
  play_env_cfg=unitree_h2_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_h2_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-H2-Tracking-No-State-Estimation",
  env_cfg=unitree_h2_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_h2_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_h2_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
