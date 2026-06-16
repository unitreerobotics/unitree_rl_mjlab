from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .encoder_ablation_rl_cfg import (
  conv1d_encoder_cfg,
  conv1d_state_encoder_cfg,
  conv2d_encoder_cfg,
  conv2d_state_encoder_cfg,
  mlp_encoder_height_only_cfg,
  mlp_encoder_with_state_cfg,
  pretrained_ae_encoder_cfg,
  raw_height_scan_cfg,
)
from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_flat_scan_env_cfg,
  unitree_go2_no_phase_env_cfg,
  unitree_go2_rough_env_cfg,
  unitree_go2_rough_no_height_scan_env_cfg,
  unitree_go2_rough_split_obs_env_cfg,
  unitree_go2_test_env_cfg,
  unitree_go2_test_train_env_cfg,
)
from .moe_env_cfgs import (
  unitree_go2_expert_climb_env_cfg,
  unitree_go2_expert_rough_env_cfg,
  unitree_go2_expert_stairs_env_cfg,
  unitree_go2_moe_mixed_env_cfg,
)
from .moe_rl_cfg import (
  unitree_go2_expert_runner_cfg,
  unitree_go2_moe_runner_cfg,
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
  task_id="Unitree-Go2-Expert-Flat",
  env_cfg=unitree_go2_flat_scan_env_cfg(),
  play_env_cfg=unitree_go2_flat_scan_env_cfg(play=True),
  rl_cfg=unitree_go2_expert_runner_cfg("go2_expert_flat"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Expert-Rough",
  env_cfg=unitree_go2_expert_rough_env_cfg(),
  play_env_cfg=unitree_go2_expert_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_expert_runner_cfg("go2_expert_rough"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Expert-Stairs",
  env_cfg=unitree_go2_expert_stairs_env_cfg(),
  play_env_cfg=unitree_go2_expert_stairs_env_cfg(play=True),
  rl_cfg=unitree_go2_expert_runner_cfg("go2_expert_stairs"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Expert-Climb",
  env_cfg=unitree_go2_expert_climb_env_cfg(),
  play_env_cfg=unitree_go2_expert_climb_env_cfg(play=True),
  rl_cfg=unitree_go2_expert_runner_cfg("go2_expert_climb"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-MoE-Mixed",
  env_cfg=unitree_go2_moe_mixed_env_cfg(),
  play_env_cfg=unitree_go2_moe_mixed_env_cfg(play=True),
  rl_cfg=unitree_go2_moe_runner_cfg(),
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

# --- Observation-encoder ablation tasks -------------------------------------
# All share the split-observation env and use MjlabOnPolicyRunner (no ONNX
# export). Switch the encoder architecture by picking a different rl_cfg.
for _suffix, _cfg_fn in (
  ("Raw", raw_height_scan_cfg),
  ("MLP", mlp_encoder_height_only_cfg),
  ("MLPState", mlp_encoder_with_state_cfg),
  ("Conv1d", conv1d_encoder_cfg),
  ("Conv1dState", conv1d_state_encoder_cfg),
  ("Conv2d", conv2d_encoder_cfg),
  ("Conv2dState", conv2d_state_encoder_cfg),
  ("AE", pretrained_ae_encoder_cfg),
):
  register_mjlab_task(
    task_id=f"Unitree-Go2-Rough-Encoder-{_suffix}",
    env_cfg=unitree_go2_rough_split_obs_env_cfg(),
    play_env_cfg=unitree_go2_rough_split_obs_env_cfg(play=True),
    rl_cfg=_cfg_fn(),
    runner_cls=MjlabOnPolicyRunner,
  )
