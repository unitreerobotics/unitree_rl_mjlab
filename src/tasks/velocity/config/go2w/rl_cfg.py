"""RL configuration for active Unitree Go2-W velocity tasks."""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def make_go2w_runner_cfg(
  *,
  experiment_name: str = "go2w_velocity",
  run_name: str | None = None,
  max_iterations: int = 10001,
  save_interval: int = 1000,
  init_noise_std: float = 1.0,
  learning_rate: float = 1.0e-3,
  entropy_coef: float = 0.01,
  max_grad_norm: float = 1.0,
  resume: bool = False,
  load_checkpoint: str | None = None,
) -> RslRlOnPolicyRunnerCfg:
  """Build a Go2-W PPO runner config from the shared lab defaults."""
  cfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": init_noise_std,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=entropy_coef,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=learning_rate,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=max_grad_norm,
    ),
    experiment_name=experiment_name,
    run_name=run_name,
    save_interval=save_interval,
    num_steps_per_env=24,
    max_iterations=max_iterations,
  )
  cfg.resume = resume
  if load_checkpoint is not None:
    cfg.load_checkpoint = load_checkpoint
  return cfg


def unitree_go2w_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return make_go2w_runner_cfg()


def unitree_go2w_rough_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return make_go2w_runner_cfg(
    experiment_name="go2w_velocity",
    init_noise_std=0.6,
    learning_rate=5.0e-4,
    entropy_coef=0.005,
    max_grad_norm=0.5,
  )


def unitree_go2w_rough_finetune_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return make_go2w_runner_cfg(
    experiment_name="go2w_velocity",
    run_name="rough_ft",
    init_noise_std=0.6,
    learning_rate=5.0e-4,
    entropy_coef=0.005,
    max_grad_norm=0.5,
    resume=True,
    load_checkpoint="model_10000.pt",
  )


def unitree_go2w_flat_legs_only_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return make_go2w_runner_cfg(
    experiment_name="go2w_legs_only",
    run_name="flat_walk",
    max_iterations=20001,
    init_noise_std=0.4,
    learning_rate=5.0e-4,
    entropy_coef=0.003,
    max_grad_norm=0.5,
  )


def unitree_go2w_flat_legs_only_omni_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return make_go2w_runner_cfg(
    experiment_name="go2w_legs_only_omni",
    run_name="flat_walk_omni",
    max_iterations=30001,
    init_noise_std=0.3,
    learning_rate=5.0e-4,
    entropy_coef=0.002,
    max_grad_norm=0.5,
  )


def unitree_go2w_flat_legs_only_omni_finetune_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return make_go2w_runner_cfg(
    experiment_name="go2w_legs_only",
    run_name="flat_walk_omni_ft",
    max_iterations=30001,
    init_noise_std=0.3,
    learning_rate=5.0e-4,
    entropy_coef=0.002,
    max_grad_norm=0.5,
    resume=True,
    load_checkpoint="model_20000.pt",
  )
