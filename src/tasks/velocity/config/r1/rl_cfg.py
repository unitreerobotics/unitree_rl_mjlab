"""RL configuration for Unitree R1 velocity task."""

from dataclasses import replace

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_r1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree R1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
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
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="r1_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def unitree_r1_steps_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree R1 blind step-up task."""
  # Started from the flat policy (--init-actor-from), so fewer iterations.
  return replace(unitree_r1_ppo_runner_cfg(), experiment_name="r1_steps", max_iterations=6001)


def unitree_r1_perceptive_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree R1 perceptive step-up task."""
  # The first run (2026-11-02) from scratch: the actor's input differs from the flat policy's.
  # Since 2026-11-03 started from its checkpoint 15800 (--init-actor-from) with mirror symmetry
  # (SymmetricVelocityOnPolicyRunner) and the changed rewards: 15001 iterations, ~17.5 h on the
  # laptop.
  return replace(unitree_r1_ppo_runner_cfg(), experiment_name="r1_perceptive", max_iterations=15001)
