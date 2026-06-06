"""Observation-encoder ablation RL configs for Unitree Go2 (velocity).

Each builder returns a runner cfg whose actor uses the same MLP head as the
standard Go2 velocity baseline. Non-raw variants prepend a configurable
observation encoder before that shared head; the raw variant uses the plain MLP
path. They target the split-observation env
(``unitree_go2_rough_split_obs_env_cfg``), which exposes ``height_scan``,
``command``, ``projected_gravity``, ``proprio`` and ``last_action`` as separate
observation groups.

The critic is a plain ``MLPModel`` over the privileged, concatenated ``critic``
group: this keeps the value function unchanged and demonstrates that actor and
critic are configured independently (the critic uses privileged observations
without affecting the actor).

These tasks should be trained with a runner that does NOT export ONNX (the
registrations use ``MjlabOnPolicyRunner``); ONNX/JIT export of encoder models is
out of scope for the ablation phase.
"""

from __future__ import annotations

import os
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

ENCODER_MODEL = "src.rl_models.encoder_mlp_model:EncoderMLPModel"

# Actor observation groups exposed by unitree_go2_rough_split_obs_env_cfg.
_ACTOR_GROUPS = (
  "height_scan",
  "command",
  "projected_gravity",
  "proprio",
  "last_action",
)


def _make_runner_cfg(
  experiment_name: str,
  observation_encoder_cfg: dict[str, Any] | None,
) -> RslRlOnPolicyRunnerCfg:
  actor_kwargs: dict[str, Any] = {}
  if observation_encoder_cfg is not None:
    actor_kwargs = {
      "class_name": ENCODER_MODEL,
      "observation_encoder_cfg": observation_encoder_cfg,
    }

  return RslRlOnPolicyRunnerCfg(
    obs_groups={"actor": _ACTOR_GROUPS, "critic": ("critic",)},
    actor=RslRlModelCfg(
      **actor_kwargs,
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
      # Plain MLP over the privileged concatenated critic group (unchanged).
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
    experiment_name=experiment_name,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def raw_height_scan_cfg() -> RslRlOnPolicyRunnerCfg:
  """Baseline: raw split observations + the standard Go2 actor MLP."""
  return _make_runner_cfg(
    "go2_velocity_encoder_raw",
    None,
  )


def mlp_encoder_height_only_cfg() -> RslRlOnPolicyRunnerCfg:
  """MLP feature encoder over height_scan only."""
  return _make_runner_cfg(
    "go2_velocity_encoder_mlp",
    {
      "type": "mlp",
      "encoder_input_keys": ["height_scan"],
      "passthrough_keys": None,
      "latent_dim": 32,
      "hidden_dims": [256, 128],
      "activation": "elu",
    },
  )


def mlp_encoder_with_state_cfg() -> RslRlOnPolicyRunnerCfg:
  """State-conditioned MLP encoder (height_scan + command + gravity)."""
  return _make_runner_cfg(
    "go2_velocity_encoder_mlp_state",
    {
      "type": "mlp",
      "encoder_input_keys": ["height_scan", "command", "projected_gravity"],
      "passthrough_keys": None,
      "latent_dim": 32,
      "hidden_dims": [256, 128],
      "activation": "elu",
    },
  )


def conv1d_encoder_cfg() -> RslRlOnPolicyRunnerCfg:
  """Conv1d encoder over height_scan only."""
  return _make_runner_cfg(
    "go2_velocity_encoder_conv1d",
    {
      "type": "conv1d",
      "encoder_input_keys": ["height_scan"],
      "passthrough_keys": None,
      "primary_key": "height_scan",
      "channels": [16, 32, 64],
      "kernel_sizes": [5, 3, 3],
      "strides": [2, 2, 1],
      "activation": "elu",
      "global_pool": "avg",
      "latent_dim": 32,
    },
  )


def conv1d_state_encoder_cfg() -> RslRlOnPolicyRunnerCfg:
  """State-conditioned Conv1d encoder (height_scan + command + gravity)."""
  return _make_runner_cfg(
    "go2_velocity_encoder_conv1d_state",
    {
      "type": "conv1d",
      "encoder_input_keys": ["height_scan", "command", "projected_gravity"],
      "passthrough_keys": None,
      "primary_key": "height_scan",
      "context_keys": ["command", "projected_gravity"],
      "channels": [16, 32, 64],
      "kernel_sizes": [5, 3, 3],
      "strides": [2, 2, 1],
      "activation": "elu",
      "global_pool": "avg",
      "context_hidden_dims": [64],
      "latent_dim": 32,
    },
  )


def conv2d_encoder_cfg() -> RslRlOnPolicyRunnerCfg:
  """Conv2d encoder over height_scan only, reshaped to a 17x11 grid."""
  return _make_runner_cfg(
    "go2_velocity_encoder_conv2d",
    {
      "type": "conv2d",
      "encoder_input_keys": ["height_scan"],
      "passthrough_keys": None,
      "primary_key": "height_scan",
      "input_hw": [17, 11],
      "channels": [16, 32, 64],
      "kernel_sizes": [3, 3, 3],
      "strides": [1, 2, 2],
      "activation": "elu",
      "global_pool": "avg",
      "latent_dim": 32,
    },
  )


def conv2d_state_encoder_cfg() -> RslRlOnPolicyRunnerCfg:
  """State-conditioned Conv2d encoder (height_scan + command + gravity)."""
  return _make_runner_cfg(
    "go2_velocity_encoder_conv2d_state",
    {
      "type": "conv2d",
      "encoder_input_keys": ["height_scan", "command", "projected_gravity"],
      "passthrough_keys": None,
      "primary_key": "height_scan",
      "context_keys": ["command", "projected_gravity"],
      "input_hw": [17, 11],
      "channels": [16, 32, 64],
      "kernel_sizes": [3, 3, 3],
      "strides": [1, 2, 2],
      "activation": "elu",
      "global_pool": "avg",
      "context_hidden_dims": [64],
      "latent_dim": 32,
    },
  )


def pretrained_ae_encoder_cfg() -> RslRlOnPolicyRunnerCfg:
  """Pretrained autoencoder encoder over height_scan, with state context."""
  checkpoint_path = os.environ.get(
    "HEIGHT_SCAN_AE_CHECKPOINT",
    "logs/pretrained_autoencoders/height_scan_ae.pt",
  )
  return _make_runner_cfg(
    "go2_velocity_encoder_ae",
    {
      "type": "pretrained_ae",
      "encoder_input_keys": ["height_scan", "command", "projected_gravity"],
      "passthrough_keys": None,
      "primary_key": "height_scan",
      "context_keys": ["command", "projected_gravity"],
      "checkpoint_path": checkpoint_path,
      "encoder_class": "src.rl_models.autoencoder:HeightScanAutoEncoder",
      "latent_dim": 32,
      "context_hidden_dims": [64],
      "freeze": True,
      "strict": False,
    },
  )

