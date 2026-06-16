"""RL configs for Go2 expert pretraining and MoE gate training."""

from __future__ import annotations

import os
from typing import Any

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .rl_cfg import unitree_go2_ppo_runner_cfg

MOE_MODEL = "src.rl_models.moe_model:MoEMLPModel"


def unitree_go2_expert_runner_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.experiment_name = experiment_name
  return cfg


def _env_flag(name: str, default: str) -> bool:
  return os.environ.get(name, default).strip().lower() not in {"0", "false", "no"}


def _expert_cfgs() -> list[dict[str, str]]:
  return [
    {"name": "flat", "checkpoint": os.environ.get("GO2_MOE_EXPERT_FLAT", "")},
    {"name": "rough", "checkpoint": os.environ.get("GO2_MOE_EXPERT_ROUGH", "")},
    {"name": "stairs", "checkpoint": os.environ.get("GO2_MOE_EXPERT_STAIRS", "")},
    {"name": "climb", "checkpoint": os.environ.get("GO2_MOE_EXPERT_CLIMB", "")},
  ]


def _moe_cfg() -> dict[str, Any]:
  return {
    "experts": _expert_cfgs(),
    "expert_hidden_dims": [512, 256, 128],
    "expert_activation": "elu",
    "expert_obs_normalization": True,
    "freeze_experts": _env_flag("GO2_MOE_FREEZE_EXPERTS", "1"),
    "gate_hidden_dims": [256, 128],
    "gate_activation": "elu",
    "gate_obs_normalization": True,
    "gate_temperature": float(os.environ.get("GO2_MOE_GATE_TEMPERATURE", "1.0")),
    "allow_missing_checkpoints": _env_flag("GO2_MOE_ALLOW_MISSING_CHECKPOINTS", "1"),
  }


def unitree_go2_moe_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name=MOE_MODEL,
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
      moe_cfg=_moe_cfg(),
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
    experiment_name="go2_moe_mixed",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
