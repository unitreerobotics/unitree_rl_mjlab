"""Tests for the configurable observation-encoder framework.

Pure-PyTorch tests: no simulator/env is needed. Observations are faked with
``TensorDict`` keyed by observation-group names, matching how rsl_rl models
receive observations.
"""

from __future__ import annotations

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from src.rl_models.autoencoder import HeightScanAutoEncoder
from src.rl_models.encoder_mlp_model import EncoderMLPModel
from src.rl_models.encoders import build_observation_encoder
from src.tasks.velocity.config.go2.encoder_ablation_rl_cfg import (
  ENCODER_MODEL,
  conv1d_encoder_cfg,
  conv1d_state_encoder_cfg,
  conv2d_encoder_cfg,
  conv2d_state_encoder_cfg,
  mlp_encoder_height_only_cfg,
  mlp_encoder_with_state_cfg,
  raw_height_scan_cfg,
)
from src.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg

B = 4
HEIGHT_L = 187  # Go2 height scan: 17 x 11 flattened.


def _shapes(obs: TensorDict, keys):
  return {k: tuple(obs[k].shape[1:]) for k in keys}


def _make_obs() -> TensorDict:
  return TensorDict(
    {
      "height_scan": torch.randn(B, HEIGHT_L),
      "command": torch.randn(B, 3),
      "projected_gravity": torch.randn(B, 3),
      "proprio": torch.randn(B, 29),
      "last_action": torch.randn(B, 12),
      "privileged": torch.randn(B, 8),
    },
    batch_size=[B],
  )


# --------------------------------------------------------------------------
# Encoders (standalone forward)
# --------------------------------------------------------------------------


def test_identity_encoder_returns_raw():
  obs = _make_obs()
  keys = ["height_scan", "command"]
  enc = build_observation_encoder(
    {"type": "identity", "flatten": True}, _shapes(obs, keys), keys
  )
  out = enc({k: obs[k] for k in keys})
  assert enc.output_dim == HEIGHT_L + 3
  assert out.shape == (B, HEIGHT_L + 3)
  assert not torch.isnan(out).any()


def test_mlp_encoder_latent_shape():
  obs = _make_obs()
  keys = ["height_scan", "command", "projected_gravity"]
  enc = build_observation_encoder(
    {
      "type": "mlp",
      "latent_dim": 32,
      "hidden_dims": [256, 128],
      "activation": "elu",
      "layer_norm": True,
    },
    _shapes(obs, keys),
    keys,
  )
  out = enc({k: obs[k] for k in keys})
  assert enc.output_dim == 32
  assert out.shape == (B, 32)
  assert not torch.isnan(out).any()


def test_conv1d_encoder_with_context():
  obs = _make_obs()
  keys = ["height_scan", "command", "projected_gravity"]
  enc = build_observation_encoder(
    {
      "type": "conv1d",
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
    _shapes(obs, keys),
    keys,
  )
  out = enc({k: obs[k] for k in keys})
  assert enc.output_dim == 32 and out.shape == (B, 32)
  assert not torch.isnan(out).any()


def test_conv1d_bad_shape_errors():
  enc = build_observation_encoder(
    {"type": "conv1d", "primary_key": "height_scan", "latent_dim": 8,
     "channels": [8], "kernel_sizes": [3], "strides": [1]},
    {"height_scan": (HEIGHT_L,)},
    ["height_scan"],
  )
  bad = torch.randn(B, 4, 4, 4)  # 4D, not [B, L] / [B, C, L]
  with pytest.raises(ValueError, match="must be"):
    enc({"height_scan": bad})


def test_conv2d_encoder_flat_reshape():
  obs = _make_obs()
  keys = ["height_scan", "command"]
  enc = build_observation_encoder(
    {
      "type": "conv2d",
      "primary_key": "height_scan",
      "context_keys": ["command"],
      "input_hw": [17, 11],
      "channels": [16, 32],
      "kernel_sizes": [3, 3],
      "strides": [1, 2],
      "context_hidden_dims": [64],
      "latent_dim": 32,
    },
    _shapes(obs, keys),
    keys,
  )
  out = enc({k: obs[k] for k in keys})
  assert enc.output_dim == 32 and out.shape == (B, 32)
  assert not torch.isnan(out).any()


def test_conv2d_flat_without_input_hw_errors():
  with pytest.raises(ValueError, match="input_hw"):
    build_observation_encoder(
      {"type": "conv2d", "primary_key": "height_scan", "latent_dim": 8},
      {"height_scan": (HEIGHT_L,)},
      ["height_scan"],
    )


def _make_ae_checkpoint(tmp_path, latent_dim=16):
  model = HeightScanAutoEncoder(
    input_dim=HEIGHT_L,
    latent_dim=latent_dim,
    hidden_dims=[64],
    decoder_hidden_dims=[64],
  )
  path = tmp_path / "height_scan_ae.pt"
  torch.save(
    {
      "state_dict": model.state_dict(),
      "model_kwargs": {
        "input_dim": HEIGHT_L,
        "latent_dim": latent_dim,
        "hidden_dims": [64],
        "decoder_hidden_dims": [64],
      },
    },
    path,
  )
  return path


def _pretrained_ae_cfg(checkpoint_path, freeze=True, latent_dim=16):
  return {
    "type": "pretrained_ae",
    "encoder_input_keys": ["height_scan"],
    "checkpoint_path": str(checkpoint_path),
    "latent_dim": latent_dim,
    "freeze": freeze,
    "encoder_kwargs": {"hidden_dims": [64], "decoder_hidden_dims": [64]},
  }


def test_pretrained_ae_encoder_frozen(tmp_path):
  obs = _make_obs()
  keys = ["height_scan"]
  enc = build_observation_encoder(
    _pretrained_ae_cfg(_make_ae_checkpoint(tmp_path), freeze=True),
    _shapes(obs, keys),
    keys,
  )
  out = enc({k: obs[k] for k in keys})
  assert enc.output_dim == 16 and out.shape == (B, 16)
  assert not torch.isnan(out).any()
  assert all(not p.requires_grad for p in enc.pretrained_encoder.parameters())

  enc.train()
  assert not enc.pretrained_encoder.training
  assert all(not p.requires_grad for p in enc.pretrained_encoder.parameters())


def test_pretrained_ae_encoder_finetune_trainable(tmp_path):
  obs = _make_obs()
  keys = ["height_scan"]
  enc = build_observation_encoder(
    _pretrained_ae_cfg(_make_ae_checkpoint(tmp_path), freeze=False),
    _shapes(obs, keys),
    keys,
  )
  assert any(p.requires_grad for p in enc.pretrained_encoder.parameters())
  assert enc({k: obs[k] for k in keys}).shape == (B, 16)


def test_pretrained_ae_encoder_with_context(tmp_path):
  obs = _make_obs()
  keys = ["height_scan", "command", "projected_gravity"]
  cfg = _pretrained_ae_cfg(_make_ae_checkpoint(tmp_path), freeze=True)
  cfg.update(
    {
      "encoder_input_keys": keys,
      "primary_key": "height_scan",
      "context_keys": ["command", "projected_gravity"],
      "context_hidden_dims": [8],
    }
  )
  enc = build_observation_encoder(cfg, _shapes(obs, keys), keys)
  out = enc({k: obs[k] for k in keys})
  assert enc.output_dim == 16 and out.shape == (B, 16)
  assert not torch.isnan(out).any()


def test_planned_encoders_not_implemented():
  with pytest.raises(NotImplementedError, match="planned"):
    build_observation_encoder({"type": "transformer"}, {"height_scan": (HEIGHT_L,)}, ["height_scan"])


def test_builder_unknown_type_errors():
  with pytest.raises(ValueError, match="Unknown observation encoder type"):
    build_observation_encoder({"type": "nope"}, {"height_scan": (HEIGHT_L,)}, ["height_scan"])


# --------------------------------------------------------------------------
# EncoderMLPModel
# --------------------------------------------------------------------------


def _obs_groups():
  return {
    "actor": ["height_scan", "command", "projected_gravity", "proprio", "last_action"],
    "critic": [
      "height_scan",
      "command",
      "projected_gravity",
      "proprio",
      "last_action",
      "privileged",
    ],
  }


def _dist_cfg():
  return {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}


def test_encoder_model_actor_critic_forward():
  obs = _make_obs()
  groups = _obs_groups()
  oe = {
    "type": "mlp",
    "encoder_input_keys": ["height_scan"],
    "passthrough_keys": None,
    "latent_dim": 32,
    "hidden_dims": [128, 64],
    "activation": "elu",
  }
  actor = EncoderMLPModel(
    obs, groups, "actor", 12, hidden_dims=[64, 64],
    distribution_cfg=_dist_cfg(), observation_encoder_cfg=oe,
  )
  critic = EncoderMLPModel(
    obs, groups, "critic", 1, hidden_dims=[64, 64],
    observation_encoder_cfg=oe,
  )

  # passthrough = all actor groups except height_scan.
  expected_pass = 3 + 3 + 29 + 12
  assert actor.obs_dim == expected_pass
  assert actor._get_latent_dim() == 32 + expected_pass

  act_out = actor(obs, stochastic_output=True)
  val_out = critic(obs)
  assert act_out.shape == (B, 12)
  assert val_out.shape == (B, 1)
  assert not torch.isnan(act_out).any()
  assert not torch.isnan(val_out).any()

  # Critic's privileged group widens its passthrough but not the actor's.
  assert critic.obs_dim == expected_pass + 8


def test_encoder_model_explicit_passthrough_and_state_conditioning():
  obs = _make_obs()
  groups = _obs_groups()
  oe = {
    "type": "mlp",
    "encoder_input_keys": ["height_scan", "command", "projected_gravity"],
    "passthrough_keys": ["proprio", "last_action"],
    "latent_dim": 16,
    "hidden_dims": [64],
    "activation": "elu",
  }
  actor = EncoderMLPModel(
    obs, groups, "actor", 12, hidden_dims=[32],
    distribution_cfg=_dist_cfg(), observation_encoder_cfg=oe,
  )
  assert actor.obs_dim == 29 + 12
  assert actor._get_latent_dim() == 16 + 29 + 12
  assert actor(obs, stochastic_output=True).shape == (B, 12)


def test_encoder_model_bad_key_errors():
  obs = _make_obs()
  groups = _obs_groups()
  oe = {"type": "mlp", "encoder_input_keys": ["does_not_exist"], "latent_dim": 8}
  with pytest.raises(ValueError, match="not in obs_groups"):
    EncoderMLPModel(obs, groups, "actor", 12, observation_encoder_cfg=oe)


def test_trainable_params_include_encoder():
  obs = _make_obs()
  groups = _obs_groups()
  oe = {"type": "mlp", "encoder_input_keys": ["height_scan"], "latent_dim": 32,
        "hidden_dims": [64]}
  model = EncoderMLPModel(obs, groups, "actor", 12, observation_encoder_cfg=oe,
                          distribution_cfg=_dist_cfg())
  enc_params = {id(p) for p in model.observation_encoder.parameters()}
  model_params = {id(p) for p in model.parameters()}
  assert enc_params and enc_params.issubset(model_params)


# --------------------------------------------------------------------------
# Backward compatibility
# --------------------------------------------------------------------------


def test_encoder_model_none_matches_mlpmodel():
  obs = _make_obs()
  groups = {"actor": ["proprio", "last_action"], "critic": ["proprio"]}
  torch.manual_seed(0)
  ref = MLPModel(obs, groups, "actor", 12, hidden_dims=[64, 64],
                 distribution_cfg=_dist_cfg())
  torch.manual_seed(0)
  enc = EncoderMLPModel(obs, groups, "actor", 12, hidden_dims=[64, 64],
                        distribution_cfg=_dist_cfg(), observation_encoder_cfg=None)
  assert enc.obs_dim == ref.obs_dim
  assert enc._get_latent_dim() == ref._get_latent_dim()
  ref_latent = ref.get_latent(obs)
  enc_latent = enc.get_latent(obs)
  assert torch.allclose(ref_latent, enc_latent)
  assert enc(obs).shape == ref(obs).shape


# --------------------------------------------------------------------------
# Go2 encoder-ablation RL configs
# --------------------------------------------------------------------------


def _encoder_ablation_cfgs():
  return (
    raw_height_scan_cfg(),
    mlp_encoder_height_only_cfg(),
    mlp_encoder_with_state_cfg(),
    conv1d_encoder_cfg(),
    conv1d_state_encoder_cfg(),
    conv2d_encoder_cfg(),
    conv2d_state_encoder_cfg(),
  )


def test_go2_encoder_ablation_configs_share_baseline_training_head():
  baseline = unitree_go2_ppo_runner_cfg()

  for cfg in _encoder_ablation_cfgs():
    assert cfg.actor.hidden_dims == baseline.actor.hidden_dims
    assert cfg.actor.activation == baseline.actor.activation
    assert cfg.actor.obs_normalization == baseline.actor.obs_normalization
    assert cfg.actor.distribution_cfg == baseline.actor.distribution_cfg
    assert cfg.critic == baseline.critic
    assert cfg.algorithm == baseline.algorithm
    assert cfg.save_interval == baseline.save_interval
    assert cfg.num_steps_per_env == baseline.num_steps_per_env
    assert cfg.max_iterations == baseline.max_iterations


def test_go2_raw_encoder_config_is_plain_baseline_mlp():
  baseline = unitree_go2_ppo_runner_cfg()
  raw = raw_height_scan_cfg()

  assert raw.experiment_name == "go2_velocity_encoder_raw"
  assert raw.actor == baseline.actor
  assert raw.actor.observation_encoder_cfg is None
  assert raw.actor.class_name == "MLPModel"


def test_go2_non_raw_encoder_configs_keep_encoder_architecture():
  expected = (
    (mlp_encoder_height_only_cfg(), "go2_velocity_encoder_mlp", "mlp"),
    (mlp_encoder_with_state_cfg(), "go2_velocity_encoder_mlp_state", "mlp"),
    (conv1d_encoder_cfg(), "go2_velocity_encoder_conv1d", "conv1d"),
    (conv1d_state_encoder_cfg(), "go2_velocity_encoder_conv1d_state", "conv1d"),
    (conv2d_encoder_cfg(), "go2_velocity_encoder_conv2d", "conv2d"),
    (conv2d_state_encoder_cfg(), "go2_velocity_encoder_conv2d_state", "conv2d"),
  )

  for cfg, experiment_name, encoder_type in expected:
    assert cfg.experiment_name == experiment_name
    assert cfg.actor.class_name == ENCODER_MODEL
    assert cfg.actor.observation_encoder_cfg is not None
    assert cfg.actor.observation_encoder_cfg["type"] == encoder_type


def test_go2_conv_encoder_configs_split_height_only_and_state_context():
  height_only_cfgs = (conv1d_encoder_cfg(), conv2d_encoder_cfg())
  state_cfgs = (conv1d_state_encoder_cfg(), conv2d_state_encoder_cfg())

  for cfg in height_only_cfgs:
    enc_cfg = cfg.actor.observation_encoder_cfg
    assert enc_cfg is not None
    assert enc_cfg["encoder_input_keys"] == ["height_scan"]
    assert enc_cfg.get("context_keys", []) == []

  for cfg in state_cfgs:
    enc_cfg = cfg.actor.observation_encoder_cfg
    assert enc_cfg is not None
    assert enc_cfg["encoder_input_keys"] == [
      "height_scan",
      "command",
      "projected_gravity",
    ]
    assert enc_cfg["context_keys"] == ["command", "projected_gravity"]
