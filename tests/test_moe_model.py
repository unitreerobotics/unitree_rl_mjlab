"""Pure-Torch tests for the Go2 MoE locomotion actor."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from tensordict import TensorDict

from src.rl_models.moe_model import MoEMLPModel

B = 5
OBS_DIM = 16
ACTION_DIM = 4


def _obs() -> TensorDict:
  return TensorDict({"actor": torch.randn(B, OBS_DIM)}, batch_size=[B])


def _obs_groups() -> dict[str, list[str]]:
  return {"actor": ["actor"], "critic": ["actor"]}


def _dist_cfg(init_std: float = 0.5) -> dict:
  return {
    "class_name": "GaussianDistribution",
    "init_std": init_std,
    "std_type": "scalar",
  }


def _expert_checkpoint(tmp_path, seed: int, std_value: float = 3.0):
  torch.manual_seed(seed)
  model = MLPModel(
    _obs(),
    _obs_groups(),
    "actor",
    ACTION_DIM,
    hidden_dims=[32, 16],
    activation="elu",
    obs_normalization=True,
    distribution_cfg=_dist_cfg(init_std=std_value),
  )
  with torch.no_grad():
    model.obs_normalizer._mean.fill_(0.1 * seed)
    model.obs_normalizer._var.fill_(1.0 + 0.01 * seed)
    model.obs_normalizer._std.copy_(torch.sqrt(model.obs_normalizer._var))
    model.obs_normalizer.count.fill_(10 + seed)

  path = tmp_path / f"expert_{seed}.pt"
  torch.save({"actor_state_dict": model.state_dict()}, path)
  return path, model


def _moe_cfg(paths, freeze=True, allow_missing=False) -> dict:
  return {
    "experts": [
      {"name": f"expert_{i}", "checkpoint": str(path)}
      for i, path in enumerate(paths)
    ],
    "expert_hidden_dims": [32, 16],
    "expert_activation": "elu",
    "expert_obs_normalization": True,
    "freeze_experts": freeze,
    "gate_hidden_dims": [12],
    "gate_activation": "elu",
    "gate_obs_normalization": True,
    "gate_temperature": 1.0,
    "allow_missing_checkpoints": allow_missing,
  }


def _make_moe(tmp_path, freeze=True, init_std=0.5):
  p0, e0 = _expert_checkpoint(tmp_path, 1)
  p1, e1 = _expert_checkpoint(tmp_path, 2)
  model = MoEMLPModel(
    _obs(),
    _obs_groups(),
    "actor",
    ACTION_DIM,
    hidden_dims=[8],
    activation="elu",
    obs_normalization=False,
    distribution_cfg=_dist_cfg(init_std=init_std),
    moe_cfg=_moe_cfg([p0, p1], freeze=freeze),
  )
  return model, (e0, e1)


def _last_linear(module: nn.Module) -> nn.Linear:
  for child in reversed(list(module.children())):
    if isinstance(child, nn.Linear):
      return child
  raise AssertionError("gate has no Linear layer")


def _zero_gate(model: MoEMLPModel) -> nn.Linear:
  for param in model.mlp.gate.parameters():
    param.data.zero_()
  return _last_linear(model.mlp.gate)


def test_construction_forward_and_stochastic_output(tmp_path):
  model, _ = _make_moe(tmp_path)
  obs = _obs()

  deterministic = model(obs)
  stochastic = model(obs, stochastic_output=True)

  assert deterministic.shape == (B, ACTION_DIM)
  assert stochastic.shape == (B, ACTION_DIM)
  assert model.output_mean.shape == (B, ACTION_DIM)
  assert not torch.isnan(stochastic).any()


def test_mixing_math_one_hot_and_uniform_gate(tmp_path):
  model, experts = _make_moe(tmp_path)
  obs = _obs()
  flat = obs["actor"]
  last = _zero_gate(model)

  with torch.no_grad():
    last.bias[:] = torch.tensor([50.0, -50.0])
  assert torch.allclose(
    model(obs),
    experts[0].mlp(experts[0].obs_normalizer(flat)),
    atol=1e-6,
  )

  _zero_gate(model)
  expected = torch.stack(
    [
      experts[0].mlp(experts[0].obs_normalizer(flat)),
      experts[1].mlp(experts[1].obs_normalizer(flat)),
    ],
    dim=0,
  ).mean(dim=0)
  assert torch.allclose(model(obs), expected, atol=1e-6)


def test_frozen_experts_leave_gate_and_std_trainable(tmp_path):
  model, _ = _make_moe(tmp_path, freeze=True)

  assert all(not p.requires_grad for expert in model.mlp.experts for p in expert.parameters())
  assert any(p.requires_grad for p in model.mlp.gate.parameters())
  assert model.distribution.std_param.requires_grad

  loss = model(_obs()).sum()
  loss.backward()

  assert all(p.grad is None for expert in model.mlp.experts for p in expert.parameters())
  assert any(p.grad is not None for p in model.mlp.gate.parameters())
  assert model.distribution.std_param.grad is None


def test_normalizer_isolation_updates_gate_only(tmp_path):
  model, _ = _make_moe(tmp_path)
  gate_count = model.mlp.gate_normalizer.count.clone()
  expert_counts = [norm.count.clone() for norm in model.mlp.expert_normalizers]

  model.update_normalization(_obs())

  assert model.mlp.gate_normalizer.count > gate_count
  for before, norm in zip(expert_counts, model.mlp.expert_normalizers, strict=True):
    assert torch.equal(norm.count, before)


def test_std_param_not_loaded_from_expert_checkpoint(tmp_path):
  model, _ = _make_moe(tmp_path, init_std=0.5)
  assert torch.allclose(
    model.distribution.std_param,
    torch.full_like(model.distribution.std_param, 0.5),
  )


def test_as_onnx_wrapper_matches_deterministic_forward(tmp_path):
  model, _ = _make_moe(tmp_path)
  model.eval()
  obs = _obs()
  onnx_model = model.as_onnx(verbose=False)

  assert torch.allclose(onnx_model(obs["actor"]), model(obs), atol=1e-6)


def test_missing_checkpoint_handling(tmp_path):
  missing = tmp_path / "missing.pt"
  cfg = _moe_cfg([missing], allow_missing=True)
  with pytest.warns(RuntimeWarning, match="checkpoint is missing"):
    MoEMLPModel(
      _obs(),
      {"actor": ["actor"]},
      "actor",
      ACTION_DIM,
      distribution_cfg=_dist_cfg(),
      moe_cfg=cfg,
    )

  cfg = _moe_cfg([missing], allow_missing=False)
  with pytest.raises(FileNotFoundError, match="checkpoint is missing"):
    MoEMLPModel(
      _obs(),
      {"actor": ["actor"]},
      "actor",
      ACTION_DIM,
      distribution_cfg=_dist_cfg(),
      moe_cfg=cfg,
    )


def test_state_dict_round_trip_with_empty_checkpoints(tmp_path):
  cfg = _moe_cfg(["", ""], allow_missing=True)
  with pytest.warns(RuntimeWarning):
    model = MoEMLPModel(
      _obs(),
      _obs_groups(),
      "actor",
      ACTION_DIM,
      distribution_cfg=_dist_cfg(),
      moe_cfg=cfg,
    )

  with pytest.warns(RuntimeWarning):
    reloaded = MoEMLPModel(
      _obs(),
      _obs_groups(),
      "actor",
      ACTION_DIM,
      distribution_cfg=_dist_cfg(),
      moe_cfg=cfg,
    )
  reloaded.load_state_dict(model.state_dict(), strict=True)

  obs = _obs()
  assert torch.allclose(reloaded(obs), model(obs), atol=1e-6)


def test_gate_weights_helper_accepts_tensordict_and_flat_tensor(tmp_path):
  model, _ = _make_moe(tmp_path)
  obs = _obs()

  weights_from_td = model.gate_weights(obs)
  weights_from_flat = model.gate_weights(obs["actor"])

  assert weights_from_td.shape == (B, 2)
  assert torch.allclose(weights_from_td, weights_from_flat)
  assert torch.allclose(weights_from_td.sum(dim=-1), torch.ones(B))
  assert isinstance(model.mlp.gate_normalizer, EmpiricalNormalization)
