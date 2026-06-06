from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from tools.network_architecture_figure import (
  load_deploy_pipeline,
  load_task_pipeline,
  render_mermaid_pipeline,
  render_pipeline,
)


def _requires_mjlab():
  pytest.importorskip("mjlab")


def test_go2_raw_pipeline_is_plain_actor():
  _requires_mjlab()
  spec = load_task_pipeline("Unitree-Go2-Rough-Encoder-Raw")

  assert spec.actor is not None
  assert spec.actor.encoder_type is None
  assert spec.actor.class_name == "MLPModel"
  assert spec.actor.obs_groups == [
    "height_scan",
    "command",
    "projected_gravity",
    "proprio",
    "last_action",
  ]
  assert spec.action.kind == "JointPositionAction"
  assert spec.action.dim == 12


def test_go2_encoder_pipeline_splits_encoder_and_passthrough():
  _requires_mjlab()
  spec = load_task_pipeline("Unitree-Go2-Rough-Encoder-MLP")

  assert spec.actor is not None
  assert spec.actor.encoder_type == "mlp"
  assert spec.actor.encoder_input_keys == ["height_scan"]
  assert spec.actor.passthrough_keys == [
    "command",
    "projected_gravity",
    "proprio",
    "last_action",
  ]
  assert spec.observations["height_scan"].dim == 187
  assert spec.observations["proprio"].dim == 29
  assert spec.observations["last_action"].dim == 12


def test_go2_conv_pipeline_is_height_only_by_default():
  _requires_mjlab()
  spec = load_task_pipeline("Unitree-Go2-Rough-Encoder-Conv2d")

  assert spec.actor is not None
  assert spec.actor.encoder_type == "conv2d"
  assert spec.actor.encoder_input_keys == ["height_scan"]
  assert spec.actor.passthrough_keys == [
    "command",
    "projected_gravity",
    "proprio",
    "last_action",
  ]
  enc_cfg = spec.actor.observation_encoder_cfg
  assert enc_cfg is not None
  assert enc_cfg["primary_key"] == "height_scan"
  assert enc_cfg.get("context_keys", []) == []
  assert enc_cfg["input_hw"] == [17, 11]
  assert enc_cfg["latent_dim"] == 32


def test_go2_conv_state_pipeline_preserves_encoder_context():
  _requires_mjlab()
  spec = load_task_pipeline("Unitree-Go2-Rough-Encoder-Conv2dState")

  assert spec.actor is not None
  assert spec.actor.encoder_type == "conv2d"
  assert spec.actor.encoder_input_keys == [
    "height_scan",
    "command",
    "projected_gravity",
  ]
  enc_cfg = spec.actor.observation_encoder_cfg
  assert enc_cfg is not None
  assert enc_cfg["primary_key"] == "height_scan"
  assert enc_cfg["context_keys"] == ["command", "projected_gravity"]
  assert enc_cfg["input_hw"] == [17, 11]
  assert enc_cfg["latent_dim"] == 32


def test_go2_deploy_pipeline_maps_raw_action_to_joint_targets():
  spec = load_deploy_pipeline(
    Path("deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml")
  )

  assert spec.action.kind == "JointPositionAction"
  assert spec.action.dim == 12
  assert spec.action.target_label == "q_target"
  assert spec.action.default_joint_pos is not None
  assert spec.action.default_joint_pos[:3] == [-0.1, 0.9, -1.8]
  assert spec.action.joint_ids_map == [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
  assert spec.observations["obs"].dim == 47


def test_render_pipeline_svg_contains_action_equation(tmp_path):
  spec = load_deploy_pipeline(
    Path("deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml")
  )
  out = render_pipeline(spec, tmp_path / "pipeline.svg")

  text = out.read_text()
  assert "q_target = a_raw * scale + offset" in text
  assert "not final q" in text
  assert "JointPositionAction" in text


def test_render_mermaid_pipeline_contains_core_flow():
  spec = load_deploy_pipeline(
    Path("deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml")
  )

  text = render_mermaid_pipeline(spec, include_critic=False)

  assert "flowchart LR" in text
  assert "a_raw (12)" in text
  assert "q_target = a_raw * scale + offset" in text
  assert "JointPositionAction" in text
  assert "motor_cmd.q" in text
  assert "policy.onnx<br/>observations -> raw action" in text
  assert "PRE -->|47| ACTOR" in text
  assert "ACTOR -->|12| ARAW" in text
  assert "ARAW -->|12| APROC" in text
  assert "APROC -->|12| QT" in text
