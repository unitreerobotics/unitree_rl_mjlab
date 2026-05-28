"""Prepare a verified Unitree B2YGX Flat policy for deployment.

This script intentionally separates validation from installation. By default it
only checks the checkpoint/policy pair and prints the play command. Copying a
policy into deploy requires --install together with --play-verified.
"""

from __future__ import annotations

import argparse
import platform
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "b2ygx_velocity"
DEFAULT_DEPLOY_ROOT = REPO_ROOT / "deploy" / "robots" / "b2ygx"
DEPLOY_POLICY_DIR = Path("config/policy/velocity/v0")
EXPECTED_JOINT_IDS_MAP = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
EXPECTED_DEFAULT_JOINT_POS = [0.0, 0.9, -1.8] * 4
EXPECTED_OBSERVATIONS = [
  "base_ang_vel",
  "projected_gravity",
  "velocity_commands",
  "gait_phase",
  "joint_pos_rel",
  "joint_vel_rel",
  "last_action",
]


def _repo_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(REPO_ROOT))
  except ValueError:
    return str(path.resolve())


def _latest_checkpoint(log_root: Path) -> Path:
  checkpoints = sorted(log_root.glob("*/model_*.pt"), key=lambda p: p.stat().st_mtime)
  if not checkpoints:
    raise FileNotFoundError(f"No B2YGX checkpoints found under {_repo_path(log_root)}")
  return checkpoints[-1]


def _policy_for_checkpoint(checkpoint: Path) -> Path:
  stem = checkpoint.stem
  if stem.startswith("model_"):
    policy = checkpoint.with_name(f"policy_{stem.removeprefix('model_')}.onnx")
  else:
    policy = checkpoint.with_name("policy.onnx")
  if not policy.exists():
    fallback = checkpoint.with_name("policy.onnx")
    if fallback.exists():
      return fallback
  return policy


def _load_yaml(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
  if not isinstance(data, dict):
    raise ValueError(f"{_repo_path(path)} did not parse as a YAML mapping")
  return data


def _as_float_list(values: Any) -> list[float]:
  if not isinstance(values, list):
    raise ValueError("expected a list")
  return [float(v) for v in values]


def _check_close(name: str, actual: list[float], expected: list[float], errors: list[str]) -> None:
  if len(actual) != len(expected) or any(abs(a - e) > 1.0e-6 for a, e in zip(actual, expected)):
    errors.append(f"{name}: expected {expected}, got {actual}")


def validate_deploy_yaml(deploy_yaml: Path) -> list[str]:
  cfg = _load_yaml(deploy_yaml)
  errors: list[str] = []

  if cfg.get("joint_ids_map") != EXPECTED_JOINT_IDS_MAP:
    errors.append(
      f"joint_ids_map: expected {EXPECTED_JOINT_IDS_MAP}, got {cfg.get('joint_ids_map')}"
    )

  if abs(float(cfg.get("step_dt", -1.0)) - 0.02) > 1.0e-6:
    errors.append(f"step_dt: expected 0.02, got {cfg.get('step_dt')}")

  expected_stiffness = [200.0, 200.0, 240.0] * 4
  expected_damping = [10.0, 10.0, 12.0] * 4
  _check_close("stiffness", _as_float_list(cfg.get("stiffness")), expected_stiffness, errors)
  _check_close("damping", _as_float_list(cfg.get("damping")), expected_damping, errors)
  _check_close(
    "default_joint_pos",
    _as_float_list(cfg.get("default_joint_pos")),
    EXPECTED_DEFAULT_JOINT_POS,
    errors,
  )

  action_cfg = cfg.get("actions", {}).get("JointPositionAction", {})
  _check_close(
    "actions.JointPositionAction.scale",
    _as_float_list(action_cfg.get("scale")),
    [0.25] * 12,
    errors,
  )
  _check_close(
    "actions.JointPositionAction.offset",
    _as_float_list(action_cfg.get("offset")),
    EXPECTED_DEFAULT_JOINT_POS,
    errors,
  )

  observations = cfg.get("observations", {})
  if not isinstance(observations, dict):
    errors.append("observations: expected mapping")
  else:
    obs_names = list(observations.keys())
    if obs_names != EXPECTED_OBSERVATIONS:
      errors.append(f"observations: expected {EXPECTED_OBSERVATIONS}, got {obs_names}")
    if "height_scan" in observations:
      errors.append("observations: height_scan must not be present for Flat deployment")

  return errors


def validate_cmake_for_host(cmake_file: Path) -> list[str]:
  errors: list[str] = []
  text = cmake_file.read_text(encoding="utf-8")
  machine = platform.machine()
  if machine == "x86_64" and "onnxruntime-linux-x64-1.22.0" not in text:
    errors.append("CMakeLists.txt does not reference the x64 ONNXRuntime package")
  if machine == "aarch64" and "onnxruntime-linux-aarch64-1.22.0" not in text:
    errors.append("CMakeLists.txt does not reference the aarch64 ONNXRuntime package")
  return errors


def write_manifest(
  manifest_path: Path,
  checkpoint: Path,
  source_policy: Path,
  installed_policy: Path,
  deploy_yaml: Path,
) -> None:
  payload = {
    "robot": "b2ygx",
    "task": "Unitree-B2YGX-Flat",
    "installed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "play_verified": True,
    "checkpoint": _repo_path(checkpoint),
    "source_policy": _repo_path(source_policy),
    "installed_policy": _repo_path(installed_policy),
    "deploy_yaml": _repo_path(deploy_yaml),
    "required_manual_checks": [
      "joint_ids_map matches the real B2YGX SDK motor order",
      "robot is suspended before entering Velocity",
      "network interface is the robot-facing interface",
    ],
  }
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  with manifest_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Validate and optionally install a verified B2YGX Flat policy."
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=None,
    help="Path to model_*.pt. Defaults to the newest checkpoint under logs/rsl_rl/b2ygx_velocity.",
  )
  parser.add_argument(
    "--policy",
    type=Path,
    default=None,
    help="Path to policy_*.onnx. Defaults to the ONNX file matching --checkpoint.",
  )
  parser.add_argument(
    "--deploy-root",
    type=Path,
    default=DEFAULT_DEPLOY_ROOT,
    help="B2YGX deploy root.",
  )
  parser.add_argument(
    "--install",
    action="store_true",
    help="Copy the selected ONNX into deploy/robots/b2ygx/.../exported/policy.onnx.",
  )
  parser.add_argument(
    "--play-verified",
    action="store_true",
    help="Confirm that the checkpoint passed play validation before installing.",
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()

  checkpoint = args.checkpoint.resolve() if args.checkpoint else _latest_checkpoint(DEFAULT_LOG_ROOT)
  policy = args.policy.resolve() if args.policy else _policy_for_checkpoint(checkpoint).resolve()
  deploy_root = args.deploy_root.resolve()
  deploy_policy_root = deploy_root / DEPLOY_POLICY_DIR
  deploy_yaml = deploy_policy_root / "params" / "deploy.yaml"
  installed_policy = deploy_policy_root / "exported" / "policy.onnx"
  manifest = deploy_policy_root / "params" / "deployment_manifest.yaml"

  errors: list[str] = []
  if not checkpoint.exists():
    errors.append(f"checkpoint not found: {_repo_path(checkpoint)}")
  if not policy.exists():
    errors.append(f"policy not found: {_repo_path(policy)}")
  if not deploy_yaml.exists():
    errors.append(f"deploy.yaml not found: {_repo_path(deploy_yaml)}")
  else:
    errors.extend(validate_deploy_yaml(deploy_yaml))

  cmake_file = deploy_root / "CMakeLists.txt"
  if not cmake_file.exists():
    errors.append(f"CMakeLists.txt not found: {_repo_path(cmake_file)}")
  else:
    errors.extend(validate_cmake_for_host(cmake_file))

  print("B2YGX deploy candidate")
  print(f"  checkpoint: {_repo_path(checkpoint)}")
  print(f"  policy:     {_repo_path(policy)}")
  print(f"  deploy:     {_repo_path(installed_policy)}")
  print()
  print("Play validation command:")
  print(f"  python scripts/play.py Unitree-B2YGX-Flat --checkpoint_file={_repo_path(checkpoint)}")
  print()

  if installed_policy.exists():
    src_mtime = policy.stat().st_mtime if policy.exists() else 0.0
    dst_mtime = installed_policy.stat().st_mtime
    relation = "newer than" if dst_mtime >= src_mtime else "older than"
    print(f"Existing deploy policy is {relation} the selected source policy.")
    print()

  if errors:
    print("Validation failed:")
    for error in errors:
      print(f"  - {error}")
    return 1

  print("Validation passed.")

  if not args.install:
    print("No files were changed. Re-run with --install --play-verified after play validation.")
    return 0

  if not args.play_verified:
    print("Refusing to install without --play-verified.")
    return 2

  installed_policy.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(policy, installed_policy)
  write_manifest(manifest, checkpoint, policy, installed_policy, deploy_yaml)
  print(f"Installed policy to {_repo_path(installed_policy)}")
  print(f"Wrote manifest to {_repo_path(manifest)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
