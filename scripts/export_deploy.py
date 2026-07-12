"""Auto-generate ``deploy.yaml`` from a trained tracking (mimic) task.

Motivation (see unitreerobotics/unitree_rl_mjlab#23): deploying a new motion
policy currently requires hand-copying ``deploy.yaml`` from another run. A
mismatched ``default_joint_pos`` / ``offset`` (the absolute joint target is
``action * scale + offset``) commands the robot to the wrong pose and can
damage the G1. This script derives every field directly from the instantiated
environment so the deployed config always matches the trained policy.

The PD gains, default joint pose, action scale/offset and observation layout
are read from the *current* task config (which must match the config used at
training time). Cross-check the exported observation dimension against the
policy ONNX input with ``--onnx``.

Example:
  python scripts/export_deploy.py \
    --task Unitree-G1-Tracking-No-State-Estimation \
    --motion-file deploy/robots/g1/config/policy/mimic/dance_train/params/dance1_subject2.npz \
    --output deploy/robots/g1/config/policy/mimic/dance_train/params/deploy.yaml \
    --onnx deploy/robots/g1/config/policy/mimic/dance_train/exported/policy.onnx
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

# Deployment-side observation naming. The C++ runtime identifies each observation
# by these names, which differ from the env's internal term names. Maps
# env-actor-term-name -> (deploy_name, deploy_params).
TRACKING_OBS_MAP: dict[str, tuple[str, dict]] = {
  "command": ("motion_command", {"command_name": "motion"}),
  "motion_anchor_ori_b": ("motion_anchor_ori_b", {"command_name": "motion"}),
  "base_ang_vel": ("base_ang_vel", {}),
  "joint_pos": ("joint_pos_rel", {}),
  "joint_vel": ("joint_vel_rel", {}),
  "actions": ("last_action", {}),
}


@dataclass(frozen=True)
class ExportConfig:
  task: str
  """Registered task id (must be the *deploy* variant, e.g. the No-State-Estimation one)."""
  motion_file: str
  """Path to the motion .npz the policy was trained on (needed to build the env)."""
  output: str
  """Where to write deploy.yaml."""
  onnx: str | None = None
  """Optional: policy .onnx to validate the observation dimension against."""
  device: str = "cpu"


def _fmt_num(v) -> str:
  """Format a number, keeping floats float-typed in YAML (e.g. ``1`` -> ``1.0``)."""
  if isinstance(v, float):
    s = f"{v:.6g}"
    if "." not in s and "e" not in s and "n" not in s:  # not nan/inf either
      s += ".0"
    return s
  return str(v)


def _fmt_list(values, per_line: int = 15, indent: int = 12) -> str:
  """Format a numeric list as inline YAML, wrapping every ``per_line`` items."""
  pad = " " * indent
  parts = [_fmt_num(v) for v in values]
  lines = []
  for i in range(0, len(parts), per_line):
    lines.append(", ".join(parts[i : i + per_line]))
  body = (",\n" + pad).join(lines)
  return f"[{body}]"


def build_deploy_dict(cfg: ExportConfig):
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  env_cfg = load_env_cfg(cfg.task, play=True)

  motion_path = Path(cfg.motion_file)
  if not motion_path.exists():
    raise FileNotFoundError(f"Motion file not found: {motion_path}")
  env_cfg.commands["motion"].motion_file = str(motion_path)
  env_cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  u = env.unwrapped
  robot = u.scene["robot"]
  joint_names = list(robot.joint_names)
  n = len(joint_names)

  # PD gains from the compiled MuJoCo model (position actuators):
  #   kp = actuator_gainprm[0], kd = -actuator_biasprm[2].
  mjm = u.sim.mj_model
  name2kp: dict[str, float] = {}
  name2kd: dict[str, float] = {}
  for i in range(mjm.nu):
    jname = mjm.joint(int(mjm.actuator_trnid[i, 0])).name.split("/")[-1]
    name2kp[jname] = float(mjm.actuator_gainprm[i][0])
    name2kd[jname] = float(-mjm.actuator_biasprm[i][2])
  stiffness = [round(name2kp[j], 4) for j in joint_names]
  damping = [round(name2kd[j], 4) for j in joint_names]

  default_joint_pos = [
    round(float(x), 6) for x in robot.data.default_joint_pos[0].tolist()
  ]

  action_term = u.action_manager.get_term("joint_pos")
  scale_t = action_term.scale
  offset_t = action_term.offset
  scale = [
    round(float(x), 6)
    for x in (scale_t[0] if scale_t.ndim > 1 else scale_t).tolist()
  ]
  offset = [
    round(float(x), 6)
    for x in (offset_t[0] if offset_t.ndim > 1 else offset_t).tolist()
  ]

  # Observations (actor group only).
  om = u.observation_manager
  term_names = list(om._group_obs_term_names["actor"])
  term_dims = [int(d[0]) for d in om._group_obs_term_dim["actor"]]
  obs_block = []
  total_obs = 0
  for name, dim in zip(term_names, term_dims):
    if name not in TRACKING_OBS_MAP:
      raise KeyError(
        f"Observation term '{name}' has no deploy mapping. "
        f"Update TRACKING_OBS_MAP in {Path(__file__).name}."
      )
    deploy_name, params = TRACKING_OBS_MAP[name]
    obs_block.append((deploy_name, params, dim))
    total_obs += dim

  step_dt = round(u.cfg.decimation * u.cfg.sim.mujoco.timestep, 6)

  env.close()

  return {
    "joint_names": joint_names,
    "n": n,
    "step_dt": step_dt,
    "stiffness": stiffness,
    "damping": damping,
    "default_joint_pos": default_joint_pos,
    "scale": scale,
    "offset": offset,
    "obs_block": obs_block,
    "total_obs": total_obs,
  }


def render_yaml(d) -> str:
  n = d["n"]
  lines = []
  lines.append(f"joint_ids_map: {list(range(n))}")
  lines.append(f"step_dt: {d['step_dt']}")
  lines.append(f"stiffness: {_fmt_list(d['stiffness'])}")
  lines.append(f"damping:   {_fmt_list(d['damping'])}")
  lines.append(f"default_joint_pos: {_fmt_list(d['default_joint_pos'], per_line=n)}")
  lines.append("commands: {}")
  lines.append("actions:")
  lines.append("  JointPositionAction:")
  lines.append("    clip: null")
  lines.append("    joint_names: [.*]")
  lines.append(f"    scale: {_fmt_list(d['scale'])}")
  lines.append(f"    offset: {_fmt_list(d['offset'], per_line=n)}")
  lines.append("    joint_ids: null")
  lines.append("observations:")
  for deploy_name, params, dim in d["obs_block"]:
    lines.append(f"  {deploy_name}:")
    pstr = (
      "{" + ", ".join(f"{k}: {v}" for k, v in params.items()) + "}" if params else "{}"
    )
    lines.append(f"    params: {pstr}")
    lines.append("    clip: null")
    lines.append(f"    scale: {_fmt_list([1.0] * dim)}")
    lines.append("    history_length: 1")
  return "\n".join(lines) + "\n"


def main():
  cfg = tyro.cli(ExportConfig)
  d = build_deploy_dict(cfg)

  print(f"[INFO] joints: {d['n']}  observation dim: {d['total_obs']}  step_dt: {d['step_dt']}")

  if cfg.onnx is not None:
    import onnx

    m = onnx.load(cfg.onnx)
    in_dim = m.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
    if in_dim != d["total_obs"]:
      print(
        f"[ERROR] ONNX input dim ({in_dim}) != exported observation dim "
        f"({d['total_obs']}). Wrong task variant or stale config — aborting.",
        file=sys.stderr,
      )
      sys.exit(1)
    print(f"[INFO] ONNX input dim {in_dim} matches observation dim. OK.")

  text = render_yaml(d)
  out = Path(cfg.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(text)
  print(f"[INFO] Wrote {out}")


if __name__ == "__main__":
  main()
