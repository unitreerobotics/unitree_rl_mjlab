"""Evaluate trained velocity policies on a rough corridor terrain."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import shutil
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import mjlab
import mjlab.entity.entity as _mjlab_entity_module
import numpy as np
import torch
import yaml
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import ViewerConfig

from src.tasks.velocity.evaluation.logger import EvaluationRunLogger, write_summary
from src.tasks.velocity.evaluation.pure_pursuit import PurePursuitVelocityCommandCfg
from src.tasks.velocity.evaluation.terrains import (
  SUPPORTED_EVAL_TERRAINS,
  make_eval_terrain_cfg,
)

setattr(_mjlab_entity_module, "<lambda>", lambda *args, **kwargs: None)


class _EvalYamlLoader(yaml.UnsafeLoader):
  pass


def _viewer_origin_type_constructor(loader: _EvalYamlLoader, node):
  value = loader.construct_sequence(node)
  return ViewerConfig.OriginType(value[0])


_EvalYamlLoader.add_constructor(
  "tag:yaml.org,2002:python/object/apply:mjlab.viewer.viewer_config.OriginType",
  _viewer_origin_type_constructor,
)


def _legacy_lambda_constructor(loader: _EvalYamlLoader, node):
  loader.construct_scalar(node)
  return lambda *args, **kwargs: None


_EvalYamlLoader.add_constructor(
  "tag:yaml.org,2002:python/name:mjlab.entity.entity.%3Clambda%3E",
  _legacy_lambda_constructor,
)


def _load_yaml(path: Path) -> Any:
  with path.open() as f:
    return yaml.load(f, Loader=_EvalYamlLoader)


def _deep_update_dict(base: dict[str, Any], saved: dict[str, Any]) -> dict[str, Any]:
  out = copy.deepcopy(base)
  for key, value in saved.items():
    if isinstance(value, dict) and isinstance(out.get(key), dict):
      out[key] = _deep_update_dict(out[key], value)
    else:
      out[key] = copy.deepcopy(value)
  return out


def _overlay_dataclass(target: Any, saved: Any, *, prune_dicts: bool = True) -> Any:
  if is_dataclass(target) and isinstance(saved, dict):
    known = {field.name for field in fields(target)}
    for key, value in saved.items():
      if key in known:
        setattr(
          target,
          key,
          _overlay_dataclass(getattr(target, key), value, prune_dicts=prune_dicts),
        )
    return target

  if isinstance(target, dict) and isinstance(saved, dict):
    if prune_dicts:
      for key in list(target.keys()):
        if key not in saved:
          target.pop(key)
    for key, value in saved.items():
      if key in target:
        target[key] = _overlay_dataclass(target[key], value, prune_dicts=prune_dicts)
    return target

  if isinstance(target, tuple) and isinstance(saved, (list, tuple)):
    target_by_name = {
      getattr(item, "name"): item
      for item in target
      if is_dataclass(item) and hasattr(item, "name")
    }
    merged = []
    for index, value in enumerate(saved):
      item = None
      if isinstance(value, dict) and "name" in value:
        item = target_by_name.get(value["name"])
      elif index < len(target):
        item = target[index]
      if item is not None:
        merged.append(_overlay_dataclass(item, value, prune_dicts=prune_dicts))
    return tuple(merged)

  if isinstance(target, list) and isinstance(saved, list):
    return [
      _overlay_dataclass(target[i], value, prune_dicts=prune_dicts)
      if i < len(target)
      else copy.deepcopy(value)
      for i, value in enumerate(saved)
    ]

  return copy.deepcopy(saved)


def _find_run_dir(checkpoint: Path) -> Path:
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
  run_dir = checkpoint.parent
  if not (run_dir / "params").is_dir():
    raise FileNotFoundError(
      f"Could not find params/ next to checkpoint. Expected: {run_dir / 'params'}"
    )
  return run_dir


def _resolve_task(cli_task: str | None, run_yaml: dict[str, Any] | None) -> str:
  run_task = None if run_yaml is None else run_yaml.get("task_id")
  if cli_task is not None:
    return cli_task
  if run_task:
    return str(run_task)
  raise ValueError("--task is required because params/run.yaml has no task_id.")


def _load_training_configs(task: str, run_dir: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
  env_yaml = run_dir / "params" / "env.yaml"
  agent_yaml = run_dir / "params" / "agent.yaml"
  run_yaml = run_dir / "params" / "run.yaml"
  if not env_yaml.exists() or not agent_yaml.exists():
    raise FileNotFoundError(
      f"Expected saved env/agent configs under {run_dir / 'params'}."
    )

  env_cfg = load_env_cfg(task, play=False)
  agent_defaults = asdict(load_rl_cfg(task))
  saved_env = _load_yaml(env_yaml)
  saved_agent = _load_yaml(agent_yaml)
  saved_run = _load_yaml(run_yaml) if run_yaml.exists() else {}

  env_cfg = _overlay_dataclass(env_cfg, saved_env, prune_dicts=True)
  agent_cfg = _deep_update_dict(agent_defaults, saved_agent)
  agent_cfg["logger"] = "tensorboard"
  agent_cfg["upload_model"] = False
  agent_cfg["resume"] = False
  return env_cfg, agent_cfg, saved_run


def _apply_eval_overrides(env_cfg: Any, args: argparse.Namespace) -> tuple[list[dict[str, float]], dict[str, Any]]:
  terrain_cfg, waypoints, terrain_metadata = make_eval_terrain_cfg(
    args.eval_terrain,
    seed=args.seed,
  )
  waypoint_values = [[p["x"], p["y"], p["z"]] for p in waypoints]

  env_cfg.scene.num_envs = 1
  env_cfg.scene.terrain.terrain_type = "generator"
  env_cfg.scene.terrain.terrain_generator = terrain_cfg
  env_cfg.scene.terrain.max_init_terrain_level = 0
  env_cfg.scene.terrain.num_envs = 1
  env_cfg.curriculum = {}

  for group in env_cfg.observations.values():
    group.enable_corruption = False

  env_cfg.events = {
    name: event
    for name, event in env_cfg.events.items()
    if getattr(event, "mode", None) == "reset" and name == "reset_robot_joints"
  }

  if "twist" not in env_cfg.commands:
    raise ValueError("Evaluation requires a velocity command named 'twist'.")
  env_cfg.commands["twist"] = PurePursuitVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    waypoints=waypoint_values,
    lookahead_distance=args.lookahead_distance,
    target_speed=args.target_speed,
    max_linear_velocity=args.max_linear_velocity,
    max_yaw_rate=args.max_yaw_rate,
    goal_tolerance=args.goal_tolerance,
    debug_vis=True,
  )

  if args.max_episode_time is not None:
    env_cfg.episode_length_s = args.max_episode_time
  elif args.max_steps is not None:
    # The exact step_dt is available after env construction; this keeps the
    # environment timeout at least as long as the explicit loop limit.
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, 1.0e9)
  else:
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, 120.0)

  env_cfg.viewer.distance = 3.5
  env_cfg.viewer.elevation = -20.0
  env_cfg.viewer.azimuth = 180.0
  return waypoints, terrain_metadata


def _write_path_csv(path: Path, waypoints: list[dict[str, float]]) -> None:
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["index", "x", "y", "z"])
    writer.writeheader()
    for i, row in enumerate(waypoints):
      writer.writerow({"index": i, **row})


def _quat_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
  w, x, y, z = q
  roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  sinp = 2 * (w * y - z * x)
  pitch = math.asin(float(np.clip(sinp, -1.0, 1.0)))
  yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return roll, pitch, yaw


def _reset_robot_to_start(env: ManagerBasedRlEnv, start_xyz: np.ndarray) -> TensorDict:
  robot = env.scene["robot"]
  env_ids = torch.zeros(1, dtype=torch.long, device=env.device)
  default_root = robot.data.default_root_state[env_ids].clone()
  default_root[:, 0] = float(start_xyz[0])
  default_root[:, 1] = float(start_xyz[1])
  default_root[:, 2] = float(default_root[:, 2].item() + start_xyz[2])
  default_root[:, 7:13] = 0.0
  robot.write_root_state_to_sim(default_root, env_ids=env_ids)
  if robot.is_articulated:
    robot.write_joint_state_to_sim(
      robot.data.default_joint_pos[env_ids].clone(),
      robot.data.default_joint_vel[env_ids].clone(),
      env_ids=env_ids,
    )
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.command_manager.compute(dt=env.step_dt)
  env.sim.sense()
  obs = env.observation_manager.compute(update_history=True)
  return TensorDict(obs, batch_size=[env.num_envs])


def _actor_expected_obs_dim(checkpoint: Path) -> int | None:
  data = torch.load(checkpoint, map_location="cpu", weights_only=False)
  actor_state = data.get("actor_state_dict", data.get("model_state_dict", {}))
  for key in ("obs_normalizer._mean", "actor_obs_normalizer._mean"):
    value = actor_state.get(key)
    if value is not None and hasattr(value, "shape"):
      return int(value.shape[-1])
  for key, value in actor_state.items():
    if key.endswith("mlp.0.weight") or key.endswith("actor.0.weight"):
      return int(value.shape[1])
  return None


def _load_policy(task: str, checkpoint: Path, env: ManagerBasedRlEnv, agent_cfg: dict[str, Any], device: str):
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, copy.deepcopy(agent_cfg), device=device)
  expected_dim = _actor_expected_obs_dim(checkpoint)
  actual_dim = getattr(runner.alg.actor, "obs_dim", None)
  try:
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  except RuntimeError as exc:
    raise RuntimeError(
      "Checkpoint actor could not be loaded. "
      f"checkpoint={checkpoint}, expected_actor_obs_dim={expected_dim}, "
      f"actual_actor_obs_dim={actual_dim}"
    ) from exc
  policy = runner.get_inference_policy(device=device)
  return policy, actual_dim, expected_dim


def _termination_reason(env: ManagerBasedRlEnv) -> str | None:
  for name in env.termination_manager.active_terms:
    if bool(env.termination_manager.get_term(name)[0].item()):
      return name
  return None


def _contact_arrays(env: ManagerBasedRlEnv) -> dict[str, Any]:
  arrays = {}
  for name, sensor in getattr(env.scene, "sensors", {}).items():
    data = getattr(sensor, "data", None)
    if data is None:
      continue
    for field_name in ("found", "force"):
      value = getattr(data, field_name, None)
      if value is not None:
        arrays[f"contact_{name}_{field_name}"] = value
  return arrays


def _obs_arrays(obs: TensorDict) -> dict[str, Any]:
  return {f"obs_{key}": obs[key] for key in obs.keys()}


def _step_row(
  env: ManagerBasedRlEnv,
  run_id: int,
  seed: int,
  checkpoint: Path,
  step: int,
  terrain_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
  robot = env.scene["robot"]
  command_term = env.command_manager.get_term("twist")
  base_pos = robot.data.root_link_pos_w[0].detach().cpu().numpy()
  base_quat = robot.data.root_link_quat_w[0].detach().cpu().numpy()
  roll, pitch, yaw = _quat_to_rpy(base_quat)
  base_lin_vel_b = robot.data.root_link_lin_vel_b[0].detach().cpu().numpy()
  base_ang_vel_b = robot.data.root_link_ang_vel_b[0].detach().cpu().numpy()
  command = command_term.command[0].detach().cpu().numpy()
  progress = float(command_term.progress[0].item())
  patch_length = float(terrain_metadata["patch_length"])
  patch_idx = min(int(progress / patch_length), terrain_metadata["num_patches"] - 1)
  patch = terrain_metadata["patches"][patch_idx]

  row = {
    "run_id": run_id,
    "seed": seed,
    "checkpoint_path": str(checkpoint),
    "sim_time": step * env.step_dt,
    "step": step,
    "base_x": base_pos[0],
    "base_y": base_pos[1],
    "base_z": base_pos[2],
    "base_qw": base_quat[0],
    "base_qx": base_quat[1],
    "base_qy": base_quat[2],
    "base_qz": base_quat[3],
    "roll": roll,
    "pitch": pitch,
    "yaw": yaw,
    "base_lin_vel_x_b": base_lin_vel_b[0],
    "base_lin_vel_y_b": base_lin_vel_b[1],
    "base_lin_vel_z_b": base_lin_vel_b[2],
    "base_ang_vel_x_b": base_ang_vel_b[0],
    "base_ang_vel_y_b": base_ang_vel_b[1],
    "base_ang_vel_z_b": base_ang_vel_b[2],
    "cmd_lin_vel_x_b": command[0],
    "cmd_lin_vel_y_b": command[1],
    "cmd_yaw_rate": command[2],
    "actual_speed_xy_b": float(np.linalg.norm(base_lin_vel_b[:2])),
    "velocity_tracking_error": float(np.linalg.norm(command[:2] - base_lin_vel_b[:2])),
    "yaw_rate_tracking_error": float(command[2] - base_ang_vel_b[2]),
    "path_lateral_error": float(command_term.lateral_error[0].item()),
    "path_progress": progress,
    "terrain_patch_index": patch_idx,
    "terrain_difficulty": patch["difficulty_level"],
    "reached_goal": bool(command_term.reached_goal[0].item()),
  }
  arrays = {
    "base_position_w": robot.data.root_link_pos_w,
    "base_quat_w": robot.data.root_link_quat_w,
    "base_lin_vel_b": robot.data.root_link_lin_vel_b,
    "base_ang_vel_b": robot.data.root_link_ang_vel_b,
    "command_b": command_term.command,
    "joint_pos": robot.data.joint_pos,
    "joint_vel": robot.data.joint_vel,
    "actuator_force": getattr(robot.data, "actuator_force", None),
  }
  arrays.update(_contact_arrays(env))
  if hasattr(env.reward_manager, "_step_reward"):
    arrays["reward_terms"] = env.reward_manager._step_reward
  if hasattr(env.action_manager, "action"):
    arrays["actions"] = env.action_manager.action
  return row, arrays


def _run_one(
  *,
  base_env: ManagerBasedRlEnv,
  policy: Any,
  args: argparse.Namespace,
  checkpoint: Path,
  output_dir: Path,
  run_id: int,
  terrain_metadata: dict[str, Any],
  waypoints: list[dict[str, float]],
) -> dict[str, Any]:
  seed = (args.seed or 0) + run_id
  run_dir = output_dir / f"run_{run_id:03d}"
  logger = EvaluationRunLogger(run_dir, run_id=run_id, seed=seed, checkpoint=checkpoint)
  base_env.seed(seed)

  interaction_env: Any = base_env
  video_recorder: VideoRecorder | None = None
  if args.video_run == run_id:
    video_recorder = VideoRecorder(
      base_env,
      video_folder=run_dir,
      step_trigger=lambda step: step == 0,
      video_length=None,
      name_prefix="video",
      disable_logger=True,
    )
    interaction_env = video_recorder

  # The reset below (RslRlVecEnvWrapper.__init__ calls env.reset()) must run under
  # inference_mode: the rollout loop updates sensor history buffers in-place via
  # .roll() inside inference_mode, which turns them into inference tensors. Resetting
  # them outside inference_mode on subsequent runs raises "Inplace update to inference
  # tensor outside InferenceMode is not allowed".
  with torch.inference_mode():
    vec_env = RslRlVecEnvWrapper(interaction_env, clip_actions=args.clip_actions)
    obs = _reset_robot_to_start(base_env, np.array([waypoints[0]["x"], waypoints[0]["y"], waypoints[0]["z"]]))
  initial_pos = base_env.scene["robot"].data.root_link_pos_w[0].detach().cpu().numpy()
  initial_quat = base_env.scene["robot"].data.root_link_quat_w[0].detach().cpu().numpy()
  _, _, initial_yaw = _quat_to_rpy(initial_quat)

  max_steps = args.max_steps or base_env.max_episode_length
  terminal_interval = max(0, int(args.terminal_log_interval))
  total_length = float(terrain_metadata["total_path_length"])
  print(
    f"[EVAL] run {run_id + 1}/ {args.num_runs} start "
    f"seed={seed} max_steps={max_steps} path={total_length:.2f}m "
    f"video={args.video_run == run_id}"
  )
  termination_reason = "max_steps"
  rows_for_summary = []
  done = False
  success = False

  with torch.inference_mode():
    for step in range(max_steps):
      action = policy(obs.to(args.device))
      obs, _reward, dones, _extras = vec_env.step(action.to(vec_env.device))
      row, arrays = _step_row(base_env, run_id, seed, checkpoint, step, terrain_metadata)
      rows_for_summary.append(row)
      arrays.update(_obs_arrays(obs))
      base_pos = arrays["base_position_w"][0].detach().cpu().numpy()
      rel_pos = base_pos - initial_pos
      c, s = math.cos(-initial_yaw), math.sin(-initial_yaw)
      rel_pos_initial = np.array(
        [c * rel_pos[0] - s * rel_pos[1], s * rel_pos[0] + c * rel_pos[1], rel_pos[2]],
        dtype=np.float32,
      )
      arrays["base_position_initial_frame"] = torch.tensor(
        rel_pos_initial,
        dtype=torch.float,
        device=base_env.device,
      ).unsqueeze(0)
      logger.log_step(row, arrays)

      done_flag = bool(dones[0].item())
      if terminal_interval > 0 and (
        step == 0 or (step + 1) % terminal_interval == 0 or row["reached_goal"] or done_flag
      ):
        rate = np.clip(float(row["path_progress"]) / max(total_length, 1e-9), 0.0, 1.0)
        cmd_speed = math.hypot(float(row["cmd_lin_vel_x_b"]), float(row["cmd_lin_vel_y_b"]))
        print(
          f"[EVAL] run {run_id + 1}/ {args.num_runs} "
          f"step={step + 1}/{max_steps} t={(step + 1) * base_env.step_dt:.1f}s "
          f"progress={float(row["path_progress"]):.2f}/{total_length:.2f}m "
          f"rate={rate:.3f} patch={int(row["terrain_patch_index"])} "
          f"diff={float(row["terrain_difficulty"]):.2f} "
          f"lat={float(row["path_lateral_error"]):.2f}m "
          f"speed={float(row["actual_speed_xy_b"]):.2f} cmd={cmd_speed:.2f} "
          f"v_err={float(row["velocity_tracking_error"]):.2f}"
        )

      if row["reached_goal"]:
        success = True
        termination_reason = "goal_reached"
        break
      if done_flag:
        done = True
        termination_reason = _termination_reason(base_env) or "done"
        break

  if video_recorder is not None and video_recorder.is_recording:
    video_recorder._finish_recording()
  if video_recorder is not None:
    videos = sorted(run_dir.glob("video-*.mp4"))
    if videos:
      shutil.move(str(videos[-1]), run_dir / "video.mp4")

  if not rows_for_summary:
    raise RuntimeError(f"Run {run_id} produced no rollout rows.")

  final = rows_for_summary[-1]
  total_length = float(terrain_metadata["total_path_length"])
  final_progress = float(final["path_progress"])
  traversal_rate = float(np.clip(final_progress / max(total_length, 1e-9), 0.0, 1.0))
  fall_flag = termination_reason in {"fell_over", "illegal_contact"}
  patch_pass = {}
  for patch in terrain_metadata["patches"]:
    idx = patch["patch_index"]
    patch_pass[f"passed_patch_{idx:02d}"] = final_progress >= float(patch["end_position"][0] - waypoints[0]["x"])
  summary = {
    "run_id": run_id,
    "seed": seed,
    "success": success,
    "traversal_rate": traversal_rate,
    "max_progress": max(float(r["path_progress"]) for r in rows_for_summary),
    "final_progress": final_progress,
    "reached_difficulty_level": max(float(r["terrain_difficulty"]) for r in rows_for_summary),
    "time_elapsed": len(rows_for_summary) * base_env.step_dt,
    "episode_steps": len(rows_for_summary),
    "mean_speed": float(np.mean([r["actual_speed_xy_b"] for r in rows_for_summary])),
    "mean_command_speed": float(np.mean([
      math.hypot(r["cmd_lin_vel_x_b"], r["cmd_lin_vel_y_b"]) for r in rows_for_summary
    ])),
    "mean_velocity_tracking_error": float(np.mean([r["velocity_tracking_error"] for r in rows_for_summary])),
    "mean_path_lateral_error": float(np.mean([r["path_lateral_error"] for r in rows_for_summary])),
    "max_roll": max(abs(float(r["roll"])) for r in rows_for_summary),
    "max_pitch": max(abs(float(r["pitch"])) for r in rows_for_summary),
    "fall": fall_flag,
    "termination_reason": termination_reason,
    **patch_pass,
  }
  logger.events.update(
    {
      "success": success,
      "done": done,
      "termination_reason": termination_reason,
      "final_progress": final_progress,
      "traversal_rate": traversal_rate,
      "episode_steps": len(rows_for_summary),
    }
  )
  logger.write()
  print(
    f"[EVAL] run {run_id + 1}/ {args.num_runs} done "
    f"success={success} reason={termination_reason} "
    f"progress={final_progress:.2f}/{total_length:.2f}m "
    f"rate={traversal_rate:.3f} steps={len(rows_for_summary)}"
  )
  return summary


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default=None)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--num-runs", type=int, default=100)
  parser.add_argument("--output-dir", default="logs/tmp")
  parser.add_argument("--video-run", type=int, default=0)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--max-episode-time", type=float, default=None)
  parser.add_argument("--max-steps", type=int, default=None)
  parser.add_argument("--lookahead-distance", type=float, default=1.0)
  parser.add_argument("--target-speed", type=float, default=0.8)
  parser.add_argument("--max-linear-velocity", type=float, default=1.5)
  parser.add_argument("--max-yaw-rate", type=float, default=1.5)
  parser.add_argument("--goal-tolerance", type=float, default=0.5)
  parser.add_argument(
    "--eval-terrain",
    default="rough_curriculum_corridor",
    choices=SUPPORTED_EVAL_TERRAINS,
  )
  parser.add_argument("--terminal-log-interval", type=int, default=500)
  parser.add_argument(
    "--gpus",
    default=None,
    help=(
      "Physical GPU id(s) to use, e.g. '0', '2', or '2,3' (like scripts/run.sh). "
      "Pins CUDA_VISIBLE_DEVICES; evaluation runs on the first selected GPU. "
      "Ignored if --device is given."
    ),
  )
  parser.add_argument(
    "--device",
    default=None,
    help="Explicit torch device override, e.g. 'cuda:0' or 'cpu'. Takes precedence over --gpus.",
  )
  return parser.parse_args()


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  args = parse_args()

  # Pin GPU selection before CUDA is initialized (mirrors scripts/run.sh's --gpus).
  # Must run before configure_torch_backends()/torch.cuda.is_available(), which
  # are the first calls that create a CUDA context.
  if args.gpus is not None:
    visible = args.gpus.replace(" ", "")
    if not re.fullmatch(r"\d+(,\d+)*", visible):
      raise ValueError(
        f"--gpus must be a comma-separated list of GPU ids, got {args.gpus!r}."
      )
    os.environ["CUDA_VISIBLE_DEVICES"] = visible

  configure_torch_backends()
  if args.device is None:
    if torch.cuda.is_available():
      args.device = "cuda:0"
    else:
      if args.gpus is not None:
        raise RuntimeError(
          f"--gpus {args.gpus!r} was requested but CUDA is not available to torch."
        )
      args.device = "cpu"
  print(
    f"[INFO] Using device={args.device} "
    f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
    f"cuda_available={torch.cuda.is_available()})"
  )
  checkpoint = Path(args.checkpoint).expanduser().resolve()
  run_dir = _find_run_dir(checkpoint)
  run_yaml_path = run_dir / "params" / "run.yaml"
  run_yaml = _load_yaml(run_yaml_path) if run_yaml_path.exists() else {}
  task = _resolve_task(args.task, run_yaml)
  if task not in list_tasks():
    raise ValueError(f"Unknown task {task!r}. Available tasks: {list_tasks()}")

  env_cfg, agent_cfg, saved_run = _load_training_configs(task, run_dir)
  args.clip_actions = agent_cfg.get("clip_actions")
  if args.seed is not None:
    env_cfg.seed = args.seed
    agent_cfg["seed"] = args.seed

  waypoints, terrain_metadata = _apply_eval_overrides(env_cfg, args)
  output_dir = Path(args.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  dump_yaml(output_dir / "eval_config.yaml", asdict(env_cfg))
  (output_dir / "checkpoint_info.json").write_text(
    json.dumps(
      {
        "checkpoint_path": str(checkpoint),
        "log_dir": str(run_dir),
        "env_config": str(run_dir / "params" / "env.yaml"),
        "agent_config": str(run_dir / "params" / "agent.yaml"),
        "run_config": str(run_yaml_path) if run_yaml_path.exists() else None,
        "task": task,
        "saved_run": saved_run,
      },
      indent=2,
      default=str,
    )
  )
  (output_dir / "terrain_metadata.json").write_text(json.dumps(terrain_metadata, indent=2))
  _write_path_csv(output_dir / "path_waypoints.csv", waypoints)

  render_mode = "rgb_array" if args.video_run >= 0 else None
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=render_mode)
  policy, actual_dim, expected_dim = _load_policy(task, checkpoint, base_env, agent_cfg, args.device)
  print(
    f"[INFO] Loaded checkpoint {checkpoint.name}; actor_obs_dim actual={actual_dim}, checkpoint={expected_dim}"
  )

  summaries = []
  try:
    for run_id in range(args.num_runs):
      summaries.append(
        _run_one(
          base_env=base_env,
          policy=policy,
          args=args,
          checkpoint=checkpoint,
          output_dir=output_dir,
          run_id=run_id,
          terrain_metadata=terrain_metadata,
          waypoints=waypoints,
        )
      )
  finally:
    base_env.close()

  write_summary(output_dir / "summary.csv", summaries)
  print(f"[INFO] Evaluation outputs written to {output_dir}")


if __name__ == "__main__":
  main()
