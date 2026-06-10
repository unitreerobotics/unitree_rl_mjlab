"""Evaluate trained velocity policies on a rough corridor terrain."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import secrets
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

# Render offscreen video on the GPU via EGL (mirrors scripts/train.py). Without
# this, MuJoCo falls back to CPU software GL (llvmpipe), which makes multi-view
# video recording over a full episode pathologically slow. Set before importing
# mjlab so the GL backend is chosen on first context creation.
os.environ.setdefault("MUJOCO_GL", "egl")

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
from mjlab.viewer import ViewerConfig

from src.tasks.velocity.evaluation.logger import EvaluationRunLogger, write_summary
from src.tasks.velocity.evaluation.pure_pursuit import PurePursuitVelocityCommandCfg
from src.tasks.velocity.evaluation.video import MultiViewVideoRecorder
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


def _relax_calf_contact(env_cfg: Any) -> list[str]:
  """Allow calf contact during evaluation by excluding the calf collision geoms
  from the sensor that drives the ``illegal_contact`` termination.

  The ``illegal_contact`` term terminates on any non-foot collision-geom contact
  above its force threshold, which counts a calf brushing rough/boxy terrain as a
  failure. For go2 the calf geoms are ``{FR,FL,RR,RL}_calf{1,2}_collision``. This
  mutates the env config in place (eval only) and returns the geoms added to the
  sensor's exclude list; it warns and no-ops if the term/sensor is not found.
  """
  term = (env_cfg.terminations or {}).get("illegal_contact")
  if term is None:
    print("[EVAL] --allow-calf-contact: no 'illegal_contact' termination; skipping.")
    return []
  sensor_name = (getattr(term, "params", None) or {}).get("sensor_name")
  sensors = env_cfg.scene.sensors or ()
  sensor = next((s for s in sensors if getattr(s, "name", None) == sensor_name), None)
  if sensor is None or not hasattr(sensor, "primary"):
    print(
      f"[EVAL] --allow-calf-contact: sensor {sensor_name!r} not found on scene; skipping."
    )
    return []
  calf_geoms = tuple(
    f"{leg}_calf{idx}_collision" for leg in ("FR", "FL", "RR", "RL") for idx in (1, 2)
  )
  existing = tuple(sensor.primary.exclude or ())
  added = [g for g in calf_geoms if g not in existing]
  sensor.primary.exclude = existing + tuple(added)
  return added


def _apply_eval_overrides(env_cfg: Any, args: argparse.Namespace) -> tuple[list[dict[str, float]], dict[str, Any]]:
  terrain_cfg, waypoints, terrain_metadata = make_eval_terrain_cfg(
    args.eval_terrain,
    seed=args.seed,
  )
  waypoint_values = [[p["x"], p["y"], p["z"]] for p in waypoints]

  # Run every requested evaluation as a parallel env in a single GPU sim
  # (run_id == env index). scene.num_envs auto-propagates to the terrain, so we
  # do NOT set terrain.num_envs here. The terrain generator uses num_cols=1, so
  # all envs share the same corridor-start origin and worlds do not collide.
  env_cfg.scene.num_envs = args.num_runs
  env_cfg.scene.terrain.terrain_type = "generator"
  env_cfg.scene.terrain.terrain_generator = terrain_cfg
  env_cfg.scene.terrain.max_init_terrain_level = 0
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

  if getattr(args, "allow_calf_contact", False):
    added = _relax_calf_contact(env_cfg)
    if added:
      print(
        f"[EVAL] allowing calf contact (excluded {len(added)} calf geoms from "
        f"illegal_contact: {', '.join(added)})"
      )

  env_cfg.viewer.distance = 3.5
  env_cfg.viewer.elevation = -20.0
  env_cfg.viewer.azimuth = 180.0
  env_cfg.viewer.width = args.video_width
  env_cfg.viewer.height = args.video_height
  # The video renders only the env at viewer.env_idx. max_extra_envs=0 avoids
  # rendering ghost robots, since all envs are stacked at the same origin.
  env_cfg.viewer.env_idx = max(args.video_run, 0)
  env_cfg.viewer.max_extra_envs = 0
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
  env_ids = torch.arange(env.num_envs, device=env.device)
  default_root = robot.data.default_root_state[env_ids].clone()
  default_root[:, 0] = float(start_xyz[0])
  default_root[:, 1] = float(start_xyz[1])
  default_root[:, 2] = default_root[:, 2] + float(start_xyz[2])
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


def _termination_reason_for_env(env: ManagerBasedRlEnv, i: int) -> str | None:
  for name in env.termination_manager.active_terms:
    if bool(env.termination_manager.get_term(name)[i].item()):
      return name
  return None


def _to_numpy(value: Any) -> np.ndarray | None:
  """One batched GPU->CPU transfer for a tensor (leading dim == num_envs)."""
  if value is None:
    return None
  return value.detach().cpu().numpy()


def _batch_step_tensors(env: ManagerBasedRlEnv, obs: TensorDict) -> dict[str, Any]:
  """Pull all per-step state for the WHOLE batch to numpy in one pass.

  Every returned value keeps its leading env dimension (N); the per-env Python
  loop slices ``[i]`` afterwards. Each tensor is moved with exactly one
  ``.detach().cpu().numpy()`` so the GPU->CPU transfer happens once per batch.
  """
  robot = env.scene["robot"]
  command_term = env.command_manager.get_term("twist")
  batch: dict[str, Any] = {}
  # Root state.
  batch["root_link_pos_w"] = _to_numpy(robot.data.root_link_pos_w)
  batch["root_link_quat_w"] = _to_numpy(robot.data.root_link_quat_w)
  batch["root_link_lin_vel_b"] = _to_numpy(robot.data.root_link_lin_vel_b)
  batch["root_link_ang_vel_b"] = _to_numpy(robot.data.root_link_ang_vel_b)
  # Command term (PurePursuit is fully vectorized over envs).
  batch["command"] = _to_numpy(command_term.command)
  batch["progress"] = _to_numpy(command_term.progress)
  batch["lateral_error"] = _to_numpy(command_term.lateral_error)
  batch["reached_goal"] = _to_numpy(command_term.reached_goal)
  # Joints / actuators (actuator_force may be None).
  batch["joint_pos"] = _to_numpy(robot.data.joint_pos)
  batch["joint_vel"] = _to_numpy(robot.data.joint_vel)
  batch["actuator_force"] = _to_numpy(getattr(robot.data, "actuator_force", None))
  # Contact sensors (per sensor: found / force).
  batch["contact"] = {}
  for name, sensor in getattr(env.scene, "sensors", {}).items():
    data = getattr(sensor, "data", None)
    if data is None:
      continue
    for field_name in ("found", "force"):
      value = getattr(data, field_name, None)
      if value is not None:
        batch["contact"][f"contact_{name}_{field_name}"] = _to_numpy(value)
  # Reward terms / actions (optional).
  if hasattr(env.reward_manager, "_step_reward"):
    batch["reward_terms"] = _to_numpy(env.reward_manager._step_reward)
  if hasattr(env.action_manager, "action"):
    batch["actions"] = _to_numpy(env.action_manager.action)
  # Observations.
  batch["obs"] = {f"obs_{key}": _to_numpy(obs[key]) for key in obs.keys()}
  # Termination term buffers (current-step values, batched [N]).
  batch["terminations"] = {
    name: _to_numpy(env.termination_manager.get_term(name))
    for name in env.termination_manager.active_terms
  }
  return batch


def _make_row(
  *,
  i: int,
  seed: int,
  checkpoint: Path,
  step: int,
  step_dt: float,
  terrain_metadata: dict[str, Any],
  base_pos: np.ndarray,
  base_quat: np.ndarray,
  base_lin_vel_b: np.ndarray,
  base_ang_vel_b: np.ndarray,
  command: np.ndarray,
  progress: float,
  lateral_error: float,
  reached_goal: bool,
) -> dict[str, Any]:
  roll, pitch, yaw = _quat_to_rpy(base_quat)
  patch_length = float(terrain_metadata["patch_length"])
  patch_idx = min(int(progress / patch_length), terrain_metadata["num_patches"] - 1)
  patch = terrain_metadata["patches"][patch_idx]
  return {
    "run_id": i,
    "seed": seed,
    "checkpoint_path": str(checkpoint),
    "sim_time": step * step_dt,
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
    "path_lateral_error": float(lateral_error),
    "path_progress": progress,
    "terrain_patch_index": patch_idx,
    "terrain_difficulty": patch["difficulty_level"],
    "reached_goal": reached_goal,
  }


def _run_batch(
  *,
  base_env: ManagerBasedRlEnv,
  policy: Any,
  args: argparse.Namespace,
  checkpoint: Path,
  output_dir: Path,
  terrain_metadata: dict[str, Any],
  waypoints: list[dict[str, float]],
) -> list[dict[str, Any]]:
  """Run all ``args.num_runs`` evaluations as parallel envs in one GPU sim.

  Each env index ``i`` corresponds to ``run_id == i``. The GPU steps every env
  for nearly the per-step cost of one, so this is a large speedup over the old
  sequential driver. Finished envs keep simulating but are ignored for logging.

  NOTE: mjlab seeding is global, so all envs share a single base seed
  (``args.seed``) and one RNG stream. Trajectories will therefore not be
  bit-identical to the old per-run-reseed sequential mode, but metric semantics
  are unchanged. The terrain layout is still seeded from ``args.seed``, so
  ``--seed`` reproduces a whole evaluation. For ``--num-runs 1`` the behavior
  matches today (base seed + 0 == base).
  """
  N = args.num_runs
  seed = args.seed or 0
  base_env.seed(seed)

  loggers = [
    EvaluationRunLogger(
      output_dir / f"run_{i:03d}", run_id=i, seed=seed, checkpoint=checkpoint
    )
    for i in range(N)
  ]

  # Video fps / capture cadence (computed in the caller, not in video.py).
  render_fps = round(1.0 / base_env.step_dt)  # control rate, ~50 Hz
  video_fps = args.video_fps if args.video_fps else render_fps
  capture_every = max(1, round(render_fps / video_fps))
  effective_fps = render_fps / capture_every

  # One video recorder for the whole batch; renders only viewer.env_idx.
  recorder: MultiViewVideoRecorder | None = None
  record_video = 0 <= args.video_run < N
  if record_video:
    video_run_dir = output_dir / f"run_{args.video_run:03d}"
    recorder = MultiViewVideoRecorder(
      base_env, video_run_dir, name_prefix="video", fps=effective_fps
    )

  # The reset below (RslRlVecEnvWrapper.__init__ calls env.reset()) must run under
  # inference_mode: the rollout loop updates sensor history buffers in-place via
  # .roll() inside inference_mode, which turns them into inference tensors. Resetting
  # them outside inference_mode raises "Inplace update to inference tensor outside
  # InferenceMode is not allowed".
  with torch.inference_mode():
    vec_env = RslRlVecEnvWrapper(base_env, clip_actions=args.clip_actions)
    obs = _reset_robot_to_start(
      base_env,
      np.array([waypoints[0]["x"], waypoints[0]["y"], waypoints[0]["z"]]),
    )

  robot = base_env.scene["robot"]
  initial_pos = robot.data.root_link_pos_w.detach().cpu().numpy()  # [N, 3]
  initial_quat = robot.data.root_link_quat_w.detach().cpu().numpy()  # [N, 4]
  initial_yaw = np.array(
    [_quat_to_rpy(initial_quat[i])[2] for i in range(N)], dtype=np.float64
  )  # [N]

  max_steps = args.max_steps or base_env.max_episode_length
  terminal_interval = max(0, int(args.terminal_log_interval))
  stuck_steps = 0
  if args.stuck_time > 0.0:
    stuck_steps = max(1, int(math.ceil(args.stuck_time / base_env.step_dt)))
  total_length = float(terrain_metadata["total_path_length"])
  print(
    f"[EVAL] batch start envs={N} seed={seed} max_steps={max_steps} "
    f"path={total_length:.2f}m video_run={args.video_run if record_video else None} "
    f"capture_every={capture_every} effective_fps={effective_fps:.2f}"
  )

  # Per-env python/numpy state.
  active = np.ones(N, dtype=bool)
  termination_reason = ["max_steps"] * N
  success = np.zeros(N, dtype=bool)
  done = np.zeros(N, dtype=bool)
  stuck = np.zeros(N, dtype=bool)
  best_progress = np.zeros(N, dtype=np.float64)
  last_progress_step = np.zeros(N, dtype=np.int64)
  rows_for_summary: list[list[dict[str, Any]]] = [[] for _ in range(N)]

  with torch.inference_mode():
    for step in range(max_steps):
      action = policy(obs.to(args.device))
      obs, _reward, dones, _extras = vec_env.step(action.to(vec_env.device))

      if (
        recorder is not None
        and step % capture_every == 0
        and active[args.video_run]
      ):
        recorder.capture()

      batch = _batch_step_tensors(base_env, obs)
      dones_np = dones.detach().cpu().numpy().astype(bool)

      command = batch["command"]
      progress = batch["progress"]
      lateral_error = batch["lateral_error"]
      reached_goal = batch["reached_goal"].astype(bool)
      base_pos_b = batch["root_link_pos_w"]
      base_quat_b = batch["root_link_quat_w"]
      lin_vel_b = batch["root_link_lin_vel_b"]
      ang_vel_b = batch["root_link_ang_vel_b"]

      for i in range(N):
        if not active[i]:
          continue
        row = _make_row(
          i=i,
          seed=seed,
          checkpoint=checkpoint,
          step=step,
          step_dt=base_env.step_dt,
          terrain_metadata=terrain_metadata,
          base_pos=base_pos_b[i],
          base_quat=base_quat_b[i],
          base_lin_vel_b=lin_vel_b[i],
          base_ang_vel_b=ang_vel_b[i],
          command=command[i],
          progress=float(progress[i]),
          lateral_error=float(lateral_error[i]),
          reached_goal=bool(reached_goal[i]),
        )
        rows_for_summary[i].append(row)

        arrays: dict[str, Any] = {
          "base_position_w": base_pos_b[i],
          "base_quat_w": base_quat_b[i],
          "base_lin_vel_b": lin_vel_b[i],
          "base_ang_vel_b": ang_vel_b[i],
          "command_b": command[i],
          "joint_pos": batch["joint_pos"][i],
          "joint_vel": batch["joint_vel"][i],
        }
        if batch["actuator_force"] is not None:
          arrays["actuator_force"] = batch["actuator_force"][i]
        for key, value in batch["contact"].items():
          arrays[key] = value[i]
        if "reward_terms" in batch:
          arrays["reward_terms"] = batch["reward_terms"][i]
        if "actions" in batch:
          arrays["actions"] = batch["actions"][i]
        for key, value in batch["obs"].items():
          arrays[key] = value[i]

        rel_pos = base_pos_b[i] - initial_pos[i]
        c, s = math.cos(-initial_yaw[i]), math.sin(-initial_yaw[i])
        arrays["base_position_initial_frame"] = np.array(
          [
            c * rel_pos[0] - s * rel_pos[1],
            s * rel_pos[0] + c * rel_pos[1],
            rel_pos[2],
          ],
          dtype=np.float32,
        )
        loggers[i].log_step(row, arrays)

        progress_now = float(row["path_progress"])
        if progress_now > best_progress[i] + args.stuck_progress_epsilon:
          best_progress[i] = progress_now
          last_progress_step[i] = step
        stuck_flag = (
          stuck_steps > 0
          and step - last_progress_step[i] >= stuck_steps
          and not row["reached_goal"]
        )

        # Per-env finish logic (same priority as the old sequential driver).
        if row["reached_goal"]:
          success[i] = True
          termination_reason[i] = "goal_reached"
          active[i] = False
          print(f"[EVAL] run {i} finished reason=goal_reached step={step + 1}")
        elif bool(dones_np[i]):
          done[i] = True
          termination_reason[i] = _termination_reason_for_env(base_env, i) or "done"
          active[i] = False
          print(
            f"[EVAL] run {i} finished reason={termination_reason[i]} step={step + 1}"
          )
        elif stuck_flag:
          stuck[i] = True
          termination_reason[i] = "stuck"
          active[i] = False
          print(f"[EVAL] run {i} finished reason=stuck step={step + 1}")

      # Aggregate progress print over currently active envs.
      if terminal_interval > 0 and (step == 0 or (step + 1) % terminal_interval == 0):
        finished = int((~active).sum())
        if active.any():
          act = active
          mean_progress = float(np.mean(progress[act]))
          mean_rate = float(
            np.mean(np.clip(progress[act] / max(total_length, 1e-9), 0.0, 1.0))
          )
          mean_speed = float(np.mean(np.linalg.norm(lin_vel_b[act][:, :2], axis=1)))
        else:
          mean_progress = mean_rate = mean_speed = 0.0
        print(
          f"[EVAL] step={step + 1}/{max_steps} t={(step + 1) * base_env.step_dt:.1f}s "
          f"finished={finished}/{N} mean_progress={mean_progress:.2f}/{total_length:.2f}m "
          f"mean_rate={mean_rate:.3f} mean_speed={mean_speed:.2f}"
        )

      if not active.any():
        break

  if recorder is not None:
    written = recorder.save()
    print(f"[EVAL] videos saved={[p.name for p in written]}")

  # Build per-env summaries (same fields/order as the old sequential driver).
  summaries: list[dict[str, Any]] = []
  for i in range(N):
    rows = rows_for_summary[i]
    if not rows:
      # Should not happen since all envs start active.
      print(f"[EVAL] run {i} produced no rollout rows; emitting empty summary.")
      summaries.append({"run_id": i, "seed": seed})
      continue
    final = rows[-1]
    final_progress = float(final["path_progress"])
    traversal_rate = float(np.clip(final_progress / max(total_length, 1e-9), 0.0, 1.0))
    fall_flag = termination_reason[i] in {"fell_over", "illegal_contact"}
    patch_pass = {}
    for patch in terrain_metadata["patches"]:
      idx = patch["patch_index"]
      patch_pass[f"passed_patch_{idx:02d}"] = final_progress >= float(
        patch["end_position"][0] - waypoints[0]["x"]
      )
    summaries.append(
      {
        "run_id": i,
        "seed": seed,
        "success": bool(success[i]),
        "traversal_rate": traversal_rate,
        "max_progress": max(float(r["path_progress"]) for r in rows),
        "final_progress": final_progress,
        "reached_difficulty_level": max(float(r["terrain_difficulty"]) for r in rows),
        "time_elapsed": len(rows) * base_env.step_dt,
        "episode_steps": len(rows),
        "mean_speed": float(np.mean([r["actual_speed_xy_b"] for r in rows])),
        "mean_command_speed": float(
          np.mean(
            [math.hypot(r["cmd_lin_vel_x_b"], r["cmd_lin_vel_y_b"]) for r in rows]
          )
        ),
        "mean_velocity_tracking_error": float(
          np.mean([r["velocity_tracking_error"] for r in rows])
        ),
        "mean_path_lateral_error": float(
          np.mean([r["path_lateral_error"] for r in rows])
        ),
        "max_roll": max(abs(float(r["roll"])) for r in rows),
        "max_pitch": max(abs(float(r["pitch"])) for r in rows),
        "fall": fall_flag,
        "stuck": bool(stuck[i]),
        "termination_reason": termination_reason[i],
        **patch_pass,
      }
    )
    loggers[i].events.update(
      {
        "success": bool(success[i]),
        "done": bool(done[i]),
        "termination_reason": termination_reason[i],
        "stuck": bool(stuck[i]),
        "stuck_time": args.stuck_time,
        "stuck_progress_epsilon": args.stuck_progress_epsilon,
        "final_progress": final_progress,
        "traversal_rate": traversal_rate,
        "episode_steps": len(rows),
      }
    )
    print(
      f"[EVAL] run {i} done success={bool(success[i])} "
      f"reason={termination_reason[i]} "
      f"progress={final_progress:.2f}/{total_length:.2f}m "
      f"rate={traversal_rate:.3f} steps={len(rows)}"
    )

  # Write all loggers in parallel; savez_compressed releases the GIL enough to
  # overlap disk IO across threads.
  with ThreadPoolExecutor(max_workers=min(N, 8)) as ex:
    list(ex.map(lambda lg: lg.write(), loggers))

  return summaries


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default=None)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--num-runs", type=int, default=100)
  parser.add_argument("--output-dir", default="logs/tmp")
  parser.add_argument("--video-run", type=int, default=0)
  parser.add_argument("--video-width", type=int, default=1920)
  parser.add_argument("--video-height", type=int, default=1080)
  parser.add_argument(
    "--video-fps",
    type=float,
    default=None,
    help=(
      "Target video frame rate. Defaults to the control rate (~50 Hz, every "
      "step). Lower values capture fewer frames at the same real-time duration, "
      "e.g. --video-fps 25 captures every 2nd step."
    ),
  )
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--max-episode-time", type=float, default=None)
  parser.add_argument("--max-steps", type=int, default=None)
  parser.add_argument(
    "--stuck-time",
    type=float,
    default=20.0,
    help="Terminate a run if path progress does not improve enough for this many seconds. Use <=0 to disable.",
  )
  parser.add_argument(
    "--stuck-progress-epsilon",
    type=float,
    default=0.25,
    help="Minimum path-progress improvement, in meters, required to reset stuck detection.",
  )
  parser.add_argument(
    "--allow-calf-contact",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
      "Allow calf contact during evaluation by excluding the calf collision geoms "
      "from the illegal_contact termination. Use --no-allow-calf-contact to restore "
      "the strict training behavior."
    ),
  )
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
  if args.stuck_progress_epsilon <= 0.0:
    raise ValueError("--stuck-progress-epsilon must be positive.")
  if args.video_width <= 0 or args.video_height <= 0:
    raise ValueError("--video-width and --video-height must be positive.")

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
  # Draw a fresh random base seed each invocation unless one is given explicitly.
  # All envs in the batch share this single global seed (mjlab seeding is global)
  # and the terrain layout is seeded from it, so passing --seed <value> reproduces
  # a whole evaluation.
  if args.seed is None:
    args.seed = secrets.randbelow(2**31 - 1)
    print(f"[INFO] No --seed given; using random seed {args.seed}")
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

  if args.video_run >= args.num_runs:
    print(
      f"[WARN] --video-run {args.video_run} >= --num-runs {args.num_runs}; "
      "disabling video for this batch."
    )
    args.video_run = -1

  try:
    summaries = _run_batch(
      base_env=base_env,
      policy=policy,
      args=args,
      checkpoint=checkpoint,
      output_dir=output_dir,
      terrain_metadata=terrain_metadata,
      waypoints=waypoints,
    )
  finally:
    base_env.close()

  write_summary(output_dir / "summary.csv", summaries)
  print(f"[INFO] Evaluation outputs written to {output_dir}")


if __name__ == "__main__":
  main()
