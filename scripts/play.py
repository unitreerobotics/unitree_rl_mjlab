"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mjlab
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from src.viz import (
  AttributionVideoRecorder,
  AttributionViserPlayViewer,
  RiskViserPlayViewer,
)


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_attribution: bool = False
  """Record a side-by-side playback video with action attribution."""
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  attribution: bool = False
  """Enable live observation attribution in the viser viewer."""
  attribution_method: Literal[
    "integrated_gradients",
    "gradient_saliency",
    "gradient_input",
    "deep_lift_rescale",
    "deep_shap",
  ] = "integrated_gradients"
  """Initial attribution method for the viser attribution panel."""
  risk_estimator: str | None = None
  """Path to a trained traversability estimator (.pt). When set with a viser
  viewer, shows live P(failure soon) + spatial risk map + 3D risk markers."""
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _expand_bool_shorthand(args: list[str], flag_name: str) -> list[str]:
  """Allow ``--flag`` shorthand while project tyro config uses FlagConversionOff."""
  expanded: list[str] = []
  index = 0
  no_flag_name = "--no-" + flag_name[2:]
  while index < len(args):
    arg = args[index]
    if arg == flag_name:
      expanded.append(arg)
      if index + 1 >= len(args) or args[index + 1].startswith("--"):
        expanded.append("True")
    elif arg == no_flag_name:
      expanded.extend([flag_name, "False"])
    else:
      expanded.append(arg)
    index += 1
  return expanded


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  record_video = cfg.video or cfg.video_attribution
  render_mode = "rgb_array" if (TRAINED_MODE and record_video) else None
  if record_video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  video_recorder: VideoRecorder | AttributionVideoRecorder | None = None
  if TRAINED_MODE and record_video:
    message = "Recording attribution video during play"
    if not cfg.video_attribution:
      message = "Recording videos during play"
    print(f"[INFO] {message}")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    recorder_cls = AttributionVideoRecorder if cfg.video_attribution else VideoRecorder
    video_recorder = recorder_cls(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    env = video_recorder

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    critic = None
    if cfg.attribution:
      try:
        runner.load(
          str(resume_path),
          load_cfg={"critic": True},
          strict=True,
          map_location=device,
        )
        critic = runner.alg.critic
      except (KeyError, RuntimeError) as exc:
        print(f"[WARN] Critic could not be loaded; value attribution disabled: {exc}")
    policy = runner.get_inference_policy(device=device)
    actor = runner.alg.actor
    if isinstance(video_recorder, AttributionVideoRecorder):
      video_recorder.configure_attribution(
        actor=actor,
        attribution_method=cfg.attribution_method,
      )

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer
  if cfg.attribution and cfg.viewer == "auto":
    resolved_viewer = "viser"
  if cfg.risk_estimator is not None and cfg.viewer == "auto":
    resolved_viewer = "viser"

  if resolved_viewer == "native":
    if cfg.attribution:
      print("[WARN] Attribution visualizer is only available with --viewer viser.")
    if cfg.risk_estimator is not None:
      print("[WARN] Risk visualizer is only available with --viewer viser.")
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    if cfg.risk_estimator is not None and TRAINED_MODE:
      if cfg.attribution:
        print("[WARN] --risk-estimator overrides --attribution.")
      RiskViserPlayViewer(env, policy, estimator_path=cfg.risk_estimator).run()
    elif cfg.risk_estimator is not None:
      print("[WARN] Risk visualizer requires --agent trained.")
      ViserPlayViewer(env, policy).run()
    elif cfg.attribution and TRAINED_MODE:
      AttributionViserPlayViewer(
        env,
        policy,
        actor=actor,
        critic=critic,
        attribution_method=cfg.attribution_method,
      ).run()
    elif cfg.attribution:
      print("[WARN] Attribution visualizer requires --agent trained.")
      ViserPlayViewer(env, policy).run()
    else:
      ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=_expand_bool_shorthand(
      _expand_bool_shorthand(remaining_args, "--attribution"),
      "--video-attribution",
    ),
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
