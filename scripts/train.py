"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = replace(load_rl_cfg(task_id), logger="tensorboard")
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(
  task_id: str, cfg: TrainConfig, log_dir: Path, resume_path: Path | None = None
) -> None:
  if cfg.video and (cfg.video_interval <= 0 or cfg.video_length <= 0):
    raise ValueError("Video interval and length must be positive.")
  rank = int(os.environ.get("RANK", "0"))
  seed = cfg.agent.seed + rank
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_file = (
      cfg.motion_file if cfg.motion_file is not None else motion_cmd.motion_file
    )
    if not motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")
    motion_path = Path(motion_file).expanduser().resolve()
    if not motion_path.is_file():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  # Resolve before constructing anything that may create the new run directory.
  if cfg.agent.resume and resume_path is None:
    resume_path = get_checkpoint_path(
      log_dir.parent, cfg.agent.load_run, cfg.agent.load_checkpoint
    )

  env = ManagerBasedRlEnv(
    cfg=cfg.env,
    device=device,
    render_mode="rgb_array" if cfg.video and rank == 0 else None,
  )

  try:
    # Only record videos on rank 0 to avoid multiple workers writing to the same files.
    if cfg.video and rank == 0:
      env = VideoRecorder(
        env,
        video_folder=Path(log_dir) / "videos" / "train",
        step_trigger=lambda step: step % cfg.video_interval == 0,
        video_length=cfg.video_length,
        disable_logger=True,
      )
      print("[INFO] Recording videos during training.")

    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    agent_cfg = asdict(cfg.agent)
    env_cfg = asdict(cfg.env)

    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
      runner_cls = MjlabOnPolicyRunner

    # RSL-RL 5 mutates agent_cfg during runner construction. Serialize it first.
    if rank == 0:
      dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
      dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    runner = runner_cls(env, agent_cfg, str(log_dir), device)

    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
      print(f"[INFO]: Loading model checkpoint from: {resume_path}")
      runner.load(str(resume_path))

    runner.learn(
      num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
    )
  finally:
    env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = (Path("logs") / "rsl_rl" / args.agent.experiment_name).resolve()
  # torchrunx creates its log directory before workers start. Resolve here so
  # load_run='.*' cannot select that new, empty directory as the latest run.
  resume_path = (
    get_checkpoint_path(log_root_path, args.agent.load_run, args.agent.load_checkpoint)
    if args.agent.resume
    else None
  )
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ.setdefault("MUJOCO_GL", "egl")

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir, resume_path)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir, resume_path)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
