"""Collect policy rollouts for policy-conditioned traversability labeling.

Rolls out a *chosen trained policy* on its own training environment (terrain
curriculum + auto-reset kept on), and logs, per timestep and per env, the actor
observation, the robot root pose, and the per-step termination flags. The output
``.npz`` is consumed by ``tools/build_traversability_labels.py`` to produce the
short-horizon failure-risk labels (scalar + optional spatial map).

Unlike ``tools/collect_height_scan.py`` (random actions), this drives the env
with the *loaded policy* so the recorded states and failures reflect what the
policy actually experiences. The env's own random twist commands plus the
difficulty curriculum push the robot across easy->hard terrain, producing a mix
of successful traversal and genuine failures (``fell_over`` / ``illegal_contact``).

The actor observation is a single concatenated vector; we also record the
per-term layout (names + flat dims) reported by the observation manager so the
labeler can split it back into named groups (``height_scan``, proprio, ...)
without hardcoding any dimension or ordering.

Example:
    CUDA_VISIBLE_DEVICES=0 python tools/collect_traversability.py \
        --checkpoint logs/rsl_rl/<exp>/<run>/model_<N>.pt \
        --num-envs 512 --steps 1500 --horizon 75 \
        --output logs/traversability/raw_rollouts.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Import the shared loaders from the sibling evaluate_policy.py. Adding the tools
# dir to sys.path makes this work regardless of the invocation CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_policy as ep  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

ACTOR_GROUP = "actor"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument(
    "--task",
    default=None,
    help="Task id. Defaults to the task_id stored in params/run.yaml.",
  )
  parser.add_argument("--num-envs", type=int, default=512)
  parser.add_argument("--steps", type=int, default=1500)
  parser.add_argument(
    "--horizon",
    type=int,
    default=75,
    help="Failure lookahead H (steps); stored as metadata for the labeler.",
  )
  parser.add_argument(
    "--output",
    default="logs/traversability/raw_rollouts.npz",
    type=Path,
  )
  parser.add_argument(
    "--device",
    default="cuda:0" if torch.cuda.is_available() else "cpu",
  )
  parser.add_argument("--seed", type=int, default=0)
  return parser.parse_args()


def main() -> None:
  # Populate the task registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  args = parse_args()
  configure_torch_backends()
  torch.manual_seed(args.seed)

  run_dir = ep._find_run_dir(args.checkpoint)
  saved_run = ep._load_yaml(run_dir / "params" / "run.yaml")
  task = ep._resolve_task(args.task, saved_run)
  print(f"[COLLECT] task={task} run_dir={run_dir}")

  env_cfg, agent_cfg, _ = ep._load_training_configs(task, run_dir)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  # Keep the training terrain curriculum + auto-reset; just disable obs noise so
  # the logged observations match what a deployed estimator would receive.
  for group in env_cfg.observations.values():
    group.enable_corruption = False

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  env.seed(args.seed)
  policy, _, _ = ep._load_policy(task, args.checkpoint, env, agent_cfg, args.device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))

  # Per-term layout of the concatenated actor observation (robust split offline).
  om = env.observation_manager
  if ACTOR_GROUP not in om.active_terms:
    raise KeyError(
      f"Observation group '{ACTOR_GROUP}' not found; available: "
      f"{list(om.active_terms.keys())}"
    )
  term_names = list(om.active_terms[ACTOR_GROUP])
  term_dims = [int(np.prod(d)) for d in om.group_obs_term_dim[ACTOR_GROUP]]
  print(f"[COLLECT] actor terms: {list(zip(term_names, term_dims))}")

  robot = env.scene["robot"]
  device = vec_env.device

  actor_buf: list[np.ndarray] = []
  pos_buf: list[np.ndarray] = []
  quat_buf: list[np.ndarray] = []
  done_buf: list[np.ndarray] = []
  timeout_buf: list[np.ndarray] = []

  obs = vec_env.get_observations()
  with torch.inference_mode():
    for step in range(args.steps):
      # Record the pre-step state (obs at time t) paired with the done flag of the
      # transition it triggers. After step the env auto-resets done envs, so the
      # returned obs belongs to the next episode -- we never log that as a state.
      actor_buf.append(obs[ACTOR_GROUP].detach().to("cpu").numpy().astype(np.float32))
      pos_buf.append(robot.data.root_link_pos_w.detach().to("cpu").numpy().astype(np.float32))
      quat_buf.append(robot.data.root_link_quat_w.detach().to("cpu").numpy().astype(np.float32))

      action = policy(obs.to(args.device))
      obs, _reward, dones, extras = vec_env.step(action.to(device))

      done_np = dones.detach().to("cpu").numpy().astype(bool)
      if "time_outs" in extras:
        timeout_np = extras["time_outs"].detach().to("cpu").numpy().astype(bool)
      else:
        # Finite-horizon env: fall back to the termination manager's time_out term.
        term = env.termination_manager.get_term("time_out")
        timeout_np = term.detach().to("cpu").numpy().astype(bool)
      done_buf.append(done_np)
      timeout_buf.append(timeout_np)

      if (step + 1) % 100 == 0:
        n_done = int(np.sum([d.sum() for d in done_buf[-100:]]))
        n_fail = int(
          np.sum(
            [
              (d & ~t).sum()
              for d, t in zip(done_buf[-100:], timeout_buf[-100:])
            ]
          )
        )
        print(
          f"[COLLECT] step {step + 1}/{args.steps}  "
          f"dones(last100)={n_done} failures(last100)={n_fail}"
        )

  vec_env.close()

  actor_obs = np.stack(actor_buf)  # [T, N, A]
  root_pos_w = np.stack(pos_buf)  # [T, N, 3]
  root_quat_w = np.stack(quat_buf)  # [T, N, 4]
  done = np.stack(done_buf)  # [T, N]
  time_out = np.stack(timeout_buf)  # [T, N]
  failure = done & ~time_out  # [T, N]

  layout = {"term_names": term_names, "term_dims": term_dims}
  n_fail_total = int(failure.sum())
  n_done_total = int(done.sum())
  print(
    f"[COLLECT] total steps={actor_obs.shape[0]} envs={actor_obs.shape[1]} "
    f"dones={n_done_total} failures={n_fail_total}"
  )
  if n_fail_total == 0:
    print(
      "[COLLECT][WARN] no failure terminations recorded. Increase --steps, raise "
      "terrain difficulty, or pick a weaker checkpoint to get positive labels."
    )

  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output,
    actor_obs=actor_obs,
    root_pos_w=root_pos_w,
    root_quat_w=root_quat_w,
    done=done,
    time_out=time_out,
    failure=failure,
    actor_layout=json.dumps(layout),
    horizon=np.int64(args.horizon),
    step_dt=np.float32(env.step_dt),
    checkpoint=str(args.checkpoint),
    task=task,
  )
  print(f"[COLLECT] saved {output}")


if __name__ == "__main__":
  main()
