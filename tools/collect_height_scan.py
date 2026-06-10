"""Collect a height_scan dataset by rolling out the split-obs env.

Steps the split-observation rough env (the one used by the encoder-ablation
tasks) with random actions and records the ``height_scan`` observation group at
every step. The resulting ``.npz`` can be fed to
``tools/train_height_scan_autoencoder.py`` to produce the checkpoint that the
``pretrained_ae`` observation encoder loads.

Random actions make the robots stumble across the terrain curriculum; combined
with env auto-resets this samples a varied distribution of height_scan patches
without needing a trained policy.

Example:
    CUDA_VISIBLE_DEVICES=0 python tools/collect_height_scan.py \
        --num-envs 1024 --steps 200 \
        --output logs/pretrained_autoencoders/height_scan_dataset.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Unitree-Go2-Rough-Encoder-AE"
OBS_KEY = "height_scan"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument(
    "--output",
    default="logs/pretrained_autoencoders/height_scan_dataset.npz",
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

  env_cfg = load_env_cfg(TASK_ID, play=False)
  agent_cfg = load_rl_cfg(TASK_ID)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  num_actions = vec_env.num_actions
  device = vec_env.device

  obs = vec_env.get_observations()
  if OBS_KEY not in obs.keys():
    raise KeyError(
      f"Observation group '{OBS_KEY}' not found; available groups: "
      f"{list(obs.keys())}"
    )

  buffers: list[torch.Tensor] = [obs[OBS_KEY].detach().to("cpu")]
  for step in range(args.steps):
    action = 2.0 * torch.rand((vec_env.num_envs, num_actions), device=device) - 1.0
    obs, _reward, _dones, _extras = vec_env.step(action)
    buffers.append(obs[OBS_KEY].detach().to("cpu"))
    if (step + 1) % 50 == 0:
      collected = sum(b.shape[0] for b in buffers)
      print(f"[INFO] step {step + 1}/{args.steps}  samples={collected}")

  vec_env.close()

  samples = torch.cat(buffers, dim=0).reshape(-1, np.prod(buffers[0].shape[1:]))
  arr = samples.numpy().astype(np.float32)

  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output, **{OBS_KEY: arr})
  print(f"[INFO] saved {arr.shape[0]} samples of dim {arr.shape[1]} to {output}")


if __name__ == "__main__":
  main()
