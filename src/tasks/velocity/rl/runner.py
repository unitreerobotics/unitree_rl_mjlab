import math
import os

import torch

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class SymmetricVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  """VelocityOnPolicyRunner with left-right mirror augmentation, for the R1 perceptive task.

  Every PPO mini-batch is learned as recorded and mirrored (config/r1/symmetry.py), so the
  policy cannot keep a preference for one side. When only the actor is taken from a checkpoint
  (train.py --init-actor-from), its action noise is set back to init_std_on_actor_load: the
  checkpoint's noise (0.55 at 15800) was too small to unlearn a habit within a few hundred
  iterations.
  """

  init_std_on_actor_load: float | None = 0.8

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
    from src.tasks.velocity.config.r1.symmetry import mirror_r1

    # A copy: train.py writes its agent config to params/agent.yaml after this, and rsl_rl adds
    # the env object to symmetry_cfg, which YAML cannot hold.
    train_cfg = {**train_cfg, "algorithm": dict(train_cfg["algorithm"])}
    train_cfg["algorithm"]["symmetry_cfg"] = {
      "use_data_augmentation": True,
      "use_mirror_loss": False,
      "mirror_loss_coeff": 0.0,
      "data_augmentation_func": mirror_r1,
    }
    super().__init__(env, train_cfg, log_dir, device)

  def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
    infos = super().load(path, load_cfg, strict, map_location)
    if load_cfg is not None and load_cfg.get("actor") and not load_cfg.get("optimizer") \
        and self.init_std_on_actor_load is not None:
      dist = self.alg.actor.distribution
      with torch.no_grad():
        if hasattr(dist, "std_param"):
          old = dist.std_param.mean().item()
          dist.std_param.fill_(self.init_std_on_actor_load)
        else:
          old = dist.log_std_param.exp().mean().item()
          dist.log_std_param.fill_(math.log(self.init_std_on_actor_load))
      print(f"[INFO]: action noise std {old:.3f} -> {self.init_std_on_actor_load}")
    return infos
