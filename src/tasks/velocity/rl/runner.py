import os
from pathlib import Path

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


def _policy_checkpoint_filename(path: str) -> str:
  model_path = Path(path)
  if model_path.stem.startswith("model_"):
    return f"policy_{model_path.stem.removeprefix('model_')}.onnx"
  return "policy.onnx"


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = _policy_checkpoint_filename(path)
    self.export_policy_to_onnx(policy_path, filename)
    self.export_policy_to_onnx(policy_path, "policy.onnx")
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    attach_metadata_to_onnx(os.path.join(policy_path, "policy.onnx"), metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
      wandb.save(policy_path + "policy.onnx", base_path=os.path.dirname(policy_path))
