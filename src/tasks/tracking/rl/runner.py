import os
from pathlib import Path
from typing import cast

import wandb
from rsl_rl.utils import WandbLogWriter

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.tracking.mdp import MotionCommand


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = str(Path(path).parent)
    filename = "policy.onnx"
    onnx_path = os.path.join(policy_path, filename)
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name
      if isinstance(self.logger.writer, WandbLogWriter) and wandb.run
      else "local"
    )  # type: ignore[assignment]
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    motion_term = cast(
      MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
    )
    metadata.update(
      {
        "anchor_body_name": motion_term.cfg.anchor_body_name,
        "body_names": list(motion_term.cfg.body_names),
      }
    )
    attach_metadata_to_onnx(onnx_path, metadata)
    if isinstance(self.logger.writer, WandbLogWriter) and self.cfg["upload_model"]:
      wandb.save(onnx_path, base_path=policy_path)
