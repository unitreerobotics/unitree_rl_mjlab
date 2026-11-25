import os
from pathlib import Path
from typing import cast

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
    metadata = get_base_metadata(self.env.unwrapped, "local")
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
