"""W&B compatibility for the pinned RSL-RL logger."""

import os
from pathlib import Path

import wandb
from rsl_rl.utils import WandbLogWriter
from torch.utils.tensorboard import SummaryWriter


class UnitreeWandbLogWriter(WandbLogWriter):
  """Use iteration steps in W&B and keep wall-clock curves in TensorBoard."""

  def __init__(self, log_dir: str, project_name: str) -> None:
    SummaryWriter.__init__(self, log_dir, flush_secs=10)
    self.logged_videos: set[str] = set()
    try:
      wandb.init(
        project=project_name,
        entity=os.environ.get("WANDB_USERNAME"),
        name=Path(log_dir).name,
        config={"log_dir": log_dir},
        settings=wandb.Settings(start_method="thread"),
      )
    except BaseException:
      self.close()
      raise

  def add_scalar(
    self, tag, scalar_value, global_step=None, walltime=None, new_style=False
  ) -> None:
    if tag.endswith("/time"):
      SummaryWriter.add_scalar(
        self, tag, scalar_value, global_step, walltime, new_style
      )
    else:
      super().add_scalar(tag, scalar_value, global_step, walltime, new_style)

  def stop(self) -> None:
    try:
      super().stop()
    finally:
      self.close()
