"""Shared Go2-W task helpers.

Robot-specific joint naming, contact patterns, and asset selection live here
so the active Go2-W task builders stay easy to read without mutating Go2 code.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.assets.robots.unitree_go2w.go2w_constants import (
  GO2W_LEG_JOINT_NAMES,
  GO2W_WHEEL_JOINT_NAMES,
  get_go2w_robot_cfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg


def get_go2w_scene_robot_cfg():
  """Return the vendored Go2-W robot asset config."""
  return get_go2w_robot_cfg()


def go2w_leg_joint_cfg() -> SceneEntityCfg:
  """Leg joints in the canonical Go2-W order."""
  return SceneEntityCfg(
    "robot",
    joint_names=GO2W_LEG_JOINT_NAMES,
    preserve_order=True,
  )


def go2w_wheel_joint_cfg() -> SceneEntityCfg:
  """Wheel joints in the canonical Go2-W order."""
  return SceneEntityCfg(
    "robot",
    joint_names=GO2W_WHEEL_JOINT_NAMES,
    preserve_order=True,
  )


def go2w_wheel_ground_contact_cfg(
  *,
  name: str = "wheel_ground_contact",
  pattern: str = r".*(FR|FL|RR|RL)_(wheel|foot).*",
  fields: Sequence[str] = ("found", "force"),
  reduce: str = "none",
  num_slots: int = 1,
  track_air_time: bool = False,
) -> ContactSensorCfg:
  """Wheel-foot terrain contact sensor used by Go2-W locomotion tasks."""
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(
      mode="body",
      pattern=pattern,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=tuple(fields),
    reduce=reduce,
    num_slots=num_slots,
    track_air_time=track_air_time,
  )
