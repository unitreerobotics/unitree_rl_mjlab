"""Unitree B2YGX constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

B2YGX_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_b2ygx" / "xmls" / "b2ygx.xml"
)
assert B2YGX_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, B2YGX_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(B2YGX_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

B2YGX_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*hip_.*",
  ),
  stiffness=200.0,
  damping=10.0,
  effort_limit=200.0,
  armature=0.1,
)
B2YGX_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*thigh_.*",
  ),
  stiffness=200.0,
  damping=10.0,
  effort_limit=200.0,
  armature=0.1,
)
B2YGX_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*calf_.*",
  ),
  stiffness=240.0,
  damping=12.0,
  effort_limit=300.0,
  armature=0.1,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.467),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*hip_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

B2YGX_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    B2YGX_ACTUATOR_HIP,
    B2YGX_ACTUATOR_THIGH,
    B2YGX_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_b2ygx_robot_cfg() -> EntityCfg:
  """Get a fresh B2YGX robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=B2YGX_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_b2ygx_robot_cfg())

  viewer.launch(robot.spec.compile())
