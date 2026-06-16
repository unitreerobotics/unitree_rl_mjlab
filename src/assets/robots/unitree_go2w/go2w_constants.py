"""Unitree Go2W (wheeled Go2) constants.

Wheel joint/inertia values come from the official description at
https://github.com/unitreerobotics/unitree_ros/tree/master/robots/go2w_description
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.xml_actuator import XmlVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

GO2W_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_go2w" / "xmls" / "go2w.xml"
)
assert GO2W_XML.exists()

# The Go2W reuses the Go2 leg/base meshes; wheels are plain cylinder geoms.
GO2_ASSETS_DIR = SRC_PATH / "assets" / "robots" / "unitree_go2" / "xmls" / "assets"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, GO2_ASSETS_DIR, meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2W_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

GO2W_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*hip_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2W_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*thigh_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2W_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*calf_.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=45,
  armature=0.02,
)
# Wheels are velocity-controlled continuous joints. The <velocity> actuators
# (kv, ctrlrange ±30.1 rad/s, forcerange ±23.7 Nm from the go2w URDF) and the
# joint armature are defined in go2w.xml because mjlab's builtin velocity
# actuator derives its ctrlrange from the joint range, which a continuous
# joint does not have.
GO2W_ACTUATOR_WHEEL = XmlVelocityActuatorCfg(
  target_names_expr=(
    ".*_foot_joint",
  ),
)

##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
  # Axle sits ~0.273 below the base at the default pose, plus 0.086 wheel radius.
  pos=(0.0, 0.0, 0.38),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
    ".*_foot_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the wheels.
# Furthermore, wheel self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions, excluding self collisions.
# Wheel collisions are given custom condim, friction and solimp.
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

GO2W_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2W_ACTUATOR_HIP,
    GO2W_ACTUATOR_THIGH,
    GO2W_ACTUATOR_CALF,
    GO2W_ACTUATOR_WHEEL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2w_robot_cfg() -> EntityCfg:
  """Get a fresh Go2W robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2W_ARTICULATION,
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2w_robot_cfg())

  viewer.launch(robot.spec.compile())
