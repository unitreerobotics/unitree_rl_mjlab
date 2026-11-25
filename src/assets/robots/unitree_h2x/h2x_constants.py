"""Unitree H2X constants."""

from copy import deepcopy
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from src import SRC_PATH

##
# MJCF and assets.
##

H2X_XML: Path = SRC_PATH / "assets" / "robots" / "unitree_h2x" / "xmls" / "h2x.xml"
assert H2X_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(H2X_XML))


##
# Actuator config.
##

# Effort limits follow the official H2X URDF.
H2X_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch.*",),
  stiffness=200.0,
  damping=4.0,
  effort_limit=220.0,
  armature=0.01,
)
H2X_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_roll.*",
    ".*_hip_yaw.*",
    ".*_knee.*",
  ),
  stiffness=200.0,
  damping=4.0,
  effort_limit=248.0,
  armature=0.01,
)
H2X_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch.*",
    ".*_ankle_roll.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=100.0,
  armature=0.01,
)
H2X_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw.*",),
  stiffness=150.0,
  damping=3.0,
  effort_limit=111.0,
  armature=0.01,
)
H2X_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_roll.*",
    "waist_pitch.*",
  ),
  stiffness=150.0,
  damping=3.0,
  effort_limit=180.0,
  armature=0.01,
)
H2X_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch.*",),
  stiffness=40.0,
  damping=2.0,
  effort_limit=111.0,
  armature=0.01,
)
H2X_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_roll.*",
    ".*_shoulder_yaw.*",
    ".*_elbow.*",
    ".*_wrist_roll.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=53.7,
  armature=0.01,
)
H2X_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_wrist_pitch.*",
    ".*_wrist_yaw.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=24.8,
  armature=0.01,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.01),
  joint_pos={
    ".*_hip_pitch_joint": -0.25,
    ".*_knee_joint": 0.5,
    ".*_ankle_pitch_joint": -0.25,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-5]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-5]_collision$": 1, ".*": 0},
  friction={r"^(left|right)_foot[1-5]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-5]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-5]_collision$": 1, ".*": 0},
  friction={r"^(left|right)_foot[1-5]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-5]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

H2X_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    H2X_ACTUATOR_HIP_PITCH,
    H2X_ACTUATOR_LEG,
    H2X_ACTUATOR_ANKLE,
    H2X_ACTUATOR_WAIST_YAW,
    H2X_ACTUATOR_WAIST,
    H2X_ACTUATOR_SHOULDER_PITCH,
    H2X_ACTUATOR_ARM,
    H2X_ACTUATOR_WRIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_h2x_robot_cfg() -> EntityCfg:
  """Get a fresh H2X robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=deepcopy(HOME_KEYFRAME),
    collisions=(deepcopy(FULL_COLLISION),),
    spec_fn=get_spec,
    articulation=deepcopy(H2X_ARTICULATION),
  )


H2X_ACTION_SCALE: dict[str, float] = {}
for a in H2X_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    H2X_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_h2x_robot_cfg())

  viewer.launch(robot.spec.compile())
