"""Unitree H2 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

H2_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_h2" / "xmls" / "h2.xml"
)
assert H2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, H2_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(H2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

H2_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch.*",
    ".*_hip_roll.*",
    ".*_hip_yaw.*",
    ".*_knee.*",
  ),
  stiffness=200.0,
  damping=4.0,
  effort_limit=360.0,
  armature=0.01,
)
H2_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_roll.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=19.0,
  armature=0.01,
)
H2_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=66.0,
  armature=0.01,
)
H2_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_.*",
  ),
  stiffness=150.0,
  damping=3.0,
  effort_limit=120.0,
  armature=0.01,
)
H2_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch.*",
    ".*_shoulder_roll.*",
    ".*_shoulder_yaw.*",
    ".*_elbow.*",
    ".*_wrist_roll.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=54.0,
  armature=0.01,
)
H2_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_wrist_pitch.*",
    ".*_wrist_yaw.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=25.0,
  armature=0.01,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.03),
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
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

H2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    H2_ACTUATOR_LEG,
    H2_ACTUATOR_ANKLE_ROLL,
    H2_ACTUATOR_ANKLE_PITCH,
    H2_ACTUATOR_WAIST,
    H2_ACTUATOR_ARM,
    H2_ACTUATOR_WRIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_h2_robot_cfg(actuator_delay: bool = False) -> EntityCfg:
  """Get a fresh H2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.

  Args:
    actuator_delay: Wrap every actuator in a DelayedActuatorCfg (0-4 physics
      steps = 0-20 ms at the 0.005 s timestep) so latency can be domain-
      randomized. Pair with the dr.sync_actuator_delays reset event; with the
      default lag of 0 the wrapper is a no-op.
  """
  articulation = H2_ARTICULATION
  if actuator_delay:
    articulation = EntityArticulationInfoCfg(
      actuators=tuple(
        DelayedActuatorCfg(base_cfg=a, delay_min_lag=0, delay_max_lag=4)
        for a in H2_ARTICULATION.actuators
      ),
      soft_joint_pos_limit_factor=0.9,
    )
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=articulation,
  )


H2_ACTION_SCALE: dict[str, float] = {}
for a in H2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    H2_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_h2_robot_cfg())

  viewer.launch(robot.spec.compile())
