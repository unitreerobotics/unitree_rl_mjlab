"""Unitree Go2 velocity environment configurations."""

from dataclasses import replace
from typing import Literal

from src.assets.robots import (
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxNarrowBeamsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  BoxSteppingStonesTerrainCfg,
  HfPyramidSlopedTerrainCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.viewer.viewer_config import ViewerConfig

from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]

# Extreme test terrain: huge step / steep hill / cliff / balance beam.
# Used by the Unitree-Go2-Test task for stress-testing trained policies.
TEST_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=10.0,
  num_rows=5,  # difficulty levels (0.0 .. 1.0), increasing along rows
  num_cols=4,  # one column per sub-terrain type
  # curriculum=True lays the grid out by type x difficulty: each column is a
  # single terrain type, difficulty rises row by row. (False = random scatter.)
  curriculum=True,
  sub_terrains={
    "huge_step": BoxPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.15, 0.40),  # up to 40 cm steps
      step_width=0.40,
      platform_width=2.0,
      border_width=1.0,
    ),
    "steep_hill": HfPyramidSlopedTerrainCfg(
      proportion=1.0,
      slope_range=(0.4, 1.2),  # steep rise/run gradient
      platform_width=2.0,
      border_width=0.25,
    ),
    "cliff": BoxSteppingStonesTerrainCfg(
      proportion=1.0,
      stone_size_range=(0.5, 0.9),
      stone_distance_range=(0.2, 0.6),
      stone_height=0.2,
      stone_height_variation=0.1,
      floor_depth=2.0,  # deep pit = cliff drop around stones
      displacement_range=0.1,
      platform_width=1.0,
      border_width=0.25,
    ),
    "balance_beam": BoxNarrowBeamsTerrainCfg(
      proportion=1.0,
      num_beams=12,
      beam_width_range=(0.15, 0.40),  # narrows with difficulty
      beam_height=0.3,
      spacing=0.8,
      floor_depth=2.0,  # deep pit on either side of the beams
      platform_width=1.0,
      border_width=0.25,
    ),
  },
  add_lights=True,
)


# Flat terrain that *keeps* the terrain_scan sensor (the rays just scan a flat
# plane). Stage 1 of the flat -> rough -> test curriculum pipeline: the
# observation space stays identical to the rough/test configs so a checkpoint
# trained here can be resumed on those terrains. Note this differs from
# `unitree_go2_flat_env_cfg`, which strips the height_scan observation.
FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=10.0,
  num_rows=5,
  num_cols=5,
  curriculum=True,
  sub_terrains={
    "flat": BoxFlatTerrainCfg(proportion=1.0),
  },
  add_lights=True,
)


def unitree_go2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 128

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Set raycast sensor frame to Go2 base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def unitree_go2_flat_scan_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Go2 velocity config on flat terrain that keeps the height scan.

  Stage 1 of the flat -> rough -> test staged training pipeline. Identical to
  `unitree_go2_rough_env_cfg` (same sim/sensors/rewards/commands and the
  terrain-level curriculum) except the terrain generator is swapped to an
  all-flat `FLAT_TERRAINS_CFG`. Because the `terrain_scan` sensor and the
  `height_scan` observation are kept, the observation space matches the rough
  and test configs, so a checkpoint trained here resumes cleanly on those.
  """
  cfg = unitree_go2_rough_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  # `replace` gives this task its own copy so play-mode mutations never touch
  # the shared module-level instance.
  cfg.scene.terrain.terrain_generator = replace(FLAT_TERRAINS_CFG)
  return cfg


def unitree_go2_rough_no_height_scan_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Go2 rough-terrain velocity config without the height scan observation."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  return cfg


def unitree_go2_test_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Go2 velocity config on an extreme test terrain.

  Terrain has four columns: huge step, steep hill, cliff (stepping stones over
  a deep pit), and balance beam. Intended for play/evaluation stress-testing.
  """
  cfg = unitree_go2_rough_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  # Swap in the test terrain. `replace` gives each task its own copy so the
  # play-mode override never mutates the shared module-level instance.
  cfg.scene.terrain.terrain_generator = replace(TEST_TERRAINS_CFG)

  # Third-person "behind the robot" camera: track the base and sit behind it,
  # looking along +x (the robot's forward travel direction).
  cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_BODY
  cfg.viewer.entity_name = "robot"
  cfg.viewer.body_name = "base_link"
  cfg.viewer.azimuth = 0.0
  cfg.viewer.elevation = -15.0
  cfg.viewer.distance = 2.5

  if play:
    # Lock the command to a steady forward walk facing world +x so the robot
    # stays in front of the tracking camera (TPS-style behind view).
    # `heading_command=True` drives yaw toward `heading`, so a heading range of
    # (0, 0) keeps the robot pointing +x; ang_vel_z is unused and left as-is.
    # lin_vel_y keeps a tiny symmetric range (averages to straight) because the
    # viser GUI requires each axis' upper bound to be >= 0.1.
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.rel_standing_envs = 0.0
    twist_cmd.ranges.lin_vel_x = (0.8, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.1, 0.1)
    twist_cmd.ranges.heading = (0.0, 0.0)

  return cfg


def unitree_go2_no_phase_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Go2 rough-terrain velocity config with the phase observation removed.

  Identical to `unitree_go2_rough_env_cfg` except the periodic gait-clock
  `phase` observation is dropped from both the actor and critic groups. Use
  this to study how much the policy depends on an explicit gait phase signal.
  The `foot_gait` reward (which also uses period 0.6) is left unchanged.
  """
  cfg = unitree_go2_rough_env_cfg(play=play)
  del cfg.observations["actor"].terms["phase"]
  del cfg.observations["critic"].terms["phase"]
  return cfg


def unitree_go2_rough_split_obs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Go2 rough-terrain config with the actor observation split into groups.

  Identical dynamics/terms to `unitree_go2_rough_env_cfg`; the only change is
  that the single concatenated ``actor`` observation group is re-bucketed into
  per-term groups (``height_scan``, ``command``, ``projected_gravity``,
  ``proprio``, ``last_action``) so the observation-encoder framework can address
  individual signals via ``encoder_input_keys``/``passthrough_keys``. The
  privileged ``critic`` group is left untouched (single concatenated tensor).

  Used by the ``Unitree-Go2-Rough-Encoder-*`` ablation tasks. Train these with a
  runner that does not export ONNX (e.g. ``MjlabOnPolicyRunner``).
  """
  cfg = unitree_go2_rough_env_cfg(play=play)

  actor_group = cfg.observations["actor"]
  terms = actor_group.terms
  corrupt = actor_group.enable_corruption  # False in play mode (handled upstream).

  def _group(term_names: tuple[str, ...]) -> ObservationGroupCfg:
    return ObservationGroupCfg(
      terms={name: terms[name] for name in term_names},
      concatenate_terms=True,
      enable_corruption=corrupt,
      history_length=1,
    )

  cfg.observations = {
    "height_scan": _group(("height_scan",)),
    "command": _group(("command",)),
    "projected_gravity": _group(("projected_gravity",)),
    "proprio": _group(("base_ang_vel", "phase", "joint_pos", "joint_vel")),
    "last_action": _group(("actions",)),
    # Privileged critic obs kept as the original single concatenated group.
    "critic": cfg.observations["critic"],
  }
  return cfg


def unitree_go2_test_train_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Go2 velocity training config on the extreme test terrain.

  Identical to `unitree_go2_rough_env_cfg` (same sim/sensors/rewards/commands
  and terrain curriculum) except the terrain generator is swapped to
  `TEST_TERRAINS_CFG`. Use this to *train* on the test terrain; for
  play/evaluation with the third-person camera use `unitree_go2_test_env_cfg`.
  """
  cfg = unitree_go2_rough_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  # Swap in the test terrain. `replace` gives each task its own copy so
  # play-mode mutations never touch the shared module-level instance.
  cfg.scene.terrain.terrain_generator = replace(TEST_TERRAINS_CFG)
  return cfg
