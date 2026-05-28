"""Unitree Go2-W velocity environment configurations."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import mdp
from .base import (
  GO2W_LEG_JOINT_NAMES,
  GO2W_WHEEL_JOINT_NAMES,
  get_go2w_scene_robot_cfg,
  go2w_leg_joint_cfg,
  go2w_wheel_ground_contact_cfg,
  go2w_wheel_joint_cfg,
)
from .velocity_env_cfg import make_velocity_env_cfg


def unitree_go2w_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the supported rough-terrain hybrid Go2-W velocity config."""
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": get_go2w_scene_robot_cfg()}

  leg_joint_cfg = go2w_leg_joint_cfg()
  wheel_joint_cfg = go2w_wheel_joint_cfg()

  cfg.actions["joint_pos"] = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=GO2W_LEG_JOINT_NAMES,
    preserve_order=True,
    scale=0.5,
    use_default_offset=True,
  )
  cfg.actions["wheel_vel"] = JointVelocityActionCfg(
    entity_name="robot",
    actuator_names=GO2W_WHEEL_JOINT_NAMES,
    preserve_order=True,
    scale=35.0,
    offset=0.0,
    use_default_offset=False,
  )

  wheel_ground_cfg = go2w_wheel_ground_contact_cfg()
  cfg.scene.sensors = (wheel_ground_cfg,)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True
    cfg.scene.terrain.max_init_terrain_level = 2
    tg = cfg.scene.terrain.terrain_generator
    tg.difficulty_range = (0.0, 0.45)
    tg.sub_terrains["flat"].proportion = 0.45
    tg.sub_terrains["random_rough"].proportion = 0.30
    tg.sub_terrains["random_rough"].noise_range = (0.01, 0.05)
    tg.sub_terrains["random_rough"].noise_step = 0.01
    tg.sub_terrains["hf_pyramid_slope"].proportion = 0.15
    tg.sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.25)
    tg.sub_terrains["wave_terrain"].proportion = 0.10
    tg.sub_terrains["wave_terrain"].amplitude_range = (0.0, 0.06)
    tg.sub_terrains["wave_terrain"].num_waves = 2.0
    tg.sub_terrains["pyramid_stairs"].proportion = 0.0
    tg.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
    tg.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0

  twist_cmd = cfg.commands["twist"]
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.ranges.heading = None
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)

  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-0.3, 0.8), "ang_vel_z": (-0.8, 0.8)},
      {"step": 5000 * 24, "lin_vel_x": (-0.8, 1.5), "ang_vel_z": (-1.0, 1.0)},
    ]

  for group_name in ("policy", "critic"):
    terms = cfg.observations[group_name].terms
    actions_term = terms.pop("actions")

    terms["joint_pos"].params = {"asset_cfg": leg_joint_cfg}
    terms["joint_vel"].params = {"asset_cfg": leg_joint_cfg}
    terms["wheel_joint_pos_rel"] = ObservationTermCfg(
      func=mdp.wheel_joint_pos_rel,
      params={"asset_cfg": wheel_joint_cfg},
      noise=Unoise(n_min=-0.01, n_max=0.01) if group_name == "policy" else None,
    )
    terms["wheel_joint_vel_rel"] = ObservationTermCfg(
      func=mdp.wheel_joint_vel_rel,
      params={"asset_cfg": wheel_joint_cfg},
      noise=Unoise(n_min=-1.0, n_max=1.0) if group_name == "policy" else None,
    )
    terms["actions"] = actions_term

  cfg.observations["policy"].terms["base_ang_vel"].func = mdp.base_ang_vel
  cfg.observations["policy"].terms["base_ang_vel"].params = {}
  cfg.observations["critic"].terms["base_ang_vel"].func = mdp.base_ang_vel
  cfg.observations["critic"].terms["base_ang_vel"].params = {}
  cfg.observations["critic"].terms["base_lin_vel"].func = mdp.base_lin_vel
  cfg.observations["critic"].terms["base_lin_vel"].params = {}

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["body_friction"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    geom_ids=slice(None),
  )

  cfg.rewards["pose"].params["asset_cfg"] = leg_joint_cfg
  cfg.rewards["joint_pos_limits"].params["asset_cfg"] = leg_joint_cfg
  cfg.rewards["joint_acc_l2"].params["asset_cfg"] = leg_joint_cfg
  cfg.rewards["pose"].params["std_standing"] = {
    r"^(FR|FL|RR|RL)_hip_joint$": 0.05,
    r"^(FR|FL|RR|RL)_thigh_joint$": 0.1,
    r"^(FR|FL|RR|RL)_calf_joint$": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r"^(FR|FL|RR|RL)_hip_joint$": 0.15,
    r"^(FR|FL|RR|RL)_thigh_joint$": 0.35,
    r"^(FR|FL|RR|RL)_calf_joint$": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r"^(FR|FL|RR|RL)_hip_joint$": 0.15,
    r"^(FR|FL|RR|RL)_thigh_joint$": 0.35,
    r"^(FR|FL|RR|RL)_calf_joint$": 0.5,
  }

  for name in (
    "foot_air_time",
    "foot_clearance",
    "foot_slip",
    "soft_landing",
    "angular_momentum",
  ):
    cfg.rewards.pop(name, None)

  cfg.rewards["wheel_roll_tracking"] = RewardTermCfg(
    func=mdp.wheel_roll_tracking,
    weight=2.0,
    params={
      "command_name": "twist",
      "wheel_radius": 0.09,
      "wheel_track": 0.19,
      "std": 8.0,
      "asset_cfg": wheel_joint_cfg,
    },
  )
  cfg.rewards["wheel_contact_bonus"] = RewardTermCfg(
    func=mdp.contact_fraction_reward,
    weight=0.5,
    params={"sensor_name": wheel_ground_cfg.name},
  )
  cfg.rewards["leg_motion_penalty"] = RewardTermCfg(
    func=mdp.adaptive_leg_motion_penalty,
    weight=-0.08,
    params={
      "command_name": "twist",
      "sensor_name": wheel_ground_cfg.name,
      "command_threshold": 0.05,
      "tilt_relax_start": 0.08,
      "tilt_relax_end": 0.30,
      "contact_target": 0.85,
      "min_penalty_scale": 0.2,
      "asset_cfg": leg_joint_cfg,
    },
  )
  cfg.rewards["flat_orientation_l2"].weight = -2.5
  cfg.rewards["body_ang_vel"].weight = -0.1
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  cfg.events.pop("push_robot", None)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2w_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the supported flat-terrain hybrid Go2-W velocity config."""
  cfg = unitree_go2w_rough_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  leg_joint_cfg = go2w_leg_joint_cfg()
  cfg.rewards["leg_motion_penalty"] = RewardTermCfg(
    func=mdp.leg_motion_penalty,
    weight=-0.15,
    params={
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": leg_joint_cfg,
    },
  )

  twist_cmd = cfg.commands["twist"]
  twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-0.5, 1.0), "ang_vel_z": (-1.0, 1.0)},
      {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "ang_vel_z": (-1.2, 1.2)},
    ]

  return cfg


def unitree_go2w_flat_legs_only_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the supported flat legs-only Go2-W walking config."""
  cfg = unitree_go2w_flat_env_cfg(play=play)

  leg_joint_cfg = go2w_leg_joint_cfg()
  wheel_joint_cfg = go2w_wheel_joint_cfg()

  cfg.actions.pop("wheel_vel", None)
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 0.35

  cfg.rewards.pop("wheel_roll_tracking", None)
  cfg.rewards.pop("wheel_contact_bonus", None)
  cfg.rewards.pop("leg_motion_penalty", None)
  cfg.rewards["wheel_spin_limit"] = RewardTermCfg(
    func=mdp.wheel_speed_limit_penalty,
    weight=-0.3,
    params={
      "max_abs_speed": 3.0,
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": wheel_joint_cfg,
    },
  )
  cfg.rewards["base_height"] = RewardTermCfg(
    func=mdp.base_height_tracking,
    weight=1.0,
    params={"target_height": 0.32, "std": 0.08},
  )
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=mdp.stand_still,
    weight=-0.2,
    params={
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": leg_joint_cfg,
    },
  )
  cfg.rewards["flat_orientation_l2"].weight = -4.0
  cfg.rewards["body_ang_vel"].weight = -0.2

  cfg.events["body_friction"].params["ranges"] = (0.8, 1.8)
  cfg.terminations["low_base_height"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={"minimum_height": 0.18},
  )

  twist_cmd = cfg.commands["twist"]
  twist_cmd.rel_standing_envs = 0.15
  twist_cmd.ranges.lin_vel_x = (-0.2, 0.5)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-0.1, 0.25), "ang_vel_z": (-0.2, 0.2)},
      {"step": 3000 * 24, "lin_vel_x": (-0.25, 0.5), "ang_vel_z": (-0.4, 0.4)},
      {"step": 8000 * 24, "lin_vel_x": (-0.5, 0.8), "ang_vel_z": (-0.6, 0.6)},
    ]

  return cfg


def unitree_go2w_flat_legs_only_omni_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the supported omni-directional flat Go2-W legs-only config."""
  cfg = unitree_go2w_flat_legs_only_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  twist_cmd.ranges.lin_vel_x = (-0.2, 0.5)
  twist_cmd.ranges.lin_vel_y = (-0.15, 0.15)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
  twist_cmd.rel_standing_envs = 0.1

  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.1, 0.25),
        "lin_vel_y": (-0.05, 0.05),
        "ang_vel_z": (-0.2, 0.2),
      },
      {
        "step": 3000 * 24,
        "lin_vel_x": (-0.25, 0.5),
        "lin_vel_y": (-0.15, 0.15),
        "ang_vel_z": (-0.4, 0.4),
      },
      {
        "step": 8000 * 24,
        "lin_vel_x": (-0.5, 0.8),
        "lin_vel_y": (-0.3, 0.3),
        "ang_vel_z": (-0.6, 0.6),
      },
      {
        "step": 14000 * 24,
        "lin_vel_x": (-0.8, 1.0),
        "lin_vel_y": (-0.4, 0.4),
        "ang_vel_z": (-0.8, 0.8),
      },
    ]

  cfg.rewards["stand_still"].weight = -0.1
  cfg.rewards["flat_orientation_l2"].weight = -3.5
  return cfg
