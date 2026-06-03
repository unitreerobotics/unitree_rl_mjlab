"""Backflip task configuration."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.back_flip.mdp as mdp


FLIP_DURATION_S = 1.6


def make_backflip_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base backflip task configuration."""

  ##
  # Observations
  ##

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.3, n_max=0.3),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "flip_phase": ObservationTermCfg(
      func=mdp.flip_phase,
      params={"duration_s": FLIP_DURATION_S},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.0, n_max=1.0),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  critic_terms = {
    **actor_terms,
    "base_height": ObservationTermCfg(func=mdp.base_height),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.03, 0.03),
          "y": (-0.03, 0.03),
          "z": (0.0, 0.0),
          "yaw": (-0.05, 0.05),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.02, 0.02),
        "velocity_range": (-0.05, 0.05),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "ranges": (0.6, 1.6),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "operation": "add",
        "ranges": {
          0: (-0.02, 0.02),
          1: (-0.02, 0.02),
          2: (-0.02, 0.02),
        },
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "rotation_progress": RewardTermCfg(
      func=mdp.backflip_progress,
      weight=4.0,
      params={
        "duration_s": FLIP_DURATION_S,
        "takeoff_s": 0.18,
        "landing_s": 1.15,
        "std": 0.18,
      },
    ),
    "backward_pitch_rate": RewardTermCfg(
      func=mdp.backward_pitch_rate,
      weight=1.2,
      params={
        "start_s": 0.15,
        "end_s": 0.95,
        "target_rate": 7.0,
        "std": 4.0,
      },
    ),
    "backward_pitch_orientation": RewardTermCfg(
      func=mdp.backward_pitch_orientation,
      weight=3.0,
      params={
        "takeoff_s": 0.18,
        "landing_s": 1.15,
        "std": 0.45,
      },
    ),
    "vertical_midflip_orientation": RewardTermCfg(
      func=mdp.vertical_midflip_orientation,
      weight=2.0,
      params={
        "center_s": 0.66,
        "width_s": 0.22,
        "std": 0.25,
      },
    ),
    "base_height_schedule": RewardTermCfg(
      func=mdp.base_height_schedule,
      weight=1.5,
      params={
        "duration_s": FLIP_DURATION_S,
        "crouch_height": 0.24,
        "air_height": 0.56,
        "landing_height": 0.32,
        "std": 0.14,
      },
    ),
    "foot_contact_schedule": RewardTermCfg(
      func=mdp.foot_contact_schedule,
      weight=1.0,
      params={
        "duration_s": FLIP_DURATION_S,
        "sensor_name": "feet_ground_contact",
      },
    ),
    "upright_landing": RewardTermCfg(
      func=mdp.upright_landing,
      weight=5.0,
      params={
        "sensor_name": "feet_ground_contact",
        "start_s": 1.15,
        "height_target": 0.32,
        "height_std": 0.12,
      },
    ),
    "rearward_displacement": RewardTermCfg(
      func=mdp.rearward_displacement,
      weight=0.3,
      params={
        "duration_s": FLIP_DURATION_S,
        "target_x": -0.15,
        "std": 0.2,
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-5e-4,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "off_axis_ang_vel_l2": RewardTermCfg(
      func=mdp.off_axis_ang_vel_l2,
      weight=-0.03,
      params={
        "start_s": 0.12,
        "end_s": 1.20,
      },
    ),
    "excess_backflip_rotation": RewardTermCfg(
      func=mdp.excess_backflip_rotation,
      weight=-6.0,
      params={"start_s": 1.05},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-25.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nonfoot_contact": TerminationTermCfg(
      func=mdp.illegal_contact_after_time,
      params={
        "sensor_name": "nonfoot_ground_touch",
        "force_threshold": 10.0,
        "grace_s": 0.2,
      },
    ),
    "base_too_low": TerminationTermCfg(
      func=mdp.base_height_below,
      params={"min_height": 0.12, "grace_s": 0.2},
    ),
    "one_flip_success": TerminationTermCfg(
      func=mdp.one_flip_success,
      time_out=True,
      params={
        "sensor_name": "feet_ground_contact",
        "start_s": 1.05,
        "max_tilt_xy": 0.35,
        "min_height": 0.22,
        "max_height": 0.48,
        "max_ang_vel": 3.0,
      },
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="plane",
        terrain_generator=None,
      ),
      sensors=(),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=128,
      njmax=300,
      contact_sensor_maxmatch=128,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=100,
      ),
    ),
    decimation=4,
    episode_length_s=FLIP_DURATION_S,
    is_finite_horizon=True,
  )
