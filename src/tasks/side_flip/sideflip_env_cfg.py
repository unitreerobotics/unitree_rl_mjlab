"""Sideflip task configuration."""

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

import src.tasks.side_flip.mdp as mdp


FLIP_DURATION_S = 1.6
SIDEFLIP_DIRECTION = 1.0


def make_sideflip_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base sideflip task configuration."""

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

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }

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

  rewards = {
    "feet_contact_before_takeoff": RewardTermCfg(
      func=mdp.feet_contact_before_takeoff,
      weight=0.5,
      params={
        "start_s": 0.0,
        "end_s": 0.25,
        "sensor_name": "feet_ground_contact",
        "min_contacts": 3,
      },
    ),
    "takeoff_vertical_velocity": RewardTermCfg(
      func=mdp.takeoff_vertical_velocity,
      weight=2.0,
      params={
        "start_s": 0.05,
        "end_s": 0.55,
        "target_vz": 1.8,
      },
    ),
    "side_roll_rate_dense": RewardTermCfg(
      func=mdp.side_roll_rate_dense,
      weight=2.0,
      params={
        "start_s": 0.08,
        "end_s": 1.05,
        "target_rate": 7.0,
        "max_rate": 12.0,
        "direction": SIDEFLIP_DIRECTION,
      },
    ),
    "sideflip_progress_delta": RewardTermCfg(
      func=mdp.sideflip_progress_delta,
      weight=8.0,
      params={
        "start_s": 0.08,
        "end_s": 1.20,
        "max_rate": 12.0,
        "max_delta": 0.05,
        "direction": SIDEFLIP_DIRECTION,
      },
    ),
    "airborne_after_takeoff": RewardTermCfg(
      func=mdp.airborne_after_takeoff,
      weight=1.0,
      params={
        "start_s": 0.20,
        "end_s": 1.05,
        "sensor_name": "feet_ground_contact",
      },
    ),
    "apex_height_reward": RewardTermCfg(
      func=mdp.apex_height_reward,
      weight=1.0,
      params={
        "start_s": 0.30,
        "target_height": 0.58,
        "std": 0.18,
        "sensor_name": "feet_ground_contact",
        "direction": SIDEFLIP_DIRECTION,
      },
    ),
    "progress_based_sideflip_orientation": RewardTermCfg(
      func=mdp.progress_based_sideflip_orientation,
      weight=2.0,
      params={
        "start_s": 0.12,
        "end_s": 1.25,
        "std": 0.55,
        "max_rate": 12.0,
        "direction": SIDEFLIP_DIRECTION,
      },
    ),
    "sideflip_progress_final": RewardTermCfg(
      func=mdp.sideflip_progress_final,
      weight=10.0,
      params={
        "start_s": 0.85,
        "std": 0.22,
        "direction": SIDEFLIP_DIRECTION,
      },
    ),
    "landing_success": RewardTermCfg(
      func=mdp.landing_success,
      weight=20.0,
      params={
        "sensor_name": "feet_ground_contact",
        "start_s": 0.95,
        "min_progress": 0.85,
        "min_contacts": 4,
        "max_tilt_xy": 0.40,
        "height_target": 0.32,
        "height_std": 0.14,
        "ang_vel_std": 3.0,
        "lin_vel_xy_std": 1.0,
        "direction": SIDEFLIP_DIRECTION,
      },
    ),
    "landing_position": RewardTermCfg(
      func=mdp.landing_position,
      weight=8.0,
      params={
        "sensor_name": "feet_ground_contact",
        "start_s": 1.00,
        "min_progress": 0.85,
        "min_contacts": 4,
        "xy_std": 0.18,
        "direction": SIDEFLIP_DIRECTION,
      },
    ),
    "landing_joint_posture": RewardTermCfg(
      func=mdp.landing_joint_posture,
      weight=6.0,
      params={
        "sensor_name": "feet_ground_contact",
        "start_s": 1.00,
        "min_progress": 0.85,
        "min_contacts": 4,
        "std": 0.25,
        "direction": SIDEFLIP_DIRECTION,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "landing_foot_stance": RewardTermCfg(
      func=mdp.landing_foot_stance,
      weight=8.0,
      params={
        "sensor_name": "feet_ground_contact",
        "start_s": 1.00,
        "min_progress": 0.85,
        "min_contacts": 4,
        "min_side_y": 0.04,
        "side_std": 0.08,
        "target_width": 0.28,
        "width_std": 0.10,
        "direction": SIDEFLIP_DIRECTION,
        "asset_cfg": SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL")),
      },
    ),
    "off_axis_ang_vel_l2": RewardTermCfg(
      func=mdp.off_axis_ang_vel_l2,
      weight=-0.02,
      params={
        "start_s": 0.08,
        "end_s": 1.20,
      },
    ),
    "excess_sideflip_rotation": RewardTermCfg(
      func=mdp.excess_sideflip_rotation,
      weight=-6.0,
      params={
        "start_s": 1.05,
        "max_rate": 12.0,
        "direction": SIDEFLIP_DIRECTION,
      },
    ),
    "soft_landing_gated": RewardTermCfg(
      func=mdp.soft_landing_gated,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "start_s": 0.95,
        "min_progress": 0.80,
        "force_scale": 800.0,
        "max_penalty": 3.0,
        "direction": SIDEFLIP_DIRECTION,
      },
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-5.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nonfoot_contact": TerminationTermCfg(
      func=mdp.illegal_contact_after_time,
      params={
        "sensor_name": "nonfoot_ground_touch",
        "force_threshold": 10.0,
        "grace_s": 0.65,
      },
    ),
    "base_too_low": TerminationTermCfg(
      func=mdp.base_height_below,
      params={"min_height": 0.08, "grace_s": 0.55},
    ),
    "one_flip_success": TerminationTermCfg(
      func=mdp.one_flip_success,
      time_out=True,
      params={
        "sensor_name": "feet_ground_contact",
        "start_s": 0.95,
        "min_progress": 0.85,
        "max_tilt_xy": 0.35,
        "min_height": 0.22,
        "max_height": 0.48,
        "max_ang_vel": 3.0,
        "min_contacts": 4,
        "direction": SIDEFLIP_DIRECTION,
      },
    ),
  }

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
