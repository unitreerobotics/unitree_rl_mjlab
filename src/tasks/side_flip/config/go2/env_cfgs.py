"""Unitree Go2 sideflip environment configuration."""

from src.assets.robots import get_go2_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

import src.tasks.side_flip.mdp as mdp
from src.tasks.side_flip.sideflip_env_cfg import FLIP_DURATION_S, make_sideflip_env_cfg


def unitree_go2_sideflip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat-ground sideflip configuration."""
  cfg = make_sideflip_env_cfg()
  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  foot_names = ("FR", "FL", "RR", "RL")
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
      pattern=r".*_collision\d*$",
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (feet_ground_cfg, nonfoot_ground_cfg)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0
  cfg.viewer.azimuth = 90.0

  cfg.terminations["nonfoot_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact_after_time,
    params={
      "sensor_name": nonfoot_ground_cfg.name,
      "force_threshold": 10.0,
      "grace_s": 0.65,
    },
  )

  if play:
    cfg.episode_length_s = FLIP_DURATION_S
    cfg.is_finite_horizon = True
    cfg.observations["actor"].enable_corruption = False

  return cfg
