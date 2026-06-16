"""Go2 expert and MoE mixed-terrain environment configs."""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  HfPyramidSlopedTerrainCfg,
  HfRandomUniformTerrainCfg,
  HfWaveTerrainCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from src.tasks.velocity.terrains import BoxHighPlatformTerrainCfg

from .env_cfgs import unitree_go2_rough_env_cfg


def _terrain_cfg(sub_terrains: dict) -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=5,
    num_cols=len(sub_terrains),
    curriculum=True,
    sub_terrains=sub_terrains,
    add_lights=True,
  )


STAIRS_TERRAINS_CFG = _terrain_cfg(
  {
    "pyramid_stairs": BoxPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.05, 0.25),
      step_width=0.30,
      platform_width=2.5,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": BoxInvertedPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.05, 0.25),
      step_width=0.30,
      platform_width=2.5,
      border_width=1.0,
    ),
  }
)

ROUGH_SLOPE_TERRAINS_CFG = _terrain_cfg(
  {
    "random_rough": HfRandomUniformTerrainCfg(
      proportion=1.0,
      noise_range=(0.02, 0.12),
      noise_step=0.02,
      border_width=0.25,
    ),
    "wave": HfWaveTerrainCfg(
      proportion=1.0,
      amplitude_range=(0.0, 0.25),
      num_waves=4,
      border_width=0.25,
    ),
    "pyramid_slope": HfPyramidSlopedTerrainCfg(
      proportion=1.0,
      slope_range=(0.0, 0.9),
      platform_width=2.0,
      border_width=0.25,
    ),
    "pyramid_slope_inv": HfPyramidSlopedTerrainCfg(
      proportion=1.0,
      slope_range=(0.0, 0.9),
      platform_width=2.0,
      border_width=0.25,
      inverted=True,
    ),
  }
)

CLIMB_TERRAINS_CFG = _terrain_cfg(
  {
    "high_ledge": BoxPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.15, 0.50),
      step_width=1.75,
      platform_width=2.5,
      border_width=1.0,
    ),
    "high_ledge_down": BoxInvertedPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.15, 0.50),
      step_width=1.75,
      platform_width=2.5,
      border_width=1.0,
    ),
    "high_platform": BoxHighPlatformTerrainCfg(
      proportion=1.0,
      platform_height_range=(0.2, 0.6),
      platform_width=2.5,
    ),
  }
)

MOE_MIXED_TERRAINS_CFG = _terrain_cfg(
  {
    "flat": BoxFlatTerrainCfg(proportion=1.0),
    **STAIRS_TERRAINS_CFG.sub_terrains,
    **ROUGH_SLOPE_TERRAINS_CFG.sub_terrains,
    **CLIMB_TERRAINS_CFG.sub_terrains,
  }
)


def _apply_climb_tweaks(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.terminations["illegal_contact"].params["force_threshold"] = 50.0
  cfg.rewards["foot_gait"].weight = 0.25
  cfg.rewards["foot_clearance"].weight = -0.5

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-0.5, 1.5)


def unitree_go2_expert_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_rough_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(ROUGH_SLOPE_TERRAINS_CFG)
  return cfg


def unitree_go2_expert_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_rough_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(STAIRS_TERRAINS_CFG)
  return cfg


def unitree_go2_expert_climb_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_rough_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(CLIMB_TERRAINS_CFG)
  _apply_climb_tweaks(cfg)
  return cfg


def unitree_go2_moe_mixed_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_rough_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(MOE_MIXED_TERRAINS_CFG)
  _apply_climb_tweaks(cfg)
  return cfg
