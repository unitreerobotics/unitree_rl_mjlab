from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  BoxRandomSpreadTerrainCfg,
  HfPerlinNoiseTerrainCfg,
  HfRandomUniformTerrainCfg,
)
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg

SUPPORTED_EVAL_TERRAINS = (
  "rough_curriculum_corridor",
  "perlin_noise_corridor",
  "random_spread_boxes_corridor",
  "stairs_corridor",
)


def _flat_patch(
  size: tuple[float, float],
  difficulty: float,
  spec: mujoco.MjSpec,
  rng: np.random.Generator,
):
  flat = BoxFlatTerrainCfg(proportion=1.0)
  flat.size = size
  return flat.function(difficulty, spec, rng)


def _is_spawn_patch(difficulty: float, num_terrain_patches: int) -> bool:
  return difficulty < 1.0 / (num_terrain_patches + 2)


def _is_finish_patch(difficulty: float, num_terrain_patches: int) -> bool:
  return difficulty >= (num_terrain_patches + 1) / (num_terrain_patches + 2)


def _terrain_progress(difficulty: float, num_terrain_patches: int) -> float:
  return float(
    np.clip(
      (difficulty * (num_terrain_patches + 2) - 1.0)
      / max(num_terrain_patches - 1, 1),
      0.0,
      1.0,
    )
  )


@dataclass(kw_only=True)
class RoughCurriculumCorridorTerrainCfg(SubTerrainCfg):
  """Single-column rough corridor with flat spawn and finish patches.

  The stock terrain generator lays rows along world x. This sub-terrain uses the
  first row as a flat spawn area, the last row as a flat finish area, and maps the
  intermediate rows to progressively rougher random-uniform heightfields.
  """

  num_rough_patches: int = 8
  min_noise: float = 0.01
  max_noise: float = 0.12
  noise_step: float = 0.01
  border_width: float = 0.0
  horizontal_scale: float = 0.08
  vertical_scale: float = 0.005

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ):
    if _is_spawn_patch(difficulty, self.num_rough_patches) or _is_finish_patch(
      difficulty, self.num_rough_patches
    ):
      return _flat_patch(self.size, difficulty, spec, rng)

    rough_progress = _terrain_progress(difficulty, self.num_rough_patches)
    noise_hi = self.min_noise + rough_progress * (self.max_noise - self.min_noise)
    rough = HfRandomUniformTerrainCfg(
      proportion=1.0,
      noise_range=(self.min_noise, float(noise_hi)),
      noise_step=self.noise_step,
      border_width=self.border_width,
      horizontal_scale=self.horizontal_scale,
      vertical_scale=self.vertical_scale,
    )
    rough.size = self.size
    return rough.function(rough_progress, spec, rng)


@dataclass(kw_only=True)
class PerlinNoiseCorridorTerrainCfg(SubTerrainCfg):
  """Single-column Perlin-noise corridor with flat spawn and finish patches."""

  num_terrain_patches: int = 8
  min_height: float = 0.01
  max_height: float = 0.18
  octaves: int = 4
  persistence: float = 0.3
  lacunarity: float = 2.0
  scale: float = 10.0
  horizontal_scale: float = 0.1
  resolution: float = 0.05
  border_width: float = 0.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ):
    if _is_spawn_patch(difficulty, self.num_terrain_patches) or _is_finish_patch(
      difficulty, self.num_terrain_patches
    ):
      return _flat_patch(self.size, difficulty, spec, rng)

    progress = _terrain_progress(difficulty, self.num_terrain_patches)
    terrain = HfPerlinNoiseTerrainCfg(
      proportion=1.0,
      height_range=(self.min_height, self.max_height),
      octaves=self.octaves,
      persistence=self.persistence,
      lacunarity=self.lacunarity,
      scale=self.scale,
      horizontal_scale=self.horizontal_scale,
      resolution=self.resolution,
      border_width=self.border_width,
    )
    terrain.size = self.size
    return terrain.function(progress, spec, rng)


@dataclass(kw_only=True)
class RandomSpreadBoxesCorridorTerrainCfg(SubTerrainCfg):
  """Single-column random-spread-box corridor with flat spawn and finish patches."""

  num_terrain_patches: int = 8
  min_boxes: int = 20
  max_boxes: int = 80
  box_width_range: tuple[float, float] = (0.1, 1.0)
  box_length_range: tuple[float, float] = (0.1, 2.0)
  min_box_height: float = 0.03
  max_box_height: float = 0.30
  box_yaw_range: tuple[float, float] = (0.0, 360.0)
  platform_width: float = 1.0
  border_width: float = 0.25

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ):
    if _is_spawn_patch(difficulty, self.num_terrain_patches) or _is_finish_patch(
      difficulty, self.num_terrain_patches
    ):
      return _flat_patch(self.size, difficulty, spec, rng)

    progress = _terrain_progress(difficulty, self.num_terrain_patches)
    target_boxes = self.min_boxes + progress * (self.max_boxes - self.min_boxes)
    scale = 0.5 + 0.5 * progress
    num_boxes = int(round(target_boxes / scale))
    terrain = BoxRandomSpreadTerrainCfg(
      proportion=1.0,
      num_boxes=num_boxes,
      box_width_range=self.box_width_range,
      box_length_range=self.box_length_range,
      box_height_range=(self.min_box_height, self.max_box_height),
      box_yaw_range=self.box_yaw_range,
      add_floor=True,
      platform_width=self.platform_width,
      border_width=self.border_width,
    )
    terrain.size = self.size
    return terrain.function(progress, spec, rng)


@dataclass(kw_only=True)
class StairsCorridorTerrainCfg(SubTerrainCfg):
  """Single-column pyramid-stairs corridor with flat spawn and finish patches.

  Each intermediate patch is a pyramid staircase (ascend to the patch center,
  then descend). The step height grows with difficulty along the corridor.
  """

  num_terrain_patches: int = 8
  min_step_height: float = 0.05
  max_step_height: float = 0.20
  step_width: float = 0.30
  platform_width: float = 1.0
  border_width: float = 0.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ):
    if _is_spawn_patch(difficulty, self.num_terrain_patches) or _is_finish_patch(
      difficulty, self.num_terrain_patches
    ):
      return _flat_patch(self.size, difficulty, spec, rng)

    progress = _terrain_progress(difficulty, self.num_terrain_patches)
    stairs = BoxPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(self.min_step_height, self.max_step_height),
      step_width=self.step_width,
      platform_width=self.platform_width,
      border_width=self.border_width,
    )
    stairs.size = self.size
    return stairs.function(progress, spec, rng)


def make_rough_curriculum_corridor_cfg(
  *,
  seed: int | None,
  patch_length: float = 4.0,
  corridor_width: float = 3.0,
  num_rough_patches: int = 8,
  min_noise: float = 0.01,
  max_noise: float = 0.12,
) -> TerrainGeneratorCfg:
  rows = num_rough_patches + 2
  return TerrainGeneratorCfg(
    seed=seed,
    curriculum=True,
    size=(patch_length, corridor_width),
    border_width=2.0,
    num_rows=rows,
    num_cols=1,
    color_scheme="height",
    difficulty_range=(0.0, 1.0),
    sub_terrains={
      "rough_curriculum_corridor": RoughCurriculumCorridorTerrainCfg(
        proportion=1.0,
        num_rough_patches=num_rough_patches,
        min_noise=min_noise,
        max_noise=max_noise,
      )
    },
    add_lights=True,
  )


def make_perlin_noise_corridor_cfg(
  *,
  seed: int | None,
  patch_length: float = 4.0,
  corridor_width: float = 3.0,
  num_terrain_patches: int = 8,
  min_height: float = 0.01,
  max_height: float = 0.18,
) -> TerrainGeneratorCfg:
  rows = num_terrain_patches + 2
  return TerrainGeneratorCfg(
    seed=seed,
    curriculum=True,
    size=(patch_length, corridor_width),
    border_width=2.0,
    num_rows=rows,
    num_cols=1,
    color_scheme="height",
    difficulty_range=(0.0, 1.0),
    sub_terrains={
      "perlin_noise_corridor": PerlinNoiseCorridorTerrainCfg(
        proportion=1.0,
        num_terrain_patches=num_terrain_patches,
        min_height=min_height,
        max_height=max_height,
      )
    },
    add_lights=True,
  )


def make_random_spread_boxes_corridor_cfg(
  *,
  seed: int | None,
  patch_length: float = 4.0,
  corridor_width: float = 3.0,
  num_terrain_patches: int = 8,
  min_boxes: int = 20,
  max_boxes: int = 80,
  min_box_height: float = 0.03,
  max_box_height: float = 0.30,
) -> TerrainGeneratorCfg:
  rows = num_terrain_patches + 2
  return TerrainGeneratorCfg(
    seed=seed,
    curriculum=True,
    size=(patch_length, corridor_width),
    border_width=2.0,
    num_rows=rows,
    num_cols=1,
    color_scheme="height",
    difficulty_range=(0.0, 1.0),
    sub_terrains={
      "random_spread_boxes_corridor": RandomSpreadBoxesCorridorTerrainCfg(
        proportion=1.0,
        num_terrain_patches=num_terrain_patches,
        min_boxes=min_boxes,
        max_boxes=max_boxes,
        min_box_height=min_box_height,
        max_box_height=max_box_height,
      )
    },
    add_lights=True,
  )


def make_stairs_corridor_cfg(
  *,
  seed: int | None,
  patch_length: float = 4.0,
  corridor_width: float = 3.0,
  num_terrain_patches: int = 8,
  min_step_height: float = 0.05,
  max_step_height: float = 0.20,
) -> TerrainGeneratorCfg:
  rows = num_terrain_patches + 2
  return TerrainGeneratorCfg(
    seed=seed,
    curriculum=True,
    size=(patch_length, corridor_width),
    border_width=2.0,
    num_rows=rows,
    num_cols=1,
    color_scheme="height",
    difficulty_range=(0.0, 1.0),
    sub_terrains={
      "stairs_corridor": StairsCorridorTerrainCfg(
        proportion=1.0,
        num_terrain_patches=num_terrain_patches,
        min_step_height=min_step_height,
        max_step_height=max_step_height,
      )
    },
    add_lights=True,
  )


def make_eval_terrain_cfg(
  eval_terrain: str,
  *,
  seed: int | None,
) -> tuple[TerrainGeneratorCfg, list[dict[str, float]], dict[str, Any]]:
  if eval_terrain == "rough_curriculum_corridor":
    terrain_cfg = make_rough_curriculum_corridor_cfg(seed=seed)
  elif eval_terrain == "perlin_noise_corridor":
    terrain_cfg = make_perlin_noise_corridor_cfg(seed=seed)
  elif eval_terrain == "random_spread_boxes_corridor":
    terrain_cfg = make_random_spread_boxes_corridor_cfg(seed=seed)
  elif eval_terrain == "stairs_corridor":
    terrain_cfg = make_stairs_corridor_cfg(seed=seed)
  else:
    supported = ", ".join(SUPPORTED_EVAL_TERRAINS)
    raise ValueError(
      f"Unsupported --eval-terrain {eval_terrain!r}; supported values: {supported}."
    )
  waypoints, terrain_metadata = make_corridor_path_and_metadata(terrain_cfg)
  return terrain_cfg, waypoints, terrain_metadata


def _patch_kind_and_params(
  corridor: SubTerrainCfg, difficulty: float
) -> tuple[str, dict[str, Any]]:
  if isinstance(corridor, RoughCurriculumCorridorTerrainCfg):
    noise_hi = corridor.min_noise + difficulty * (
      corridor.max_noise - corridor.min_noise
    )
    return "random_rough", {"roughness": {"noise_range": [corridor.min_noise, noise_hi]}}
  if isinstance(corridor, PerlinNoiseCorridorTerrainCfg):
    target_height = corridor.min_height + difficulty * (
      corridor.max_height - corridor.min_height
    )
    return (
      "perlin_noise",
      {
        "perlin_noise": {
          "height_range": [corridor.min_height, target_height],
          "octaves": corridor.octaves,
          "persistence": corridor.persistence,
          "lacunarity": corridor.lacunarity,
          "scale": corridor.scale,
        }
      },
    )
  if isinstance(corridor, RandomSpreadBoxesCorridorTerrainCfg):
    num_boxes = int(
      round(corridor.min_boxes + difficulty * (corridor.max_boxes - corridor.min_boxes))
    )
    height_scale = 0.2 + 0.8 * difficulty
    return (
      "random_spread_boxes",
      {
        "random_spread_boxes": {
          "num_boxes": num_boxes,
          "box_height_range": [
            corridor.min_box_height * height_scale,
            corridor.max_box_height * height_scale,
          ],
          "box_width_range": list(corridor.box_width_range),
          "box_length_range": list(corridor.box_length_range),
        }
      },
    )
  if isinstance(corridor, StairsCorridorTerrainCfg):
    step_height = corridor.min_step_height + difficulty * (
      corridor.max_step_height - corridor.min_step_height
    )
    return (
      "pyramid_stairs",
      {
        "pyramid_stairs": {
          "step_height": step_height,
          "step_width": corridor.step_width,
        }
      },
    )
  raise TypeError(f"Unsupported corridor terrain config: {type(corridor).__name__}")


def make_corridor_path_and_metadata(
  terrain_cfg: TerrainGeneratorCfg,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
  patch_length = float(terrain_cfg.size[0])
  corridor_width = float(terrain_cfg.size[1])
  rows = int(terrain_cfg.num_rows)
  x0 = -rows * patch_length * 0.5
  y_center = 0.0

  waypoints = [
    {"x": x0 + row * patch_length + patch_length * 0.5, "y": y_center, "z": 0.0}
    for row in range(rows)
  ]

  patches = []
  terrain_rows = max(rows - 2, 1)
  terrain_mode = next(iter(terrain_cfg.sub_terrains.keys()))
  corridor = next(iter(terrain_cfg.sub_terrains.values()))
  for row in range(rows):
    start_x = x0 + row * patch_length
    end_x = start_x + patch_length
    if row == 0:
      kind = "flat_spawn"
      difficulty = 0.0
      terrain_params: dict[str, Any] = {}
    elif row == rows - 1:
      kind = "flat_finish"
      difficulty = 1.0
      terrain_params = {}
    else:
      difficulty = (row - 1) / max(terrain_rows - 1, 1)
      kind, terrain_params = _patch_kind_and_params(corridor, float(difficulty))
    patch = {
      "patch_index": row,
      "kind": kind,
      "difficulty_level": float(difficulty),
      "start_position": [float(start_x), -corridor_width * 0.5, 0.0],
      "end_position": [float(end_x), corridor_width * 0.5, 0.0],
    }
    patch.update(terrain_params)
    patches.append(patch)

  total_length = max(0.0, (rows - 1) * patch_length)
  metadata = {
    "terrain_mode": terrain_mode,
    "forward_axis": "x",
    "patch_length": patch_length,
    "corridor_width": corridor_width,
    "num_patches": rows,
    "num_terrain_patches": rows - 2,
    "total_path_length": total_length,
    "patches": patches,
  }
  if terrain_mode == "rough_curriculum_corridor":
    metadata["num_rough_patches"] = rows - 2
  return waypoints, metadata
