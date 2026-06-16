"""Project-local terrain primitives for velocity tasks."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains import SubTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane


@dataclass(kw_only=True)
class BoxHighPlatformTerrainCfg(SubTerrainCfg):
  """Flat terrain with one centered high platform."""

  platform_height_range: tuple[float, float]
  """Min and max platform height, in meters. Interpolated by difficulty."""
  platform_width: float = 2.5
  """Side length of the centered platform top, in meters."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng  # Unused.
    body = spec.body("terrain")
    platform_height = self.platform_height_range[0] + difficulty * (
      self.platform_height_range[1] - self.platform_height_range[0]
    )

    plane = make_plane(body, self.size, 0.0, center_zero=False)[0]
    center = (self.size[0] / 2.0, self.size[1] / 2.0)
    platform = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.platform_width / 2.0, self.platform_width / 2.0, platform_height / 2.0),
      pos=(center[0], center[1], platform_height / 2.0),
    )

    origin = np.array([center[0], center[1], platform_height])
    return TerrainOutput(
      origin=origin,
      geometries=[
        TerrainGeometry(geom=plane, color=(0.45, 0.45, 0.45, 1.0)),
        TerrainGeometry(geom=platform, color=(0.22, 0.52, 0.85, 1.0)),
      ],
    )
