"""EUR pallets (EPAL 1, 1200 x 800 x 144 mm) as a training terrain, built like the test pallet.

The box steps of the other terrains have closed risers; a real pallet is open in front: between
its blocks the space under the top board is free from 22 mm (bottom board) to 100 mm (blocks),
under the stringers to 122 mm, and its top has 41 mm gaps between the boards. In the offline
tests every fall of the perceptive policies v2 and v3 began with a toe under a top board
(perception/tools: the pallet of src/assets/robots/unitree_r1/xmls/euro_pallet.xml). This
terrain puts those pallets in training.

Layout: a closed ring of 12 pallets around the spawn point, laid as a pinwheel so that a robot
walking 3 m in any direction (the curriculum's move_up_distance) has to cross one: up its near
edge, over the top, down its far edge. Each side of the ring has two pallets with the long side
and one with the short side towards the centre, in random order; the ring is turned by a random
angle. Difficulty shortens the run-up from 1.4 m to 1.0 m.

Scan cover: the deployed height map (heightmap_node.py) shows a pallet's top as closed (the
board gaps are narrower than its cells; sim: 0.3 cm median error against a closed top), while
the training scan's rays, 0.1 m apart, would fall into gaps. Each pallet therefore gets a thin
box on its top that rays hit (group 0) and nothing collides with (contype = conaffinity = 0).

Collisions: the pallets collide only with geoms carrying contype/conaffinity bit 2 (the R1's
legs in the perceptive task); see PALLET_CONTYPE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane

PALLET_LENGTH, PALLET_WIDTH, PALLET_HEIGHT = 1.2, 0.8, 0.144
WOOD = (0.78, 0.62, 0.42, 1.0)

# Collision bits: pallets collide only with geoms that have bit 2, the R1's legs in the
# perceptive task (env_cfgs.py). Every terrain geom is checked against every colliding robot
# geom in each world; with all 31 R1 geoms the pallets pushed mujoco_warp past 250k pairs, off
# its direct (NXN) broad phase onto sweep-and-prune: 23x slower. Trunk and arms reach a pallet
# only in a fall, which ends the episode anyway.
PALLET_CONTYPE = PALLET_CONAFFINITY = 2

# (centre x, y, z, half sizes) in the pallet's frame: x along the 1200 mm boards, y along the
# 800 mm side, z up from the floor. euro_pallet.xml's pallet with fewer boxes, the same from
# outside: the blocks reach up to the top boards and take the stringers' place (from the long
# side the openings between them are the same; from the short side the openings under the outer
# stringer reach 122 instead of 100 mm), and the centre block and middle bottom board, which no
# foot can reach, are left out. 15 boxes instead of 20.
_BLOCK_Z = (0.022, 0.122)          # on the bottom boards, up to the top boards
_PALLET_BOXES: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
  # top deck: 145 / 100 / 145 / 100 / 145 mm boards, 41.25 mm gaps
  ((0.0, -0.3275, 0.133), (0.6, 0.0725, 0.011)),
  ((0.0, -0.16375, 0.133), (0.6, 0.05, 0.011)),
  ((0.0, 0.0, 0.133), (0.6, 0.0725, 0.011)),
  ((0.0, 0.16375, 0.133), (0.6, 0.05, 0.011)),
  ((0.0, 0.3275, 0.133), (0.6, 0.0725, 0.011)),
  # blocks of the front and back rows, on the bottom boards
  *(((x, y, sum(_BLOCK_Z) / 2), (0.0725, 0.05, (_BLOCK_Z[1] - _BLOCK_Z[0]) / 2))
    for x in (-0.5275, 0.0, 0.5275) for y in (-0.35, 0.35)),
  # the middle row's outer blocks, on the floor (no middle bottom board)
  *(((x, 0.0, _BLOCK_Z[1] / 2), (0.0725, 0.0725, _BLOCK_Z[1] / 2)) for x in (-0.5275, 0.5275)),
  # bottom boards along the long sides: the 22 mm lip in front of the opening
  ((0.0, -0.35, 0.011), (0.6, 0.05, 0.011)),
  ((0.0, 0.35, 0.011), (0.6, 0.05, 0.011)),
)


def add_euro_pallet(body: mujoco.MjsBody, centre_xy: tuple[float, float], yaw: float,
                    scan_cover: bool = True) -> list[mujoco.MjsGeom]:
  """One pallet standing on z = 0, its centre at centre_xy, its 1200 mm side turned by yaw."""
  c, s = math.cos(yaw), math.sin(yaw)
  quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
  geoms = []
  for (x, y, z), half in _PALLET_BOXES:
    pos = (centre_xy[0] + c * x - s * y, centre_xy[1] + s * x + c * y, z)
    geoms.append(body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos, quat=quat, rgba=WOOD,
                               contype=PALLET_CONTYPE, conaffinity=PALLET_CONAFFINITY))
  if scan_cover:
    geoms.append(body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(PALLET_LENGTH / 2, PALLET_WIDTH / 2, 0.0005),
      pos=(centre_xy[0], centre_xy[1], PALLET_HEIGHT - 0.0005),
      # Nearly, not fully transparent: mj_ray and mujoco_warp's rays skip geoms with alpha 0.
      quat=quat, contype=0, conaffinity=0, group=0, rgba=(WOOD[0], WOOD[1], WOOD[2], 0.02),
    ))
  return geoms


@dataclass(kw_only=True)
class EuroPalletRingTerrainCfg(SubTerrainCfg):
  """A ring of EUR pallets around the spawn point (see the module docstring)."""

  run_up_range: tuple[float, float] = (1.4, 1.0)
  """Distance from the spawn point to the ring's inner edge at difficulty 0 and 1, in metres."""
  scan_cover: bool = True
  """A closed top for the rays, as the deployed height map has; no collision."""

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    body = spec.body("terrain")
    centre = np.array([self.size[0] / 2, self.size[1] / 2])
    geoms = list(make_plane(body, self.size, 0.0, center_zero=False))

    a = self.run_up_range[0] + difficulty * (self.run_up_range[1] - self.run_up_range[0])
    turn = rng.uniform(0.0, math.pi / 2)
    for side in range(4):
      phi = turn + side * math.pi / 2
      c, s = math.cos(phi), math.sin(phi)
      # In the side's frame the ring's edge runs along +t at r = -a (in front of the robot
      # when it faces -r); this side covers t from -a, and its far end the next corner.
      kinds = ["long", "long", "short"]
      rng.shuffle(kinds)
      t = -a
      for kind in kinds:
        along, depth = (PALLET_LENGTH, PALLET_WIDTH) if kind == "long" else (PALLET_WIDTH, PALLET_LENGTH)
        t_c, r_c = t + along / 2, -(a + depth / 2)
        t += along
        xy = centre + np.array([c * t_c - s * r_c, s * t_c + c * r_c])
        # The pallet's 1200 mm side along t (long side towards the centre) or along r.
        yaw = phi + (0.0 if kind == "long" else math.pi / 2)
        geoms += add_euro_pallet(body, (float(xy[0]), float(xy[1])), yaw, self.scan_cover)

    return TerrainOutput(
      origin=np.array([centre[0], centre[1], 0.0]),
      geometries=[TerrainGeometry(geom=g) for g in geoms],
    )
