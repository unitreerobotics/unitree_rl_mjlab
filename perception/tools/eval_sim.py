#!/usr/bin/env python3
"""Simulator only: walks the R1 towards the EUR pallet and scores every rt/heightmap against the truth.

    .venv/bin/python tools/eval_sim.py --out run.npz

Needs the simulator with scene_arena.xml, r1_ctrl (sport service) and heightmap_node.py running.
The course (robot starts at the origin facing +x, pallet 1.2 m to its left): turn left to face it,
step back to 0.8 m behind the start, walk up to the pallet and stop 0.25 m before its edge.

Truth comes from the simulator itself, which only a test may use: the pelvis position it
publishes on rt/sportmodestate (unitree_go type, sim only), the yaw from rt/lowstate's IMU (exact
in sim) and rays into scene_arena.xml without the robot. The pallet counts as a solid block, as
the training terrain is and as the map sees it.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r1_perception.mapper import SCAN_OFFSETS, yaw_of  # noqa: E402
from r1_perception.robot_model import DEFAULT_MJCF  # noqa: E402

SCENE = DEFAULT_MJCF.parent / "scene_arena.xml"
# What the climb tests walk over: the EUR pallet (euro_pallet.xml) or, with R1_CLIMB_TARGET=block, the
# solid block of the same outline next to it (solid_block.xml). The names keep "PALLET" for both.
TARGETS = {"pallet": np.array([0.0, 1.6]), "block": np.array([1.5, 1.6])}
TARGET = os.environ.get("R1_CLIMB_TARGET", "pallet")
PALLET_CENTRE, PALLET_HALF, PALLET_TOP = TARGETS[TARGET], np.array([0.6, 0.4]), 0.144
EDGE_Y = PALLET_CENTRE[1] - PALLET_HALF[1]


class Truth:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        root = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.robot = {g for g in range(self.m.ngeom) if self.m.body_rootid[self.m.geom_bodyid[g]] == root}
        self.groups = np.array([1, 1, 1, 0, 0, 0], np.uint8)   # what the depth camera sees

    def ground(self, xy) -> float:
        if np.all(np.abs(xy - PALLET_CENTRE) <= PALLET_HALF):
            return PALLET_TOP
        # From 1.5 m, not higher: the KUKA's arm overhangs the block's starts at about 2.4 m, which
        # a ray from above took for the ground (and the robot for fallen).
        start, geom = np.array([xy[0], xy[1], 1.5]), np.zeros(1, np.int32)
        while True:
            dist = mujoco.mj_ray(self.m, self.d, start, np.array([0, 0, -1.0]), self.groups, 1, -1, geom)
            if dist < 0:
                return 0.0
            hit = start[2] - dist
            if geom[0] not in self.robot:   # the robot in the scene stands at its start pose
                return hit
            start = np.array([xy[0], xy[1], hit - 1e-4])

    def scan(self, pelvis, yaw) -> np.ndarray:
        c, s = math.cos(yaw), math.sin(yaw)
        xy = pelvis[:2] + SCAN_OFFSETS @ np.array([[c, s], [-s, c]])
        return np.array([pelvis[2] - self.ground(p) for p in xy]), xy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--out", default="eval_sim.npz")
    args = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_, SportModeState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient

    truth = Truth()
    ChannelFactoryInitialize(0, args.iface)
    state = {"pos": None, "yaw": None, "tick": None}
    phase = ["stand"]
    maps: list = []
    lock = threading.Lock()

    def on_truth(msg):
        with lock:
            state["pos"] = np.array(msg.position, dtype=float)

    def on_low(msg):
        with lock:
            state["yaw"] = yaw_of(msg.imu_state.quaternion)
            state["tick"] = msg.tick

    def on_map(msg):
        with lock:
            if state["pos"] is not None and state["yaw"] is not None:
                maps.append((time.time(), np.array(msg.data), state["pos"].copy(), state["yaw"], phase[0]))

    subs = [ChannelSubscriber("rt/sportmodestate", SportModeState_), ChannelSubscriber("rt/lowstate", LowState_),
            ChannelSubscriber("rt/heightmap", HeightMap_)]
    for sub, cb in zip(subs, (on_truth, on_low, on_map)):
        sub.Init(cb, 10)
    loco = LocoClient()
    loco.SetTimeout(0.5)
    loco.Init()

    def drive(name, vx, w, until, timeout):
        phase[0] = name
        t_end = time.time() + timeout
        while time.time() < t_end:
            with lock:
                pos, yaw = state["pos"], state["yaw"]
            if pos is not None and until(pos, yaw):
                break
            loco.SetVelocity(vx, 0.0, w, 0.5)
            time.sleep(0.1)
        loco.SetVelocity(0.0, 0.0, 0.0, 0.5)

    # The simulator needs about half a minute to reach real time (model, head camera, ffmpeg).
    print("waiting for the simulator to run in real time and for height maps ...", flush=True)
    t_wait = time.time() + 90
    while time.time() < t_wait:
        with lock:
            tick0 = state["tick"]
        time.sleep(2.0)
        with lock:
            rtf = (state["tick"] - tick0) / 2000.0 if tick0 is not None and state["tick"] is not None else 0.0
            n_maps = len(maps)
        if rtf > 0.9 and n_maps > 0:
            break
    else:
        sys.exit("the simulator did not reach real time or no height maps arrived")
    with lock:
        maps.clear()
    print(f"running in real time ({rtf:.2f}); driving the course", flush=True)
    time.sleep(3.0)
    drive("turn", 0.0, 1.0, lambda p, y: y > math.radians(85), 15)
    drive("back", -0.5, 0.0, lambda p, y: p[1] < -0.8, 15)
    drive("stand", 0.0, 0.0, lambda p, y: False, 2)
    drive("approach", 0.3, 0.0, lambda p, y: p[1] > EDGE_Y - 0.35, 20)
    drive("stand", 0.0, 0.0, lambda p, y: False, 3)

    with lock:
        recorded = list(maps)
    rows = []
    for _, scan, pos, yaw, ph in recorded:
        true, xy = truth.scan(pos, yaw)
        pallet = np.all(np.abs(xy - PALLET_CENTRE) <= PALLET_HALF, axis=1)
        edge = np.abs(xy[:, 1] - EDGE_Y) < 0.05
        rows.append((ph, np.abs(scan - true), pallet & ~edge, pos, scan, true))
    print(f"{len(rows)} maps scored")
    for ph in ("turn", "back", "stand", "approach"):
        errs = [r[1] for r in rows if r[0] == ph]
        if errs:
            e = np.concatenate(errs)
            print(f"  {ph:9s}: {len(errs):3d} maps, error median {np.median(e) * 100:4.1f} cm, 90% {np.percentile(e, 90) * 100:4.1f} cm")
    last = rows[-1]
    print(f"  final: pelvis {np.round(last[3], 2).tolist()}, pallet points in the scan {int(last[2].sum())}, "
          f"their error median {np.median(last[1][last[2]]) * 100 if last[2].any() else float('nan'):.1f} cm, "
          f"max {last[1][last[2]].max() * 100 if last[2].any() else float('nan'):.1f} cm")
    np.savez(args.out, phase=[r[0] for r in rows], err=np.array([r[1] for r in rows]),
             pallet=np.array([r[2] for r in rows]), pos=np.array([r[3] for r in rows]),
             scan=np.array([r[4] for r in rows]), true=np.array([r[5] for r in rows]))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
