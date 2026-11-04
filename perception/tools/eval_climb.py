#!/usr/bin/env python3
"""Simulator only: one attempt to walk the R1 over the EUR pallet, scored from the simulator's truth.

    .venv/bin/python tools/eval_climb.py --distance 1.5 --offset 0.1 --angle 5 --speed 0.4 --out runs.jsonl

Needs the simulator with scene_arena.xml and r1_ctrl (sport service) running, for the perceptive
policy also lowstate_shm and heightmap_node.py; tools/eval_climb_runs.py starts them for a series.

The robot starts at the origin facing +x, the pallet (1.2 m wide in x, 0.8 m deep in y) 1.2 m to
its left. It turns to face the pallet and walks to its start: distance before the near edge, placed
so that its straight path, turned by angle, crosses the near edge at x = offset. From there it
walks at speed over the pallet, holding its heading as an operator would, and stops once its pelvis
is 0.4 m past the far edge. One JSON line per attempt goes to --out:

    success   past the far edge and standing 2 s later
    fall      pelvis less than 0.45 m above the ground under it, or tilted more than 45 deg
    stuck     less than 5 cm progress along the path in 5 s
    missed    walked beside the pallet instead of over it
    timeout   30 s of walking without one of the above
    setup     did not reach its start (the attempt says nothing about the climb)

Truth comes from the simulator, which only a test may use: rt/sportmodestate's pelvis position
(sim only), rt/lowstate's IMU (exact in sim), rays into scene_arena.xml (eval_sim.Truth).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_sim import EDGE_Y, PALLET_CENTRE, PALLET_HALF, Truth, yaw_of  # noqa: E402

FAR_Y = PALLET_CENTRE[1] + PALLET_HALF[1]
GOAL_Y = FAR_Y + 0.4
FACING = math.pi / 2        # facing the pallet: +y
FALL_HEIGHT, FALL_TILT = 0.45, math.radians(45)
STUCK_S, STUCK_M, WALK_TIMEOUT_S, STAND_S = 5.0, 0.05, 30.0, 2.0


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def tilt_of(quat_wxyz) -> float:
    """Angle between the pelvis' z axis and the world's."""
    _, x, y, _ = quat_wxyz
    return math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--distance", type=float, default=1.5, help="start before the near edge (m, along y)")
    ap.add_argument("--offset", type=float, default=0.0, help="x where the path crosses the near edge (m)")
    ap.add_argument("--angle", type=float, default=0.0, help="path turned from straight on (deg, + to the left)")
    ap.add_argument("--speed", type=float, default=0.4, help="walking speed (m/s)")
    ap.add_argument("--label", default="", help="kept in the result, e.g. the policy and run number")
    ap.add_argument("--out", default="eval_climb.jsonl", help="appends one JSON line")
    args = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_, SportModeState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient

    truth = Truth()
    ChannelFactoryInitialize(0, args.iface)
    state = {"pos": None, "quat": None, "tick": None, "maps": 0}
    maps: list = []   # (arrival, stamp, data, pelvis, yaw) while walking
    lock = threading.Lock()
    recording = threading.Event()

    def on_map(msg):
        with lock:
            state["maps"] += 1
            if recording.is_set() and state["pos"] is not None and state["quat"] is not None:
                maps.append((time.time(), msg.stamp, np.array(msg.data), state["pos"].copy(), yaw_of(state["quat"])))

    # rt/sportmodestate and rt/lowstate come 1000 times a second: a callback for each would keep
    # Python busy decoding (holding the GIL), late for the height maps and the walking commands.
    # Their readers keep only the latest sample, which this thread takes 50 times a second.
    truth_sub, low_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_), ChannelSubscriber("rt/lowstate", LowState_)
    truth_sub.Init()
    low_sub.Init()
    ChannelSubscriber("rt/heightmap", HeightMap_).Init(on_map, 10)

    def poll():
        while True:
            truth_msg, low_msg = truth_sub.Read(0.02), low_sub.Read(0.02)
            with lock:
                if truth_msg is not None:
                    state["pos"] = np.array(truth_msg.position, dtype=float)
                if low_msg is not None:
                    state["quat"] = tuple(low_msg.imu_state.quaternion)
                    state["tick"] = low_msg.tick
            time.sleep(0.02)

    threading.Thread(target=poll, daemon=True).start()
    loco = LocoClient()
    loco.SetTimeout(0.5)
    loco.Init()
    codes: Counter = Counter()

    def pose():
        with lock:
            pos, quat = state["pos"], state["quat"]
        return (None, None, None) if pos is None or quat is None else (pos, yaw_of(quat), tilt_of(quat))

    def command(vx, vy, w):
        code = loco.SetVelocity(float(vx), float(vy), float(w), 0.5)
        if code != 0:
            codes[str(code)] += 1

    def fallen(pos, tilt) -> bool:
        return pos[2] - truth.ground(pos[:2]) < FALL_HEIGHT or tilt > FALL_TILT

    result = dict(label=args.label, distance=args.distance, offset=args.offset, angle=args.angle, speed=args.speed)

    def finish(outcome, **extra):
        command(0.0, 0.0, 0.0)
        result.update(outcome=outcome, refused=dict(codes), **extra)
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"[climb] {outcome}: {json.dumps(result)}", flush=True)
        sys.exit(0)

    # The simulator needs about half a minute to reach real time (model, head camera, ffmpeg).
    print("[climb] waiting for the simulator to run in real time ...", flush=True)
    t_wait = time.time() + 90
    while True:
        with lock:
            tick0 = state["tick"]
        time.sleep(2.0)
        with lock:
            rtf = (state["tick"] - tick0) / 2000.0 if tick0 is not None and state["tick"] is not None else 0.0
        if rtf > 0.9 and pose()[0] is not None:
            break
        if time.time() > t_wait:
            finish("setup", reason="the simulator did not reach real time")
    time.sleep(3.0)
    with lock:
        maps_before = state["maps"]
    floor_z = pose()[0][2]
    print(f"[climb] real time ({rtf:.2f}), pelvis {floor_z:.3f} m; {maps_before} height maps so far", flush=True)

    # ---- to the start ---------------------------------------------------------------------------
    alpha = math.radians(args.angle)
    heading = FACING + alpha
    direction = np.array([math.cos(heading), math.sin(heading)])
    start = np.array([PALLET_CENTRE[0] + args.offset + args.distance * math.tan(alpha), EDGE_Y - args.distance])

    def go_to(target_xy, target_yaw, timeout, tol_m=0.10, tol_yaw=math.radians(3)) -> bool:
        t_end, settled = time.time() + timeout, 0
        while time.time() < t_end:
            pos, yaw, tilt = pose()
            if fallen(pos, tilt):
                finish("setup", reason="fell while getting to the start")
            err_yaw = wrap(target_yaw - yaw)
            w = float(np.clip(2.0 * err_yaw, -0.8, 0.8))
            vx = vy = 0.0
            if target_xy is not None and abs(err_yaw) < math.radians(20):
                e = target_xy - pos[:2]
                c, s = math.cos(yaw), math.sin(yaw)
                v = 1.5 * np.array([c * e[0] + s * e[1], -s * e[0] + c * e[1]])
                speed = np.linalg.norm(v)   # at least 0.15 m/s: the policy does not take slower steps
                vx, vy = v * np.clip(speed, 0.15, 0.4) / speed if speed > 1e-6 else v
            done = abs(err_yaw) < tol_yaw and (target_xy is None or np.linalg.norm(target_xy - pos[:2]) < tol_m)
            settled = settled + 1 if done else 0
            if settled >= 5:
                command(0.0, 0.0, 0.0)
                return True
            command(vx, vy, w)
            time.sleep(0.1)
        return False

    print(f"[climb] to the start {np.round(start, 2).tolist()}, then {args.angle:+.0f} deg at {args.speed} m/s", flush=True)
    if not (go_to(None, FACING, 15) and go_to(start, FACING, 25) and go_to(None, heading, 10)):
        finish("setup", reason="did not reach the start", pelvis=np.round(pose()[0], 3).tolist())
    time.sleep(1.0)

    # ---- over the pallet ------------------------------------------------------------------------
    recording.set()
    t0 = time.time()
    origin = pose()[0][:2].copy()
    progress = deque()          # (t, metres along the path)
    peak_rise, max_tilt, beside, edge_x, path_angle = 0.0, 0.0, False, None, None
    outcome = None
    while outcome is None:
        pos, yaw, tilt = pose()
        now = time.time()
        along = float((pos[:2] - origin) @ direction)
        max_tilt = max(max_tilt, tilt)
        if edge_x is None and pos[1] >= EDGE_Y:   # where the pelvis crossed the near edge, and its path's angle
            edge_x = float(pos[0])
            path_angle = math.degrees(math.atan2(-(pos[0] - origin[0]), pos[1] - origin[1]))
        if EDGE_Y <= pos[1] <= FAR_Y:
            peak_rise = max(peak_rise, pos[2] - floor_z)
            beside |= abs(pos[0] - PALLET_CENTRE[0]) > PALLET_HALF[0]
        progress.append((now, along))
        while progress and progress[0][0] < now - STUCK_S:
            progress.popleft()
        if fallen(pos, tilt):
            outcome = "fall"
        elif pos[1] > GOAL_Y:
            outcome = "past"
        elif now - t0 > STUCK_S and along - progress[0][1] < STUCK_M:
            outcome = "stuck"
        elif now - t0 > WALK_TIMEOUT_S:
            outcome = "timeout"
        else:
            command(args.speed, 0.0, float(np.clip(2.0 * wrap(heading - yaw), -0.5, 0.5)))
            time.sleep(0.1)
    walk_s = time.time() - t0
    if outcome == "past":
        command(0.0, 0.0, 0.0)
        t_end = time.time() + STAND_S
        while time.time() < t_end and outcome == "past":
            pos, _, tilt = pose()
            max_tilt = max(max_tilt, tilt)
            if fallen(pos, tilt):
                outcome = "fall"
            command(0.0, 0.0, 0.0)
            time.sleep(0.1)
        if outcome == "past":
            outcome = "missed" if beside else "success"
    recording.clear()

    # ---- the height maps on the way -------------------------------------------------------------
    with lock:
        recorded = list(maps)
    latency = [arrived - stamp for arrived, stamp, *_ in recorded]
    near = []   # scan points within 0.1 m of the pallet's outline (its edges and top)
    for _, _, scan, pos, yaw in recorded:
        true, xy = truth.scan(pos, yaw)
        close = np.all(np.abs(xy - PALLET_CENTRE) <= PALLET_HALF + 0.1, axis=1)
        near.append(np.abs(scan - true)[close])
    near = np.concatenate(near) if near else np.array([])
    pos = pose()[0]
    finish(outcome, start=np.round(origin, 3).tolist(), edge_x=round(edge_x, 3) if edge_x is not None else None,
           path_angle=round(path_angle, 1) if path_angle is not None else None, walk_s=round(walk_s, 2),
           pelvis=np.round(pos, 3).tolist(), peak_rise=round(peak_rise, 3),
           max_tilt_deg=round(math.degrees(max_tilt), 1), maps=len(recorded),
           maps_per_s=round(len(recorded) / walk_s, 1) if walk_s > 0 else 0.0,
           latency_ms_median=round(1000 * float(np.median(latency)), 1) if latency else None,
           latency_ms_p90=round(1000 * float(np.percentile(latency, 90)), 1) if latency else None,
           pallet_error_cm_median=round(100 * float(np.median(near)), 1) if near.size else None,
           pallet_error_cm_p90=round(100 * float(np.percentile(near, 90)), 1) if near.size else None)


if __name__ == "__main__":
    main()
