#!/usr/bin/env python3
"""The R1's height map for the perceptive policy: depth frames and rt/lowstate in, rt/heightmap out.

    cpp/build/lowstate_shm &                            # rt/lowstate -> /dev/shm/r1_lowstate (first)
    .venv/bin/python heightmap_node.py                 # simulator: depth from /dev/shm/r1_depth, DDS on lo
    .venv/bin/python heightmap_node.py --iface eth0    # (robot: needs a depth source for its camera module)

rt/lowstate comes through cpp/lowstate_shm's ring buffer, every sample into the odometry: reading
it with Python's DDS library starved the mapping (the library decodes all of the simulator's 1000
messages a second while holding the GIL). Only rt/heightmap goes out through Python's DDS.

Each depth frame (10 Hz) goes into the map and a scan goes out on rt/heightmap (unitree_go
HeightMap_): 17 x 11 values, 0.1 m apart, origin (-0.8, -0.5) m, frame "pelvis_yaw", data rows
along y with x within a row, each the pelvis height minus the ground height in metres; stamp is
the depth frame's capture time. The layout is r1_perception/mapper.py's, Unitree-R1-Perceptive's
training scan. Ctrl+C ends it.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time

import numpy as np

from r1_perception.depth_source import ShmDepthSource
from r1_perception.lowstate_source import LowstateShmSource
from r1_perception.mapper import SCAN_X, SCAN_Y, HeightMapper

TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HEIGHTMAP = "rt/heightmap"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iface", default="lo", help="DDS network interface (lo: the simulator)")
    ap.add_argument("--shm", default="/r1_depth", help="the simulator's depth shared memory")
    ap.add_argument("--lowstate-shm", default="/r1_lowstate", help="cpp/lowstate_shm's ring buffer")
    ap.add_argument("--timing-log", default=None,
                    help="CSV with the timing of every published map (see the [heightmap] timing lines)")
    args = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_

    mapper = HeightMapper()
    counts = {"lowstate": 0, "depth": 0, "unmatched": 0}
    lowstate = LowstateShmSource(args.lowstate_shm)

    ChannelFactoryInitialize(0, args.iface)
    pub = ChannelPublisher(TOPIC_HEIGHTMAP, HeightMap_)
    pub.Init()
    source = ShmDepthSource(args.shm)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    print(f"[heightmap] {TOPIC_LOWSTATE} from /dev/shm{args.lowstate_shm}, depth from /dev/shm{args.shm}, "
          f"publishing {TOPIC_HEIGHTMAP} on {args.iface}", flush=True)
    # Timing per published map, in ms: where the time from the depth frame's capture to rt/heightmap
    # goes. wait: capture until this loop picked the frame up (the simulator writing it, and this
    # loop busy with rt/lowstate or the previous frame); lowstate: this loop's rt/lowstate work just
    # before; depth: mapper.on_depth; scan; pub: the DDS write; cpu: this process's CPU time for
    # depth + scan + pub. depth + scan + pub well above cpu means the process waited for a CPU core
    # (other processes), close to it means the work itself is slow.
    timing_log = open(args.timing_log, "w") if args.timing_log else None
    if timing_log:
        timing_log.write("stamp,total_ms,wait_ms,lowstate_ms,depth_ms,scan_ms,pub_ms,cpu_ms,skipped\n")
    timing: list[tuple] = []
    report_at, latency = time.time() + 2.0, []
    skipped_at_report = 0
    while not stop.is_set():
        t_low = time.perf_counter()
        for row in lowstate.read_new():
            if mapper.on_lowstate(row["q"], row["quat"], row["tick"] / 1000.0, row["wall"]):
                counts["lowstate"] += 1
        low_ms = (time.perf_counter() - t_low) * 1000
        frame = source.read()
        if frame is None:
            time.sleep(0.005)
        else:
            depth, k, stamp = frame
            picked, cpu0, t0 = time.time(), time.process_time(), time.perf_counter()
            cells = mapper.on_depth(depth, k, stamp)
            t1 = time.perf_counter()
            scan = mapper.scan() if cells else None
            t2 = time.perf_counter()
            if scan is None:
                counts["unmatched"] += 1
            else:
                pub.Write(HeightMap_(stamp=stamp, frame_id="pelvis_yaw", resolution=0.1,
                                     width=len(SCAN_X), height=len(SCAN_Y),
                                     origin=[float(SCAN_X[0]), float(SCAN_Y[0])],
                                     data=[float(v) for v in scan]))
                t3 = time.perf_counter()
                counts["depth"] += 1
                total = time.time() - stamp
                latency.append(total)
                row_t = (total * 1000, (picked - stamp) * 1000, low_ms, (t1 - t0) * 1000, (t2 - t1) * 1000,
                         (t3 - t2) * 1000, (time.process_time() - cpu0) * 1000)
                timing.append(row_t)
                if timing_log:
                    timing_log.write(f"{stamp:.3f}," + ",".join(f"{v:.1f}" for v in row_t) + f",{source.skipped}\n")
        if time.time() >= report_at:
            pos, fixes = mapper.odometry.position.copy(), mapper.height_fixes[-20:]
            print(f"[heightmap] maps {counts['depth']}, lowstate {counts['lowstate']} (lost {lowstate.lost}), "
                  f"no pose for {counts['unmatched']}; latency {np.median(latency) * 1000 if latency else float('nan'):.0f} ms; "
                  f"odometry {np.round(pos, 3).tolist()}; last height fixes {np.round(np.array(fixes) * 100, 1).tolist()} cm",
                  flush=True)
            if timing:
                t = np.array(timing)
                med, p90, mx = np.median(t, axis=0), np.percentile(t, 90, axis=0), t.max(axis=0)
                names = ("total", "wait", "lowstate", "depth", "scan", "pub", "cpu")
                print("[heightmap] timing ms (median/p90/max): " + ", ".join(
                    f"{n} {a:.0f}/{b:.0f}/{c:.0f}" for n, a, b, c in zip(names, med, p90, mx))
                    + f"; frames skipped {source.skipped - skipped_at_report}", flush=True)
                if timing_log:
                    timing_log.flush()
            skipped_at_report = source.skipped
            report_at, latency, timing = time.time() + 2.0, [], []
    if timing_log:
        timing_log.close()
    print("[heightmap] stopped", flush=True)


if __name__ == "__main__":
    main()
