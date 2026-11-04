#!/usr/bin/env python3
"""Simulator only: a series of eval_climb.py attempts over the EUR pallet, each in a fresh simulator.

    .venv/bin/python tools/eval_climb_runs.py --policy perceptive --runs 20 --out results/perceptive
    .venv/bin/python tools/eval_climb_runs.py --policy velocity --runs 20 --out results/flat

Per attempt it starts what start_quest_sim.sh starts, without the Quest's stream.py: mediamtx,
r1_ctrl (--start-state Perceptive or Velocity), the simulator and, for the perceptive policy,
lowstate_shm and heightmap_node.py; runs one attempt and stops them all. The start (distance,
offset, angle, speed) is drawn per attempt from --seed, so a series can be repeated. Needs a
desktop (DISPLAY) for the simulator's window, and nothing of the simulator running yet.

Writes <out>/runs.jsonl (eval_climb.py's lines), <out>/runs.csv and the logs of each attempt in
<out>/logs/<n>/, and prints the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PERCEPTION = Path(__file__).resolve().parents[1]
MJLAB = PERCEPTION.parent
MEDIAMTX = Path.home() / "tools/mediamtx/mediamtx"
PYTHON = PERCEPTION / ".venv/bin/python"
# The same patterns as start_quest_sim.sh's leftovers: never an r1_ctrl for the real robot.
LEFTOVERS = (["-f", "r1_ctrl --network lo"], ["-x", "unitree_mujoco"], ["-f", "mediamtx mediamtx_sim.yml"],
             ["-f", "heightmap_node.py"], ["-x", "lowstate_shm"])
RANGES = dict(distance=(1.0, 2.0), offset=(-0.3, 0.3), angle=(-15.0, 15.0), speed=(0.3, 0.6))
OUTCOMES = ("success", "fall", "stuck", "missed", "timeout", "setup", "error")


def running() -> list[str]:
    found = []
    for pattern in LEFTOVERS:
        out = subprocess.run(["pgrep", "-a", *pattern], capture_output=True, text=True).stdout.strip()
        found += out.splitlines() if out else []
    return found


class Stack:
    """The simulator's processes for one attempt, each in its own process group."""

    def __init__(self, logs: Path):
        self.logs, self.procs = logs, []

    def start(self, name: str, cmd: list[str], cwd: Path, ready: str, timeout: float, env=None) -> None:
        log = open(self.logs / f"{name}.log", "w")
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                env={**os.environ, **(env or {})})
        self.procs.append((name, proc))
        t_end = time.time() + timeout
        while time.time() < t_end:
            if ready in (self.logs / f"{name}.log").read_text(errors="replace"):
                return
            if proc.poll() is not None:
                raise RuntimeError(f"{name} ended (exit {proc.returncode}); see {self.logs / name}.log")
            time.sleep(0.2)
        raise RuntimeError(f"{name} not ready after {timeout:.0f} s (no '{ready}' in its log)")

    def wait_log(self, name: str, ready: str, timeout: float) -> None:
        t_end = time.time() + timeout
        while ready not in (self.logs / f"{name}.log").read_text(errors="replace"):
            if time.time() > t_end:
                raise RuntimeError(f"{name}: no '{ready}' after {timeout:.0f} s")
            time.sleep(0.2)

    def stop(self) -> None:
        """Last started first: SIGTERM to each group, SIGKILL after 5 s."""
        for _, proc in reversed(self.procs):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        t_end = time.time() + 5
        for _, proc in reversed(self.procs):
            try:
                proc.wait(max(0.1, t_end - time.time()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        self.procs = []


def attempt(n: int, policy: str, start: dict, stereo: int, out: Path) -> dict:
    logs = out / "logs" / f"{n:02d}"
    logs.mkdir(parents=True, exist_ok=True)
    stack = Stack(logs)
    label = f"{policy}/{n:02d}"
    try:
        stack.start("mediamtx", [str(MEDIAMTX), "mediamtx_sim.yml"], MJLAB / "simulate", "WebRTC] started", 10)
        state = "Perceptive" if policy == "perceptive" else "Velocity"
        stack.start("controller", ["./r1_ctrl", "--network", "lo", "--cmd-source", "sport", "--arm-sdk",
                                   "--start-state", state], MJLAB / "deploy/robots/r1/build", "Waiting for connection", 10)
        stack.start("simulator", ["./unitree_mujoco", "--joystick", "0", "--stereo", str(stereo)],
                    MJLAB / "simulate/build", "head camera] publishing", 30, env={"DISPLAY": os.environ.get("DISPLAY", ":0")})
        stack.wait_log("controller", "walking service", 30)
        if policy == "perceptive":
            stack.start("lowstate_shm", ["cpp/build/lowstate_shm", "--iface", "lo"], PERCEPTION, "r1_lowstate", 10)
            stack.start("heightmap", [str(PYTHON), "-u", "heightmap_node.py", "--iface", "lo"], PERCEPTION,
                        "[heightmap]", 20)
        args = [f"--{k}={v:.3f}" for k, v in start.items()]
        with open(logs / "climb.log", "w") as log:
            subprocess.run([str(PYTHON), "-u", str(PERCEPTION / "tools/eval_climb.py"), *args, "--label", label,
                            "--out", str(out / "runs.jsonl")], stdout=log, stderr=subprocess.STDOUT, timeout=240)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        line = dict(label=label, **start, outcome="error", reason=str(e))
        with open(out / "runs.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
    finally:
        stack.stop()
    for _ in range(50):     # until the ports and shared memory are free again
        if not running():
            break
        time.sleep(0.2)
    lines = (out / "runs.jsonl").read_text().splitlines() if (out / "runs.jsonl").exists() else []
    last = json.loads(lines[-1]) if lines else {}
    if last.get("label") != label:   # eval_climb.py ended without its line
        last = dict(label=label, **start, outcome="error", reason=f"no result; see {logs}/climb.log")
        with open(out / "runs.jsonl", "a") as f:
            f.write(json.dumps(last) + "\n")
    return last


def summary(results: list[dict]) -> str:
    n = len(results)
    counts = Counter(r["outcome"] for r in results)
    climbs = [r for r in results if r["outcome"] not in ("setup", "error")]
    lines = [f"{n} attempts: " + ", ".join(f"{o} {counts[o]}" for o in OUTCOMES if counts[o])]
    if climbs:
        ok = sum(r["outcome"] == "success" for r in climbs)
        lines.append(f"success rate {ok}/{len(climbs)} = {100 * ok / len(climbs):.0f} % (without setup/error)")
        for key, unit in (("peak_rise", "m"), ("max_tilt_deg", "deg"), ("walk_s", "s"), ("latency_ms_median", "ms"),
                          ("pallet_error_cm_median", "cm"), ("maps_per_s", "/s")):
            vals = [r[key] for r in climbs if r.get(key) is not None]
            if vals:
                lines.append(f"  {key:24s} median {np.median(vals):7.2f} {unit}, min {min(vals):7.2f}, max {max(vals):7.2f}")
        refused = Counter()
        for r in climbs:
            refused.update(r.get("refused", {}))
        if refused:
            lines.append(f"  refused walking commands: {dict(refused)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", choices=("perceptive", "velocity"), default="perceptive")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stereo", type=int, default=0, choices=(0, 1))
    ap.add_argument("--out", type=Path, default=Path("eval_climb"))
    ap.add_argument("--target", choices=("pallet", "block"), default="pallet",
                    help="the EUR pallet or the solid block of its outline beside it (eval_sim.TARGETS)")
    args = ap.parse_args()
    os.environ["R1_CLIMB_TARGET"] = args.target    # eval_climb.py, started per attempt, reads it

    if not os.environ.get("DISPLAY"):
        sys.exit("no DISPLAY: the simulator needs a desktop")
    if leftovers := running():
        sys.exit("the simulator's processes are still running; stop them first:\n  " + "\n  ".join(leftovers))
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    starts = [{k: float(rng.uniform(*r)) for k, r in RANGES.items()} for _ in range(args.runs)]

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))   # through finally: stops the stack
    results = []
    for n, start in enumerate(starts):
        t0 = time.time()
        print(f"[runs] {n + 1}/{args.runs} {args.policy}: " + ", ".join(f"{k} {v:+.2f}" for k, v in start.items()),
              flush=True)
        r = attempt(n, args.policy, start, args.stereo, args.out)
        results.append(r)
        print(f"[runs]   -> {r['outcome']}" + (f" ({r['reason']})" if r.get("reason") else "")
              + f", {time.time() - t0:.0f} s", flush=True)

    keys = ["label", "outcome", *RANGES, "edge_x", "path_angle", "walk_s", "peak_rise", "max_tilt_deg", "maps_per_s",
            "latency_ms_median", "latency_ms_p90", "pallet_error_cm_median", "pallet_error_cm_p90", "refused", "reason"]
    with open(args.out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    text = summary(results)
    (args.out / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
