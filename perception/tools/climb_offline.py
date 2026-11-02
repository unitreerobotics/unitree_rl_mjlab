#!/usr/bin/env python3
"""Offline: a policy alone over the EUR pallet, in MuJoCo from Python, no simulator, controller or camera.

    .venv/bin/python tools/climb_offline.py --policy ../deploy/robots/r1/config/policy/perceptive/v2 --runs 20
    .venv/bin/python tools/climb_offline.py --policy .../v2 .../v3 --runs 50 --out results/offline

Tests what the policy can do, faster than real time and without a window: the chain to the robot
(depth camera, height map, DDS) is replaced by the scan computed from the scene, as in training.
The controller is rebuilt from deploy/ as it runs in the simulator: r1_ctrl --start-state
Perceptive (FixStand's gains and pose for the joints the policy does not drive, the head tilted to
0.45 rad), the policy every 20 ms with deploy.yaml's observations, and unitree_mujoco's PD on
every motor at each 2 ms physics step. The scan is the pelvis height minus the ground under the
17 x 11 grid, new every 100 ms and --scan-latency old, as heightmap_node.py delivers it.

The attempts are eval_climb_runs.py's: the same starts from --seed (distance, offset, angle,
speed), the same course and scoring as eval_climb.py, but the robot is put at its start directly
instead of walking there. Writes <out>/runs.jsonl and prints a summary per policy.

--steer heading (eval_climb.py's): hold the heading only, so the robot may drift off its line.
--steer line: also keep to the line as an operator would, with a sideways command against the
offset from it. Together they tell whether a policy cannot climb or only drifts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_climb import FACING, FALL_HEIGHT, FALL_TILT, FAR_Y, GOAL_Y, STAND_S, STUCK_M, STUCK_S, WALK_TIMEOUT_S, tilt_of, wrap  # noqa: E402
from eval_climb_runs import OUTCOMES, RANGES  # noqa: E402
from eval_sim import EDGE_Y, PALLET_CENTRE, PALLET_HALF, SCENE, Truth  # noqa: E402
from r1_perception.mapper import SCAN_OFFSETS  # noqa: E402

DEPLOY = Path(__file__).resolve().parents[2] / "deploy/robots/r1"
POLICY_DT, SCAN_PERIOD = 0.02, 0.1
HEAD_PITCH_MOTOR, HEAD_PITCH = 29, 0.45     # config.yaml: Perceptive's head_pitch
SETTLE_S = 2.0                              # standing at the start before walking
COMMAND_PERIOD = 0.1                        # eval_climb.py sends a command every 100 ms
LINE_GAIN, LINE_MAX_VY = 1.5, 0.3           # --steer line: m/s sideways per m off the line, at most


class Controller:
    """r1_ctrl's Perceptive state for one policy directory (exported/policy.onnx, params/deploy.yaml)."""

    def __init__(self, policy_dir: Path):
        cfg = yaml.safe_load(open(policy_dir / "params/deploy.yaml"))
        self.session = ort.InferenceSession(str(policy_dir / "exported/policy.onnx"), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.ids = np.array(cfg["joint_ids_map"])
        self.kp, self.kd = np.array(cfg["stiffness"], float), np.array(cfg["damping"], float)
        self.default = np.array(cfg["default_joint_pos"], float)
        act = cfg["actions"]["JointPositionAction"]
        self.scale, self.offset = np.array(act["scale"], float), np.array(act["offset"], float)
        self.ranges = cfg["commands"]["base_velocity"]["ranges"]
        self.obs_cfg = cfg["observations"]
        self.period = self.obs_cfg["gait_phase"]["params"]["period"]
        scan = self.obs_cfg["height_scan"]["params"]
        assert (scan["width"], scan["height"], scan["resolution"], scan["origin"]) == (17, 11, 0.1, [-0.8, -0.5])
        self.step_dt = cfg["step_dt"]
        self.reset()

    def reset(self) -> None:
        self.phase, self.action = 0.0, np.zeros(len(self.ids))

    def target(self) -> np.ndarray:
        return self.action * self.scale + self.offset

    def step(self, gyro, quat_wxyz, q, dq, command, scan) -> np.ndarray:
        """One policy step from the motor-order state; returns the policy joints' targets."""
        cmd = np.array([np.clip(command[0], *self.ranges["lin_vel_x"]), np.clip(command[1], *self.ranges["lin_vel_y"]),
                        np.clip(command[2], *self.ranges["ang_vel_z"])])
        self.phase = math.fmod(self.phase + self.step_dt / self.period, 1.0)
        phase = np.array([math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase)])
        if np.linalg.norm(cmd) < 0.1:
            phase[:] = 0.0
        w, x, y, z = quat_wxyz      # gravity (0, 0, -1) into the body frame: R^T g
        gravity = -np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
        terms = dict(base_ang_vel=gyro, projected_gravity=gravity, velocity_commands=cmd, gait_phase=phase,
                     joint_pos_rel=q[self.ids] - self.default, joint_vel_rel=dq[self.ids], last_action=self.action,
                     height_scan=scan)
        obs = np.concatenate([np.asarray(terms[name], float) * np.array(c["scale"], float)
                              for name, c in self.obs_cfg.items()]).astype(np.float32)
        self.action = self.session.run(None, {self.input: obs[None]})[0][0].astype(float)
        return self.target()


class Scene:
    """scene_arena.xml with the robot, stepped as unitree_mujoco does, plus the ground for the scan."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.truth = Truth()
        fix = yaml.safe_load(open(DEPLOY / "config/config.yaml"))["FSM"]["FixStand"]
        self.kp, self.kd = np.array(fix["kp"], float), np.array(fix["kd"], float)
        self.q_hold = np.array(fix["qs"][-1], float)
        self.q_hold[HEAD_PITCH_MOTOR] = HEAD_PITCH
        self.nu = self.m.nu
        assert self.nu == len(self.kp), f"{self.nu} motors, FixStand has {len(self.kp)}"
        sensor = lambda name: self.m.sensor_adr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, name)]
        self.quat_adr, self.gyro_adr = sensor("imu_quat"), sensor("imu_gyro")
        self.ground_cache: dict[tuple[int, int], float] = {}

    def place(self, xy, yaw) -> None:
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:2] = xy
        self.d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        # Joints as FixStand leaves them; free joint's qpos is 7, then the motors' joints in order.
        self.d.qpos[7:7 + self.nu] = self.q_hold
        mujoco.mj_forward(self.m, self.d)

    def state(self):
        s = self.d.sensordata
        return (s[self.gyro_adr:self.gyro_adr + 3].copy(), s[self.quat_adr:self.quat_adr + 4].copy(),
                s[:self.nu].copy(), s[self.nu:2 * self.nu].copy())

    def step(self, q_target, kp, kd) -> None:
        s = self.d.sensordata
        self.d.ctrl[:] = kp * (q_target - s[:self.nu]) + kd * (0.0 - s[self.nu:2 * self.nu])
        mujoco.mj_step(self.m, self.d)

    def pose(self):
        return self.d.qpos[:3].copy(), self.d.qpos[3:7].copy()

    def ground(self, xy) -> float:
        key = (round(xy[0] * 100), round(xy[1] * 100))   # 1 cm cells: the scene is static
        if key not in self.ground_cache:
            self.ground_cache[key] = self.truth.ground(np.array(xy))
        return self.ground_cache[key]

    def scan(self, pelvis, yaw) -> np.ndarray:
        c, s = math.cos(yaw), math.sin(yaw)
        xy = pelvis[:2] + SCAN_OFFSETS @ np.array([[c, s], [-s, c]])
        return np.array([pelvis[2] - self.ground(p) for p in xy])


def yaw_of_wxyz(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def attempt(scene: Scene, ctrl: Controller, start: dict, scan_latency: float, steer: str) -> dict:
    """One attempt as eval_climb.py, in simulated time."""
    alpha = math.radians(start["angle"])
    heading = FACING + alpha
    direction = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-direction[1], direction[0]])
    start_xy = np.array([start["offset"] + start["distance"] * math.tan(alpha), EDGE_Y - start["distance"]])
    scene.place(start_xy, heading)
    ctrl.reset()

    dt = scene.m.opt.timestep
    per_policy = round(ctrl.step_dt / dt)
    kp, kd = scene.kp.copy(), scene.kd.copy()
    kp[ctrl.ids], kd[ctrl.ids] = ctrl.kp, ctrl.kd
    q_target = scene.q_hold.copy()
    q_target[ctrl.ids] = ctrl.target()

    history = deque()       # (t, pelvis, yaw): the scan is taken from the pose scan_latency ago
    scan, next_scan, next_cmd, command = None, 0.0, 0.0, np.zeros(3)
    t, walk_t0 = 0.0, None
    origin, floor_z, progress = None, None, deque()
    peak_rise, max_tilt, beside, edge_x, path_angle = 0.0, 0.0, False, None, None
    outcome, stand_until = None, None

    def fallen(pos, tilt) -> bool:
        return pos[2] - scene.ground(pos[:2]) < FALL_HEIGHT or tilt > FALL_TILT

    while outcome is None:
        pos, quat = scene.pose()
        yaw, tilt = yaw_of_wxyz(quat), tilt_of(quat)
        history.append((t, pos, yaw))
        while len(history) > 1 and history[1][0] <= t - scan_latency:
            history.popleft()
        if t >= next_scan:
            _, p, y = history[0]
            scan = scene.scan(p, y)
            next_scan += SCAN_PERIOD

        if t >= next_cmd:   # eval_climb.py's walk: settle, then speed ahead and hold the heading (and line)
            next_cmd += COMMAND_PERIOD
            if walk_t0 is None and t >= SETTLE_S:
                walk_t0, origin, floor_z = t, pos[:2].copy(), pos[2]
            if walk_t0 is not None:
                max_tilt = max(max_tilt, tilt)
                if stand_until is not None:
                    command = np.zeros(3)
                    if fallen(pos, tilt):
                        outcome = "fall"
                    elif t >= stand_until:
                        outcome = "missed" if beside else "success"
                else:
                    along = float((pos[:2] - origin) @ direction)
                    if edge_x is None and pos[1] >= EDGE_Y:
                        edge_x = float(pos[0])
                        path_angle = math.degrees(math.atan2(-(pos[0] - origin[0]), pos[1] - origin[1]))
                    if EDGE_Y <= pos[1] <= FAR_Y:
                        peak_rise = max(peak_rise, pos[2] - floor_z)
                        beside |= abs(pos[0] - PALLET_CENTRE[0]) > PALLET_HALF[0]
                    progress.append((t, along))
                    while progress and progress[0][0] < t - STUCK_S:
                        progress.popleft()
                    if fallen(pos, tilt):
                        outcome = "fall"
                    elif pos[1] > GOAL_Y:
                        stand_until, walk_s = t + STAND_S, t - walk_t0
                        command = np.zeros(3)
                    elif t - walk_t0 > STUCK_S and along - progress[0][1] < STUCK_M:
                        outcome = "stuck"
                    elif t - walk_t0 > WALK_TIMEOUT_S:
                        outcome = "timeout"
                    else:
                        vy = 0.0
                        if steer == "line":   # the heading is held, so the body's y is the line's left
                            vy = float(np.clip(-LINE_GAIN * (pos[:2] - origin) @ left, -LINE_MAX_VY, LINE_MAX_VY))
                        command = np.array([start["speed"], vy, float(np.clip(2.0 * wrap(heading - yaw), -0.5, 0.5))])
            elif fallen(pos, tilt):
                return dict(**start, outcome="setup", reason="fell while standing at the start")
            if outcome is not None:
                break

        gyro, imu_quat, q, dq = scene.state()
        if round(t / dt) % per_policy == 0:
            q_target[ctrl.ids] = ctrl.step(gyro, imu_quat, q, dq, command, scan)
        scene.step(q_target, kp, kd)
        t += dt

    if stand_until is None:
        walk_s = t - walk_t0
    pos, _ = scene.pose()
    return dict(**start, outcome=outcome, start=np.round(origin, 3).tolist(),
                edge_x=round(edge_x, 3) if edge_x is not None else None,
                path_angle=round(path_angle, 1) if path_angle is not None else None, walk_s=round(walk_s, 2),
                pelvis=np.round(pos, 3).tolist(), peak_rise=round(peak_rise, 3),
                max_tilt_deg=round(math.degrees(max_tilt), 1))


def summary(name: str, results: list[dict]) -> str:
    counts = Counter(r["outcome"] for r in results)
    climbs = [r for r in results if r["outcome"] != "setup"]
    lines = [f"{name}: {len(results)} attempts: " + ", ".join(f"{o} {counts[o]}" for o in OUTCOMES if counts[o])]
    if climbs:
        ok = sum(r["outcome"] == "success" for r in climbs)
        lines.append(f"  success rate {ok}/{len(climbs)} = {100 * ok / len(climbs):.0f} %")
        for key, unit in (("peak_rise", "m"), ("max_tilt_deg", "deg"), ("walk_s", "s")):
            vals = [r[key] for r in climbs if r.get(key) is not None]
            lines.append(f"  {key:14s} median {np.median(vals):6.2f} {unit}, min {min(vals):6.2f}, max {max(vals):6.2f}")
        drift = [abs(r["path_angle"] - r["angle"]) for r in climbs if r.get("path_angle") is not None]
        if drift:
            lines.append(f"  {'drift':14s} median {np.median(drift):6.1f} deg, max {max(drift):6.1f} (path minus commanded angle)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", type=Path, nargs="+", required=True, help="policy directories (exported/, params/)")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="eval_climb_runs.py's: the same starts")
    ap.add_argument("--scan-latency", type=float, default=0.07, help="s; the chain measured 61-70 ms")
    ap.add_argument("--steer", choices=("heading", "line"), default="heading",
                    help="heading: eval_climb.py's; line: also correct sideways like an operator")
    ap.add_argument("--out", type=Path, default=Path("climb_offline"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    starts = [{k: float(rng.uniform(*r)) for k, r in RANGES.items()} for _ in range(args.runs)]
    scene = Scene()
    texts = []
    for policy_dir in args.policy:
        name = policy_dir.resolve().name
        ctrl = Controller(policy_dir)
        results = []
        for n, start in enumerate(starts):
            r = dict(label=f"{name}/{n:02d}", **attempt(scene, ctrl, start, args.scan_latency, args.steer))
            results.append(r)
            with open(args.out / "runs.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"[offline] {r['label']}: " + ", ".join(f"{k} {start[k]:+.2f}" for k in RANGES)
                  + f" -> {r['outcome']}" + (f", rise {r['peak_rise']:.3f} m, path {r['path_angle']} deg"
                                             if r.get("peak_rise") is not None else ""), flush=True)
        texts.append(summary(name, results))
    text = "\n".join(texts)
    (args.out / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
