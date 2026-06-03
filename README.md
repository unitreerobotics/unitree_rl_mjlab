# Unitree Go2 RL Mjlab

## Overview

Unitree Go2 RL Mjlab is a reinforcement learning project built upon [mjlab](https://github.com/mujocolab/mjlab.git), using MuJoCo as its physics simulation backend for the Unitree Go2 quadruped robot.

Mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s proven API with best-in-class [MuJoCo](https://github.com/google-deepmind/mujoco_warp) physics to provide lightweight, modular abstractions for RL robotics research and sim-to-real deployment.

<div align="center">

| <div align="center"> MuJoCo </div> | <div align="center"> Physical </div> |
|---|---|
| <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/go2-velocity.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> | <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/go2-velocity-real.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> |

</div>

## Installation

Please refer to [setup.md](doc/setup_en.md) for installation and configuration steps.

## Workflow

The basic RL motion control pipeline is:

`Train` → `Play` → `Sim2Real`

- **Train**: The agent interacts with the MuJoCo simulation and optimizes policies through reward maximization.
- **Play**: Replay trained policies to verify expected behavior (with optional video recording).
- **Sim2Real**: Deploy trained policies to physical Unitree Go2 for real-world execution.

## Training

### Velocity Tracking

Train a velocity tracking policy:

```bash
python scripts/train.py Unitree-Go2-Flat --env.scene.num-envs=4096
```

Multi-GPU training with `scripts/run.sh`:

```bash
./scripts/run.sh Unitree-Go2-Flat --num_gpus 2
./scripts/run.sh Unitree-Go2-Flat --num_gpus 1
```

### Resume Training

Resume from a previous run using the experiment/run directory:

```bash
./scripts/run.sh Unitree-Go2-Flat --resume logs/rsl_rl/go2_velocity/2026-04-22_18-54-05
```

### Parameters

| Flag | Description |
|---|---|
| `--env.scene` | Simulation scene config (num_envs, dt, ground type, gravity, disturbances) |
| `--env.observations` | Observation space (joint state, IMU, commands, etc.) |
| `--env.rewards` | Reward terms for policy optimization |
| `--env.commands` | Task commands (velocity, pose, or motion targets) |
| `--env.terminations` | Episode termination conditions |
| `--agent.seed` | Random seed for reproducibility |
| `--agent.policy` | Policy network architecture |
| `--agent.algorithm` | RL algorithm config (PPO, hyperparameters, etc.) |

**Training results are stored at**: `logs/rsl_rl/go2_velocity/<date_time>/model_<iteration>.pt`

## Play & Video Recording

### Simulation Validation

```bash
python scripts/play.py Unitree-Go2-Flat --checkpoint_file=logs/rsl_rl/go2_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

### Video Recording

Record playback as MP4 using `scripts/play.sh`:

```bash
./scripts/play.sh Unitree-Go2-Flat --checkpoint <path> --video
./scripts/play.sh Unitree-Go2-Flat --checkpoint <path> --video --video-length 400 --video-width 800 --video-height 600
./scripts/play.sh Unitree-Go2-Flat --checkpoint <path> --video-attribution --attribution-method gradient_saliency
```

Videos are saved under `<checkpoint_dir>/videos/play/rl-video-step-0.mp4`. Attribution videos are saved as `<checkpoint_dir>/videos/play/rl-video-attribution-step-0.mp4`. On headless machines, `MUJOCO_GL=egl` is set automatically.

### Visualization Results

| MuJoCo | Physical |
|---|---|
| ![go2](doc/gif/go2-velocity.gif) | <img src="doc/gif/go2-velocity-real.gif" width="300"/> |

## Sim2Real Deployment

### Prerequisites

Install the required communication tools:
- [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git)
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)

### Steps

1. **Power On** — Start the robot and wait until it enters `zero-torque` mode.
2. **Enable Debug Mode** — While in `zero-torque`, press `L2 + R2` on the controller to enter debug mode with joint damping enabled.
3. **Connect** — Connect your PC via Ethernet:
   - Address: `192.168.123.222`
   - Netmask: `255.255.255.0`

### Compilation

Place `policy.onnx` and `policy.onnx.data` into `deploy/robots/go2/config/policy/velocity/v0/exported`, then compile:

```bash
cd deploy/robots/go2
mkdir build && cd build
cmake .. && make
```

### Deployment

Simulation (using [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)):
```bash
# Build unitree_mujoco first
cd simulate && mkdir build && cd build && cmake .. && make -j8
./simulate/build/unitree_mujoco   # gamepad must be connected

# Launch control program
cd deploy/robots/go2/build
./go2_ctrl --network=lo
```

Real robot:
```bash
cd deploy/robots/go2/build
./go2_ctrl --network=enp5s0   # use ifconfig to find your interface
```

## Train Log Manager

Browse and compare rsl_rl training runs with a Streamlit app:

```bash
pip install streamlit pyyaml tensorboard
streamlit run tools/train_log_manager/app.py -- --logs-root logs/rsl_rl
```

Open http://localhost:8501 (port-forward if running over SSH).

Features:
- **Sortable table** — one row per training run with `task_id` from `run.yaml` plus user-defined columns from `agent.yaml`, `env.yaml`, and git diffs.
- **Play selected checkpoint** — select a run, choose a Go2 environment/checkpoint, and launch `scripts/play.py --viewer viser` in the background with an `Open Viser` link.
- **TensorBoard view** — auto-start TensorBoard for `logs/rsl_rl` and open the selected run filter.
- **Column management** — add, rename, remove columns dynamically; peek values before committing.
- **YAML / Git diff comparison** — select two table rows to compare their configs and code changes side by side.
- **URL persistence** — column configuration is encoded in the URL for easy sharing.

See [tools/train_log_manager/README.md](tools/train_log_manager/README.md) for details.

## Acknowledgements

This project would not be possible without the contributions of:

- [mjlab](https://github.com/mujocolab/mjlab.git) — training and execution framework
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git) — RL algorithm implementation
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git) — GPU-accelerated rendering and simulation
- [mujoco](https://github.com/google-deepmind/mujoco.git) — high-fidelity rigid-body physics engine
