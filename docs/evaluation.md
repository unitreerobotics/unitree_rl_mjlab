# Policy Evaluation

Run trained velocity policies on deterministic corridor terrains:

```bash
python tools/evaluate_policy.py \
  --task Unitree-Go2-Rough \
  --checkpoint /path/to/logs/rsl_rl/.../model_2000.pt \
  --num-runs 100 \
  --eval-terrain rough_curriculum_corridor \
  --video-run 0 \
  --output-dir logs/tmp
```

Supported `--eval-terrain` values:

- `rough_curriculum_corridor`: random-uniform rough heightfield patches with increasing roughness.
- `perlin_noise_corridor`: Perlin heightfield patches with increasing height amplitude.
- `random_spread_boxes_corridor`: randomly spread box obstacles with increasing density and height.

The evaluator infers the run directory from `--checkpoint`, loads
`params/env.yaml`, `params/agent.yaml`, and `params/run.yaml`, reconstructs the
registered task config with those saved values, then overrides only evaluation
runtime settings: one environment, the selected corridor terrain, disabled
observation corruption/curriculum/random pushes, and a Pure Pursuit `twist`
command. The trained policy architecture and observation setup come from the
saved run config.

Outputs are written under `--output-dir`, which defaults to `logs/tmp`:

- `eval_config.yaml`: final resolved evaluation environment config.
- `checkpoint_info.json`: checkpoint, run directory, config paths, and task.
- `terrain_metadata.json`: patch layout, terrain settings, and difficulty.
- `path_waypoints.csv`: centerline waypoints.
- `summary.csv`: one row per evaluation run.
- `run_000/raw.csv` and `run_000/raw.npz`: raw per-step data.
- `run_000/events.json`: run result and termination metadata.
- `run_000/video.mp4`: only for the selected `--video-run`.

`traversal_rate` is `final path progress / total path length`, clipped to
`[0, 1]`.

Useful options:

- `--eval-terrain perlin_noise_corridor`: evaluate on the Perlin corridor.
- `--eval-terrain random_spread_boxes_corridor`: evaluate on the random box corridor.
- `--num-runs 2`: change the number of evaluation episodes.
- `--video-run 0`: record exactly that run.
- `--video-run -1`: disable video recording.
- `--max-steps N` or `--max-episode-time T`: limit episode length.
- `--lookahead-distance`, `--target-speed`, `--max-linear-velocity`,
  `--max-yaw-rate`, and `--goal-tolerance`: tune Pure Pursuit behavior.

The evaluation corridors are defined in
`src/tasks/velocity/evaluation/terrains.py`. Adjust the corresponding
`make_*_corridor_cfg()` helper to change patch length, corridor width, terrain
patch count, or terrain-specific difficulty ranges. The corridor uses the
repository terrain generator's row direction, which is world `x` for the current
local `mjlab` terrain API.
