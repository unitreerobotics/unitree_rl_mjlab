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
runtime settings: all `--num-runs` episodes run **simultaneously as parallel
environments** in a single GPU MuJoCo-Warp scene (`run_id` == env index), with
the selected corridor terrain, disabled observation corruption/curriculum/random
pushes, and a Pure Pursuit `twist` command. The trained policy architecture and
observation setup come from the saved run config.

All envs share the same corridor-start origin (terrain uses `num_cols=1`,
`max_init_terrain_level=0`) and the same world-coordinate Pure Pursuit waypoints.
Once an env finishes its episode, simulation continues but its data is no longer
recorded. This parallelism is the source of a large wall-time speedup: a 10-run
job drops from ~25–30 min to roughly one episode's wall time; 100-run jobs see
50×+ speedup, because the GPU sim steps thousands of envs for nearly the same
per-step cost as one.

Outputs are written under `--output-dir`, which defaults to `logs/tmp`:

- `eval_config.yaml`: final resolved evaluation environment config.
- `checkpoint_info.json`: checkpoint, run directory, config paths, and task.
- `terrain_metadata.json`: patch layout, terrain settings, and difficulty.
- `path_waypoints.csv`: centerline waypoints.
- `summary.csv`: one row per evaluation run. The `seed` column records the
  global base seed shared by all runs (see [Seed semantics](#seed-semantics)
  below).
- `run_000/raw.csv` and `run_000/raw.npz`: raw per-step data. The `seed` column
  is the global base seed (not base+run_id).
- `run_000/events.json`: run result and termination metadata. `seed` is the
  global base seed.
- `run_000/video.mp4`, `run_000/video_side.mp4`, `run_000/video_behind.mp4`:
  front, side, and behind camera views of the tracked robot (1920×1080),
  recorded only for the env selected by `--video-run`. Frames stream to disk
  during the rollout. If `--video-run >= --num-runs`, video is disabled with a
  warning.

`traversal_rate` is `final path progress / total path length`, clipped to
`[0, 1]`.

### Seed semantics

mjlab seeding is global, so there is a single base seed (`--seed`) shared by
the whole parallel batch — there are no per-env seeds. The `seed` column in
`summary.csv`, `raw.csv`, and `events.json` records this base seed for every
run. Previously it stored `base + run_id`. Passing the same `--seed` still
fully reproduces a terrain layout and evaluation, but per-step trajectories
differ from the old sequential per-run-reseed mode because the RNG is now a
single global stream. Metric semantics (traversal rate, distance, etc.) are
unchanged.

Useful options:

- `--eval-terrain perlin_noise_corridor`: evaluate on the Perlin corridor.
- `--eval-terrain random_spread_boxes_corridor`: evaluate on the random box corridor.
- `--num-runs 2`: change the number of evaluation episodes (all run as parallel
  envs; `--num-runs 1` behaves exactly like the old sequential mode).
- `--seed <value>`: global base seed for the whole batch. Terrain layout and the
  RNG stream are both seeded from this value, so passing the same seed
  reproduces a complete evaluation. **Changed from before**: the `seed` column
  in output files is now this single base seed for all runs (previously
  base+run_id). Trajectories are not bit-identical to the old per-run-reseed
  mode (single global RNG stream now), but metric semantics are unchanged.
- `--video-run 0`: record exactly that env index; must be `< --num-runs`.
- `--video-run -1`: disable video recording.
- `--video-fps <fps>`: capture cadence for video frames (default: control rate,
  ~50 fps). `capture_every = round(control_rate / video_fps)`; playback stays
  real-time. E.g. `--video-fps 25` halves frame count and file size with the
  same playback duration.
- `--video-width 1920 --video-height 1080`: set video resolution.
- `--max-steps N` or `--max-episode-time T`: limit episode length.
- `--stuck-time 20` and `--stuck-progress-epsilon 0.25`: stop stalled runs when path progress does not improve enough for the configured time window. Use `--stuck-time 0` to disable.
- `--lookahead-distance`, `--target-speed`, `--max-linear-velocity`,
  `--max-yaw-rate`, and `--goal-tolerance`: tune Pure Pursuit behavior.

The evaluation corridors are defined in
`src/tasks/velocity/evaluation/terrains.py`. Adjust the corresponding
`make_*_corridor_cfg()` helper to change patch length, corridor width, terrain
patch count, or terrain-specific difficulty ranges. The corridor uses the
repository terrain generator's row direction, which is world `x` for the current
local `mjlab` terrain API.
