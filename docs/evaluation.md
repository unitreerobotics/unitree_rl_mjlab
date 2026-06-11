# Policy Evaluation

Use `scripts/run_eval.sh` for batch evaluation across the standard velocity
corridor terrains. The script evaluates every checkpoint listed in a CSV, writes
per-checkpoint/per-terrain outputs, aggregates summaries, and generates the grid
video and annotated result image.

```bash
scripts/run_eval.sh \
  --checkpoints-csv tmp/checkpoints.csv \
  --num-runs 10 \
  --gpus 1,2,3
```

The checkpoint CSV is intentionally explicit. The current CSV can stay at
`tmp/checkpoints.csv`, but the script will not use it unless you pass
`--checkpoints-csv tmp/checkpoints.csv`.

## Batch Script Options

Required options:

- `--checkpoints-csv PATH`: CSV with a `checkpoint` header and one checkpoint
  path per row.
- `--gpus GPU[,GPU...]`: comma-separated GPU IDs. Jobs are assigned round-robin
  and concurrency is capped to the number of listed GPUs.

Common options:

- `--num-runs N` or `--runs N`: number of parallel evaluation episodes per
  checkpoint/terrain. Defaults to `100`.
- `--output-dir DIR`: output root. Defaults to
  `logs/data/eval/YYYYMMDD_eval_${NUM_RUNS}runs`.
- `--video-run N`: run index to record video for. Defaults to `0`; use `-1` to
  disable video.
- `--skip-artifacts`: skip grid artifact generation.
- `--help`: print usage.

The script evaluates these terrains for every checkpoint:

- `rough_curriculum_corridor`: random-uniform rough heightfield patches with
  increasing roughness.
- `perlin_noise_corridor`: Perlin heightfield patches with increasing height
  amplitude.
- `random_spread_boxes_corridor`: randomly spread box obstacles with increasing
  density and height.

After all evaluations finish successfully, the script writes:

- `<output-dir>/combined_summary.csv`: aggregate metrics from all
  `<label>/<terrain>/summary.csv` files.
- `<output-dir>/couple.mp4`: tiled side-view video using `run_000/video_side.mp4`
  from every checkpoint/terrain cell.
- `<output-dir>/result.jpg`: final still extracted from `couple.mp4`.
- `<output-dir>/result_annotated.jpg`: final still with per-cell outcome badges.

## Output Layout

For each checkpoint and terrain, outputs are written under
`<output-dir>/<checkpoint_label>/<terrain>/`:

- `eval_config.yaml`: final resolved evaluation environment config.
- `checkpoint_info.json`: checkpoint, run directory, config paths, and task.
- `terrain_metadata.json`: patch layout, terrain settings, and difficulty.
- `path_waypoints.csv`: centerline waypoints.
- `summary.csv`: one row per evaluation run. The `seed` column records the
  global base seed shared by all runs.
- `run_000/raw.csv` and `run_000/raw.npz`: raw per-step data for run 0.
- `run_000/events.json`: run result and termination metadata.
- `run_000/video.mp4`, `run_000/video_side.mp4`, `run_000/video_behind.mp4`:
  front, side, and behind camera views, recorded only for the env selected by
  `--video-run`.

`traversal_rate` is `final path progress / total path length`, clipped to
`[0, 1]`.

## Manual Single-Policy Evaluation

For one checkpoint and one terrain, call `tools/evaluate_policy.py` directly:

```bash
uv run python tools/evaluate_policy.py \
  --task Unitree-Go2-Rough \
  --checkpoint /path/to/logs/rsl_rl/.../model_2000.pt \
  --num-runs 100 \
  --eval-terrain rough_curriculum_corridor \
  --video-run 0 \
  --output-dir logs/tmp
```

The evaluator infers the run directory from `--checkpoint`, loads
`params/env.yaml`, `params/agent.yaml`, and `params/run.yaml`, reconstructs the
registered task config with those saved values, then overrides only evaluation
runtime settings: all `--num-runs` episodes run simultaneously as parallel
environments in a single GPU MuJoCo-Warp scene (`run_id` is the env index), with
the selected corridor terrain, disabled observation corruption/curriculum/random
pushes, and a Pure Pursuit `twist` command. The trained policy architecture and
observation setup come from the saved run config.

All envs share the same corridor-start origin (terrain uses `num_cols=1`,
`max_init_terrain_level=0`) and the same world-coordinate Pure Pursuit waypoints.
Once an env finishes its episode, simulation continues but its data is no longer
recorded. This parallelism is the source of a large wall-time speedup: a 10-run
job drops from roughly 25-30 min to about one episode's wall time; 100-run jobs
see 50x+ speedup, because the GPU sim steps many envs for nearly the same
per-step cost as one.

Useful low-level evaluator options:

- `--eval-terrain perlin_noise_corridor`: evaluate on the Perlin corridor.
- `--eval-terrain random_spread_boxes_corridor`: evaluate on the random box
  corridor.
- `--num-runs 2`: change the number of evaluation episodes. All runs execute as
  parallel envs; `--num-runs 1` behaves like the old sequential mode.
- `--seed VALUE`: global base seed for the whole batch. Terrain layout and the
  RNG stream are both seeded from this value.
- `--video-run 0`: record exactly that env index; must be `< --num-runs`.
- `--video-run -1`: disable video recording.
- `--video-fps FPS`: target video frame rate. For example, `--video-fps 25`
  halves frame count and file size while keeping real-time playback duration.
- `--video-width 1920 --video-height 1080`: set video resolution.
- `--max-steps N` or `--max-episode-time T`: limit episode length.
- `--stuck-time 20` and `--stuck-progress-epsilon 0.25`: stop stalled runs when
  path progress does not improve enough for the configured time window. Use
  `--stuck-time 0` to disable.
- `--lookahead-distance`, `--target-speed`, `--max-linear-velocity`,
  `--max-yaw-rate`, and `--goal-tolerance`: tune Pure Pursuit behavior.

## Seed Semantics

mjlab seeding is global, so there is a single base seed (`--seed`) shared by the
whole parallel batch; there are no per-env seeds. The `seed` column in
`summary.csv`, `raw.csv`, and `events.json` records this base seed for every run.
Previously it stored `base + run_id`. Passing the same `--seed` still reproduces
a terrain layout and evaluation, but per-step trajectories differ from the old
sequential per-run-reseed mode because the RNG is now a single global stream.
Metric semantics are unchanged.

The evaluation corridors are defined in
`src/tasks/velocity/evaluation/terrains.py`. Adjust the corresponding
`make_*_corridor_cfg()` helper to change patch length, corridor width, terrain
patch count, or terrain-specific difficulty ranges. The corridor uses the
repository terrain generator's row direction, which is world `x` for the current
local `mjlab` terrain API.
