# Policy-Conditioned Traversability Estimation

A learning pipeline that predicts, for a **specific chosen locomotion policy**, how
traversable the current situation is — the controller's *true* capability envelope,
not a geometric slope/roughness heuristic. A planner using generic traversability
doesn't know what a given Go2W policy can actually do; an estimator trained on the
policy's *own* rollouts does. The scalar output also doubles as a runtime safety
monitor.

## What it predicts

- **Label = short-horizon failure risk.** For each rollout timestep `t`,
  `label = 1` if the policy falls / gets stuck within the next `H` steps
  (`H ≈ 75 ≈ 1.5 s @ 50 Hz`), else `0`. The model outputs `P(failure soon) ∈ [0, 1]`.
- **Inputs are configurable** (default: actor obs, deployable): `height_scan` plus
  any subset of proprioception (`base_ang_vel`, `projected_gravity`, `command`,
  `joint_pos`, `joint_vel`, `actions`). Selected via `--input-keys`.
- **Two heads on a shared encoder:**
  1. **scalar** `P(failure soon)` — core, validated first.
  2. **spatial** per-cell failure map — extension, labeled by back-projecting future
     failures onto cells. The map size is configurable and **decoupled** from the
     fixed 17×11 height-scan input grid: real-world extent `--spatial-size-m W H`
     (metres) and resolution `--spatial-grid NW NH` (cells).

## Components

| File | Role |
|------|------|
| `tools/collect_traversability.py` | Roll out the chosen policy on its training env (curriculum + auto-reset on) and log per-step obs, root pose, and `failure = dones & ~time_outs`. |
| `tools/build_traversability_labels.py` | Offline labeler: scalar short-horizon failure labels (clipped at episode boundaries — no leakage) + optional configurable spatial map. Reports `pos_weight`. |
| `src/rl_models/traversability.py` | `TraversabilityEstimator`: shared encoder (`build_observation_encoder`, conv2d over height_scan + proprio context) with scalar + spatial heads. |
| `tools/train_traversability.py` | BCE(`pos_weight`) scalar loss + masked per-cell spatial loss; val metrics (AUC / PR-AUC / accuracy / Brier); `--eval-only`. |

### Design notes

- The collector records the observation manager's **per-term layout**
  (`active_terms` + `group_obs_term_dim`) so the concatenated actor vector is split
  by name offline — robust to Go2W's wheel joints changing the obs dimensions.
- The failure signal is `dones & ~time_outs` (a truncation/timeout is *not* a
  failure), read straight from the `RslRlVecEnvWrapper` step contract.
- Scalar labeling segments each env timeline into episodes at `done==True` and clips
  the `H`-step lookahead at the episode end, so labels never cross the auto-reset.
- Spatial labeling back-projects each grid cell to a world location, marks cells near
  the upcoming failure, and masks to cells the robot actually traverses within the
  window (masked BCE).

## End-to-end usage

```bash
# 1. Collect rollouts from a chosen policy (GPU). Curriculum + random twist commands
#    push the robot across easy->hard terrain, producing real failures.
CUDA_VISIBLE_DEVICES=1 python tools/collect_traversability.py \
  --checkpoint logs/rsl_rl/<exp>/<run>/model_<N>.pt \
  --num-envs 1024 --steps 1500 --horizon 75 \
  --output logs/traversability/raw_rollouts.npz

# 2. Build labels (scalar core; add --spatial for the map). Prints pos/neg + pos_weight.
python tools/build_traversability_labels.py \
  --input logs/traversability/raw_rollouts.npz --horizon 75 \
  --spatial --spatial-size-m 2.0 1.0 --spatial-grid 20 10 \
  --output logs/traversability/labels.npz

# 3a. Train the scalar head (spatial disabled by default).
python tools/train_traversability.py \
  --input logs/traversability/labels.npz \
  --output logs/traversability/traversability.pt --epochs 50

# 3b. Enable the spatial head (grid size read from labels.npz).
python tools/train_traversability.py --input logs/traversability/labels.npz \
  --output logs/traversability/traversability_spatial.pt --spatial-weight 1.0

# 4. Held-out metrics for a trained checkpoint.
python tools/train_traversability.py --input logs/traversability/labels.npz \
  --eval-only --checkpoint logs/traversability/traversability.pt
```

### Restrict the input set (e.g. height-scan + command only)

```bash
python tools/train_traversability.py --input logs/traversability/labels.npz \
  --output logs/traversability/trav_hs_cmd.pt \
  --input-keys height_scan command
```

## Loading the estimator for inference

```python
from src.rl_models.traversability import load_traversability_estimator
model = load_traversability_estimator("logs/traversability/traversability.pt", map_location="cuda")
risk = model.predict_proba(obs)            # [B] P(failure soon)
risk_map = model.predict_spatial_proba(obs)  # [B, NW, NH] per-cell risk
```
`obs` is a mapping of the selected group names (e.g. `height_scan`, `command`, …) to
tensors — the same groups passed via `--input-keys` at train time.

## Troubleshooting

- **No positive labels** (`[COLLECT][WARN] no failure terminations`): the policy is
  too strong on its training distribution. Increase `--steps`, raise terrain
  difficulty, or use an earlier/weaker checkpoint.
- **`auc`/`ap` report `nan`**: the held-out split has only one class. Collect more
  data or lower `--val-frac`.
- **Memory**: the collector holds `steps × num_envs × obs_dim` floats on CPU. Scale
  `--steps` / `--num-envs` accordingly.
```

## Evaluation & visualization

The training metrics (`--eval-only` prints `auc / ap / acc / brier`) tell you the
*ranking* quality but not *what* the estimator does. `tools/eval_traversability.py`
makes the behaviour legible, and the same engine is exposed in the log-manager UI.

```bash
# 1. Static report: ROC / PR / calibration / score histogram + threshold sweep CSV.
#    AUC/AP here match train_traversability.py --eval-only on the same split.
python tools/eval_traversability.py report \
  --labels logs/traversability/labels.npz \
  --checkpoint logs/traversability/traversability_go2w_moe.pt

# 2. Per-episode risk timelines + lead-time / false-alarm stats ("safety monitor").
python tools/eval_traversability.py timelines \
  --rollouts logs/traversability/raw_rollouts.npz \
  --checkpoint logs/traversability/traversability_go2w_moe.pt --threshold 0.5

# 3. Spatial head: predicted risk map vs GT label / path mask / height-scan.
python tools/eval_traversability.py spatial \
  --rollouts logs/traversability/raw_rollouts.npz \
  --checkpoint logs/traversability/traversability_go2w_moe.pt --num-samples 8

# 4. Live overlay: roll the policy out, composite the risk gauge + spatial map to mp4.
CUDA_VISIBLE_DEVICES=0 python tools/eval_traversability.py live \
  --policy-checkpoint logs/rsl_rl/<exp>/<run>/model_<N>.pt \
  --estimator logs/traversability/traversability_go2w_moe.pt \
  --steps 600 --output logs/traversability/eval/live_overlay.mp4
```

Outputs land in `logs/traversability/eval/` (`report.png`, `threshold_sweep.csv`,
`timelines.png`, `lead_time.png`, `spatial.png`, `live_overlay.mp4`).

### In the log manager (Streamlit)

```bash
streamlit run tools/train_log_manager/app.py -- --logs-root logs/rsl_rl
```

### Live "Play with risk" viewer

Watch the policy walk **live** with the estimator running in real time:

```bash
python scripts/play.py <Task> --viewer viser \
  --checkpoint-file logs/rsl_rl/<exp>/<run>/model_<N>.pt \
  --risk-estimator logs/traversability/traversability_go2w_moe.pt
```

Open the printed Viser URL: the **Traversability** tab shows a `P(failure soon)` gauge + rolling sparkline and the spatial risk heatmap, and colored **risk markers** are drawn on the terrain ahead of the robot (`src/viz/risk_viewer.py`, a `ViserPlayViewer` subclass mirroring `AttributionViserPlayViewer`). The play task must match the task the estimator was trained on. In the log manager, the same is launched from the Evaluator tab's **Play (live)** sub-tab (→ *Open Viser*).

The app now has two top-level tabs: **Runs** (the original run browser) and
**Evaluator**. The Evaluator tab picks an estimator + artifacts from
`logs/traversability/` and runs the same four views in-browser — Report / Timelines
/ Spatial render in-process (CPU); **Live overlay** launches the GPU `live` command
through the app's process manager and shows the mp4 when it finishes.

> Implementation: `tools/_trav_eval_common.py` is the shared engine (scoring +
> figure builders returning `matplotlib` Figures); the CLI saves them, the UI
> `st.pyplot`s them. It reuses `train_traversability._binary_metrics`,
> `build_traversability_labels.{parse_layout,compute_episode_geometry,build_spatial_labels}`,
> and `traversability.load_traversability_estimator`.
