"""Evaluate / visualize a policy-conditioned traversability estimator.

Four subcommands, sharing the engine in ``tools/_trav_eval_common.py`` (the same
engine the "Evaluator" tab in ``tools/train_log_manager/app.py`` uses):

* ``report``    -- ROC / PR / calibration / score-histogram + threshold sweep,
                  from ``labels.npz`` (no GPU rollout needed).
* ``timelines`` -- per-episode risk(t) + lead-time / false-alarm stats, from
                  ``raw_rollouts.npz``.
* ``spatial``   -- spatial head's predicted risk map vs GT label / mask / height
                  scan, from ``raw_rollouts.npz`` (needs a trained spatial head).
* ``live``      -- roll the policy out and write an mp4 with the scalar risk gauge
                  and spatial heatmap composited onto each frame (needs GPU).

Examples:
    python tools/eval_traversability.py report \
        --labels logs/traversability/labels.npz \
        --checkpoint logs/traversability/traversability_go2w_moe.pt

    python tools/eval_traversability.py timelines \
        --rollouts logs/traversability/raw_rollouts.npz \
        --checkpoint logs/traversability/traversability_go2w_moe.pt --threshold 0.5

    python tools/eval_traversability.py spatial \
        --rollouts logs/traversability/raw_rollouts.npz \
        --checkpoint logs/traversability/traversability_go2w_moe.pt --num-samples 8

    CUDA_VISIBLE_DEVICES=0 python tools/eval_traversability.py live \
        --policy-checkpoint logs/rsl_rl/<exp>/<run>/model_<N>.pt \
        --estimator logs/traversability/traversability_go2w_moe.pt \
        --steps 600 --output logs/traversability/eval/live_overlay.mp4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never open a window.

import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

import _trav_eval_common as tec  # noqa: E402

DEFAULT_OUT = Path("logs/traversability/eval")


def _write_sweep_csv(path: Path, sweep: list[dict]) -> None:
  if not sweep:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
    w.writeheader()
    w.writerows(sweep)


def cmd_report(args: argparse.Namespace) -> None:
  scores, labels, info = tec.score_labels_file(
    args.labels, args.checkpoint, device=args.device, val_frac=args.val_frac,
    seed=args.seed, split=args.split,
  )
  fig, result = tec.build_report_figure(scores, labels)
  args.output.mkdir(parents=True, exist_ok=True)
  png = args.output / "report.png"
  fig.savefig(png, dpi=130)
  _write_sweep_csv(args.output / "threshold_sweep.csv", result["sweep"])
  m = result["metrics"]
  print(f"[REPORT] split={info['split']} n={info['n_samples']} keys={info['keys']}")
  print(f"[REPORT] auc={m['auc']:.4f} ap={m['ap']:.4f} acc={m['acc']:.4f} "
        f"brier={m['brier']:.4f} pos_rate={result['pos_rate']:.4f}")
  if result["best_f1"]:
    b = result["best_f1"]
    print(f"[REPORT] best-F1 thr={b['threshold']:.2f} P={b['precision']:.3f} "
          f"R={b['recall']:.3f} F1={b['f1']:.3f} FPR={b['fpr']:.4f} "
          f"(tp={b['tp']} fp={b['fp']} fn={b['fn']})")
  print(f"[REPORT] saved {png} and threshold_sweep.csv")


def cmd_timelines(args: argparse.Namespace) -> None:
  scored = tec.score_rollouts_file(args.rollouts, args.checkpoint, device=args.device)
  args.output.mkdir(parents=True, exist_ok=True)
  fig_tl, _ = tec.build_timeline_figure(
    scored, threshold=args.threshold, max_fail=args.max_fail, n_safe=args.n_safe, seed=args.seed,
  )
  fig_tl.savefig(args.output / "timelines.png", dpi=130)
  fig_lt, stats = tec.build_leadtime_figure(scored, threshold=args.threshold)
  fig_lt.savefig(args.output / "lead_time.png", dpi=130)
  print(f"[TIMELINES] failures={stats['n_failures']} detected={stats['n_detected']} "
        f"missed={stats['n_missed']} detection_rate={stats['detection_rate']:.3f}")
  print(f"[TIMELINES] lead median={stats['lead_median_s']:.2f}s "
        f"(IQR {stats['lead_iqr_steps']:.0f} steps)  "
        f"false_alarms/min={stats['false_alarms_per_min']:.2f} "
        f"(safe-step rate {stats['false_alarm_rate']:.4f})")
  print(f"[TIMELINES] saved timelines.png and lead_time.png in {args.output}")


def cmd_spatial(args: argparse.Namespace) -> None:
  scored = tec.score_rollouts_file(
    args.rollouts, args.checkpoint, device=args.device, spatial=True,
  )
  args.output.mkdir(parents=True, exist_ok=True)
  fig, info = tec.build_spatial_figure(scored, num_samples=args.num_samples, seed=args.seed)
  png = args.output / "spatial.png"
  fig.savefig(png, dpi=130)
  print(f"[SPATIAL] grid={scored['model'].spatial_grid} "
        f"size_m={scored['model'].spatial_size_m} samples={len(info['samples'])}")
  print(f"[SPATIAL] saved {png}")


def cmd_live(args: argparse.Namespace) -> None:
  _run_live(args)


def _run_live(args: argparse.Namespace) -> None:
  """Roll the policy out and composite scalar-risk + spatial-map overlays to mp4."""
  import os

  os.environ.setdefault("MUJOCO_GL", "egl")

  import imageio.v2 as imageio
  import matplotlib.pyplot as plt
  import torch

  import evaluate_policy as ep  # noqa: E402
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  torch.manual_seed(args.seed)
  device = args.device

  run_dir = ep._find_run_dir(Path(args.policy_checkpoint))
  saved_run = ep._load_yaml(run_dir / "params" / "run.yaml")
  task = ep._resolve_task(args.task, saved_run)
  env_cfg, agent_cfg, _ = ep._load_training_configs(task, run_dir)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.seed
  for group in env_cfg.observations.values():
    group.enable_corruption = False

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env.seed(args.seed)
  policy, _, _ = ep._load_policy(task, Path(args.policy_checkpoint), env, agent_cfg, device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))

  om = env.observation_manager
  term_names = list(om.active_terms["actor"])
  term_dims = [int(np.prod(d)) for d in om.group_obs_term_dim["actor"]]
  layout, off = [], 0
  for name, dim in zip(term_names, term_dims):
    layout.append((name, slice(off, off + dim)))
    off += dim

  model = tec.load_traversability_estimator(args.estimator, map_location=device).to(device)
  keys = list(model.encoder_input_keys)
  has_spatial = float(getattr(model, "spatial_grid", (0, 0))[0]) > 0

  args.output.parent.mkdir(parents=True, exist_ok=True)
  writer = imageio.get_writer(args.output, fps=args.fps, macro_block_size=None)

  obs = vec_env.get_observations()
  risk_hist: list[float] = []
  with torch.inference_mode():
    for step in range(args.steps):
      actor = obs["actor"].detach().to(device)
      groups = {
        name: actor[..., sl].float() for name, sl in layout if name in keys
      }
      risk = float(model.predict_proba(groups)[0].item())
      risk_hist.append(risk)
      risk_map = (model.predict_spatial_proba(groups)[0].cpu().numpy()
                  if has_spatial else None)

      frame = env.render()  # [H, W, 3]
      composite = _compose_overlay(frame, risk, risk_hist, risk_map, plt)
      writer.append_data(composite)

      action = policy(obs.to(device))
      obs, _r, _d, _e = vec_env.step(action.to(vec_env.device))
      if (step + 1) % 100 == 0:
        print(f"[LIVE] step {step + 1}/{args.steps} risk={risk:.3f}")

  writer.close()
  vec_env.close()
  print(f"[LIVE] saved {args.output} ({len(risk_hist)} frames)")


def _compose_overlay(frame, risk, risk_hist, risk_map, plt):
  """Render the sim frame + a risk gauge / time-series / spatial map to one RGB array."""
  H, W = frame.shape[:2]
  fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
  ax_img = fig.add_axes([0, 0, 1, 1])
  ax_img.imshow(frame)
  ax_img.axis("off")

  color = "C3" if risk >= 0.5 else "C2"

  # Risk gauge (top-left).
  ax_g = fig.add_axes([0.02, 0.80, 0.18, 0.05])
  ax_g.barh([0], [risk], color=color)
  ax_g.barh([0], [1], color="none", edgecolor="white")
  ax_g.set_xlim(0, 1); ax_g.set_ylim(-0.5, 0.5)
  ax_g.set_xticks([]); ax_g.set_yticks([])
  ax_g.set_title(f"P(failure soon) = {risk:.2f}", color="white", fontsize=10, loc="left")

  # Risk time-series (bottom strip).
  ax_t = fig.add_axes([0.02, 0.04, 0.40, 0.14])
  tail = risk_hist[-200:]
  ax_t.plot(tail, color=color)
  ax_t.axhline(0.5, ls="--", color="white", lw=0.6)
  ax_t.set_ylim(0, 1); ax_t.set_xticks([])
  ax_t.tick_params(colors="white", labelsize=7)
  ax_t.patch.set_alpha(0.3)

  # Spatial map (top-right) if present.
  if risk_map is not None:
    ax_s = fig.add_axes([0.80, 0.70, 0.16, 0.18])
    ax_s.imshow(risk_map.T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax_s.set_xticks([]); ax_s.set_yticks([])
    ax_s.set_title("risk map", color="white", fontsize=8)

  fig.canvas.draw()
  buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
  plt.close(fig)
  return buf


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="command", required=True)

  pr = sub.add_parser("report", help="ROC/PR/calibration/sweep from labels.npz")
  pr.add_argument("--labels", required=True, type=Path)
  pr.add_argument("--checkpoint", required=True, type=Path)
  pr.add_argument("--split", default="val", choices=["val", "train", "all"])
  pr.add_argument("--val-frac", type=float, default=0.15)
  pr.add_argument("--seed", type=int, default=0)
  pr.add_argument("--device", default="cpu")
  pr.add_argument("--output", type=Path, default=DEFAULT_OUT)
  pr.set_defaults(func=cmd_report)

  tl = sub.add_parser("timelines", help="risk timelines + lead-time from raw_rollouts.npz")
  tl.add_argument("--rollouts", required=True, type=Path)
  tl.add_argument("--checkpoint", required=True, type=Path)
  tl.add_argument("--threshold", type=float, default=0.5)
  tl.add_argument("--max-fail", type=int, default=6)
  tl.add_argument("--n-safe", type=int, default=4)
  tl.add_argument("--seed", type=int, default=0)
  tl.add_argument("--device", default="cpu")
  tl.add_argument("--output", type=Path, default=DEFAULT_OUT)
  tl.set_defaults(func=cmd_timelines)

  sp = sub.add_parser("spatial", help="spatial risk map vs GT from raw_rollouts.npz")
  sp.add_argument("--rollouts", required=True, type=Path)
  sp.add_argument("--checkpoint", required=True, type=Path)
  sp.add_argument("--num-samples", type=int, default=6)
  sp.add_argument("--seed", type=int, default=0)
  sp.add_argument("--device", default="cpu")
  sp.add_argument("--output", type=Path, default=DEFAULT_OUT)
  sp.set_defaults(func=cmd_spatial)

  lv = sub.add_parser("live", help="rollout with risk overlay -> mp4")
  lv.add_argument("--policy-checkpoint", required=True, type=Path)
  lv.add_argument("--estimator", required=True, type=Path)
  lv.add_argument("--task", default=None)
  lv.add_argument("--steps", type=int, default=600)
  lv.add_argument("--fps", type=int, default=30)
  lv.add_argument("--seed", type=int, default=0)
  import torch
  lv.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  lv.add_argument("--output", type=Path, default=DEFAULT_OUT / "live_overlay.mp4")
  lv.set_defaults(func=cmd_live)

  return p.parse_args()


def main() -> None:
  args = parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
