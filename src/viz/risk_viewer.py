"""Viser viewer extension that renders live policy-conditioned traversability risk.

Mirrors :class:`src.viz.attribution_viewer.AttributionViserPlayViewer`: a
:class:`~mjlab.viewer.ViserPlayViewer` subclass that, each playback step, runs a
trained :class:`~src.rl_models.traversability.TraversabilityEstimator` on the same
observation the policy sees and surfaces the result live:

* a scalar ``P(failure soon)`` gauge + rolling sparkline,
* the spatial risk heatmap (the estimator's robot-frame grid), and
* colored risk markers drawn on the terrain in the 3D scene in front of the robot.

The estimator consumes named observation *terms* (``height_scan`` + proprio) that
live inside the policy's concatenated ``actor`` observation group, so we read the
per-term layout from the observation manager (exactly like
``tools/collect_traversability.py``) and slice ``obs["actor"]`` accordingly.
"""

from __future__ import annotations

import html
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import viser

from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.base import VerbosityLevel

from src.rl_models.traversability import load_traversability_estimator

ACTOR_GROUP = "actor"
_EMPTY_HEATMAP = np.zeros((200, 100, 3), dtype=np.uint8)


def _colormap(values: np.ndarray) -> np.ndarray:
  """Map ``values`` in ``[0, 1]`` to an RGB ramp (blue -> green -> yellow -> red)."""
  stops = np.array(
    [
      [20, 30, 92],
      [42, 151, 194],
      [157, 217, 86],
      [255, 214, 74],
      [218, 70, 54],
    ],
    dtype=np.float64,
  )
  x = np.clip(values, 0.0, 1.0) * (len(stops) - 1)
  lo = np.floor(x).astype(np.int64)
  hi = np.clip(lo + 1, 0, len(stops) - 1)
  t = x[..., None] - lo[..., None]
  rgb = (1.0 - t) * stops[lo] + t * stops[hi]
  return rgb.astype(np.uint8)


def _term_layout(observation_manager) -> list[tuple[str, slice]]:
  """Per-term (name, slice) layout of the concatenated ``actor`` observation."""
  if ACTOR_GROUP not in observation_manager.active_terms:
    raise KeyError(
      f"Observation group {ACTOR_GROUP!r} not found; available: "
      f"{list(observation_manager.active_terms.keys())}"
    )
  names = list(observation_manager.active_terms[ACTOR_GROUP])
  dims = [int(np.prod(d)) for d in observation_manager.group_obs_term_dim[ACTOR_GROUP]]
  layout: list[tuple[str, slice]] = []
  offset = 0
  for name, dim in zip(names, dims):
    layout.append((name, slice(offset, offset + dim)))
    offset += dim
  return layout


def _quat_to_yaw(quat: np.ndarray) -> float:
  """Yaw (rad) from a wxyz quaternion."""
  w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
  return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class RiskViserPlayViewer(ViserPlayViewer):
  """Viser playback viewer with a live traversability-risk panel + 3D markers."""

  def __init__(
    self,
    env,
    policy,
    *,
    estimator_path: str | Path,
    frame_rate: float = 60.0,
    verbosity: VerbosityLevel = VerbosityLevel.SILENT,
    viser_server: viser.ViserServer | None = None,
  ) -> None:
    super().__init__(env, policy, frame_rate, verbosity, viser_server)
    device = env.unwrapped.device
    model = load_traversability_estimator(estimator_path, map_location=device).to(device)
    model.eval()
    layout = _term_layout(env.unwrapped.observation_manager)
    self._risk_panel = _RiskPanel(env, model, layout, device)

  def setup(self) -> None:
    super().setup()
    tabs = self._server.gui.add_tab_group()
    self._risk_panel.setup(self._server, tabs)

  def _execute_step(self) -> bool:
    try:
      with torch.no_grad():
        obs = self.env.get_observations()
      self._risk_panel.maybe_compute(obs, self._scene.env_idx)
      with torch.no_grad():
        actions = self.policy(obs)
        self.env.step(actions)
      self._step_count += 1
      self._stats_steps += 1
      return True
    except Exception:
      self._last_error = traceback.format_exc()
      self.log(f"[ERROR] Exception during step:\n{self._last_error}", VerbosityLevel.SILENT)
      self.pause()
      return False

  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    self._risk_panel.update_display(
      self._server, self._scene.env_idx, self._scene._scene_offset
    )


@dataclass
class _Handles:
  gauge: viser.GuiHtmlHandle
  sparkline: viser.GuiHtmlHandle
  heatmap: viser.GuiImageHandle
  status: viser.GuiHtmlHandle


class _RiskPanel:
  def __init__(self, env, model, layout: list[tuple[str, slice]], device) -> None:
    self._env = env
    self._model = model
    self._layout = {name: sl for name, sl in layout}
    self._keys = list(model.encoder_input_keys)
    self._device = device

    missing = [k for k in self._keys if k not in self._layout]
    if missing:
      raise KeyError(
        f"Estimator needs observation terms {missing} that are not in the policy's "
        f"'{ACTOR_GROUP}' group {list(self._layout)}. The play task likely does not "
        "match the task the estimator was trained on."
      )
    self._has_spatial = int(model.spatial_grid[0]) * int(model.spatial_grid[1]) > 0
    self._cells_b = self._build_cell_grid()  # [P, 2] robot-frame cell centres

    self._history: deque[float] = deque(maxlen=200)
    self._latest_risk: float | None = None
    self._latest_map: np.ndarray | None = None  # [NW, NH]
    self._step_counter = 0
    self._last_error: str | None = None

    # GUI handles (created in setup()).
    self._enabled = None
    self._show_markers = None
    self._every_n = None
    self._handles: _Handles | None = None
    self._marker_handle = None

  # -- cell grid (robot frame), mirrors build_traversability_labels -----------
  def _build_cell_grid(self) -> np.ndarray:
    nw, nh = (int(x) for x in self._model.spatial_grid)
    w, hm = (float(x) for x in self._model.spatial_size_m)
    res_w, res_h = w / max(nw, 1), hm / max(nh, 1)
    xs = (np.arange(nw) + 0.5) * res_w - w / 2.0
    ys = (np.arange(nh) + 0.5) * res_h - hm / 2.0
    cx, cy = np.meshgrid(xs, ys, indexing="ij")  # [NW, NH]
    return np.stack([cx.reshape(-1), cy.reshape(-1)], axis=1)  # [P, 2]

  # -- setup ------------------------------------------------------------------
  def setup(self, server: viser.ViserServer, tabs: viser.GuiTabGroupHandle) -> None:
    with tabs.add_tab("Traversability", icon=viser.Icon.ALERT_TRIANGLE):
      with server.gui.add_folder("Controls", expand_by_default=True):
        self._enabled = server.gui.add_checkbox(
          "Enabled", initial_value=True, hint="Run the estimator during playback."
        )
        self._show_markers = server.gui.add_checkbox(
          "Risk markers in 3D",
          initial_value=self._has_spatial,
          disabled=not self._has_spatial,
          hint="Draw the spatial risk map as colored markers on the terrain.",
        )
        self._every_n = server.gui.add_number(
          "Every N steps", initial_value=1, min=1, step=1,
          hint="Run the estimator every N simulation steps.",
        )
      gauge = server.gui.add_html("")
      sparkline = server.gui.add_html("")
      server.gui.add_markdown("<small>Spatial risk — front +x at top, left +y at left.</small>")
      heatmap = server.gui.add_image(_EMPTY_HEATMAP, label="Spatial risk map")
      status = server.gui.add_html("")
      self._handles = _Handles(gauge=gauge, sparkline=sparkline, heatmap=heatmap, status=status)

    if self._has_spatial:
      self._marker_handle = server.scene.add_point_cloud(
        "/traversability_risk",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        point_size=0.045,
        point_shape="circle",
      )

  # -- per-step inference -----------------------------------------------------
  def maybe_compute(self, obs, env_idx: int) -> None:
    if self._enabled is None or not self._enabled.value:
      return
    self._step_counter += 1
    every_n = max(1, int(self._every_n.value)) if self._every_n is not None else 1
    if self._step_counter % every_n != 0:
      return
    try:
      actor = obs[ACTOR_GROUP][env_idx]  # [A]
      groups = {k: actor[self._layout[k]].unsqueeze(0) for k in self._keys}
      with torch.no_grad():
        self._latest_risk = float(self._model.predict_proba(groups)[0].item())
        if self._has_spatial:
          self._latest_map = (
            self._model.predict_spatial_proba(groups)[0].detach().cpu().numpy()
          )
      self._history.append(self._latest_risk)
      self._last_error = None
    except Exception:
      self._last_error = traceback.format_exc().strip().splitlines()[-1]

  # -- display ----------------------------------------------------------------
  def update_display(self, server, env_idx: int, scene_offset) -> None:
    if self._handles is None:
      return
    if self._latest_risk is not None:
      self._handles.gauge.content = _gauge_html(self._latest_risk)
      self._handles.sparkline.content = _sparkline_html(self._history)
    if self._has_spatial and self._latest_map is not None:
      self._handles.heatmap.image = _render_risk_map(self._latest_map)
      self._update_markers(server, env_idx, scene_offset)
    if self._last_error:
      self._handles.status.content = f"<small>Risk error: {html.escape(self._last_error)}</small>"
    else:
      self._handles.status.content = (
        f"<small>scalar head{' + spatial head' if self._has_spatial else ''}; "
        f"grid {tuple(self._model.spatial_grid)} over {tuple(self._model.spatial_size_m)} m</small>"
      )

  def _update_markers(self, server, env_idx: int, scene_offset) -> None:
    show = self._show_markers is None or self._show_markers.value
    if not show:
      if self._marker_handle is not None:
        self._marker_handle.visible = False
      return
    if self._latest_map is None:
      return
    robot = self._env.unwrapped.scene["robot"]
    pos = robot.data.root_link_pos_w[env_idx].detach().cpu().numpy()  # [3]
    quat = robot.data.root_link_quat_w[env_idx].detach().cpu().numpy()  # wxyz
    yaw = _quat_to_yaw(quat)
    cos, sin = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos, -sin], [sin, cos]])
    world_xy = pos[:2] + self._cells_b @ rot.T  # [P, 2]
    offset = np.asarray(scene_offset, dtype=np.float64).reshape(3)
    z = np.full((world_xy.shape[0],), pos[2] - 0.30) + offset[2]
    points = np.stack(
      [world_xy[:, 0] + offset[0], world_xy[:, 1] + offset[1], z], axis=1
    ).astype(np.float32)
    colors = _colormap(self._latest_map.reshape(-1))
    # PointCloudHandle has no settable points/colors; re-adding by the same name
    # replaces the node in place.
    self._marker_handle = server.scene.add_point_cloud(
      "/traversability_risk",
      points=points,
      colors=colors,
      point_size=0.045,
      point_shape="circle",
    )


def _gauge_html(risk: float) -> str:
  pct = max(0.0, min(1.0, risk)) * 100.0
  rgb = _colormap(np.array([risk]))[0]
  color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
  return (
    f"<div style='font-size:0.8em;color:#888;'>P(failure soon)</div>"
    f"<div style='font-size:1.9em;font-weight:700;font-variant-numeric:tabular-nums;"
    f"color:{color};line-height:1.1;'>{risk:.2f}</div>"
    f"<div style='height:0.6em;background:#eceff3;border-radius:3px;overflow:hidden;margin-top:0.2em;'>"
    f"<div style='height:100%;width:{pct:.1f}%;background:{color};'></div></div>"
  )


def _sparkline_html(history: deque[float]) -> str:
  if not history:
    return ""
  vals = list(history)
  n = len(vals)
  w, h = 240.0, 44.0
  pts = " ".join(
    f"{(i / max(n - 1, 1)) * w:.1f},{h - max(0.0, min(1.0, v)) * h:.1f}"
    for i, v in enumerate(vals)
  )
  thr_y = h - 0.5 * h
  return (
    f"<svg width='{w:.0f}' height='{h:.0f}' style='background:#f5f6f8;border-radius:3px;'>"
    f"<line x1='0' y1='{thr_y:.0f}' x2='{w:.0f}' y2='{thr_y:.0f}' "
    f"stroke='#bbb' stroke-dasharray='3,3'/>"
    f"<polyline fill='none' stroke='#d84636' stroke-width='1.5' points='{pts}'/>"
    f"</svg><div style='font-size:0.72em;color:#888;'>risk over last {n} steps "
    f"(dashed = 0.5)</div>"
  )


def _render_risk_map(risk_map: np.ndarray) -> np.ndarray:
  """Render the ``[NW, NH]`` risk map to an upscaled RGB image (front +x at top)."""
  grid = np.flipud(np.fliplr(risk_map.T))  # orient: front +x up, left +y left
  rgb = _colormap(np.clip(grid, 0.0, 1.0))
  rgb = np.repeat(np.repeat(rgb, 10, axis=0), 10, axis=1)
  rgb[::10, :, :] = 32
  rgb[:, ::10, :] = 32
  return rgb
