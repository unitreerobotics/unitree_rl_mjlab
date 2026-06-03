"""Viser viewer extension that renders live observation attribution."""

from __future__ import annotations

import html
import time
import traceback
from dataclasses import dataclass

import numpy as np
import torch
import viser

from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.base import VerbosityLevel

from src.viz.attribution import (
  AttributionMapResult,
  AttributionTerm,
  AttributionMethodName,
  DeepLiftRescale,
  DeepShap,
  GradientInput,
  GradientSaliency,
  IntegratedGradients,
  ObservationAttributionComputer,
)


_EMPTY_HEATMAP = np.zeros((170, 110, 3), dtype=np.uint8)
_METHOD_OPTIONS: dict[str, AttributionMethodName] = {
  "Integrated Gradients": "integrated_gradients",
  "Gradient Saliency": "gradient_saliency",
  "Gradient x Input": "gradient_input",
  "DeepLIFT Rescale": "deep_lift_rescale",
  "DeepSHAP": "deep_shap",
}


@dataclass
class _MapHandles:
  heatmap: viser.GuiImageHandle
  bars: viser.GuiHtmlHandle
  status: viser.GuiHtmlHandle


class AttributionViserPlayViewer(ViserPlayViewer):
  """Viser playback viewer with an attribution tab."""

  def __init__(
    self,
    env,
    policy,
    *,
    actor: torch.nn.Module,
    critic: torch.nn.Module | None = None,
    attribution_method: AttributionMethodName = "integrated_gradients",
    frame_rate: float = 60.0,
    verbosity: VerbosityLevel = VerbosityLevel.SILENT,
    viser_server: viser.ViserServer | None = None,
  ) -> None:
    super().__init__(env, policy, frame_rate, verbosity, viser_server)
    self._attribution_panel = _AttributionPanel(
      env, actor, critic, attribution_method
    )

  def setup(self) -> None:
    super().setup()
    tabs = self._server.gui.add_tab_group()
    self._attribution_panel.setup(self._server, tabs)

  def _execute_step(self) -> bool:
    """Run one playback step and compute attribution from the same observation."""
    try:
      with torch.no_grad():
        obs = self.env.get_observations()

      self._attribution_panel.maybe_compute(obs, self._scene.env_idx)

      with torch.no_grad():
        actions = self.policy(obs)
        self.env.step(actions)
        self._step_count += 1
        self._stats_steps += 1
        return True
    except Exception:
      self._last_error = traceback.format_exc()
      self.log(
        f"[ERROR] Exception during step:\n{self._last_error}",
        VerbosityLevel.SILENT,
      )
      self.pause()
      return False

  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    self._attribution_panel.update_display()


class _AttributionPanel:
  def __init__(
    self,
    env,
    actor: torch.nn.Module,
    critic: torch.nn.Module | None,
    attribution_method: AttributionMethodName,
  ) -> None:
    self._env = env
    self._actor = actor
    self._critic = critic
    self._computer = ObservationAttributionComputer(
      env.unwrapped.observation_manager, attribution_method
    )

    self._enabled = None
    self._show_action = None
    self._show_value = None
    self._method = None
    self._scale_mode = None
    self._ig_steps = None
    self._deep_shap_samples = None
    self._every_n_steps = None
    self._status = None
    self._map_handles: dict[str, _MapHandles] = {}
    self._latest: dict[str, AttributionMapResult] = {}
    self._rolling_max: dict[str, float] = {}
    self._selected_method = _method_label_from_key(attribution_method)
    self._step_counter = 0
    self._last_compute_ms = 0.0
    self._last_error: str | None = None

  def setup(self, server: viser.ViserServer, tabs: viser.GuiTabGroupHandle) -> None:
    with tabs.add_tab("Attribution", icon=viser.Icon.CHART_BAR):
      with server.gui.add_folder("Controls", expand_by_default=True):
        self._enabled = server.gui.add_checkbox(
          "Enabled",
          initial_value=True,
          hint="Compute attribution during playback.",
        )
        self._show_action = server.gui.add_checkbox(
          "Action map",
          initial_value=True,
          hint="Attribution of the action-vector norm.",
        )
        self._show_value = server.gui.add_checkbox(
          "Value map",
          initial_value=self._critic is not None,
          disabled=self._critic is None,
          hint="Attribution of critic value.",
        )
        self._method = server.gui.add_dropdown(
          "Attribution method",
          options=tuple(_METHOD_OPTIONS.keys()),
          initial_value=self._selected_method,
        )
        self._scale_mode = server.gui.add_dropdown(
          "Color scale",
          options=("Per-step max", "Rolling max"),
          initial_value="Rolling max",
        )
        self._ig_steps = server.gui.add_number(
          "IG steps",
          initial_value=16,
          min=2,
          step=1,
          hint="Number of Integrated Gradients samples from baseline to current observation.",
        )
        self._deep_shap_samples = server.gui.add_number(
          "DeepSHAP samples",
          initial_value=16,
          min=1,
          step=1,
          hint="Number of background references for DeepSHAP-style attribution.",
        )
        self._every_n_steps = server.gui.add_number(
          "Every N steps",
          initial_value=4,
          min=1,
          step=1,
          hint="Compute attribution every N simulation steps.",
        )
        self._status = server.gui.add_html("")

      self._map_handles["action"] = self._create_map_section(server, "Action")
      self._map_handles["value"] = self._create_map_section(server, "Value")
      self._sync_visibility()

      for handle in (self._enabled, self._show_action, self._show_value):
        handle.on_update(lambda _: self._sync_visibility())
      if self._method is not None:
        self._method.on_update(lambda _: self._sync_method_controls())
      self._sync_method_controls()

      if self._critic is None:
        self._set_status("Value map disabled: critic weights were not loaded.")
      else:
        self._set_status("Waiting for playback step.")

  def maybe_compute(self, obs, env_idx: int) -> None:
    if self._enabled is None or not self._enabled.value:
      return

    self._step_counter += 1
    every_n = 1
    if self._every_n_steps is not None:
      every_n = max(1, int(self._every_n_steps.value))
    if self._step_counter % every_n != 0:
      return

    compute_action = self._show_action is not None and self._show_action.value
    compute_value = (
      self._critic is not None
      and self._show_value is not None
      and self._show_value.value
    )
    if not compute_action and not compute_value:
      return

    start = time.perf_counter()
    try:
      self._sync_method_controls()
      if compute_action:
        self._latest["action"] = self._computer.compute_action(
          self._actor, obs, env_idx
        )
      if compute_value and self._critic is not None:
        self._latest["value"] = self._computer.compute_value(
          self._critic, obs, env_idx
        )
      self._last_error = None
    except Exception:
      self._last_error = traceback.format_exc().strip().splitlines()[-1]
    finally:
      self._last_compute_ms = (time.perf_counter() - start) * 1000.0

  def update_display(self) -> None:
    self._sync_visibility()
    for map_name, result in self._latest.items():
      handles = self._map_handles.get(map_name)
      if handles is None:
        continue
      scale = self._scale_for(result)
      height_scan = result.get_term("height_scan")
      handles.heatmap.image = (
        _render_height_scan(height_scan, scale) if height_scan else _EMPTY_HEATMAP
      )
      handles.bars.content = _render_bars_html(result, scale)
      handles.status.content = _map_status_html(result, scale)

    if self._last_error:
      self._set_status(f"Attribution error: {html.escape(self._last_error)}")
    else:
      every_n = int(self._every_n_steps.value) if self._every_n_steps else 4
      self._set_status(
        f"{_method_status(self._computer.method)}; "
        f"every {every_n} sim steps; "
        f"last pass {self._last_compute_ms:.1f} ms"
      )

  def _create_map_section(
    self, server: viser.ViserServer, title: str
  ) -> _MapHandles:
    with server.gui.add_folder(title, expand_by_default=(title == "Action")):
      status = server.gui.add_html("")
      server.gui.add_markdown(
        "<small>Height scan: front +x at top, left +y at left.</small>"
      )
      heatmap = server.gui.add_image(_EMPTY_HEATMAP, label=f"{title} height scan")
      bars = server.gui.add_html("")
      return _MapHandles(heatmap=heatmap, bars=bars, status=status)

  def _sync_visibility(self) -> None:
    action_visible = self._show_action is not None and self._show_action.value
    value_visible = self._show_value is not None and self._show_value.value
    for key, visible in (("action", action_visible), ("value", value_visible)):
      handles = self._map_handles.get(key)
      if handles is None:
        continue
      handles.heatmap.visible = visible
      handles.bars.visible = visible
      handles.status.visible = visible

  def _scale_for(self, result: AttributionMapResult) -> float:
    current = max(result.max_score, 1.0e-12)
    if self._scale_mode is not None and self._scale_mode.value == "Per-step max":
      return current
    previous = self._rolling_max.get(result.name, 0.0)
    rolling = max(previous * 0.98, current)
    self._rolling_max[result.name] = rolling
    return max(rolling, 1.0e-12)

  def _set_status(self, message: str) -> None:
    if self._status is not None:
      self._status.content = f"<small>{message}</small>"

  def _sync_method_controls(self) -> None:
    if self._method is not None and self._method.value != self._selected_method:
      self._selected_method = self._method.value
      self._computer.set_method(_METHOD_OPTIONS[self._selected_method])
      self._latest.clear()
      self._rolling_max.clear()

    method = self._computer.method
    if isinstance(method, IntegratedGradients) and self._ig_steps is not None:
      method.steps = max(2, int(self._ig_steps.value))
    if isinstance(method, DeepShap) and self._deep_shap_samples is not None:
      method.samples = max(1, int(self._deep_shap_samples.value))

    if self._ig_steps is not None:
      self._ig_steps.visible = isinstance(method, IntegratedGradients)
    if self._deep_shap_samples is not None:
      self._deep_shap_samples.visible = isinstance(method, DeepShap)


def _render_height_scan(term: AttributionTerm | None, scale: float) -> np.ndarray:
  if term is None or term.values.size == 0:
    return _EMPTY_HEATMAP

  values = term.values.astype(np.float64, copy=False)
  if values.size == 187:
    grid = values.reshape(11, 17).T
    grid = np.flipud(np.fliplr(grid))
  else:
    side = int(np.ceil(np.sqrt(values.size)))
    grid = np.zeros((side, side), dtype=np.float64)
    grid.reshape(-1)[: values.size] = values

  normalized = np.clip(grid / max(scale, 1.0e-12), 0.0, 1.0)
  rgb = _colormap(normalized)
  rgb = np.repeat(np.repeat(rgb, 10, axis=0), 10, axis=1)

  rgb[::10, :, :] = 32
  rgb[:, ::10, :] = 32
  return rgb


def _colormap(values: np.ndarray) -> np.ndarray:
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


def _render_bars_html(result: AttributionMapResult, scale: float) -> str:
  terms = [term for term in result.terms if term.name != "height_scan"]
  if not terms:
    return "<small>No non-height-scan observation terms.</small>"

  max_term = max((term.mean_score for term in terms), default=1.0e-12)
  max_term = max(max_term, 1.0e-12)
  rows = []
  for term in terms:
    width = 100.0 * term.mean_score / max_term
    color_value = min(term.max_score / max(scale, 1.0e-12), 1.0)
    rgb = _colormap(np.array([[color_value]], dtype=np.float64))[0, 0]
    color = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
    rows.append(
      """
      <div style="display:grid;grid-template-columns:9.5em 1fr 6.5em;gap:0.5em;
                  align-items:center;margin:0.18em 0;font-size:0.82em;">
        <div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
             title="{name}">{name}</div>
        <div style="height:0.75em;background:#eceff3;border-radius:2px;overflow:hidden;">
          <div style="height:100%;width:{width:.1f}%;background:{color};"></div>
        </div>
        <div style="font-variant-numeric:tabular-nums;text-align:right;">{score:.3g}</div>
      </div>
      """.format(
        name=html.escape(term.name),
        width=width,
        color=color,
        score=term.mean_score,
      )
    )
  return "<div>" + "\n".join(rows) + "</div>"


def _map_status_html(result: AttributionMapResult, scale: float) -> str:
  height = result.get_term("height_scan")
  height_text = "no height_scan"
  if height is not None:
    height_text = f"height_scan max {height.max_score:.3g}"
  return (
    f"<small>{html.escape(result.group_name)} observations, "
    f"scale {scale:.3g}, {height_text}</small>"
  )


def _method_label_from_key(method: AttributionMethodName) -> str:
  for label, key in _METHOD_OPTIONS.items():
    if key == method:
      return label
  return "Integrated Gradients"


def _method_status(method) -> str:
  if isinstance(method, IntegratedGradients):
    return f"Integrated Gradients from normalizer mean; {method.steps} steps"
  if isinstance(method, DeepLiftRescale):
    return "DeepLIFT Rescale from normalizer mean"
  if isinstance(method, DeepShap):
    return f"DeepSHAP-style background attribution; {method.samples} samples"
  if isinstance(method, GradientInput):
    return "Gradient x Input"
  if isinstance(method, GradientSaliency):
    return "Gradient Saliency"
  return method.__class__.__name__
