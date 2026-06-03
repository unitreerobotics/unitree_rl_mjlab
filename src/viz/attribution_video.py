"""Video recorder that composites playback frames with attribution maps."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tensordict import TensorDict
from PIL import Image, ImageDraw, ImageFont

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.wrappers import VideoRecorder

from src.viz.attribution import (
  AttributionMapResult,
  AttributionMethodName,
  AttributionTerm,
  ObservationAttributionComputer,
)


_PADDING = 18
_ROW_HEIGHT = 28
_PANEL_BG = (248, 250, 252)
_TEXT = (20, 24, 31)
_MUTED = (88, 96, 111)
_TRACK = (226, 232, 240)
_EMPTY_HEATMAP = np.zeros((170, 110, 3), dtype=np.uint8)


class AttributionVideoRecorder(VideoRecorder):
  """Record a side-by-side simulation and action-attribution MP4."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    video_folder: str | Path,
    episode_trigger: Callable[[int], bool] | None = None,
    step_trigger: Callable[[int], bool] | None = None,
    video_length: int | None = None,
    disable_logger: bool = False,
  ) -> None:
    super().__init__(
      env,
      video_folder=video_folder,
      episode_trigger=episode_trigger,
      step_trigger=step_trigger,
      video_length=video_length,
      name_prefix="rl-video-attribution",
      disable_logger=disable_logger,
    )
    self._actor: torch.nn.Module | None = None
    self._computer: ObservationAttributionComputer | None = None
    self._latest: AttributionMapResult | None = None
    self._rolling_max = 0.0
    self._last_error: str | None = None
    self._method_name: AttributionMethodName = "integrated_gradients"

  def configure_attribution(
    self,
    *,
    actor: torch.nn.Module,
    attribution_method: AttributionMethodName,
  ) -> None:
    self._actor = actor
    self._method_name = attribution_method
    self._computer = ObservationAttributionComputer(
      self.unwrapped.observation_manager,
      attribution_method,
    )

  def _record_frame(self) -> None:
    if self._wrapped_env.render_mode != "rgb_array":
      return

    frame = self._wrapped_env.render()
    if frame is None:
      return

    rgb_frame = frame[0] if isinstance(frame, np.ndarray) and frame.ndim == 4 else frame
    rgb_frame = _to_uint8(rgb_frame)
    self.current_video_frames.append(self._compose_frame(rgb_frame))

  def _compose_frame(self, rgb_frame: np.ndarray) -> np.ndarray:
    result = self._compute_action_attribution()
    panel = render_action_attribution_panel(
      result,
      height=rgb_frame.shape[0],
      method_name=self._method_name,
      error=self._last_error,
      rolling_scale=self._scale_for(result) if result is not None else None,
    )
    return _ensure_even_dimensions(np.concatenate((rgb_frame, panel), axis=1))

  def _compute_action_attribution(self) -> AttributionMapResult | None:
    if self._actor is None or self._computer is None:
      self._last_error = "Attribution recorder was not configured."
      return self._latest

    try:
      with torch.no_grad():
        obs_dict = self.unwrapped.observation_manager.compute()
      obs = TensorDict(obs_dict, batch_size=[self.unwrapped.num_envs])
      self._latest = self._computer.compute_action(self._actor, obs, env_idx=0)
      self._last_error = None
    except Exception:
      self._last_error = traceback.format_exc().strip().splitlines()[-1]
    return self._latest

  def _scale_for(self, result: AttributionMapResult | None) -> float:
    if result is None:
      return 1.0
    current = max(result.max_score, 1.0e-12)
    self._rolling_max = max(self._rolling_max * 0.98, current)
    return max(self._rolling_max, 1.0e-12)


def render_action_attribution_panel(
  result: AttributionMapResult | None,
  *,
  height: int,
  method_name: str,
  error: str | None = None,
  rolling_scale: float | None = None,
) -> np.ndarray:
  """Render a fixed-height attribution side panel as RGB uint8 pixels."""
  width = max(360, int(height * 0.72))
  image = Image.new("RGB", (width, height), _PANEL_BG)
  draw = ImageDraw.Draw(image)
  font = ImageFont.load_default()

  y = _PADDING
  draw.text((_PADDING, y), "Action attribution", fill=_TEXT, font=font)
  y += 18
  draw.text((_PADDING, y), method_name.replace("_", " "), fill=_MUTED, font=font)
  y += 22

  if error:
    y = _draw_wrapped(draw, f"Attribution error: {error}", _PADDING, y, width, font)
    y += 8

  if result is None:
    _draw_wrapped(draw, "Waiting for attribution data.", _PADDING, y, width, font)
    return np.asarray(image, dtype=np.uint8)

  scale = max(rolling_scale or result.max_score, 1.0e-12)
  height_scan = result.get_term("height_scan")
  if height_scan is not None:
    heatmap = _render_height_scan(height_scan, scale)
    heatmap_h = max(56, min(180, height - y - 110))
    heatmap_w = min(150, width - 2 * _PADDING)
    heatmap_image = Image.fromarray(heatmap).resize((heatmap_w, heatmap_h))
    image.paste(heatmap_image, (_PADDING, y))
    draw.text(
      (_PADDING + heatmap_image.width + 14, y),
      f"scale {scale:.3g}",
      fill=_MUTED,
      font=font,
    )
    y += heatmap_image.height + 12
  else:
    draw.text((_PADDING, y), f"scale {scale:.3g}", fill=_MUTED, font=font)
    y += 16

  terms = [term for term in result.terms if term.name != "height_scan"]
  if not terms:
    _draw_wrapped(draw, "No non-height-scan observation terms.", _PADDING, y, width, font)
    return np.asarray(image, dtype=np.uint8)

  max_term = max(max(term.mean_score for term in terms), 1.0e-12)
  row_height = max(13, min(_ROW_HEIGHT, (height - y - _PADDING) // len(terms)))
  bar_left = _PADDING + 116
  value_right = width - _PADDING
  bar_right = value_right - 64
  for term in terms:
    label = _truncate(term.name, 18)
    text_y = y + max(1, (row_height - 11) // 2)
    bar_y0 = y + max(3, (row_height - 9) // 2)
    bar_y1 = min(y + row_height - 2, bar_y0 + 9)
    draw.text((_PADDING, text_y), label, fill=_TEXT, font=font)
    draw.rounded_rectangle(
      (bar_left, bar_y0, bar_right, bar_y1),
      radius=2,
      fill=_TRACK,
    )
    width_frac = float(term.mean_score / max_term)
    color_value = min(term.max_score / scale, 1.0)
    color = tuple(int(v) for v in _colormap(np.array([[color_value]]))[0, 0])
    draw.rounded_rectangle(
      (bar_left, bar_y0, bar_left + (bar_right - bar_left) * width_frac, bar_y1),
      radius=2,
      fill=color,
    )
    draw.text((bar_right + 8, text_y), f"{term.mean_score:.2g}", fill=_MUTED, font=font)
    y += row_height

  return np.asarray(image, dtype=np.uint8)


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


def _to_uint8(frame: np.ndarray) -> np.ndarray:
  frame = np.asarray(frame)
  if frame.ndim == 3 and frame.shape[2] > 3:
    frame = frame[:, :, :3]
  if frame.dtype == np.uint8:
    return frame
  return (np.clip(frame, 0, 1) * 255).astype(np.uint8)


def _ensure_even_dimensions(frame: np.ndarray) -> np.ndarray:
  pad_h = frame.shape[0] % 2
  pad_w = frame.shape[1] % 2
  if not pad_h and not pad_w:
    return frame
  return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _truncate(text: str, max_chars: int) -> str:
  if len(text) <= max_chars:
    return text
  return text[: max_chars - 1] + "."


def _draw_wrapped(
  draw: ImageDraw.ImageDraw,
  text: str,
  x: int,
  y: int,
  width: int,
  font: ImageFont.ImageFont,
) -> int:
  max_chars = max(18, (width - 2 * x) // 7)
  words = text.split()
  line = ""
  for word in words:
    candidate = f"{line} {word}".strip()
    if len(candidate) > max_chars and line:
      draw.text((x, y), line, fill=_MUTED, font=font)
      y += 14
      line = word
    else:
      line = candidate
  if line:
    draw.text((x, y), line, fill=_MUTED, font=font)
    y += 14
  return y
