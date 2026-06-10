"""Multi-view video recording for policy evaluation.

The training-time :class:`mjlab.utils.wrappers.VideoRecorder` records a single
camera (the env viewer config). For evaluation we want to see the robot from
several angles, so this module drives the env's offscreen renderer directly and
re-renders each simulation frame from multiple orbit azimuths around the tracked
body.

The viewer uses an ``ASSET_BODY`` tracking camera, so ``azimuth`` simply sets the
orbit angle around the robot; ``elevation``/``distance`` are shared across views.
Views are expressed as azimuth offsets relative to the configured base azimuth
(the existing "front" view), so they stay consistent if the base view changes.

Frames are streamed directly to disk via ``mediapy.VideoWriter`` (one ffmpeg
subprocess per view, lazily opened on the first frame) rather than buffered in
Python lists. This avoids the ~110 GB RAM blowup that buffering causes for a
3-view 1080p episode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mediapy as media
import numpy as np

from mjlab.envs import ManagerBasedRlEnv

# (view_name, azimuth_offset_degrees) relative to the configured base azimuth.
DEFAULT_EVAL_VIEWS: tuple[tuple[str, float], ...] = (
  ("front", 0.0),
  ("side", 90.0),
  ("behind", 180.0),
)


class MultiViewVideoRecorder:
  """Records the tracked robot from several camera azimuths in one rollout.

  Each :meth:`capture` re-renders the current simulation state from every
  configured view by re-pointing the offscreen renderer's camera.  Frames are
  streamed to disk immediately via ``mediapy.VideoWriter`` (one ffmpeg process
  per view, lazily spawned on the first frame) so memory usage stays constant
  regardless of episode length.

  The "front" view (zero azimuth offset) is written as ``<prefix>.mp4`` to keep
  backward compatibility with the previous single-view output; other views are
  written as ``<prefix>_<view>.mp4``.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    run_dir: str | Path,
    *,
    views: Sequence[tuple[str, float]] = DEFAULT_EVAL_VIEWS,
    name_prefix: str = "video",
    fps: float = 50.0,
  ) -> None:
    renderer = getattr(env, "_offline_renderer", None)
    if renderer is None:
      raise ValueError(
        "Environment has no offscreen renderer; render_mode must be 'rgb_array' "
        "to record evaluation video."
      )
    self._env = env
    self._renderer = renderer
    self._run_dir = Path(run_dir)
    self._name_prefix = name_prefix
    self._fps = fps
    self._debug_cb = getattr(env, "update_visualizers", None)

    base_azimuth = float(env.cfg.viewer.azimuth)

    # Precompute (name, azimuth, filename, path) for each view.
    self._views: list[tuple[str, float, str, Path]] = []
    for name, offset in views:
      filename = (
        f"{name_prefix}.mp4" if name == "front" else f"{name_prefix}_{name}.mp4"
      )
      path = self._run_dir / filename
      self._views.append((name, base_azimuth + offset, filename, path))

    # One writer slot per view; opened lazily on the first frame.
    self._writers: dict[str, media.VideoWriter | None] = {
      name: None for name, _, _, _ in self._views
    }

  def capture(self) -> None:
    """Render the current sim state from every view and stream one frame to disk."""
    renderer = self._renderer
    data = self._env.sim.data
    original_azimuth = renderer._cam.azimuth
    try:
      for name, azimuth, _filename, path in self._views:
        renderer._cam.azimuth = azimuth
        renderer.update(data, debug_vis_callback=self._debug_cb)
        frame = renderer.render()
        if frame is None:
          continue
        if isinstance(frame, np.ndarray) and frame.ndim == 4:
          frame = frame[0]
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
          frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        # Lazily open the writer on the first frame for this view.
        if self._writers[name] is None:
          h, w = frame.shape[:2]
          vw = media.VideoWriter(path, shape=(h, w), fps=self._fps)
          vw.__enter__()  # spawns the ffmpeg subprocess
          self._writers[name] = vw
        self._writers[name].add_image(frame)
    finally:
      renderer._cam.azimuth = original_azimuth

  def save(self) -> list[Path]:
    """Close all open writers and return the list of Paths that received frames."""
    written: list[Path] = []
    for name, _azimuth, _filename, path in self._views:
      vw = self._writers[name]
      if vw is not None:
        vw.close()
        self._writers[name] = None
        written.append(path)
    return written
