#!/usr/bin/env python3
"""Build tiled evaluation videos and annotated outcome images.

The expected input layout is the one produced by scripts/run_eval.sh:

  <base>/<checkpoint_label>/<terrain>/run_000/video_side.mp4
  <base>/<checkpoint_label>/<terrain>/run_000/events.json

Outputs are written to <base>/couple.mp4, <base>/result.jpg, and
<base>/result_annotated.jpg.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_TERRAINS = [
  "perlin_noise_corridor",
  "random_spread_boxes_corridor",
  "rough_curriculum_corridor",
]
DEFAULT_CELL_WIDTH = 640
DEFAULT_CELL_HEIGHT = 360
DEFAULT_MAX_OUTPUT_WIDTH = 3840
DEFAULT_TARGET_SECONDS = 60
DEFAULT_FPS = 50
PREFERRED_ENCODER_ORDER = [
  "conv1d",
  "conv1d_state",
  "conv2d",
  "conv2d_state",
  "mlp",
  "mlp_state",
  "raw",
  "ae",
]

OUTCOME_COLORS = {
  "goal_reached": (34, 160, 60),
  "illegal_contact": (200, 40, 40),
  "fell_over": (200, 40, 40),
  "stuck": (210, 130, 20),
  "max_steps": (120, 120, 120),
  "done": (160, 60, 160),
}
DEFAULT_COLOR = (90, 90, 90)


@dataclass(frozen=True)
class Cell:
  video_path: Path
  events_path: Path
  encoder_label: str
  terrain: str
  frame_count: int

  @property
  def terrain_label(self) -> str:
    return self.terrain.removesuffix("_corridor")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
  return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _frame_count(path: Path) -> int:
  result = _run([
    "ffprobe",
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-show_entries",
    "stream=nb_frames",
    "-of",
    "csv=p=0",
    str(path),
  ])
  if result.returncode != 0:
    raise RuntimeError(f"ffprobe failed for {path}:\n{result.stderr[-2000:]}")
  raw = result.stdout.strip()
  if raw.isdigit():
    return int(raw)

  result = _run([
    "ffprobe",
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-count_frames",
    "-show_entries",
    "stream=nb_read_frames",
    "-of",
    "csv=p=0",
    str(path),
  ])
  if result.returncode != 0:
    raise RuntimeError(f"ffprobe failed for {path}:\n{result.stderr[-2000:]}")
  raw = result.stdout.strip()
  if not raw.isdigit():
    raise RuntimeError(f"ffprobe did not return a frame count for {path}: {raw!r}")
  return int(raw)


def _ffmpeg_text(text: str) -> str:
  return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "")


def _encoder_label(checkpoint_label: str) -> str:
  match = re.match(r"^go2_velocity_encoder_(.+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", checkpoint_label)
  if match:
    return match.group(1)
  return checkpoint_label


def _encoder_sort_key(path: Path) -> tuple[int, str]:
  label = _encoder_label(path.name)
  try:
    return (PREFERRED_ENCODER_ORDER.index(label), label)
  except ValueError:
    return (len(PREFERRED_ENCODER_ORDER), label)


def _even_at_least(value: int, minimum: int = 2) -> int:
  value = value if value % 2 == 0 else value - 1
  return max(minimum, value)


def _effective_cell_size(cols: int, cell_width: int, cell_height: int, max_output_width: int) -> tuple[int, int]:
  if max_output_width <= 0 or cols * cell_width <= max_output_width:
    return cell_width, cell_height

  scale = max_output_width / (cols * cell_width)
  return _even_at_least(int(cell_width * scale)), _even_at_least(int(cell_height * scale))


def _checkpoint_dirs(base: Path, terrains: list[str]) -> list[Path]:
  dirs = [
    p for p in base.iterdir()
    if p.is_dir() and not p.name.startswith("_") and any((p / terrain).is_dir() for terrain in terrains)
  ]
  return sorted(dirs, key=_encoder_sort_key)


def _collect_cells(base: Path, terrains: list[str]) -> tuple[list[Cell], list[Path]]:
  checkpoint_dirs = _checkpoint_dirs(base, terrains)
  if not checkpoint_dirs:
    raise FileNotFoundError(f"No checkpoint result directories found under {base}.")

  cells: list[Cell] = []
  missing: list[str] = []
  for terrain in terrains:
    for checkpoint_dir in checkpoint_dirs:
      video_path = checkpoint_dir / terrain / "run_000" / "video_side.mp4"
      events_path = checkpoint_dir / terrain / "run_000" / "events.json"
      if not video_path.exists():
        missing.append(str(video_path))
        continue
      frame_count = _frame_count(video_path)
      encoder_label = _encoder_label(checkpoint_dir.name)
      cells.append(Cell(
        video_path=video_path,
        events_path=events_path,
        encoder_label=encoder_label,
        terrain=terrain,
        frame_count=frame_count,
      ))
      print(f"frames={frame_count:5}  {terrain:28} {encoder_label}")

  if missing:
    formatted = "\n".join(f"  {path}" for path in missing)
    raise FileNotFoundError(f"Missing expected side-view videos:\n{formatted}")
  return cells, checkpoint_dirs


def _build_grid_video(
  cells: list[Cell],
  out_path: Path,
  cols: int,
  cell_width: int,
  cell_height: int,
  target_seconds: int,
  fps: int,
) -> None:
  inputs: list[str] = []
  for cell in cells:
    inputs.extend(["-i", str(cell.video_path)])

  filters: list[str] = []
  labels: list[str] = []
  for i, cell in enumerate(cells):
    text = _ffmpeg_text(f"{cell.encoder_label} | {cell.terrain_label}")
    label = f"v{i}"
    end_frame = max(1, cell.frame_count - 1)
    filters.append(
      f"[{i}:v]trim=end_frame={end_frame},setpts=PTS-STARTPTS,"
      f"scale={cell_width}:{cell_height},"
      f"tpad=stop_mode=clone:stop_duration={target_seconds},"
      f"drawtext=text='{text}':x=8:y=8:fontsize=20:fontcolor=white:"
      f"box=1:boxcolor=black@0.5:boxborderw=4[{label}]"
    )
    labels.append(f"[{label}]")

  rows = len(cells) // cols
  layout = "|".join(f"{(i % cols) * cell_width}_{(i // cols) * cell_height}" for i in range(len(cells)))
  filters.append("".join(labels) + f"xstack=inputs={len(cells)}:layout={layout}:fill=black[out]")

  cmd = ["ffmpeg", "-y"] + inputs + [
    "-filter_complex",
    ";".join(filters),
    "-map",
    "[out]",
    "-t",
    str(target_seconds),
    "-r",
    str(fps),
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-crf",
    "23",
    "-preset",
    "medium",
    "-movflags",
    "+faststart",
    str(out_path),
  ]
  print("OUT:", out_path, "grid:", cols, "x", rows, "->", cols * cell_width, "x", rows * cell_height)
  result = _run(cmd)
  print("ffmpeg rc", result.returncode)
  if result.returncode != 0:
    raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-3000:]}")


def _extract_still(video_path: Path, still_path: Path) -> None:
  result = _run([
    "ffmpeg",
    "-y",
    "-sseof",
    "-0.1",
    "-i",
    str(video_path),
    "-update",
    "1",
    "-q:v",
    "2",
    str(still_path),
  ])
  print("still rc", result.returncode, "->", still_path)
  if result.returncode != 0:
    raise RuntimeError(f"still extraction failed:\n{result.stderr[-2000:]}")


def _outcome(events_path: Path) -> tuple[str, tuple[int, int, int]]:
  if not events_path.exists():
    return "NO DATA", DEFAULT_COLOR
  with events_path.open() as f:
    event = json.load(f)
  reason = event.get("termination_reason", "?")
  success = bool(event.get("success", False))
  color = OUTCOME_COLORS.get(reason, (34, 160, 60) if success else DEFAULT_COLOR)
  label = "SUCCESS" if success else str(reason).replace("_", " ").upper()
  return label, color


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
  if font_path.exists():
    return ImageFont.truetype(str(font_path), size)
  return ImageFont.load_default()


def _annotate_still(
  cells: list[Cell],
  still_path: Path,
  annotated_path: Path,
  cols: int,
  cell_width: int,
  cell_height: int,
) -> None:
  rows = len(cells) // cols
  image = Image.open(still_path).convert("RGB")
  expected_size = (cols * cell_width, rows * cell_height)
  if image.size != expected_size:
    raise RuntimeError(f"Unexpected still size {image.size}; expected {expected_size}.")

  draw = ImageDraw.Draw(image, "RGBA")
  font = _load_font(26)

  for i, cell in enumerate(cells):
    col = i % cols
    row = i // cols
    label, color = _outcome(cell.events_path)
    x0 = col * cell_width
    y0 = row * cell_height
    tb = draw.textbbox((0, 0), label, font=font)
    text_width = tb[2] - tb[0]
    text_height = tb[3] - tb[1]
    pad = 8
    bx1 = x0 + cell_width - 8
    bx0 = bx1 - text_width - 2 * pad
    by0 = y0 + 8
    by1 = by0 + text_height + 2 * pad
    draw.rectangle([bx0, by0, bx1, by1], fill=color + (235,))
    draw.text((bx0 + pad - tb[0], by0 + pad - tb[1]), label, font=font, fill=(255, 255, 255))

  image.save(annotated_path, quality=92)
  print("wrote", annotated_path, image.size)


def _print_outcome_matrix(cells: list[Cell], terrains: list[str], encoders: list[str]) -> None:
  by_cell = {(cell.encoder_label, cell.terrain): _outcome(cell.events_path)[0] for cell in cells}
  for terrain in terrains:
    row = [by_cell[(encoder, terrain)] for encoder in encoders]
    print(f"{terrain.removesuffix('_corridor'):22}", " | ".join(f"{value:14}" for value in row))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--base", required=True, help="Evaluation output root.")
  parser.add_argument("--target-seconds", type=int, default=DEFAULT_TARGET_SECONDS)
  parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
  parser.add_argument("--cell-width", type=int, default=DEFAULT_CELL_WIDTH)
  parser.add_argument("--cell-height", type=int, default=DEFAULT_CELL_HEIGHT)
  parser.add_argument(
    "--max-output-width",
    type=int,
    default=DEFAULT_MAX_OUTPUT_WIDTH,
    help="Downscale cells if the tiled video would exceed this width. Use 0 to disable.",
  )
  parser.add_argument(
    "--terrains",
    nargs="+",
    default=DEFAULT_TERRAINS,
    help="Terrain subdirectories (grid rows) to include. Defaults to the standard corridors.",
  )
  args = parser.parse_args()

  base = Path(args.base).expanduser().resolve()
  if not base.is_dir():
    raise FileNotFoundError(f"Evaluation output directory not found: {base}")
  if args.target_seconds <= 0 or args.fps <= 0 or args.cell_width <= 0 or args.cell_height <= 0:
    raise ValueError("target seconds, fps, cell width, and cell height must be positive.")
  if args.max_output_width < 0:
    raise ValueError("max output width must be non-negative.")

  cells, checkpoint_dirs = _collect_cells(base, args.terrains)
  checkpoint_count = len(checkpoint_dirs)
  if len(cells) % checkpoint_count != 0:
    raise RuntimeError("Collected an uneven grid; check terrain outputs.")

  cell_width, cell_height = _effective_cell_size(
    checkpoint_count,
    args.cell_width,
    args.cell_height,
    args.max_output_width,
  )
  if (cell_width, cell_height) != (args.cell_width, args.cell_height):
    print(
      "downscaled cells:",
      f"{args.cell_width}x{args.cell_height}",
      "->",
      f"{cell_width}x{cell_height}",
      f"(max output width {args.max_output_width})",
    )

  couple_path = base / "couple.mp4"
  still_path = base / "result.jpg"
  annotated_path = base / "result_annotated.jpg"
  _build_grid_video(cells, couple_path, checkpoint_count, cell_width, cell_height, args.target_seconds, args.fps)
  _extract_still(couple_path, still_path)
  _annotate_still(cells, still_path, annotated_path, checkpoint_count, cell_width, cell_height)

  encoders = [cell.encoder_label for cell in cells[:checkpoint_count]]
  _print_outcome_matrix(cells, args.terrains, encoders)


if __name__ == "__main__":
  main()
