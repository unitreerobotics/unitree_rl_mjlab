import pytest

from src.tasks.velocity.evaluation.terrains import (
  SUPPORTED_EVAL_TERRAINS,
  make_eval_terrain_cfg,
)


@pytest.mark.parametrize("terrain_name", SUPPORTED_EVAL_TERRAINS)
def test_eval_terrain_factory_preserves_corridor_metadata_contract(terrain_name):
  terrain_cfg, waypoints, metadata = make_eval_terrain_cfg(terrain_name, seed=123)

  assert metadata["terrain_mode"] == terrain_name
  assert terrain_cfg.seed == 123
  assert terrain_cfg.num_cols == 1
  assert terrain_cfg.curriculum is True
  assert len(waypoints) == metadata["num_patches"] == terrain_cfg.num_rows
  assert metadata["patch_length"] == pytest.approx(4.0)
  assert metadata["corridor_width"] == pytest.approx(3.0)
  assert metadata["total_path_length"] == pytest.approx(36.0)

  patches = metadata["patches"]
  assert patches[0]["kind"] == "flat_spawn"
  assert patches[0]["difficulty_level"] == pytest.approx(0.0)
  assert patches[-1]["kind"] == "flat_finish"
  assert patches[-1]["difficulty_level"] == pytest.approx(1.0)

  for patch_index, patch in enumerate(patches):
    assert patch["patch_index"] == patch_index
    assert "start_position" in patch
    assert "end_position" in patch
    assert "difficulty_level" in patch


@pytest.mark.parametrize(
  ("terrain_name", "terrain_kind", "param_key"),
  [
    ("rough_curriculum_corridor", "random_rough", "roughness"),
    ("perlin_noise_corridor", "perlin_noise", "perlin_noise"),
    ("random_spread_boxes_corridor", "random_spread_boxes", "random_spread_boxes"),
  ],
)
def test_eval_terrain_metadata_records_terrain_specific_patch_params(
  terrain_name, terrain_kind, param_key
):
  _terrain_cfg, _waypoints, metadata = make_eval_terrain_cfg(terrain_name, seed=123)
  terrain_patches = metadata["patches"][1:-1]

  assert terrain_patches
  assert {patch["kind"] for patch in terrain_patches} == {terrain_kind}
  assert all(param_key in patch for patch in terrain_patches)
  assert terrain_patches[0]["difficulty_level"] == pytest.approx(0.0)
  assert terrain_patches[-1]["difficulty_level"] == pytest.approx(1.0)


def test_rough_eval_terrain_keeps_legacy_num_rough_patches_metadata():
  _terrain_cfg, _waypoints, metadata = make_eval_terrain_cfg(
    "rough_curriculum_corridor",
    seed=123,
  )

  assert metadata["num_terrain_patches"] == 8
  assert metadata["num_rough_patches"] == 8


def test_eval_terrain_factory_rejects_unknown_terrain():
  with pytest.raises(ValueError, match="Unsupported --eval-terrain"):
    make_eval_terrain_cfg("unknown", seed=123)
