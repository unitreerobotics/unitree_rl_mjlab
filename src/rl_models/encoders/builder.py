"""Factory that builds an observation encoder from config."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.rl_models.encoders.base import BaseObservationEncoder, concat_dim


def build_observation_encoder(
  observation_encoder_cfg: Mapping[str, Any],
  obs_shapes: Mapping[str, tuple[int, ...]],
  encoder_input_keys: Sequence[str],
) -> BaseObservationEncoder:
  """Build an observation encoder.

  Args:
    observation_encoder_cfg: Encoder config. Must contain ``type`` and the
      type-specific keys documented in ``docs/observation_encoders.md``.
    obs_shapes: Maps each encoder-input group name to its per-sample feature
      shape (without the batch dim), e.g. ``{"height_scan": (187,)}``.
    encoder_input_keys: Ordered group names the encoder consumes.

  Returns:
    A :class:`BaseObservationEncoder` exposing ``output_dim``.
  """
  cfg = dict(observation_encoder_cfg)
  enc_type = cfg.pop("type", None)
  if enc_type is None:
    raise ValueError("observation_encoder_cfg must specify 'type'.")
  # 'encoder_input_keys'/'passthrough_keys' are consumed by the model wrapper.
  cfg.pop("encoder_input_keys", None)
  cfg.pop("passthrough_keys", None)

  keys = list(encoder_input_keys)
  input_dim = concat_dim(obs_shapes, keys)

  if enc_type == "identity":
    from src.rl_models.encoders.identity import IdentityObservationEncoder

    return IdentityObservationEncoder(keys, input_dim, **cfg)

  if enc_type == "mlp":
    from src.rl_models.encoders.mlp_encoder import MLPObservationEncoder

    return MLPObservationEncoder(keys, input_dim, **cfg)

  if enc_type == "conv1d":
    from src.rl_models.encoders.conv1d_encoder import Conv1dObservationEncoder

    return Conv1dObservationEncoder(keys, obs_shapes, **cfg)

  if enc_type == "conv2d":
    from src.rl_models.encoders.conv2d_encoder import Conv2dObservationEncoder

    return Conv2dObservationEncoder(keys, obs_shapes, **cfg)

  if enc_type in ("pretrained_ae", "transformer"):
    raise NotImplementedError(
      f"Observation encoder type '{enc_type}' is planned but not implemented. "
      "See docs/observation_encoders_planned.md for the intended design."
    )

  raise ValueError(
    f"Unknown observation encoder type '{enc_type}'. Valid types: "
    "identity, mlp, conv1d, conv2d (pretrained_ae and transformer are planned)."
  )
