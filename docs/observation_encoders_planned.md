# Planned Observation Encoders (not yet implemented)

`transformer` was part of the original design but is **not implemented** in the
current phase. `build_observation_encoder` raises `NotImplementedError` for this
type. This document records the intended design so it can be added later without
re-deriving it.

It fits the existing framework unchanged: it would subclass
`BaseObservationEncoder`, expose `output_dim`, accept the same `primary_key` /
`context_keys` fusion convention as the conv encoders, and be dispatched from
`builder.py`. `EncoderMLPModel` needs **no** changes -- it only relies on
`encoder.output_dim`.

To enable it later: implement the module under `src/rl_models/encoders/`, then
replace the corresponding branch in `builder.py` (currently the
`NotImplementedError`) with a constructor call, mirroring the existing encoder
branches.

---

## `transformer` -- Transformer encoder

**Purpose.** Treat height-scan elements (or patches) as tokens and encode them
with a Transformer.

**Behavior.**
- Primary input `[B, L]` treated as `[B, L, 1]` (or accept `[B, L, token_dim]`).
- Linear token embedding + positional encoding (`learned` or `sinusoidal`).
- `nn.TransformerEncoder` (`num_layers`, `num_heads`, `mlp_ratio`, `dropout`).
- Pooling: `mean` or `cls` (prepend a learned CLS token).
- Optional `context_keys` fusion -> project to `latent_dim`.
- Raise clear errors for incompatible shapes.

**Config sketch.**
```python
{"type": "transformer",
 "encoder_input_keys": ["height_scan", "command", "projected_gravity"],
 "passthrough_keys": None,
 "primary_key": "height_scan",
 "context_keys": ["command", "projected_gravity"],
 "token_dim": 1, "embed_dim": 64, "num_heads": 4, "num_layers": 2,
 "mlp_ratio": 4, "dropout": 0.0,
 "pooling": "mean", "positional_encoding": "learned",
 "context_hidden_dims": [64], "latent_dim": 32}
```

**Implementation notes.**
- Build positional encodings sized to `L` (learned `nn.Parameter` of shape
  `[1, L(+1 for cls), embed_dim]`, or precomputed sinusoidal buffer).
- For `cls` pooling, prepend a learned token and read its output row; for `mean`,
  average over the token dimension.
- Use `batch_first=True` in `nn.TransformerEncoderLayer`.
- Concatenate the pooled token latent with the optional context latent before
  the final linear projection to `latent_dim`.
