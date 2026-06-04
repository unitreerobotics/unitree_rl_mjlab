# Planned Observation Encoders (not yet implemented)

`pretrained_ae` and `transformer` were part of the original design but are
**not implemented** in the current phase (they are heavier and add dependencies
/ checkpoint plumbing). `build_observation_encoder` raises `NotImplementedError`
for these types. This document records the intended design so they can be added
later without re-deriving it.

Both fit the existing framework unchanged: each would subclass
`BaseObservationEncoder`, expose `output_dim`, accept the same
`primary_key` / `context_keys` fusion convention as the conv encoders, and be
dispatched from `builder.py`. `EncoderMLPModel` needs **no** changes — it only
relies on `encoder.output_dim`.

To enable one later: implement the module under `src/rl_models/encoders/`, then
replace the corresponding branch in `builder.py`
(currently the `NotImplementedError`) with a constructor call, mirroring the
`conv1d`/`conv2d` branches.

---

## `pretrained_ae` — pretrained AutoEncoder encoder

**Purpose.** Reuse the encoder half of a pretrained autoencoder as a frozen (or
fine-tuned) feature extractor for `height_scan`.

**Behavior.**
- Load a pretrained autoencoder/encoder checkpoint; use **only** the encoder
  output (no decoder at policy inference time).
- `freeze=True` → `requires_grad_(False)` + `eval()` on the pretrained encoder.
- `freeze=False` → allow PPO to fine-tune it.
- Optional `primary_key` / `context_keys` fusion (context MLP → concat →
  project to `latent_dim`), identical to the conv encoders.
- **No reconstruction loss** during PPO — feature encoder only.

**Config sketch.**
```python
{"type": "pretrained_ae",
 "encoder_input_keys": ["height_scan"],
 "passthrough_keys": None,
 "checkpoint_path": "/path/to/ae_checkpoint.pt",
 "encoder_class": "src.rl_models.autoencoder:HeightScanAutoEncoder",
 "latent_dim": 32, "freeze": True, "strict": False}
```
With context fusion: add `primary_key`, `context_keys`, `context_hidden_dims`.

**Implementation notes.**
- Resolve `encoder_class` via `rsl_rl.utils.resolve_callable` (supports
  `"module.path:ClassName"`), instantiate, then
  `load_state_dict(torch.load(checkpoint_path), strict=strict)`.
- Keep only the encoder submodule; set `output_dim` from the AE latent (project
  to `latent_dim` if they differ, or if context is fused).
- If `freeze=True`, also guard against `train()` re-enabling grads (re-assert
  `eval()` / `requires_grad_(False)` as needed).
- Ship a tiny example `HeightScanAutoEncoder` (encoder+decoder MLP) under
  `src/rl_models/autoencoder.py` so tests can build a dummy checkpoint.
- Tests should assert `freeze=True` leaves all pretrained-encoder params with
  `requires_grad=False`.

---

## `transformer` — Transformer encoder

**Purpose.** Treat height-scan elements (or patches) as tokens and encode them
with a Transformer.

**Behavior.**
- Primary input `[B, L]` treated as `[B, L, 1]` (or accept `[B, L, token_dim]`).
- Linear token embedding + positional encoding (`learned` or `sinusoidal`).
- `nn.TransformerEncoder` (`num_layers`, `num_heads`, `mlp_ratio`, `dropout`).
- Pooling: `mean` or `cls` (prepend a learned CLS token).
- Optional `context_keys` fusion → project to `latent_dim`.
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
