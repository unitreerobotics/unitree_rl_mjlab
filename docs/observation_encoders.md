# Configurable Observation Encoders

A small, **opt-in** framework for swapping the observation-encoder architecture
of an rsl_rl actor/critic **by changing config only**. It lets you compare, e.g.,
a raw height-scan baseline against MLP / Conv1d / Conv2d feature encoders without
writing new policy classes.

## Architecture

```
        selected observation groups (encoder_input_keys)
                          |
                          v
            configurable observation encoder
                          |
                          v
                   encoded latent  z  ──┐
                                        ├── concat ── policy/value MLP ── action/value
   remaining groups (passthrough_keys) ─┘
   (optionally normalized, raw features)
```

- The encoder consumes a configurable subset of observation **groups**
  (`encoder_input_keys`) and returns a fixed-size latent `z` of shape
  `[B, output_dim]`.
- The remaining groups (`passthrough_keys`) are concatenated as raw features.
- `final_latent = concat([z, passthrough])` is fed to the MLP head.
- The model wrapper only relies on `encoder.output_dim`; it does not care which
  encoder type is used.

The implementation lives in `src/rl_models/`:

```
rl_models/
  encoder_mlp_model.py        # EncoderMLPModel (subclasses rsl_rl MLPModel)
  encoders/
    base.py                   # BaseObservationEncoder + helpers
    identity.py  mlp_encoder.py  conv1d_encoder.py  conv2d_encoder.py
    builder.py                # build_observation_encoder(...)
```

## This feature is opt-in — old configs are unchanged

`RslRlModelCfg` gained one optional field, `observation_encoder_cfg`
(default `None`), alongside the existing `cnn_cfg`/`distribution_cfg`. When it is
`None` it is stripped by `MjlabOnPolicyRunner` before the model is built, so
`MLPModel`/`CNNModel` configs never see it and behave **exactly** as before.

`EncoderMLPModel` is also a literal drop-in: with `observation_encoder_cfg=None`
it delegates entirely to `MLPModel` (same latent, same output, same
normalization). No existing task config was modified.

## How observation groups work here

rsl_rl models receive a `TensorDict` keyed by **observation-group** names, and
`obs_groups[obs_set]` lists the groups each model uses. `encoder_input_keys` and
`passthrough_keys` are these group names.

The stock velocity envs concatenate every term into single `actor`/`critic`
tensors, so individual terms are not separately addressable. The example task
uses `unitree_go2_rough_split_obs_env_cfg`, which re-buckets the actor
observation into per-term groups (`height_scan`, `command`, `projected_gravity`,
`proprio`, `last_action`) — dynamics are identical, only the grouping changes.

## `observation_encoder_cfg` schema

Common keys (all encoder types):

| Key                  | Meaning                                                            |
|----------------------|-------------------------------------------------------------------|
| `type`               | `identity` \| `mlp` \| `conv1d` \| `conv2d`                        |
| `encoder_input_keys` | ordered observation groups fed to the encoder                     |
| `passthrough_keys`   | groups concatenated raw after the latent. `null`/omitted ⇒ **all groups in `obs_groups[obs_set]` except `encoder_input_keys`** |

`encoder_input_keys` must all exist in `obs_groups[obs_set]` (clear error
otherwise). Passthrough groups must be 1D (`[B, D]`).

### `encoder_input_keys` vs `passthrough_keys`

- Put a signal in `encoder_input_keys` to **encode** it (compress / extract
  features). Typical: `height_scan`, optionally `command` + `projected_gravity`
  for state-conditioned terrain encoding.
- Leave robot proprioception (`proprio`, `last_action`) in `passthrough_keys` so
  the policy sees it directly.

## Encoder types and shapes

### `identity` — raw baseline
Flattens + concatenates `encoder_input_keys`. `output_dim = total raw dim`.
Equivalent to "raw observation + MLP".
```python
{"type": "identity", "encoder_input_keys": ["height_scan"], "flatten": True}
```

### `mlp` — feature encoder
Flatten + concat → MLP → `latent_dim`.
```python
{"type": "mlp", "encoder_input_keys": ["height_scan"],
 "latent_dim": 32, "hidden_dims": [256, 128], "activation": "elu",
 "layer_norm": False}
```
State-conditioned variant: add `command`, `projected_gravity` to
`encoder_input_keys`.

### `conv1d` — ordered height-scan vector
Primary group processed with Conv1d + global pool; optional `context_keys` go
through a small MLP; both are concatenated and projected to `latent_dim`.
Accepts primary `[B, L]` (reshaped to `[B, 1, L]`) or `[B, C, L]`; other shapes
raise a clear error.
```python
{"type": "conv1d", "encoder_input_keys": ["height_scan", "command"],
 "primary_key": "height_scan", "context_keys": ["command"],
 "channels": [16, 32, 64], "kernel_sizes": [5, 3, 3], "strides": [2, 2, 1],
 "activation": "elu", "global_pool": "avg",
 "context_hidden_dims": [64], "latent_dim": 32}
```

### `conv2d` — grid-like height map
Like `conv1d` but 2D. Accepts primary `[B, H, W]` or `[B, C, H, W]`. If the
group is stored flat (Go2 `height_scan` is `[B, 187]` for a 17x11 grid), set
`input_hw: [H, W]` to reshape it; a flat input without `input_hw` raises a clear
error.
```python
{"type": "conv2d", "encoder_input_keys": ["height_scan", "command"],
 "primary_key": "height_scan", "context_keys": ["command"],
 "input_hw": [17, 11],
 "channels": [16, 32, 64], "kernel_sizes": [3, 3, 3], "strides": [1, 2, 2],
 "activation": "elu", "global_pool": "avg",
 "context_hidden_dims": [64], "latent_dim": 32}
```

> `pretrained_ae` and `transformer` are **planned but not implemented** in this
> phase. The builder raises `NotImplementedError` for them. See
> [`observation_encoders_planned.md`](observation_encoders_planned.md).

## Switching encoders from config

Configs are Python dataclasses (not YAML). Set the actor's `class_name` to the
encoder model and supply `observation_encoder_cfg`:

```python
from mjlab.rl import RslRlModelCfg

RslRlModelCfg(
    class_name="src.rl_models.encoder_mlp_model:EncoderMLPModel",
    observation_encoder_cfg={
        "type": "mlp",                       # <- change this to compare
        "encoder_input_keys": ["height_scan"],
        "passthrough_keys": None,
        "latent_dim": 32, "hidden_dims": [256, 128], "activation": "elu",
    },
    hidden_dims=(256, 256), activation="elu", obs_normalization=False,
)
```

Switch `type` (and its type-specific keys) to compare architectures; change
`encoder_input_keys` to feed height-only vs height + robot state.

See `src/tasks/velocity/config/go2/encoder_ablation_rl_cfg.py` for ready-made
builders and `__init__.py` for the registered tasks:

```
Unitree-Go2-Rough-Encoder-Raw        # identity
Unitree-Go2-Rough-Encoder-MLP        # mlp, height only
Unitree-Go2-Rough-Encoder-MLPState   # mlp, height + command + gravity
Unitree-Go2-Rough-Encoder-Conv1d
Unitree-Go2-Rough-Encoder-Conv2d
```

## Actor and critic are independent

Actor and critic each have their own `RslRlModelCfg`, so they can use different
encoders — or none. In the example tasks the **actor** uses an encoder over the
split groups while the **critic** is a plain `MLPModel` over the privileged,
concatenated `critic` group. The critic's privileged observations therefore do
not affect the actor. The same `EncoderMLPModel` class works for both: a critic
encoder is just another `observation_encoder_cfg`.

## Observation normalization

When an encoder is active, `obs_normalization` applies **only to the passthrough
groups** (the encoder inputs bypass the normalizer; MLP encoders can use their
own `layer_norm`). This keeps the normalizer dimension consistent
(`= passthrough_dim`). The example configs set `obs_normalization=False` on the
actor for simplicity; the critic keeps `obs_normalization=True` unchanged.

## PPO / training

- Encoder parameters are submodules of the model, so they are included in the
  actor/critic parameter groups and optimized by PPO automatically. No PPO or
  runner-loop changes are required.
- ONNX/JIT export of `EncoderMLPModel` is **not** supported in this phase
  (`as_onnx`/`as_jit` raise a clear error when an encoder is active). The
  ablation tasks use `MjlabOnPolicyRunner`, which saves `.pt` checkpoints and
  does not export ONNX.

## Out of scope

No reconstruction / auxiliary losses. The pretrained-AE encoder (planned) is
intended purely as a feature encoder, not trained with a reconstruction loss
during PPO. Auxiliary losses can be added later but are out of scope here.

## Running the tests

```sh
cd unitree_rl_mjlab
uv run pytest tests/test_observation_encoders.py -q
```
