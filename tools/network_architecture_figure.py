"""Generate policy pipeline architecture figures.

The tool renders the full path from observations to the final joint-position
target. It can inspect a registered mjlab task, a deploy YAML, or both.
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


GO2_ACTION_DIM = 12

_TERM_DIMS: dict[str, int] = {
  "base_ang_vel": 3,
  "base_lin_vel": 3,
  "builtin_sensor": 3,
  "command": 3,
  "generated_commands": 3,
  "gait_phase": 2,
  "phase": 2,
  "projected_gravity": 3,
  "velocity_commands": 3,
  "foot_air_time": 4,
  "foot_contact": 4,
  "foot_contact_forces": 12,
  "foot_height": 4,
  "height_scan": 187,
}

_ACTION_SIZED_TERMS = {
  "actions",
  "joint_pos",
  "joint_pos_rel",
  "joint_vel",
  "joint_vel_rel",
  "last_action",
}


@dataclass
class ObservationTermSpec:
  name: str
  dim: int | None = None
  scale: Any = None
  clip: Any = None
  history_length: int = 1


@dataclass
class ObservationGroupSpec:
  name: str
  terms: list[ObservationTermSpec] = field(default_factory=list)
  concatenate: bool = True

  @property
  def dim(self) -> int | None:
    if any(t.dim is None for t in self.terms):
      return None
    return sum(int(t.dim) * int(t.history_length) for t in self.terms if t.dim is not None)


@dataclass
class ModelSpec:
  name: str
  class_name: str
  obs_groups: list[str]
  hidden_dims: list[int]
  activation: str
  obs_normalization: bool
  distribution_cfg: dict[str, Any] | None = None
  observation_encoder_cfg: dict[str, Any] | None = None

  @property
  def encoder_type(self) -> str | None:
    if not self.observation_encoder_cfg:
      return None
    return str(self.observation_encoder_cfg.get("type"))

  @property
  def encoder_input_keys(self) -> list[str]:
    if not self.observation_encoder_cfg:
      return []
    return list(self.observation_encoder_cfg.get("encoder_input_keys") or [])

  @property
  def passthrough_keys(self) -> list[str]:
    if not self.observation_encoder_cfg:
      return list(self.obs_groups)
    configured = self.observation_encoder_cfg.get("passthrough_keys")
    if configured is not None:
      return list(configured)
    enc = set(self.encoder_input_keys)
    return [g for g in self.obs_groups if g not in enc]


@dataclass
class ActionSpec:
  kind: str = "JointPositionAction"
  dim: int = GO2_ACTION_DIM
  scale: Any = 1.0
  offset: Any = 0.0
  clip: Any = None
  joint_ids_map: list[int] | None = None
  default_joint_pos: list[float] | None = None
  stiffness: list[float] | None = None
  damping: list[float] | None = None

  @property
  def target_label(self) -> str:
    if "Position" in self.kind:
      return "q_target"
    if "Velocity" in self.kind:
      return "dq_target"
    if "Effort" in self.kind:
      return "tau_target"
    return "processed_action"


@dataclass
class PipelineSpec:
  source: str
  task_id: str | None = None
  deploy_yaml: str | None = None
  observations: dict[str, ObservationGroupSpec] = field(default_factory=dict)
  actor: ModelSpec | None = None
  critic: ModelSpec | None = None
  action: ActionSpec = field(default_factory=ActionSpec)


def load_task_pipeline(task_id: str, obs_dim_overrides: dict[str, int] | None = None) -> PipelineSpec:
  """Load pipeline metadata from a registered mjlab task."""
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  env_cfg = load_env_cfg(task_id)
  rl_cfg = load_rl_cfg(task_id)

  action = _extract_task_action(env_cfg, task_id)
  observations = _extract_task_observations(env_cfg, action.dim, obs_dim_overrides or {})
  obs_groups = {k: list(v) for k, v in dict(getattr(rl_cfg, "obs_groups", {}) or {}).items()}

  actor_groups = obs_groups.get("actor", ["actor"])
  critic_groups = obs_groups.get("critic", ["critic"])
  actor = _model_from_cfg("Actor", getattr(rl_cfg, "actor"), actor_groups)
  critic = _model_from_cfg("Critic", getattr(rl_cfg, "critic"), critic_groups)

  return PipelineSpec(
    source="task",
    task_id=task_id,
    observations=observations,
    actor=actor,
    critic=critic,
    action=action,
  )


def load_deploy_pipeline(
  deploy_yaml: str | Path,
  obs_dim_overrides: dict[str, int] | None = None,
) -> PipelineSpec:
  """Load pipeline metadata from a deploy YAML."""
  import yaml

  path = Path(deploy_yaml)
  data = yaml.safe_load(path.read_text())
  action = _extract_deploy_action(data)
  observations = _extract_deploy_observations(data, action.dim, obs_dim_overrides or {})
  actor = ModelSpec(
    name="Actor",
    class_name="policy.onnx",
    obs_groups=list(observations),
    hidden_dims=[],
    activation="",
    obs_normalization=False,
    distribution_cfg=None,
  )

  return PipelineSpec(
    source="deploy",
    deploy_yaml=str(path),
    observations=observations,
    actor=actor,
    critic=None,
    action=action,
  )


def merge_deploy_metadata(spec: PipelineSpec, deploy_yaml: str | Path) -> PipelineSpec:
  """Overlay deploy-side observation/action metadata onto a task spec."""
  deploy_spec = load_deploy_pipeline(deploy_yaml)
  spec.source = "task+deploy"
  spec.deploy_yaml = deploy_spec.deploy_yaml
  spec.action = deploy_spec.action
  if deploy_spec.observations:
    spec.observations = deploy_spec.observations
    if spec.actor is not None and spec.actor.class_name == "policy.onnx":
      spec.actor.obs_groups = list(deploy_spec.observations)
  return spec


def render_pipeline(
  spec: PipelineSpec,
  out: str | Path,
  output_format: str | None = None,
  include_critic: bool = True,
) -> Path:
  """Render a pipeline figure to SVG, PNG, or Mermaid text."""
  path = Path(out)
  fmt = output_format or path.suffix.lstrip(".") or "svg"
  if fmt in ("mmd", "mermaid"):
    if not path.suffix:
      path = path.with_suffix(".mmd")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_mermaid_pipeline(spec, include_critic=include_critic))
    return path
  if not path.suffix:
    path = path.with_suffix(f".{fmt}")

  plt.rcParams["svg.fonttype"] = "none"
  fig, ax = plt.subplots(figsize=(18, 10.5))
  ax.set_axis_off()
  fig.patch.set_facecolor("#f8fafc")
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)

  title = _title(spec)
  ax.text(0.03, 0.965, title, fontsize=18, fontweight="bold", color="#0f172a")
  ax.text(
    0.03,
    0.925,
    "Main path: observations are converted to a normalized raw action; JointPositionAction then converts it to the joint-position target.",
    fontsize=10,
    color="#475569",
  )

  top_y = 0.76
  xs = [0.08, 0.23, 0.39, 0.55, 0.69, 0.83, 0.94]
  labels = [
    _sensor_summary_label(spec),
    _observation_summary_label(spec),
    _preprocess_label(spec),
    _actor_summary_label(spec),
    _raw_action_label(spec),
    _action_processing_summary_label(spec),
    _target_label(spec),
  ]
  colors = ["#dbeafe", "#e0f2fe", "#dcfce7", "#fef3c7", "#fee2e2", "#ede9fe", "#e2e8f0"]

  for x, label, color in zip(xs, labels, colors, strict=True):
    _box(ax, x, top_y, label, color=color, fontsize=9)
  for start, end in zip(xs[:-1], xs[1:], strict=False):
    _arrow(ax, start + 0.055, top_y, end - 0.055, top_y)

  _panel(ax, 0.25, 0.47, _observation_detail_label(spec), color="#ffffff", fontsize=8.3)
  _panel(ax, 0.56, 0.47, _actor_detail_label(spec), color="#fffbeb", fontsize=8.5)
  _panel(ax, 0.83, 0.47, _action_detail_label(spec), color="#faf5ff", fontsize=8.5)

  if include_critic and spec.critic is not None:
    critic_y = 0.17
    critic_xs = [0.25, 0.49, 0.70, 0.86]
    critic_labels = [
      _critic_obs_label(spec),
      "Critic preprocessing\nnormalization: " + _yes_no(spec.critic.obs_normalization),
      _critic_model_label(spec),
      "Value output\nV(s) scalar",
    ]
    for x, label in zip(critic_xs, critic_labels, strict=True):
      _box(ax, x, critic_y, label, color="#f1f5f9", fontsize=8.5)
    for start, end in zip(critic_xs[:-1], critic_xs[1:], strict=False):
      _arrow(ax, start + 0.055, critic_y, end - 0.055, critic_y)

  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, format=fmt, bbox_inches="tight")
  plt.close(fig)
  return path


def render_mermaid_pipeline(spec: PipelineSpec, include_critic: bool = True) -> str:
  """Return a Mermaid flowchart for the policy pipeline."""
  actor = spec.actor
  action = spec.action
  lines = [
    f"%% {_title(spec)}",
    "flowchart LR",
    "  classDef sensor fill:#dbeafe,stroke:#2563eb,color:#0f172a;",
    "  classDef obs fill:#e0f2fe,stroke:#0284c7,color:#0f172a;",
    "  classDef prep fill:#dcfce7,stroke:#16a34a,color:#0f172a;",
    "  classDef net fill:#fef3c7,stroke:#d97706,color:#0f172a;",
    "  classDef action fill:#fee2e2,stroke:#dc2626,color:#0f172a;",
    "  classDef target fill:#e2e8f0,stroke:#475569,color:#0f172a;",
    "  classDef critic fill:#f1f5f9,stroke:#64748b,color:#0f172a;",
    "",
    f"  S[\"{_mmd_label(_sensor_summary_label(spec))}\"]:::sensor",
  ]

  actor_groups = actor.obs_groups if actor is not None else list(spec.observations)
  lines += ["", "  subgraph OBS[Observation groups]"]
  for group_name in actor_groups:
    group = spec.observations.get(group_name)
    if group is None:
      label = group_name
    else:
      terms = _compact_terms(group, max_terms=4)
      dim = group.dim if group.dim is not None else "?"
      label = f"{group_name} ({dim})\\n{terms}"
    lines.append(f"    {_mmd_id('obs_' + group_name)}[\"{_mmd_label(label)}\"]:::obs")
  lines.append("  end")
  lines.append("  S --> OBS")

  lines += [
    f"  PRE[\"{_mmd_label(_preprocess_label(spec))}\"]:::prep",
    "  OBS --> PRE",
  ]

  if actor is not None and actor.encoder_type:
    enc_in, enc_out = _encoder_io_dims(spec, actor)
    pass_dim = _group_total_dim(spec, actor.passthrough_keys)
    enc_label = (
      "Observation encoder\\n"
      f"{_encoder_summary(actor)}\\n"
      "inputs: " + ", ".join(actor.encoder_input_keys)
    )
    pass_label = "Passthrough + normalizer\\n" + (", ".join(actor.passthrough_keys) or "none")
    lines += [
      f"  ENC[\"{_mmd_label(enc_label)}\"]:::net",
      f"  PASS[\"{_mmd_label(pass_label)}\"]:::net",
      "  CAT[\"Concat<br/>encoder latent + passthrough\"]:::net",
      f"  PRE -->|{_fmt_dim(enc_in)}| ENC",
      f"  PRE -->|{_fmt_dim(pass_dim)}| PASS",
      f"  ENC -->|{_fmt_dim(enc_out)}| CAT",
      f"  PASS -->|{_fmt_dim(pass_dim)}| CAT",
    ]
    actor_input = "CAT"
  else:
    actor_input = "PRE"

  actor_label = _actor_mermaid_label(spec)
  actor_in = _actor_input_dim(spec, actor) if actor is not None else None
  lines += [
    f"  ACTOR[\"{_mmd_label(actor_label)}\"]:::net",
    f"  {actor_input} -->|{_fmt_dim(actor_in)}| ACTOR",
  ]
  if actor is not None and actor.distribution_cfg:
    lines += [
      "  DIST[\"Gaussian distribution<br/>deterministic mean for inference\"]:::net",
      f"  ACTOR -->|{action.dim}| DIST",
      f"  ARAW[\"a_raw ({action.dim})<br/>normalized action delta<br/>not final q\"]:::action",
      f"  DIST -->|{action.dim}| ARAW",
    ]
  else:
    lines += [
      f"  ARAW[\"a_raw ({action.dim})<br/>policy output<br/>not final q\"]:::action",
      f"  ACTOR -->|{action.dim}| ARAW",
    ]

  action_label = f"{action.kind}\\n{action.target_label} = a_raw * scale + offset"
  if action.default_joint_pos is not None or action.offset == "default_joint_pos":
    action_label += "\\noffset = default_joint_pos"
  if action.clip:
    action_label += "\\nclip processed target"
  target_label = f"{action.target_label} ({action.dim})"
  if action.joint_ids_map is not None:
    target_label += "\\njoint_ids_map " + _short_list(action.joint_ids_map, max_items=12)
  if action.stiffness and action.damping:
    target_label += "\\nwrite motor_cmd.q; kp/kd separate"
  else:
    target_label += "\\njoint position target"

  lines += [
    f"  APROC[\"{_mmd_label(action_label)}\"]:::action",
    f"  QT[\"{_mmd_label(target_label)}\"]:::target",
    f"  ARAW -->|{action.dim}| APROC",
    f"  APROC -->|{action.dim}| QT",
  ]

  if include_critic and spec.critic is not None:
    critic = spec.critic
    critic_groups = ", ".join(
      f"{name}({spec.observations[name].dim})" if name in spec.observations else name
      for name in critic.obs_groups
    )
    critic_in = _group_total_dim(spec, critic.obs_groups)
    lines += [
      "",
      "  subgraph CRITIC[Critic / value branch]",
      f"    COBS[\"critic observations<br/>{_mmd_label(critic_groups)}\"]:::critic",
      f"    CNORM[\"normalization: {_yes_no(critic.obs_normalization)}\"]:::critic",
      f"    CMLP[\"{_mmd_label(_critic_model_label(spec))}\"]:::critic",
      "    V[\"V(s) scalar\"]:::critic",
      f"    COBS -->|{_fmt_dim(critic_in)}| CNORM",
      f"    CNORM -->|{_fmt_dim(critic_in)}| CMLP",
      "    CMLP -->|1| V",
      "  end",
      "  OBS -. privileged/state obs .-> COBS",
    ]

  lines.append("")
  return "\n".join(lines)


def _actor_mermaid_label(spec: PipelineSpec) -> str:
  actor = spec.actor
  if actor is None:
    return "Actor policy\\nunknown"
  if actor.class_name == "policy.onnx":
    return "policy.onnx\\nobservations -> raw action"
  return (
    f"{_short_class(actor.class_name)}\\n"
    f"MLP: {_mlp_dims(actor.hidden_dims, spec.action.dim)}\\n"
    f"activation: {actor.activation}\\n"
    "normalization on MLP path"
  )


def _encoder_io_dims(spec: PipelineSpec, actor: ModelSpec) -> tuple[int | None, int | None]:
  in_dim = _group_total_dim(spec, actor.encoder_input_keys)
  cfg = actor.observation_encoder_cfg or {}
  if actor.encoder_type == "identity":
    out_dim = in_dim
  else:
    latent_dim = cfg.get("latent_dim")
    out_dim = int(latent_dim) if latent_dim is not None else None
  return in_dim, out_dim


def _actor_input_dim(spec: PipelineSpec, actor: ModelSpec) -> int | None:
  if not actor.encoder_type:
    return _group_total_dim(spec, actor.obs_groups)
  _, enc_out = _encoder_io_dims(spec, actor)
  pass_dim = _group_total_dim(spec, actor.passthrough_keys)
  if enc_out is None or pass_dim is None:
    return None
  return enc_out + pass_dim


def _group_total_dim(spec: PipelineSpec, group_names: list[str]) -> int | None:
  total = 0
  for group_name in group_names:
    group = spec.observations.get(group_name)
    if group is None or group.dim is None:
      return None
    total += group.dim
  return total


def _fmt_dim(dim: int | None) -> str:
  return str(dim) if dim is not None else "?"


def _compact_terms(group: ObservationGroupSpec, max_terms: int = 4) -> str:
  terms = []
  for term in group.terms[:max_terms]:
    suffix = f"({term.dim})" if term.dim is not None else "(?)"
    terms.append(f"{term.name}{suffix}")
  if len(group.terms) > max_terms:
    terms.append("...")
  return ", ".join(terms)


def _mmd_id(raw: str) -> str:
  return "N_" + "".join(ch if ch.isalnum() else "_" for ch in raw)


def _mmd_label(raw: str) -> str:
  return raw.replace("\\n", "<br/>").replace("\n", "<br/>").replace('"', "'")


def _extract_task_action(env_cfg: Any, task_id: str) -> ActionSpec:
  action_cfgs = dict(getattr(env_cfg, "actions", {}) or {})
  if not action_cfgs:
    return ActionSpec(dim=_default_action_dim(task_id))

  name, cfg = next(iter(action_cfgs.items()))
  kind = type(cfg).__name__.removesuffix("Cfg")
  scale = getattr(cfg, "scale", 1.0)
  offset = getattr(cfg, "offset", 0.0)
  default_joint_pos = None
  if kind == "JointPositionAction" and bool(getattr(cfg, "use_default_offset", False)):
    offset = "default_joint_pos"
  dim = _dim_from_scale_or_offset(scale, offset) or _default_action_dim(task_id)
  return ActionSpec(kind=kind, dim=dim, scale=scale, offset=offset, default_joint_pos=default_joint_pos)


def _extract_deploy_action(data: dict[str, Any]) -> ActionSpec:
  actions = data.get("actions") or {}
  if actions:
    kind, cfg = next(iter(actions.items()))
    scale = cfg.get("scale", 1.0)
    offset = cfg.get("offset", data.get("default_joint_pos", 0.0))
    clip = cfg.get("clip")
    dim = _dim_from_scale_or_offset(scale, offset) or len(data.get("joint_ids_map") or []) or GO2_ACTION_DIM
  else:
    kind, scale, offset, clip = "JointPositionAction", 1.0, data.get("default_joint_pos", 0.0), None
    dim = len(data.get("joint_ids_map") or []) or GO2_ACTION_DIM

  return ActionSpec(
    kind=kind,
    dim=dim,
    scale=scale,
    offset=offset,
    clip=clip,
    joint_ids_map=data.get("joint_ids_map"),
    default_joint_pos=data.get("default_joint_pos"),
    stiffness=data.get("stiffness"),
    damping=data.get("damping"),
  )


def _extract_task_observations(
  env_cfg: Any,
  action_dim: int,
  overrides: dict[str, int],
) -> dict[str, ObservationGroupSpec]:
  groups: dict[str, ObservationGroupSpec] = {}
  for group_name, group_cfg in dict(getattr(env_cfg, "observations", {}) or {}).items():
    group_history = getattr(group_cfg, "history_length", None)
    terms = []
    for term_name, term_cfg in dict(getattr(group_cfg, "terms", {}) or {}).items():
      history = int(group_history or getattr(term_cfg, "history_length", 1) or 1)
      terms.append(
        ObservationTermSpec(
          name=term_name,
          dim=_infer_term_dim(term_name, action_dim, overrides, term_cfg),
          scale=getattr(term_cfg, "scale", None),
          clip=getattr(term_cfg, "clip", None),
          history_length=history,
        )
      )
    groups[group_name] = ObservationGroupSpec(
      name=group_name,
      terms=terms,
      concatenate=bool(getattr(group_cfg, "concatenate_terms", True)),
    )
  return groups


def _extract_deploy_observations(
  data: dict[str, Any],
  action_dim: int,
  overrides: dict[str, int],
) -> dict[str, ObservationGroupSpec]:
  obs_cfg = data.get("observations") or {}
  if not obs_cfg:
    return {}

  first = next(iter(obs_cfg.values()))
  grouped = not (isinstance(first, dict) and "params" in first)
  if grouped:
    groups = {}
    for group_name, group_terms in obs_cfg.items():
      groups[group_name] = ObservationGroupSpec(
        name=group_name,
        terms=[_deploy_term(name, cfg, action_dim, overrides) for name, cfg in group_terms.items()],
      )
    return groups

  return {
    "obs": ObservationGroupSpec(
      name="obs",
      terms=[_deploy_term(name, cfg, action_dim, overrides) for name, cfg in obs_cfg.items()],
    )
  }


def _deploy_term(
  name: str,
  cfg: dict[str, Any],
  action_dim: int,
  overrides: dict[str, int],
) -> ObservationTermSpec:
  return ObservationTermSpec(
    name=name,
    dim=_infer_term_dim(name, action_dim, overrides, cfg),
    scale=cfg.get("scale"),
    clip=cfg.get("clip"),
    history_length=int(cfg.get("history_length", 1) or 1),
  )


def _model_from_cfg(name: str, cfg: Any, obs_groups: list[str]) -> ModelSpec:
  return ModelSpec(
    name=name,
    class_name=str(getattr(cfg, "class_name", type(cfg).__name__)),
    obs_groups=list(obs_groups),
    hidden_dims=[int(v) for v in getattr(cfg, "hidden_dims", ())],
    activation=str(getattr(cfg, "activation", "")),
    obs_normalization=bool(getattr(cfg, "obs_normalization", False)),
    distribution_cfg=_copy_dict(getattr(cfg, "distribution_cfg", None)),
    observation_encoder_cfg=_copy_dict(getattr(cfg, "observation_encoder_cfg", None)),
  )


def _infer_term_dim(
  name: str,
  action_dim: int,
  overrides: dict[str, int],
  cfg: Any | None = None,
) -> int | None:
  if name in overrides:
    return overrides[name]
  scale = _get_cfg_value(cfg, "scale")
  if isinstance(scale, (list, tuple)) and scale:
    return len(scale)
  if name in _ACTION_SIZED_TERMS:
    return action_dim
  return _TERM_DIMS.get(name)


def _dim_from_scale_or_offset(scale: Any, offset: Any) -> int | None:
  if isinstance(scale, (list, tuple)) and scale:
    return len(scale)
  if isinstance(offset, (list, tuple)) and offset:
    return len(offset)
  return None


def _default_action_dim(task_id: str | None) -> int:
  if task_id and "Go2" in task_id:
    return GO2_ACTION_DIM
  return GO2_ACTION_DIM


def _copy_dict(value: Any) -> dict[str, Any] | None:
  if value is None:
    return None
  return dict(value)


def _get_cfg_value(cfg: Any | None, key: str) -> Any:
  if cfg is None:
    return None
  if isinstance(cfg, dict):
    return cfg.get(key)
  return getattr(cfg, key, None)


def _title(spec: PipelineSpec) -> str:
  if spec.task_id and spec.deploy_yaml:
    return f"Policy Pipeline: {spec.task_id} + {Path(spec.deploy_yaml).name}"
  if spec.task_id:
    return f"Policy Pipeline: {spec.task_id}"
  if spec.deploy_yaml:
    return f"Deploy Policy Pipeline: {Path(spec.deploy_yaml).name}"
  return "Policy Pipeline"


def _sensor_label(spec: PipelineSpec) -> str:
  terms = {t.name for g in spec.observations.values() for t in g.terms}
  sensors = []
  if {"base_ang_vel", "projected_gravity", "gait_phase", "phase"} & terms:
    sensors.append("IMU/body state")
  if {"command", "velocity_commands"} & terms:
    sensors.append("command/joystick")
  if {"joint_pos", "joint_pos_rel", "joint_vel", "joint_vel_rel"} & terms:
    sensors.append("joint state")
  if {"actions", "last_action"} & terms:
    sensors.append("previous action")
  if "height_scan" in terms:
    sensors.append("terrain scan")
  return "Robot / command state\n" + "\n".join(sensors or ["configured observations"])


def _sensor_summary_label(spec: PipelineSpec) -> str:
  lines = _sensor_label(spec).splitlines()
  return "Robot / command state\n" + ", ".join(lines[1:])


def _observation_summary_label(spec: PipelineSpec) -> str:
  actor = spec.actor
  group_names = actor.obs_groups if actor is not None else list(spec.observations)
  total = 0
  known = True
  parts = []
  for name in group_names:
    dim = spec.observations.get(name).dim if name in spec.observations else None
    if dim is None:
      known = False
      parts.append(f"{name}(?)")
    else:
      total += dim
      parts.append(f"{name}({dim})")
  total_label = str(total) if known else "?"
  return "Observation groups\nactor input dim: " + total_label + "\n" + _wrap_text(", ".join(parts), 40)


def _actor_summary_label(spec: PipelineSpec) -> str:
  actor = spec.actor
  if actor is None:
    return "Actor policy\nunknown"
  if actor.class_name == "policy.onnx":
    return "ONNX policy\nobs tensors -> a_raw"
  model = _short_class(actor.class_name)
  if actor.encoder_type:
    return f"Actor policy\n{model} + {actor.encoder_type} encoder\nMLP: {_mlp_dims(actor.hidden_dims, spec.action.dim)}"
  return f"Actor policy\n{model}\nMLP: {_mlp_dims(actor.hidden_dims, spec.action.dim)}"


def _action_processing_summary_label(spec: PipelineSpec) -> str:
  return spec.action.kind + "\nscale + offset\nraw action -> target"


def _observation_detail_label(spec: PipelineSpec) -> str:
  lines = ["Observation detail"]
  for group in spec.observations.values():
    terms = []
    for term in group.terms:
      suffix = f"({term.dim})" if term.dim is not None else "(?)"
      terms.append(f"{term.name}{suffix}")
    dim = f"{group.dim}" if group.dim is not None else "?"
    lines.append(f"{group.name}: {dim}")
    lines.extend("  " + line for line in textwrap.wrap(", ".join(terms), width=58))
  return "\n".join(lines)


def _actor_detail_label(spec: PipelineSpec) -> str:
  actor = spec.actor
  if actor is None:
    return "Actor detail\nunknown"
  if actor.class_name == "policy.onnx":
    return "Actor detail\npolicy.onnx\ninput names match observation groups\noutput: deterministic raw action vector"
  lines = [f"Actor detail: {_short_class(actor.class_name)}"]
  if actor.encoder_type:
    lines.append(f"encoder: {_encoder_summary(actor)}")
    lines.append("encoder inputs: " + ", ".join(actor.encoder_input_keys))
    lines.append("passthrough: " + (", ".join(actor.passthrough_keys) or "none"))
    lines.append("fusion: latent + normalized passthrough -> concat")
  else:
    lines.append("plain path: concat obs groups -> normalizer -> MLP")
  lines.append(f"normalization: {_yes_no(actor.obs_normalization)}")
  lines.append(f"activation: {actor.activation}")
  lines.append(f"MLP: {_mlp_dims(actor.hidden_dims, spec.action.dim)}")
  if actor.distribution_cfg:
    lines.append(f"distribution: {actor.distribution_cfg.get('class_name', 'distribution')}")
  return "\n".join(_wrap_preserving_prefix(lines, 58))


def _action_detail_label(spec: PipelineSpec) -> str:
  action = spec.action
  lines = [
    "Action detail",
    "a_raw is the actor output, not q.",
    f"{action.target_label} = a_raw * scale + offset",
  ]
  if action.default_joint_pos is not None or action.offset == "default_joint_pos":
    lines.append("offset = default_joint_pos")
  if action.clip:
    lines.append("clip applied after affine transform")
  if action.joint_ids_map is not None:
    lines.append("joint_ids_map = " + _short_list(action.joint_ids_map, max_items=12))
  if action.stiffness and action.damping:
    lines.append("deploy writes q_target into motor_cmd.q")
    lines.append("PD gains kp/kd are configured separately")
  else:
    lines.append("training applies JointPositionAction target")
  return "\n".join(_wrap_preserving_prefix(lines, 54))


def _observation_label(spec: PipelineSpec) -> str:
  lines = ["Observation terms"]
  for group in spec.observations.values():
    term_bits = []
    for term in group.terms:
      suffix = f"({term.dim})" if term.dim is not None else "(?)"
      term_bits.append(f"{term.name}{suffix}")
    joined = ", ".join(term_bits)
    if len(joined) > 86:
      joined = joined[:83] + "..."
    dim = f" -> {group.dim}" if group.dim is not None else ""
    lines.append(f"{group.name}{dim}: {joined}")
  return "\n".join(lines)


def _preprocess_label(spec: PipelineSpec) -> str:
  has_clip = any(t.clip for g in spec.observations.values() for t in g.terms)
  has_scale = any(t.scale is not None for g in spec.observations.values() for t in g.terms)
  has_history = any(t.history_length > 1 for g in spec.observations.values() for t in g.terms)
  steps = ["concat/group"]
  if has_scale:
    steps.insert(0, "scale")
  if has_clip:
    steps.insert(0, "clip")
  if has_history:
    steps.append("history")
  if spec.source.startswith("task"):
    steps.insert(0, "noise during training")
  return "Observation preprocessing\n" + " -> ".join(steps)


def _actor_label(spec: PipelineSpec) -> str:
  actor = spec.actor
  if actor is None:
    return "Actor policy\nunknown"
  if actor.class_name == "policy.onnx":
    return "ONNX policy\ninput names from observation groups\noutput: raw action"

  lines = [f"Actor: {_short_class(actor.class_name)}"]
  if actor.encoder_type:
    lines.append(f"encoder: {_encoder_summary(actor)}")
    lines.append(f"encoder inputs: {', '.join(actor.encoder_input_keys)}")
    lines.append(f"passthrough: {', '.join(actor.passthrough_keys) or 'none'}")
    lines.append("concat encoder latent + passthrough")
  else:
    lines.append("plain observation path")
  lines.append(f"normalization: {_yes_no(actor.obs_normalization)}")
  lines.append(f"MLP: {_mlp_dims(actor.hidden_dims, spec.action.dim)}")
  if actor.distribution_cfg:
    lines.append(f"distribution: {actor.distribution_cfg.get('class_name', 'distribution')}")
  return "\n".join(lines)


def _raw_action_label(spec: PipelineSpec) -> str:
  return f"Actor output\na_raw ({spec.action.dim})\nnormalized action delta\nnot final q"


def _action_processing_label(spec: PipelineSpec) -> str:
  lines = [spec.action.kind, "q_target = a_raw * scale + offset"]
  if spec.action.clip:
    lines.append("clip processed action")
  if spec.action.default_joint_pos is not None or spec.action.offset == "default_joint_pos":
    lines.append("offset = default_joint_pos")
  return "\n".join(lines)


def _target_label(spec: PipelineSpec) -> str:
  action = spec.action
  lines = [f"Final command: {action.target_label} ({action.dim})"]
  if action.joint_ids_map is not None:
    lines.append(f"joint_ids_map: {_short_list(action.joint_ids_map)}")
  if action.stiffness and action.damping:
    lines.append("PD gains set separately")
    lines.append("motor q field in deploy")
  else:
    lines.append("joint position target")
  return "\n".join(lines)


def _equation_strip(ax: Any, spec: PipelineSpec) -> None:
  action = spec.action
  text = (
    "Key distinction: actor output is a_raw. "
    f"{action.target_label} = a_raw * scale + offset"
  )
  if "Position" in action.kind and spec.deploy_yaml:
    text += "; deploy writes q_target to lowcmd.motor_cmd[...].q()."
  elif "Position" in action.kind:
    text += "; JointPositionAction applies it as a joint position target."
  _box(ax, 0.5, 0.43, text, color="#ffffff", width=0.76, height=0.08, fontsize=11)


def _critic_obs_label(spec: PipelineSpec) -> str:
  if spec.critic is None:
    return "Critic observations"
  bits = []
  for group_name in spec.critic.obs_groups:
    group = spec.observations.get(group_name)
    if group is None:
      bits.append(group_name)
    else:
      dim = f" ({group.dim})" if group.dim is not None else ""
      bits.append(f"{group_name}{dim}")
  return "Critic observations\n" + ", ".join(bits)


def _critic_model_label(spec: PipelineSpec) -> str:
  critic = spec.critic
  if critic is None:
    return "Critic model"
  return (
    f"Critic: {_short_class(critic.class_name)}\n"
    f"MLP: {_mlp_dims(critic.hidden_dims, 1)}\n"
    f"activation: {critic.activation}"
  )


def _encoder_summary(actor: ModelSpec) -> str:
  cfg = actor.observation_encoder_cfg or {}
  enc_type = actor.encoder_type or "unknown"
  if enc_type == "mlp":
    return f"MLP {cfg.get('hidden_dims', [])} -> latent {cfg.get('latent_dim')}"
  if enc_type in {"conv1d", "conv2d"}:
    parts = [
      enc_type,
      f"channels {cfg.get('channels')}",
      f"k {cfg.get('kernel_sizes')}",
      f"s {cfg.get('strides')}",
      f"pool {cfg.get('global_pool')}",
      f"latent {cfg.get('latent_dim')}",
    ]
    if cfg.get("input_hw"):
      parts.insert(1, f"input_hw {cfg.get('input_hw')}")
    if cfg.get("context_keys"):
      parts.append(f"context {cfg.get('context_keys')}")
    return ", ".join(parts)
  if enc_type == "identity":
    return "identity flatten/concat"
  return enc_type


def _mlp_dims(hidden_dims: list[int], output_dim: int) -> str:
  if hidden_dims:
    return " -> ".join([*(str(v) for v in hidden_dims), str(output_dim)])
  return f"ONNX/output {output_dim}"


def _short_class(class_name: str) -> str:
  return class_name.split(":")[-1].split(".")[-1]


def _short_list(values: list[int] | list[float], max_items: int = 6) -> str:
  shown = ", ".join(str(v) for v in values[:max_items])
  if len(values) > max_items:
    shown += ", ..."
  return f"[{shown}]"


def _yes_no(value: bool) -> str:
  return "yes" if value else "no"


def _wrap_text(text: str, width: int) -> str:
  return "\n".join(textwrap.wrap(text, width=width)) if text else ""


def _wrap_preserving_prefix(lines: list[str], width: int) -> list[str]:
  wrapped: list[str] = []
  for line in lines:
    if len(line) <= width:
      wrapped.append(line)
    else:
      wrapped.extend(textwrap.wrap(line, width=width))
  return wrapped


def _panel(
  ax: Any,
  x: float,
  y: float,
  text: str,
  *,
  color: str,
  fontsize: float = 8.5,
) -> None:
  ax.text(
    x,
    y,
    text,
    ha="center",
    va="center",
    fontsize=fontsize,
    color="#0f172a",
    linespacing=1.22,
    bbox={
      "boxstyle": "round,pad=0.55,rounding_size=0.04",
      "facecolor": color,
      "edgecolor": "#94a3b8",
      "linewidth": 1.0,
    },
  )


def _box(
  ax: Any,
  x: float,
  y: float,
  text: str,
  *,
  color: str,
  width: float = 0.12,
  height: float = 0.26,
  fontsize: int = 9,
) -> None:
  ax.text(
    x,
    y,
    text,
    ha="center",
    va="center",
    fontsize=fontsize,
    color="#0f172a",
    wrap=True,
    bbox={
      "boxstyle": "round,pad=0.42,rounding_size=0.04",
      "facecolor": color,
      "edgecolor": "#64748b",
      "linewidth": 1.0,
    },
  )


def _arrow(ax: Any, x0: float, y0: float, x1: float, y1: float) -> None:
  ax.annotate(
    "",
    xy=(x1, y1),
    xytext=(x0, y0),
    arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 1.8},
  )


def _parse_obs_dim(values: list[str] | None) -> dict[str, int]:
  dims: dict[str, int] = {}
  for item in values or []:
    if "=" not in item:
      raise ValueError(f"--obs-dim must be key=value, got: {item}")
    key, value = item.split("=", 1)
    dims[key] = int(value)
  return dims


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task_id", nargs="?", help="Registered mjlab task id.")
  parser.add_argument("--task", dest="task_opt", help="Registered mjlab task id.")
  parser.add_argument("--deploy-yaml", type=Path, help="Deploy params YAML.")
  parser.add_argument("--out", required=True, type=Path, help="Output SVG/PNG/Mermaid path. SVG/PNG exports also write <stem>_mmd.mmd.")
  parser.add_argument("--format", choices=("svg", "png", "mmd", "mermaid"), default=None)
  parser.add_argument("--no-critic", action="store_true", help="Hide critic row.")
  parser.add_argument(
    "--obs-dim",
    action="append",
    default=[],
    help="Override an observation term dimension, e.g. --obs-dim height_scan=187.",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  task_id = args.task_opt or args.task_id
  overrides = _parse_obs_dim(args.obs_dim)

  if task_id:
    spec = load_task_pipeline(task_id, overrides)
    if args.deploy_yaml:
      spec = merge_deploy_metadata(spec, args.deploy_yaml)
  elif args.deploy_yaml:
    spec = load_deploy_pipeline(args.deploy_yaml, overrides)
  else:
    raise SystemExit("Provide a task id, --task, or --deploy-yaml.")

  out = render_pipeline(
    spec,
    args.out,
    output_format=args.format,
    include_critic=not args.no_critic,
  )
  print(f"Wrote {out}")

  fmt = args.format or out.suffix.lstrip(".")
  if fmt not in ("mmd", "mermaid"):
    mmd_out = out.with_name(f"{out.stem}_mmd.mmd")
    mmd_out.write_text(render_mermaid_pipeline(spec, include_critic=not args.no_critic))
    print(f"Wrote {mmd_out}")


if __name__ == "__main__":
  main()
