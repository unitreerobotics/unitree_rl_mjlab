"""Observation attribution utilities for live policy playback."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Literal

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn


TargetFn = Callable[[torch.Tensor], torch.Tensor]
AttributionMethodName = Literal[
  "gradient_saliency",
  "gradient_input",
  "integrated_gradients",
  "deep_lift_rescale",
  "deep_shap",
]


@dataclass(frozen=True)
class ObservationTermSlice:
  """Metadata for one flattened observation term."""

  name: str
  shape: tuple[int, ...]
  start: int
  stop: int


@dataclass(frozen=True)
class AttributionTerm:
  """Attribution scores for one named observation term."""

  name: str
  values: np.ndarray
  shape: tuple[int, ...]
  start: int
  stop: int

  @property
  def mean_score(self) -> float:
    return float(np.mean(self.values)) if self.values.size else 0.0

  @property
  def max_score(self) -> float:
    return float(np.max(self.values)) if self.values.size else 0.0

  @property
  def sum_score(self) -> float:
    return float(np.sum(self.values)) if self.values.size else 0.0


@dataclass(frozen=True)
class AttributionMapResult:
  """Grouped attribution result for one target map."""

  name: str
  group_name: str
  terms: tuple[AttributionTerm, ...]

  def get_term(self, term_name: str) -> AttributionTerm | None:
    for term in self.terms:
      if term.name == term_name:
        return term
    return None

  @property
  def max_score(self) -> float:
    if not self.terms:
      return 0.0
    return max(term.max_score for term in self.terms)


class AttributionMethod(ABC):
  """Interface for swappable attribution methods."""

  @abstractmethod
  def compute(
    self,
    model: nn.Module,
    obs: TensorDictBase,
    group_name: str,
    target_fn: TargetFn,
    env_idx: int,
  ) -> torch.Tensor:
    """Return a flat attribution score for each raw element in ``group_name``."""


class GradientSaliency(AttributionMethod):
  """Absolute input-gradient saliency."""

  def compute(
    self,
    model: nn.Module,
    obs: TensorDictBase,
    group_name: str,
    target_fn: TargetFn,
    env_idx: int,
  ) -> torch.Tensor:
    if group_name not in obs.keys():
      raise KeyError(f"Observation group '{group_name}' is not present.")

    obs_for_grad = _clone_obs_for_grad(obs, group_name)
    raw_group = obs_for_grad[group_name]
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
      output = model(obs_for_grad)
      target = target_fn(output)[env_idx]
      if target.ndim != 0:
        target = target.sum()
      grad = torch.autograd.grad(
        target,
        raw_group,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
      )[0]

    return grad[env_idx].detach().abs().reshape(-1)


class GradientInput(AttributionMethod):
  """Absolute gradient times input attribution."""

  def compute(
    self,
    model: nn.Module,
    obs: TensorDictBase,
    group_name: str,
    target_fn: TargetFn,
    env_idx: int,
  ) -> torch.Tensor:
    if group_name not in obs.keys():
      raise KeyError(f"Observation group '{group_name}' is not present.")

    obs_for_grad = _clone_obs_for_grad(obs, group_name)
    raw_group = obs_for_grad[group_name]
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
      output = model(obs_for_grad)
      target = target_fn(output)[env_idx]
      if target.ndim != 0:
        target = target.sum()
      grad = torch.autograd.grad(
        target,
        raw_group,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
      )[0]

    contribution = raw_group[env_idx].detach() * grad[env_idx].detach()
    return contribution.abs().reshape(-1)


class IntegratedGradients(AttributionMethod):
  """Integrated Gradients from a baseline to the current observation."""

  def __init__(self, steps: int = 16) -> None:
    self.steps = steps
    self.baseline_source = "normalizer mean"

  def compute(
    self,
    model: nn.Module,
    obs: TensorDictBase,
    group_name: str,
    target_fn: TargetFn,
    env_idx: int,
  ) -> torch.Tensor:
    if group_name not in obs.keys():
      raise KeyError(f"Observation group '{group_name}' is not present.")

    steps = max(2, int(self.steps))
    current = obs[group_name][env_idx].detach()
    baseline = _normalizer_mean_baseline(model, current)
    alphas = torch.linspace(
      0.0, 1.0, steps + 1, device=current.device, dtype=current.dtype
    )
    path = baseline.unsqueeze(0) + alphas.reshape(-1, 1) * (
      current - baseline
    ).unsqueeze(0)
    path.requires_grad_(True)

    obs_for_grad = _clone_obs_for_path(obs, group_name, env_idx, path)
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
      output = model(obs_for_grad)
      targets = target_fn(output)
      target = targets.sum()
      grads = torch.autograd.grad(
        target,
        path,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
      )[0]

    weights = torch.ones(steps + 1, device=current.device, dtype=current.dtype)
    weights[0] = 0.5
    weights[-1] = 0.5
    avg_grad = (grads * weights.reshape(-1, 1)).sum(dim=0) / steps
    integrated = (current - baseline) * avg_grad
    return integrated.detach().abs().reshape(-1)


class DeepShap(AttributionMethod):
  """Background-reference DeepSHAP-style attribution.

  This uses the paper's background-distribution framing with differentiable
  PyTorch modules by averaging contribution multipliers across reference
  observations. It is intentionally dependency-free for live playback.
  """

  def __init__(self, samples: int = 16) -> None:
    self.samples = samples
    self.baseline_source = "batch observations + normalizer mean"

  def compute(
    self,
    model: nn.Module,
    obs: TensorDictBase,
    group_name: str,
    target_fn: TargetFn,
    env_idx: int,
  ) -> torch.Tensor:
    if group_name not in obs.keys():
      raise KeyError(f"Observation group '{group_name}' is not present.")

    current = obs[group_name][env_idx].detach()
    refs = _background_references(
      model,
      obs,
      group_name,
      env_idx,
      current,
      max_refs=max(1, int(self.samples)),
    )
    if refs.numel() == 0:
      return torch.zeros_like(current).reshape(-1)

    alphas = torch.linspace(
      0.0,
      1.0,
      refs.shape[0] + 2,
      device=current.device,
      dtype=current.dtype,
    )[1:-1]
    path = refs + alphas.reshape(-1, *([1] * current.ndim)) * (
      current.unsqueeze(0) - refs
    )
    path.requires_grad_(True)

    obs_for_grad = _clone_obs_for_path(obs, group_name, env_idx, path)
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
      output = model(obs_for_grad)
      target = target_fn(output).sum()
      grads = torch.autograd.grad(
        target,
        path,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
      )[0]

    contributions = (current.unsqueeze(0) - refs) * grads
    return contributions.mean(dim=0).detach().abs().reshape(-1)


class DeepLiftRescale(AttributionMethod):
  """DeepLIFT attribution using the Rescale rule.

  The implementation runs the current observation and a normalizer-mean
  reference together, then replaces elementwise activation derivatives with
  finite-difference multipliers during backpropagation.
  """

  def __init__(self, eps: float = 1.0e-6) -> None:
    self.eps = eps
    self.baseline_source = "normalizer mean"

  def compute(
    self,
    model: nn.Module,
    obs: TensorDictBase,
    group_name: str,
    target_fn: TargetFn,
    env_idx: int,
  ) -> torch.Tensor:
    if group_name not in obs.keys():
      raise KeyError(f"Observation group '{group_name}' is not present.")

    current = obs[group_name][env_idx].detach()
    baseline = _normalizer_mean_baseline(model, current)
    path = torch.stack((current, baseline), dim=0).requires_grad_(True)
    obs_for_grad = _clone_obs_for_path(obs, group_name, env_idx, path)
    model.zero_grad(set_to_none=True)

    with _deeplift_rescale_hooks(model, self.eps):
      with torch.enable_grad():
        output = model(obs_for_grad)
        target = target_fn(output)[0]
        if target.ndim != 0:
          target = target.sum()
        multipliers = torch.autograd.grad(
          target,
          path,
          retain_graph=False,
          create_graph=False,
          allow_unused=False,
        )[0]

    contribution = (current - baseline) * multipliers[0].detach()
    return contribution.abs().reshape(-1)


class ObservationAttributionComputer:
  """Compute and group attribution maps from ObservationManager metadata."""

  def __init__(
    self,
    observation_manager,
    method: AttributionMethod | AttributionMethodName | None = None,
  ):
    self._observation_manager = observation_manager
    self._method = create_attribution_method(method)
    self._metadata_cache: dict[str, tuple[ObservationTermSlice, ...]] = {}

  @property
  def method(self) -> AttributionMethod:
    return self._method

  def set_method(self, method: AttributionMethod | AttributionMethodName) -> None:
    self._method = create_attribution_method(method)

  def compute_action(
    self,
    actor: nn.Module,
    obs: TensorDictBase,
    env_idx: int,
  ) -> AttributionMapResult:
    scores = self._method.compute(
      actor,
      obs,
      "actor",
      lambda output: output.norm(p=2, dim=-1),
      env_idx,
    )
    return self._group_scores("action", "actor", scores)

  def compute_value(
    self,
    critic: nn.Module,
    obs: TensorDictBase,
    env_idx: int,
  ) -> AttributionMapResult:
    scores = self._method.compute(
      critic,
      obs,
      "critic",
      lambda output: output.reshape(output.shape[0], -1)[:, 0],
      env_idx,
    )
    return self._group_scores("value", "critic", scores)

  def _group_scores(
    self, map_name: str, group_name: str, scores: torch.Tensor
  ) -> AttributionMapResult:
    metadata = self._metadata(group_name)
    scores_np = scores.detach().cpu().numpy()
    terms = []
    for term in metadata:
      values = scores_np[term.start : term.stop]
      terms.append(
        AttributionTerm(
          name=term.name,
          values=values,
          shape=term.shape,
          start=term.start,
          stop=term.stop,
        )
      )
    return AttributionMapResult(
      name=map_name, group_name=group_name, terms=tuple(terms)
    )

  def _metadata(self, group_name: str) -> tuple[ObservationTermSlice, ...]:
    if group_name in self._metadata_cache:
      return self._metadata_cache[group_name]

    obs_manager = self._observation_manager
    if not obs_manager.group_obs_concatenate.get(group_name, False):
      raise ValueError(
        f"Attribution currently supports concatenated observation groups only; "
        f"'{group_name}' is not concatenated."
      )

    names = obs_manager.active_terms[group_name]
    dims = obs_manager.group_obs_term_dim[group_name]

    offset = 0
    metadata: list[ObservationTermSlice] = []
    for name, shape in zip(names, dims, strict=True):
      length = int(np.prod(shape))
      metadata.append(
        ObservationTermSlice(
          name=name,
          shape=tuple(int(v) for v in shape),
          start=offset,
          stop=offset + length,
        )
      )
      offset += length

    self._metadata_cache[group_name] = tuple(metadata)
    return self._metadata_cache[group_name]


def _clone_obs_for_grad(obs: TensorDictBase, grad_group: str) -> TensorDict:
  cloned = {}
  for key in obs.keys():
    value = obs[key]
    if not isinstance(value, torch.Tensor):
      raise TypeError(f"Observation group '{key}' is not a tensor.")
    tensor = value.detach().clone()
    if key == grad_group:
      tensor.requires_grad_(True)
    cloned[key] = tensor
  return TensorDict(cloned, batch_size=obs.batch_size, device=obs.device)


def _clone_obs_for_path(
  obs: TensorDictBase,
  grad_group: str,
  env_idx: int,
  path: torch.Tensor,
) -> TensorDict:
  cloned = {}
  batch_size = path.shape[0]
  for key in obs.keys():
    value = obs[key]
    if not isinstance(value, torch.Tensor):
      raise TypeError(f"Observation group '{key}' is not a tensor.")
    if key == grad_group:
      cloned[key] = path
    else:
      selected = value[env_idx].detach().clone()
      cloned[key] = selected.unsqueeze(0).expand(batch_size, *selected.shape).clone()
  return TensorDict(cloned, batch_size=[batch_size], device=obs.device)


def create_attribution_method(
  method: AttributionMethod | AttributionMethodName | str | None,
) -> AttributionMethod:
  if method is None:
    return IntegratedGradients()
  if isinstance(method, AttributionMethod):
    return method

  normalized = method.lower().replace("-", "_").replace(" ", "_")
  if normalized in {"gradient", "gradient_saliency", "saliency"}:
    return GradientSaliency()
  if normalized in {
    "gradient_input",
    "gradient_x_input",
    "grad_input",
    "grad_x_input",
    "simple_gradient_input",
    "simple_gradient_x_input",
  }:
    return GradientInput()
  if normalized in {"integrated_gradients", "ig"}:
    return IntegratedGradients()
  if normalized in {
    "deep_lift",
    "deeplift",
    "deep_lift_rescale",
    "deeplift_rescale",
  }:
    return DeepLiftRescale()
  if normalized in {"deep_shap", "deepshap", "deep_shapley"}:
    return DeepShap()
  raise ValueError(f"Unsupported attribution method: {method}")


_DEEPLIFT_NONLINEAR_MODULES = (
  nn.CELU,
  nn.ELU,
  nn.GELU,
  nn.LeakyReLU,
  nn.Mish,
  nn.ReLU,
  nn.SELU,
  nn.Sigmoid,
  nn.SiLU,
  nn.Softplus,
  nn.Tanh,
)


@contextmanager
def _deeplift_rescale_hooks(model: nn.Module, eps: float) -> Iterator[None]:
  handles = []
  hooked_modules = []

  def forward_hook(module, inputs, output) -> None:
    if not inputs or not isinstance(inputs[0], torch.Tensor):
      return
    if not isinstance(output, torch.Tensor):
      return
    if inputs[0].shape[0] < 2 or output.shape[0] < 2:
      return
    contexts = getattr(module, "_deeplift_rescale_contexts", None)
    if contexts is None:
      contexts = []
      setattr(module, "_deeplift_rescale_contexts", contexts)
    contexts.append((inputs[0].detach(), output.detach()))

  def backward_hook(module, grad_input, grad_output):
    contexts = getattr(module, "_deeplift_rescale_contexts", None)
    if not contexts or not grad_input or not grad_output:
      return None
    if grad_input[0] is None or grad_output[0] is None:
      return None

    in_pair, out_pair = contexts.pop()
    grad_in = grad_input[0]
    grad_out = grad_output[0]
    if in_pair.shape[0] < 2 or out_pair.shape[0] < 2 or grad_out.shape[0] < 1:
      return None

    delta_in = in_pair[0] - in_pair[1]
    delta_out = out_pair[0] - out_pair[1]
    safe_delta_in = torch.where(
      delta_in.abs() > eps, delta_in, torch.ones_like(delta_in)
    )
    rescale_slope = delta_out / safe_delta_in

    default_slope = torch.zeros_like(rescale_slope)
    current_grad_out = grad_out[0]
    current_grad_in = grad_in[0]
    nonzero_grad_out = current_grad_out.abs() > eps
    safe_grad_out = torch.where(
      nonzero_grad_out, current_grad_out, torch.ones_like(current_grad_out)
    )
    default_slope = torch.where(
      nonzero_grad_out, current_grad_in / safe_grad_out, default_slope
    )
    local_slope = torch.where(delta_in.abs() > eps, rescale_slope, default_slope)

    modified_grad_in = grad_in.clone()
    modified_grad_in[0] = current_grad_out * local_slope
    return (modified_grad_in,) + tuple(grad_input[1:])

  for module in model.modules():
    if isinstance(module, _DEEPLIFT_NONLINEAR_MODULES):
      setattr(module, "_deeplift_rescale_contexts", [])
      hooked_modules.append(module)
      handles.append(module.register_forward_hook(forward_hook))
      handles.append(module.register_full_backward_hook(backward_hook))

  try:
    yield
  finally:
    for handle in handles:
      handle.remove()
    for module in hooked_modules:
      if hasattr(module, "_deeplift_rescale_contexts"):
        delattr(module, "_deeplift_rescale_contexts")


def _background_references(
  model: nn.Module,
  obs: TensorDictBase,
  group_name: str,
  env_idx: int,
  current: torch.Tensor,
  max_refs: int,
) -> torch.Tensor:
  references = [
    _normalizer_mean_baseline(model, current),
    torch.zeros_like(current),
  ]

  batch = obs[group_name].detach()
  if batch.ndim >= 1:
    for index in range(batch.shape[0]):
      if index == env_idx and batch.shape[0] > 1:
        continue
      candidate = batch[index]
      if candidate.numel() == current.numel():
        references.append(candidate.reshape_as(current).to(current))

  unique_refs = []
  for reference in references:
    if len(unique_refs) >= max_refs:
      break
    reference = reference.reshape_as(current).to(
      device=current.device, dtype=current.dtype
    )
    if torch.allclose(reference, current):
      continue
    if any(torch.allclose(reference, existing) for existing in unique_refs):
      continue
    unique_refs.append(reference)

  if not unique_refs:
    unique_refs.append(torch.zeros_like(current))
  return torch.stack(unique_refs, dim=0)


def _normalizer_mean_baseline(model: nn.Module, current: torch.Tensor) -> torch.Tensor:
  normalizer = getattr(model, "obs_normalizer", None)
  mean = getattr(normalizer, "mean", None)
  if isinstance(mean, torch.Tensor) and mean.numel() == current.numel():
    return mean.reshape_as(current).to(device=current.device, dtype=current.dtype)
  return torch.zeros_like(current)
