"""Left-right mirror symmetry for the R1 velocity tasks (rsl_rl's symmetry_cfg).

PPO learns from every transition twice: as recorded and mirrored about the robot's sagittal
plane (x-z plane of the pelvis). The policy can then no longer prefer one side; checkpoint 15300
of the perceptive task turned left in 20 of 20 offline approaches to a step, at 30-40 deg.

Mirror rules, checked by forward kinematics on r1.xml (all bodies mirror exactly):
- joints: left and right swap; roll and yaw joints change sign (all R1 joint axes are x, y or
  z in their body frames, and those frames mirror), pitch, knee and elbow keep it. The home
  pose mirrors itself (shoulder roll +0.18 / -0.18), so do positions relative to it, and the
  action scales are equal per left/right pair, so the actions mirror like the joints.
- vectors in the pelvis frame: polar (velocity, gravity) negate y; axial (angular velocity)
  negate x and z.
- command (vx, vy, wz): vy and wz change sign.
- gait phase (sin, cos of a clock with the legs half a period apart): mirroring swaps the legs,
  i.e. shifts the clock by half a period: both change sign (0, 0 when standing stays).
- height scan: the grid's rows (y) reverse; the grid is centred on the pelvis.
- per-foot terms swap; foot contact forces (world frame) also negate y, exact when the robot
  faces world x and approximate otherwise (critic only).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

_POLAR = torch.tensor([1.0, -1.0, 1.0])
_AXIAL = torch.tensor([-1.0, 1.0, -1.0])


def _partner(name: str) -> str:
  return name.replace("left_", "\0").replace("right_", "left_").replace("\0", "right_")


class _Mirror:
  """Index permutation and signs for one observation group, and for the actions."""

  def __init__(self, env) -> None:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    joints = list(robot.joint_names)
    self.joint_perm = torch.tensor([joints.index(_partner(n)) for n in joints])
    self.joint_sign = torch.tensor([-1.0 if ("_roll" in n or "_yaw" in n) else 1.0 for n in joints])

    action_term = unwrapped.action_manager._terms["joint_pos"]
    action_joints = list(action_term._target_names)
    assert action_joints == joints, "actions are expected in the robot's joint order"

    scan = next(s for s in unwrapped.cfg.scene.sensors if s.name == "terrain_scan")
    size_x, size_y = scan.pattern.size
    res = scan.pattern.resolution
    nx, ny = round(size_x / res) + 1, round(size_y / res) + 1
    # mjlab's GridPatternCfg: meshgrid(x, y, indexing="xy").flatten() -> index iy * nx + ix
    self.scan_perm = torch.tensor([(ny - 1 - iy) * nx + ix for iy in range(ny) for ix in range(nx)])

    om = unwrapped.observation_manager
    self.groups: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for group in om.active_terms:
      perm, sign, start = [], [], 0
      for name, dims in zip(om.active_terms[group], om.group_obs_term_dim[group]):
        n = int(torch.tensor(dims).prod())
        idx = torch.arange(start, start + n)
        p, s = self._term(name, n)
        perm.append(idx[p])
        sign.append(s)
        start += n
      self.groups[group] = (torch.cat(perm), torch.cat(sign))
    self.device = None

  def _term(self, name: str, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    ar = torch.arange(n)
    ones = torch.ones(n)
    if name in ("base_lin_vel", "projected_gravity"):
      return ar, _POLAR.clone()
    if name == "base_ang_vel":
      return ar, _AXIAL.clone()
    if name == "command":
      return ar, torch.tensor([1.0, -1.0, -1.0])
    if name == "phase":
      return ar, -ones
    if name in ("joint_pos", "joint_vel", "actions"):
      return self.joint_perm.clone(), self.joint_sign.clone()
    if name == "height_scan":
      return self.scan_perm.clone(), ones
    if name in ("foot_height", "foot_air_time", "foot_contact"):
      assert n == 2, name
      return torch.tensor([1, 0]), ones
    if name == "foot_contact_forces":
      assert n == 6, name
      return torch.tensor([3, 4, 5, 0, 1, 2]), torch.cat([_POLAR, _POLAR])
    raise KeyError(f"no mirror rule for observation term '{name}'")

  def to(self, device) -> None:
    if self.device != device:
      self.joint_perm, self.joint_sign = self.joint_perm.to(device), self.joint_sign.to(device)
      self.groups = {g: (p.to(device), s.to(device)) for g, (p, s) in self.groups.items()}
      self.device = device


_MIRRORS: dict[int, _Mirror] = {}


def mirror_r1(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
  """rsl_rl data_augmentation_func: returns [original; mirrored] for obs and actions."""
  key = id(env)
  if key not in _MIRRORS:
    _MIRRORS[key] = _Mirror(env)
  mirror = _MIRRORS[key]

  obs_aug = None
  if obs is not None:
    device = next(iter(obs.values())).device
    mirror.to(device)
    mirrored = {}
    for group, value in obs.items():
      perm, sign = mirror.groups[group]
      mirrored[group] = value[..., perm] * sign
    obs_aug = torch.cat([obs, TensorDict(mirrored, batch_size=obs.batch_size, device=obs.device)], dim=0)

  actions_aug = None
  if actions is not None:
    mirror.to(actions.device)
    actions_aug = torch.cat([actions, actions[..., mirror.joint_perm] * mirror.joint_sign], dim=0)
  return obs_aug, actions_aug
