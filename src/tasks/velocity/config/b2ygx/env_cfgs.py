"""Unitree B2YGX velocity environment configurations."""

from typing import Literal

from src.assets.robots import get_b2ygx_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
import mjlab.terrains as terrain_gen

TerrainType = Literal["rough", "obstacles"]


def unitree_b2ygx_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree B2YGX rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 256
  cfg.sim.njmax = 2000

  cfg.scene.entities = {"robot": get_b2ygx_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=tuple(geom_names) + (r".*calf\d_collision",),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  calves_ground_cfg = ContactSensorCfg(
    name="calves_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*calf\d_collision",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
    calves_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 0.25

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["track_linear_velocity"].weight = 1.5
  cfg.rewards["track_angular_velocity"].weight = 1.0

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards.pop("foot_clearance", None)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg

def custom_terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  # 计算机器人行进物理位移
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # 升级条件：行进距离超过地形块长度的一半（4.0米）
  move_up = distance > terrain_generator.size[0] / 2

  # 降级条件：完全禁用降级！避免被无理重置降级回 Level 0
  move_down = torch.zeros_like(move_up, dtype=torch.bool)

  # 更新地形关卡
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def unitree_b2ygx_rough_no_height_actor_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create B2YGX rough terrain config without actor height scan."""
  cfg = unitree_b2ygx_rough_env_cfg(play=play)
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["actor"].history_length = 24

  # 注册防降级的自适应课程函数，并彻底剥离危险的速度膨胀课程项
  if "terrain_levels" in cfg.curriculum:
    cfg.curriculum["terrain_levels"].func = custom_terrain_levels_vel
  cfg.curriculum.pop("command_vel", None)

  # 1. 覆盖地形生成器，配置专属的自适应上下台阶场景
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.sub_terrains = {
      "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.02),
      "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
        proportion=0.48,
        step_height_range=(0.0, 0.14),
        step_width=0.32,
        platform_width=3.0,
        border_width=1.0,
      ),
      "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
        proportion=0.30,
        step_height_range=(0.0, 0.14),
        step_width=0.32,
        platform_width=3.0,
        border_width=1.0,
      ),
      "random_stairs": terrain_gen.BoxRandomStairsTerrainCfg(
        proportion=0.20,
        step_width=0.8,
        step_height_range=(0.05, 0.14),
        platform_width=1.0,
        border_width=0.25,
      ),
    }

  # 2. 已彻底从基础奖励中移除了基于绝对 Z 坐标的 foot_clearance，避免盲爬时因绝对高度偏差导致双脚锁死。

  # 3. 放宽大腿/小腿在行走与运行状态下的姿态标准差约束，允许更大范围的抬脚迈步动作
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.40,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.55,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.40,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.55,
  }

  # 4. 限制训练时的速度指令范围以适配盲爬上下楼梯：前向 1.0 m/s，后退 -0.5 m/s，侧向 0.5 m/s，转向 0.3 rad/s
  twist_cmd = cfg.commands.get("twist", None)
  if twist_cmd is not None and isinstance(twist_cmd, UniformVelocityCommandCfg):
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)

  # 5. 减低小腿接触的惩罚：排除小腿触发 Hendrick_contact 终止，替换为轻微的碰撞惩罚 (weight=-0.1)
  cfg.rewards["calf_contact_penalty"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": "calves_ground_touch", "force_threshold": 10.0},
  )

  # 6. 减低 angular_momentum 惩罚权重，给躯干解绑以在台阶阶跃中灵活扭动身体和调整重心
  cfg.rewards["angular_momentum"].weight = -0.015



  return cfg


def unitree_b2ygx_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree B2YGX flat terrain velocity configuration."""
  cfg = unitree_b2ygx_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.curriculum.pop("terrain_levels", None)
  cfg.observations["actor"].history_length = 24

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
