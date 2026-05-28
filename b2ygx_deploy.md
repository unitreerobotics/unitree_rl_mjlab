# Unitree B2YGX 部署代码分析

本文档详细分析 Unitree B2YGX 四足机器人的部署代码架构，包括目录结构、核心模块、数据流、状态机设计和关键配置。

## 1. 目录结构

```
deploy/robots/b2ygx/
├── main.cpp                      # 程序入口
├── CMakeLists.txt                # CMake 构建配置
├── README.md                     # 部署说明
├── config/
│   └── policy/velocity/v0/
│       ├── params/deploy.yaml    # 部署参数配置
│       └── exported/policy.onnx # 神经网络策略
├── include/
│   └── Types.h                  # 类型定义 (复用 go2 的 LowCmd/LowState)
├── src/
│   └── State_RLBase.cpp         # RL 状态实现
└── build/                       # 编译产物
```

## 2. 核心模块架构

部署系统由以下核心模块组成：

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.cpp                                │
│  - 初始化 Unitree DDS 通信                                       │
│  - 释放运动控制服务                                               │
│  - 初始化 FSM 状态机                                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CtrlFSM (状态机)                            │
│  - 管理所有状态 (Passive, FixStand, FixDown, Velocity)           │
│  - 处理状态切换                                                  │
│  - 1000Hz 控制循环                                               │
└──────────┬──────────────────────────────────────┬──────────────┘
           │                                      │
           ▼                                      ▼
┌──────────────────────┐           ┌──────────────────────────────┐
│   FSMState           │           │     State_RLBase           │
│   - 手柄输入处理     │           │     - 加载 ONNX 策略         │
│   - 状态转换条件     │           │     - 50Hz 推理循环          │
│   - 低级命令发布     │           │     - 观测/动作处理          │
└──────────────────────┘           └──────────────────────────────┘
```

## 3. 程序入口 (main.cpp)

### 3.1 初始化流程

```cpp
int main(int argc, char** argv)
{
    // 1. 解析命令行参数
    auto vm = param::helper(argc, argv);  // 支持 --network, --log 等

    // 2. 初始化 Unitree DDS 网络
    unitree::robot::ChannelFactory::Instance()->Init(0, network);

    // 3. 释放运动控制服务 (避免冲突)
    release_motion_control_service();

    // 4. 初始化 FSM 状态
    init_fsm_state();

    // 5. 启动状态机
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    // 6. 主循环 (保持进程运行)
    while (true) sleep(1);
}
```

### 3.2 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--network` (-n) | 网络接口名 | 空 |
| `--log` | 启用日志记录 | - |
| `--help` (-h) | 显示帮助 | - |
| `--version` (-v) | 显示版本 | - |

### 3.3 运动控制服务释放

```cpp
void release_motion_control_service()
{
    // 检查并释放已激活的运动控制服务
    while(query_motion_status(msc)) {
        msc.ReleaseMode();
        sleep(5);
    }
}
```

## 4. FSM 状态机设计

### 4.1 状态定义

| 状态 ID | 状态名 | 类型 | 说明 |
|---------|--------|------|------|
| 1 | Passive | BaseState | 电机零力矩模式 |
| 2 | FixStand | FixStand | 固定站立，关节位置控制 |
| 3 | FixDown | FixStand | 蹲下状态 |
| 4 | Velocity | RLBase | RL 策略控制 |

### 4.2 状态转换图

```
                    ┌─────────────┐
                    │  Passive    │
                    │ (零力矩)     │
                    └──────┬──────┘
                           │ LT + Up
                           ▼
┌─────────────┐   LT + Down   ┌─────────────┐
│  FixDown    │◄──────────────│  FixStand   │
│   (蹲下)    │──────────────►│  (站立)     │
└─────────────┘   LT + Up     └──────┬──────┘
                                      │ RT + A
                                      ▼
                               ┌─────────────┐
                               │  Velocity   │
                               │  (RL 控制)   │
                               └─────────────┘
```

### 4.3 手柄操作映射

```yaml
Passive:
  transitions:
    FixStand: LT + up.on_pressed

FixStand:
  transitions:
    FixDown: LT + down.on_pressed
    Velocity: RT + A.on_pressed

FixDown:
  transitions:
    Passive: LT + B.on_pressed
    FixStand: LT + up.on_pressed

Velocity:
  transitions:
    FixDown: LT + down.on_pressed
```

### 4.4 控制器周期

- **FSM 主循环**: 1000 Hz (1ms)
- **Policy 推理线程**: 50 Hz (20ms)
- **DDS 通信**: 实时

## 5. 核心状态类实现

### 5.1 FSMState 基类

```cpp
class FSMState : public BaseState
{
    void pre_run()  // 更新传感器数据、手柄输入
    void run()      // 执行状态逻辑
    void post_run() // 发布低层命令
};
```

**关键职责**:
1. 读取 `lowstate` (电机状态、IMU、手柄)
2. 解析状态转换条件
3. 写入 `lowcmd` (目标关节位置、PD 增益)

### 5.2 State_FixStand

固定站立状态，使用位置控制。

```cpp
// 配置参数
kp: [400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400]
kd: [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]

// 姿态插值序列
qs: [
  [],                                    // 初始位置 (从当前读取)
  [0.0, 1.36, -2.60, ...],              // 蜷腿
  [0.0, 0.8, -1.5, ...],                // 站立姿态
]
ts: [0, 1, 2]  // 时间节点 (秒)
```

**姿态目标**:
```
hip:   0.0 rad
thigh: 0.8 rad  (或 1.36 rad 蜷腿)
calf: -1.5 rad  (或 -2.60 rad 蜷腿)
```

### 5.3 State_RLBase

RL 策略执行状态。

```cpp
State_RLBase::State_RLBase(int state_mode, std::string state_string)
{
    // 1. 加载部署配置
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = cfg["policy_dir"].as<std::string>();

    // 2. 创建 RL 环境
    env = std::make_unique<ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(lowstate)
    );

    // 3. 加载 ONNX 策略
    env->alg = std::make_unique<isaaclab::OrtRunner>(
        policy_dir / "exported" / "policy.onnx"
    );
}
```

**执行循环**:
```cpp
void State_RLBase::run()
{
    // 从 action_manager 获取处理后的动作
    auto action = env->action_manager->processed_actions();

    // 映射到实际电机顺序并写入 lowcmd
    for(int i = 0; i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
```

## 6. RL 环境与策略

### 6.1 ManagerBasedRLEnv

```cpp
class ManagerBasedRLEnv
{
    // 初始化
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot)
    {
        step_dt = cfg["step_dt"];  // 0.02 (50Hz)
        joint_ids_map = cfg["joint_ids_map"];
        joint_stiffness = cfg["stiffness"];
        joint_damping = cfg["damping"];

        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
    }

    // 策略步进
    void step()
    {
        robot->update();                    // 读取传感器
        auto obs = observation_manager->compute();  // 构建观测
        auto action = alg->act(obs);       // 推理
        action_manager->process_action(action);      // 处理动作
    }
};
```

### 6.2 关节 ID 映射

```yaml
joint_ids_map: [3,4,5,0,1,2,9,10,11,6,7,8]
```

**映射关系**:
| 索引 | 训练环境 | 实机 SDK |
|------|---------|---------|
| 0 | FL_hip | FR_hip |
| 1 | FL_thigh | FR_thigh |
| 2 | FL_calf | FR_calf |
| 3 | FR_hip | FL_hip |
| 4 | FR_thigh | FL_thigh |
| 5 | FR_calf | FL_calf |
| 6 | RL_hip | RR_hip |
| 7 | RL_thigh | RR_thigh |
| 8 | RL_calf | RR_calf |
| 9 | RR_hip | RL_hip |
| 10 | RR_thigh | RL_thigh |
| 11 | RR_calf | RL_calf |

### 6.3 观测构建 (observations)

| 观测项 | 维度 | 说明 |
|--------|------|------|
| base_ang_vel | 3 | 机体角速度 (body frame) |
| projected_gravity | 3 | 重力向量投影 |
| velocity_commands | 3 | 速度命令 (lin_x, lin_y, ang_z) |
| gait_phase | 2 | 步态相位 (sin, cos) |
| joint_pos_rel | 12 | 关节位置相对值 |
| joint_vel_rel | 12 | 关节速度 |
| last_action | 12 | 上一帧动作 |

**velocity_commands 映射**:
```cpp
obs[0] = clamp(joystick->ly(), -0.5, 1.0);   // lin_vel_x
obs[1] = clamp(-joystick->lx(), -0.5, 0.5);  // lin_vel_y
obs[2] = clamp(-joystick->rx(), -0.5, 0.5);  // ang_vel_z
```

### 6.4 动作处理 (actions)

```yaml
JointPositionAction:
  scale: [0.25, 0.25, ...]    # 动作缩放
  offset: [0.0, 0.8, -1.5, ...]  # 默认姿态偏移
```

**处理流程**:
```
神经网络输出 action ∈ [-1, 1]
       │
       ▼
  action * scale = [0, 0.25]  (假设 action=1)
       │
       ▼
  + offset = [0, 0.8, -1.5]    (加上默认姿态)
       │
       ▼
  = 目标关节位置
```

## 7. 通信接口

### 7.1 类型定义

```cpp
// 复用 go2 的 DDS 类型
using LowCmd_t = unitree::robot::go2::publisher::LowCmd;
using LowState_t = unitree::robot::go2::subscription::LowState;
```

### 7.2 LowCmd 结构

```cpp
lowcmd->msg_.motor_cmd()[motor_id].q()  // 目标位置
lowcmd->msg_.motor_cmd()[motor_id].dq() // 目标速度
lowcmd->msg_.motor_cmd()[motor_id].tau() // 目标力矩
lowcmd->msg_.motor_cmd()[motor_id].kp() // 刚度
lowcmd->msg_.motor_cmd()[motor_id].kd() // 阻尼
```

### 7.3 LowState 结构

```cpp
lowstate->msg_.imu_state().quaternion()    // IMU 四元数
lowstate->msg_.imu_state().gyroscope()    // 陀螺仪
lowstate->msg_.motor_state()[id].q()       // 关节位置
lowstate->msg_.motor_state()[id].dq()     // 关节速度
lowstate->joystick                         // 手柄状态
```

## 8. 本地手柄支持

### 8.1 XBox 手柄映射

```cpp
class XBoxJoystick : public unitree::common::UnitreeJoystick
{
    // 按钮映射
    back   = button_[6];
    start  = button_[7];
    LB     = button_[4];
    RB     = button_[5];
    A      = button_[0];
    B      = button_[1];
    X      = button_[2];
    Y      = button_[3];
    up     = axis_[7] < 0;
    down   = axis_[7] > 0;
    LT/RT  = axis_[2]/axis_[5] > 0;

    // 摇杆
    lx = axis_[0] / max_value;   // 左摇杆 X
    ly = -axis_[1] / max_value;   // 左摇杆 Y
    rx = axis_[3] / max_value;    // 右摇杆 X
    ry = -axis_[4] / max_value;   // 右摇杆 Y
};
```

### 8.2 配置

```yaml
Joystick:
  enabled: true
  type: xbox           # 或 switch
  device: /dev/input/js0
  bits: 16
```

## 9. 真机部署 vs 仿真部署 — 手柄输入差异

### 9.1 架构对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        真机部署 (b2ygx_ctrl)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌─────────────────┐     ┌───────────────────────┐    │
│   │ 无线手柄     │     │ 机器人 SDK     │     │   本地手柄 (/dev/)   │    │
│   │ (遥控器)     │────►│ DDS 接收        │────►│   XBoxJoystick      │    │
│   │ WirelessCtrl │     │ lowstate->js    │     │ extract(combine())  │    │
│   └──────────────┘     └─────────────────┘     └───────────┬───────────┘    │
│                                                           │               │
│                                                  local_joystick            │
│                                                  pre_run() 中合并           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        仿真部署 (MuJoCo + unitree_bridge)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌─────────────────┐                                 │
│   │ 本地手柄    │────►│ 仿真桥接        │                                 │
│   │ (/dev/input)│     │ RobotBridge     │                                 │
│   │ XBoxJoystick│     │ lowstate->js=   │                                 │
│   │             │     │ joystick (直连)  │                                 │
│   └──────────────┘     └─────────────────┘                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 关键代码对比

| 方面 | 真机部署 | 仿真部署 |
|------|---------|---------|
| **手柄数据源** | `lowstate->joystick` (来自机器人 DDS) | `joystick` (直接读取 /dev/input) |
| **本地手柄注入** | `local_joystick` → `lowstate->joystick.extract()` | `lowstate->joystick = joystick` |
| **初始化** | `lowstate->wait_for_connection()` | `joystick = std::make_shared<XBoxJoystick>()` |
| **配置位置** | `config.yaml` → `Joystick.enabled: true` | `config.yaml` → `use_joystick: 1` |

### 9.3 真机部署手柄合并逻辑

```cpp
// FSMState::pre_run() — 真机部署
void pre_run()
{
    lowstate->update();  // 从机器人 DDS 接收 lowstate (含无线手柄)

    if(local_joystick)  // 如果启用了本地手柄
    {
        local_joystick->update();  // 读取本地手柄
        // 关键: extract() 合并本地手柄到 DDS 手柄数据
        lowstate->joystick.extract(local_joystick->combine());
    }
}
```

**作用**: 当使用真实网卡连接机器人时，本地手柄可以通过 `extract()` 将按键状态注入到 `lowstate->joystick`。

### 9.4 仿真部署手柄直连

```cpp
// RobotBridge 构造函数 — 仿真
RobotBridge(mjModel *model, mjData *data)
{
    lowstate = std::make_unique<LowState_t>();
    lowstate->joystick = joystick;  // 本地手柄直接使用，无 extract()
}
```

### 9.5 配置对比

```yaml
# 真机部署 config.yaml
Joystick:
  enabled: true
  type: xbox
  device: /dev/input/js0
  bits: 16

# 仿真 config.yaml
use_joystick: 1
joystick_type: "xbox"
joystick_device: "/dev/input/js0"
joystick_bits: 16
```

### 9.6 extract() vs combine() 方法

```cpp
// combine() = 读取本地手柄 → 得到 REMOTE_DATA_RX
REMOTE_DATA_RX combine()
{
    REMOTE_DATA_RX key;
    key.RF_RX.btn.components.A = A();
    key.RF_RX.lx = lx();
    // ...
    return key;
}

// extract() = 从 REMOTE_DATA_RX 写入到目标对象
void extract(const REMOTE_DATA_RX& key)
{
    A(key.RF_RX.btn.components.A);
    lx(key.RF_RX.lx);
    // ...
}
```

### 9.7 注意事项

1. **仿真不需要 `extract()`**: 仿真中 `lowstate->joystick` 直接指向本地手柄，不存在数据来源冲突

2. **真机手柄优先级**: 如果机器人固件发送的手柄状态和本地手柄同时有效，`extract()` 用本地覆盖

3. **网络模式差异**:
   - `--network=lo`: 不连接机器人，本地手柄必须启用
   - `--network=<iface>`: 连接机器人，无线手柄 + 可选本地手柄

4. **Joystick SDK 路径**: 手柄数据结构定义在 `unitree_sdk2/include/unitree/dds_wrapper/common/unitree_joystick.hpp`

## 10. 编译与部署

### 9.1 编译

```bash
cd deploy/robots/b2ygx
mkdir -p build && cd build
cmake ..
make
```

**依赖**:
- unitree_sdk2
- yaml-cpp
- Boost (program_options)
- Eigen3
- ONNXRuntime (1.22.0)

### 9.2 仿真部署

```bash
# 1. 启动仿真
cd simulate/build
./unitree_mujoco

# 2. 启动控制器
cd deploy/robots/b2ygx/build
./b2ygx_ctrl --network=lo
```

### 9.3 实机部署

```bash
# 前置条件
# - 机器人吊装
# - 进入零力矩/调试模式
# - 主机配置 192.168.123.222/24

# 启动
cd deploy/robots/b2ygx/build
./b2ygx_ctrl --network=<interface>
# 例如: enp5s0, wlp4s0
```

## 11. 安全机制

### 10.1 自动状态检查

```cpp
// 所有状态都注册了超时检查
registered_checks.emplace_back(
    []()->bool{ return lowstate->isTimeout(); },
    FSMStringMap.right.at("Passive")  // 超时自动切 Passive
);
```

### 10.2 姿态异常检测

```cpp
// State_RLBase 构造函数中注册
registered_checks.emplace_back(
    []()->bool{ return bad_orientation(env.get(), 1.0); },
    FSMStringMap.right.at("Passive")
);
```

### 10.3 紧急操作

| 操作 | 按键 | 效果 |
|------|------|------|
| 急停 | LT + B | 切 Passive，电机零力矩 |
| 蹲下 | LT + Down | 切 FixDown |
| 站立 | LT + Up | 切 FixStand |

## 12. 部署参数核对清单

部署前必须核对以下参数与训练配置一致：

| 参数 | 期望值 | 说明 |
|------|--------|------|
| step_dt | 0.02 | 50Hz |
| stiffness | [200, 200, 240, ...] | 12 个关节 |
| damping | [10, 10, 12, ...] | 12 个关节 |
| action scale | 0.25 | 12 个关节 |
| default_joint_pos | [0.0, 0.8, -1.5, ...] | 站立姿态 |
| joint_ids_map | [3,4,5,0,1,2,9,10,11,6,7,8] | 关节顺序 |
| observations | 不含 height_scan | Flat 地形配置 |

## 13. 数据流图

```
┌────────────────────────────────────────────────────────────────────┐
│                        部署控制器 (1000Hz)                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐   │
│  │  手柄输入    │────►│  FSMState    │────►│  状态转换检查    │   │
│  │  (joystick)  │     │   pre_run()  │     │  registered_check│   │
│  └──────────────┘     └──────────────┘     └────────┬─────────┘   │
│                                                      │             │
│                     ┌───────────────────────────────┘             │
│                     ▼                                          │
│            ┌──────────────────────┐                               │
│            │   当前状态 run()     │                               │
│            └──────────┬───────────┘                               │
│                       │                                           │
│  ┌────────────────────┼────────────────────┐                      │
│  │                    │                    │                      │
│  ▼                    ▼                    ▼                      │
│ FixStand          State_RLBase        State_Passive               │
│ (PD 控制)         (RL 推理)           (零力矩)                     │
│                       │                                          │
│                       ▼                                          │
│            ┌──────────────────────┐                               │
│            │  ManagerBasedRLEnv   │                               │
│            │       step()        │◄────── Policy 线程 (50Hz)      │
│            └──────────┬───────────┘                               │
│                       │                                          │
│      ┌────────────────┼────────────────┐                         │
│      ▼                ▼                ▼                         │
│  ┌────────┐     ┌────────────┐    ┌──────────────┐               │
│  │Obs Mgr │     │ OrtRunner  │    │ Action Mgr   │               │
│  │观测构建│     │ ONNX 推理  │    │ 动作处理     │               │
│  └────────┘     └────────────┘    └──────┬───────┘               │
│                                          │                       │
└──────────────────────────────────────────┼───────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │              LowCmd 发布                     │
                    │  motor_cmd[id].q = action[i]                │
                    │  motor_cmd[id].kp/kd = stiffness/damping    │
                    └─────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │            DDS 网络 (LOWCMD)                │
                    └─────────────────────────────────────────────┘
```

## 14. 关键文件索引

| 文件 | 位置 | 说明 |
|------|------|------|
| main.cpp | [deploy/robots/b2ygx/main.cpp](deploy/robots/b2ygx/main.cpp) | 程序入口 |
| State_RLBase.cpp | [deploy/robots/b2ygx/src/State_RLBase.cpp](deploy/robots/b2ygx/src/State_RLBase.cpp) | RL 状态实现 |
| CtrlFSM.h | [deploy/include/FSM/CtrlFSM.h](deploy/include/FSM/CtrlFSM.h) | 状态机核心 |
| FSMState.h | [deploy/include/FSM/FSMState.h](deploy/include/FSM/FSMState.h) | 状态基类 |
| State_RLBase.h | [deploy/include/FSM/State_RLBase.h](deploy/include/FSM/State_RLBase.h) | RL 状态类 |
| manager_based_rl_env.h | [deploy/include/isaaclab/envs/manager_based_rl_env.h](deploy/include/isaaclab/envs/manager_based_rl_env.h) | RL 环境 |
| algorithms.h | [deploy/include/isaaclab/algorithms/algorithms.h](deploy/include/isaaclab/algorithms/algorithms.h) | ONNX 推理 |
| observations.h | [deploy/include/isaaclab/envs/mdp/observations/observations.h](deploy/include/isaaclab/envs/mdp/observations/observations.h) | 观测构建 |
| action_manager.h | [deploy/include/isaaclab/manager/action_manager.h](deploy/include/isaaclab/manager/action_manager.h) | 动作管理 |
| deploy.yaml | [deploy/robots/b2ygx/config/policy/velocity/v0/params/deploy.yaml](deploy/robots/b2ygx/config/policy/velocity/v0/params/deploy.yaml) | 部署配置 |
| config.yaml | [deploy/robots/b2ygx/config/config.yaml](deploy/robots/b2ygx/config/config.yaml) | 控制器配置 |
| unitree_joystick.hpp | `unitree_sdk2/include/unitree/dds_wrapper/common/unitree_joystick.hpp` | 手柄数据结构 |
