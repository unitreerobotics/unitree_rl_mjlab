# Unitree B2YGX

## 机器人参数

| 参数 | 值 |
|------|-----|
| base collision box | 0.315 x 0.162 x 0.092 |
| base 质量 | 40.84 kg |
| hip 质量 | 2.93 kg |
| thigh 质量 | 7.9 kg |
| thigh 长度 | 0.354 m |
| calf 长度 | 0.35 m |
| foot 半径 | 0.032 m |
| hip 位置 (前后) | ±0.3945 m |
| hip 位置 (左右) | ±0.09 m |
| thigh 偏移 (Y) | ±0.13225 m |

## 训练参数

| 参数 | 值 |
|------|-----|
| stiffness (kp) | 160.0 |
| damping (kd) | 8.0 |
| effort_limit (hip/thigh) | 200.0 |
| effort_limit (calf) | 300.0 |
| armature | 0.1 |
| action scale | 0.25 |
| 初始高度 | 0.546 m |
| 默认关节角 | hip=0.0, thigh=0.8, calf=-1.5 |

### 训练命令

```bash
# Flat 地形
python scripts/train.py Unitree-B2YGX-Flat --agent.logger=tensorboard --env.scene.num-envs=4096

# Rough 地形
python scripts/train.py Unitree-B2YGX-Rough --agent.logger=tensorboard --env.scene.num-envs=4096
```

### 仿真回放

```bash
python scripts/play.py Unitree-B2YGX-Flat --checkpoint_file=<path_to_model>
python scripts/play.py Unitree-B2YGX-Rough --checkpoint_file=<path_to_model>
```

## 与 B2 的区别

| 参数 | B2 | B2YGX |
|------|-----|-------|
| stiffness (hip/thigh) | 200.0 | 160.0 |
| damping (hip/thigh) | 10.0 | 8.0 |
| stiffness (calf) | 240.0 | 160.0 |
| damping (calf) | 12.0 | 8.0 |
| base collision box | 0.25x0.14x0.075 | 0.315x0.162x0.092 |
| hip 位置 | ±0.3285, ±0.072 | ±0.3945, ±0.09 |
| hip 质量 | 2.53 kg | 2.93 kg |
| thigh 质量 | 7.46 kg | 7.9 kg |
| calf 长度偏移 | -0.35 | -0.354 |

B2YGX 体型更大、更重，关节更柔软。

## 部署

本目录默认部署 `Unitree-B2YGX-Flat` 速度跟踪策略。Flat 策略不使用 `height_scan`，部署侧观测由 IMU 角速度、重力投影、速度命令、步态相位、关节位置/速度和上一帧动作组成。

部署前不要直接沿用 `config/policy/velocity/v0/exported/policy.onnx` 里的旧文件。训练保存 checkpoint 时会同时导出同目录下的 `policy_<iter>.onnx` 和覆盖式 `policy.onnx`，必须先选定通过 `play` 验证的 checkpoint，再安装对应 ONNX。

### 1. 选择并验证策略

先让准备脚本找出最新候选策略，并检查部署参数是否仍然匹配 Flat 配置：

```bash
python scripts/prepare_b2ygx_deploy.py
```

脚本会打印需要执行的回放命令，例如：

```bash
python scripts/play.py Unitree-B2YGX-Flat --checkpoint_file=logs/rsl_rl/b2ygx_velocity/<run>/model_<iter>.pt
```

在 `play` 中至少确认：
- 零命令站立不持续漂移或放大抖动。
- 小速度前进、横移和转向方向正确。
- 关节没有持续打限位，身体没有明显摔倒趋势。

如果要指定某个 checkpoint：

```bash
python scripts/prepare_b2ygx_deploy.py \
  --checkpoint logs/rsl_rl/b2ygx_velocity/<run>/model_<iter>.pt
```

### 2. 安装已验证 ONNX

只有完成上面的 `play` 验证后，才安装策略到 deploy 路径：

```bash
python scripts/prepare_b2ygx_deploy.py \
  --checkpoint logs/rsl_rl/b2ygx_velocity/<run>/model_<iter>.pt \
  --install \
  --play-verified
```

该命令会复制对应的 `policy_<iter>.onnx` 到：

```text
deploy/robots/b2ygx/config/policy/velocity/v0/exported/policy.onnx
```

并写入：

```text
deploy/robots/b2ygx/config/policy/velocity/v0/params/deployment_manifest.yaml
```

用于记录 checkpoint、源 ONNX、安装时间和必须人工确认的安全检查项。

### 3. 部署参数核对

准备脚本会自动核对以下参数：

| 参数 | 期望值 |
|------|-----|
| `step_dt` | `0.02` (50Hz) |
| `stiffness` | 12 个关节均为 `160` |
| `damping` | 12 个关节均为 `8` |
| `action scale` | 12 个关节均为 `0.25` |
| `default_joint_pos` | `hip=0.0, thigh=0.8, calf=-1.5` |
| `joint_ids_map` | `[3,4,5,0,1,2,9,10,11,6,7,8]` |
| `observations` | 不包含 `height_scan` |

`joint_ids_map` 必须再由人工结合实机 SDK 电机顺序核对一次；该项错误会导致腿序或关节序错误，是最关键的部署风险。

### 4. 编译

```bash
cd deploy/robots/b2ygx
mkdir -p build && cd build
cmake ..
make
```

产物: `b2ygx_ctrl`，依赖 unitree_sdk2 + onnxruntime (支持 x86_64 / aarch64)。

### 5. 仿真部署

先启动 Unitree MuJoCo 仿真：

```bash
cd simulate
mkdir -p build && cd build
cmake ..
make -j8
./unitree_mujoco
```

确认 `simulate/config.yaml` 中机器人为 `b2ygx` 后，启动控制器：

```bash
cd deploy/robots/b2ygx/build
./b2ygx_ctrl --network=lo
```

### 6. 实机部署

实机部署前：
- 机器人吊装。
- 机器人进入零力矩/调试模式。
- 外部 x86_64 主机网口配置为 `192.168.123.222/24`。
- 确认没有其他进程占用 lowcmd 通道。

启动：

```bash
cd deploy/robots/b2ygx/build
./b2ygx_ctrl --network=<robot_iface>
```

其中 `<robot_iface>` 是连接机器人网口的网卡名，例如 `enp5s0`。

### 实机手柄输入

实机运行时，`b2ygx_ctrl` 默认读取本机手柄 `/dev/input/js0`，并把按键状态注入到 FSM 使用的 `lowstate->joystick`。这样使用真实网卡，例如 `wlp4s0`，也可以通过本机手柄触发状态切换。

```yaml
Joystick:
  enabled: true
  type: xbox
  device: /dev/input/js0
  bits: 16
```

如果 `--network=lo` 正常，但 `--network=<robot_iface>` 下按手柄不切状态，通常说明真实网卡模式下没有本机手柄输入。检查 `config/config.yaml` 中 `Joystick.enabled` 是否为 `true`，并用 `ls /dev/input/js*` 确认设备路径。若启动时出现 `Joystick open failed`，将 `device` 改成实际路径，例如 `/dev/input/js1`，或检查当前用户是否有 input 设备权限。

### FixStand 参数

| 参数 | 值 |
|------|-----|
| kp | 400 |
| kd | 12 |

FixStand 分两阶段: 先蜷腿 (thigh=1.36, calf=-2.60)，再展开到默认姿态 (thigh=0.8, calf=-1.5)。

### 手柄操作

| 操作 | 按键 |
|------|------|
| Passive -> FixStand | LT + D-pad 上 |
| FixStand -> Velocity | RT + A |
| Velocity -> FixDown | LT + D-pad 下 |
| FixDown -> Passive | LT + B |
| 速度控制 | 左摇杆 |

### 安全机制

- bad_orientation 检测: 倾斜超过阈值自动切回 Passive
- lowstate 超时: 自动切回 Passive
- 出现姿态异常、关节顺序异常、抖动放大或通信异常时，立即 `LT + D-pad 下` 回 FixDown，必要时切回 Passive 或急停
