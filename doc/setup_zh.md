# 安装配置文档

## 系统要求

- **操作系统**：推荐使用 Ubuntu 22.04
- **显卡**：Nvidia 显卡  
- **驱动版本**：建议使用 550 或更高版本  

---

## 1. 创建虚拟环境

建议在虚拟环境中运行训练或部署程序，推荐使用 Conda 创建虚拟环境。如果您的系统中已经安装了 Conda，可以跳过步骤 1.1。

### 1.1 下载并安装 MiniConda

MiniConda 是 Conda 的轻量级发行版，适用于创建和管理虚拟环境。使用以下命令下载并安装：

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

安装完成后，初始化 Conda：

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

### 1.2 创建新环境

使用以下命令创建虚拟环境：

```bash
conda create -n unitree_rl_mjlab python=3.11
```

### 1.3 激活虚拟环境

```bash
conda activate unitree_rl_mjlab
```

---

## 2. 安装

### 2.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
```

### 2.2 安装依赖

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

`setup.py` 会自动读取固定版本清单 `requirements/runtime.txt`，
进入 unitree_rl_mjlab 项目根目录并安装：

```bash
cd unitree_rl_mjlab
pip install -e .
```

本项目固定使用 `mjlab==1.6.0`、`mujoco==3.11.0`、`mujoco-warp==3.11.0`、
`rsl-rl-lib==5.4.2`、`torch==2.9.1` 和 `torchvision==0.24.1`。
直接和间接运行依赖均使用 `==` 固定；Python 3.10–3.13 的差异通过环境标记选择对应的固定版本。
推荐上面的 Python 3.11 环境。CPU/CUDA wheel 可以使用相同公开版本号的不同构建。
上游发布新版不会使上述安装命令自动选择新版；固定版本不可用或依赖冲突时，安装会报错。
请保留完整仓库或源码包中的 `requirements` 目录，不要单独复制 `setup.py`。
已有旧版本环境可在项目根目录执行：

```bash
python -m pip install --upgrade -e .
python -m pip check
```

升级后，多 GPU 参数使用 `--gpu-ids '[0, 1]'`；元组参数同样使用 Python 语法，例如
`--agent.actor.hidden-dims '(512, 256, 128)'`。

G1 模仿任务现在使用仓库本地模型及其默认关节角；H2 速度任务使用 `H2_ACTION_SCALE`。
这些修改会影响策略输入或动作含义，旧 checkpoint 应配合原训练配置验证；使用修正后的任务重新训练时，
应同时更新部署策略及导出的动作缩放、默认关节角。

## 总结

按照上述步骤完成后，您已经准备好在虚拟环境中运行相关程序。若遇到问题，请参考各组件的官方文档或检查依赖安装是否正确。

