# Installation Guide

## System Requirements

- **Operating System**: Recommended Ubuntu 22.04 
- **GPU**: Nvidia GPU  
- **Driver Version**: Recommended version 550 or later  

---

## 1. Creating a Virtual Environment

It is recommended to run training or deployment programs in a virtual environment. Conda is recommended for creating virtual environments. If Conda is already installed on your system, you can skip step 1.1.

### 1.1 Download and Install MiniConda

MiniConda is a lightweight distribution of Conda, suitable for creating and managing virtual environments. Use the following commands to download and install:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

After installation, initialize Conda:

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

### 1.2 Create a New Environment

Use the following command to create a virtual environment:

```bash
conda create -n unitree_rl_mjlab python=3.11
```

### 1.3 Activate the Virtual Environment

```bash
conda activate unitree_rl_mjlab
```

---

## 2. Installing

### 2.1 Download the Project

Clone the repository using Git:

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
```

### 2.2 Install Dependencies

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

`setup.py` reads the exact runtime pins from `requirements/runtime.txt` automatically.
Navigate to the project root directory and install them with:

```bash
cd unitree_rl_mjlab
pip install -e .
```

This project pins `mjlab==1.6.0`, `mujoco==3.11.0`, `mujoco-warp==3.11.0`,
`rsl-rl-lib==5.4.2`, `torch==2.9.1`, and `torchvision==0.24.1`.
Direct and transitive runtime dependencies use exact `==` pins. Environment markers select
compatible fixed versions for Python 3.10–3.13; Python 3.11 is recommended.
CPU/CUDA wheels may use different builds of the same public version.
New upstream releases will not be selected automatically. Missing pinned versions or dependency
conflicts cause installation to fail. Keep the `requirements` directory with `setup.py`.
To upgrade an existing environment, run from the repository root:

```bash
python -m pip install --upgrade -e .
python -m pip check
```

The v1.6 CLI uses Python collection syntax, e.g. `--gpu-ids '[0, 1]'` and
`--agent.actor.hidden-dims '(512, 256, 128)'`.

G1 tracking now uses the local model and its default joint angles. H2 velocity tasks use
`H2_ACTION_SCALE`. These changes affect policy inputs or action semantics: validate existing
checkpoints against their original training configuration. Retrained policies must be deployed
with their exported default joint angles and action scales.

## Summary

After completing the above steps, you are ready to run the related programs in the virtual environment. If you encounter any issues, refer to the official documentation of each component or check if the dependencies are installed correctly.

