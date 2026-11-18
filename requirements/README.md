# 固定安装版本

`runtime.in` 记录直接依赖的目标版本，`runtime.txt` 包含解析后的直接和间接依赖。
`setup.py` 将 `runtime.txt` 的每个有效条目作为安装依赖，因此普通的
`python -m pip install -e .` 或 wheel 安装都会应用这些版本，无需额外传入 constraints 文件。
源码包通过 `MANIFEST.in` 携带版本清单；构建依赖另在 `pyproject.toml` 中固定。

清单针对 Python 3.10–3.13 进行 universal 解析，使用环境标记处理不同 Python 版本的依赖差异。
PyTorch 与 torchvision 使用公开版本号，允许安装相同版本的 CPU/CUDA 构建；
对应 CUDA wheel 的原生 CUDA 库版本由该 wheel 的元数据固定。

上游更新不会自动更新清单。主动升级时，修改 `runtime.in`，重新解析整个依赖集合，
审阅 `runtime.txt` 的全部变化，再运行 [模型与训练检查](../tests/README.md) 和安装包构建检查。
不要只放宽 `mjlab` 的版本，也不要用 `pip install --no-deps` 绕过安装依赖检查。
