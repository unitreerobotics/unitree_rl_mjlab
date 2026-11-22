# Pinned runtime versions

`runtime.in` records the target versions of the direct dependencies, and `runtime.txt`
contains the resolved direct and transitive dependencies. `setup.py` uses every valid
entry of `runtime.txt` as an install requirement, so a plain
`python -m pip install -e .` or a wheel install applies these versions without an extra
constraints file. The source distribution ships the manifests through `MANIFEST.in`;
build dependencies are pinned separately in `pyproject.toml`.

The manifest is resolved universally for Python 3.10–3.13 and uses environment markers
for dependencies that differ between Python versions. PyTorch and torchvision use their
public version numbers, so matching CPU or CUDA builds can be installed; the native CUDA
library versions of a CUDA wheel are pinned by that wheel's own metadata.

Upstream releases do not update the manifest automatically. To upgrade on purpose, edit
`runtime.in`, re-resolve the whole dependency set, review every change in `runtime.txt`,
then re-run the training, play and package build checks. Do not loosen only the `mjlab`
version, and do not bypass the dependency check with `pip install --no-deps`.

[中文](README_zh.md)
