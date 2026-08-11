"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
    # mujoco-warp 3.5.0 is built against the matching mujoco 3.5.x C API. mjlab
    # only requires "mujoco>=3.5.0", so a fresh resolve pulls the newest mujoco
    # (3.11.0), where mjtEnableBit.mjENBL_MULTICCD no longer exists and
    # `import mujoco_warp` fails outright. Pin the pair together.
    "mujoco==3.5.0",
    # Same story for warp. mjlab 1.2.0 calls `wp.context.runtime` (sim/sim.py)
    # and `wp.context.Device` (sensor/raycast_sensor.py), but declares only
    # "warp-lang>=1.12.0". warp moved that namespace to `warp._src.context`, so
    # any resolve picking up >=1.13 fails with
    # "module 'warp' has no attribute 'context'" the moment an env is built.
    "warp-lang==1.12.0",
    # mjlab's terrain module imports scipy (terrains/heightfield_terrains.py)
    # but never declares it, so a clean install cannot construct any terrain.
    "scipy",
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
