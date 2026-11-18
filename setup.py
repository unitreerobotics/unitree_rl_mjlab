"""Installation script for the 'unitree_rl_mjlab' python package."""

from pathlib import Path

from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent
# Pin the whole runtime, not just mjlab: its own dependencies have open ranges.
INSTALL_REQUIRES = [
    line.strip()
    for line in (ROOT / "requirements" / "runtime.txt").read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=find_packages(include=["src", "src.*"]),
    package_data={
        "src": [
            "assets/robots/**/*.xml",
            "assets/robots/**/*.stl",
            "assets/robots/**/*.STL",
            "assets/robots/**/*.obj",
        ],
    },
    version="0.0.1",
    python_requires=">=3.10,<3.14",
    install_requires=INSTALL_REQUIRES,
)
