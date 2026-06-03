"""Small process helpers for the Streamlit train log manager."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

RUNTIME_DIR = Path("/tmp/unitree_rl_mjlab_train_log_manager")


@dataclass
class ManagedProcess:
    label: str
    command: list[str]
    cwd: Path
    env_overrides: dict[str, str]
    log_path: Path
    process: subprocess.Popen
    url: str | None = None
    started_at: float = 0.0

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def is_running(self) -> bool:
        return self.process.poll() is None

    @property
    def returncode(self) -> int | None:
        return self.process.poll()


def is_port_listening(host: str, port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def find_free_port(start: int, end: int, host: str = "127.0.0.1") -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found in {start}-{end}")


def start_process(
    *,
    label: str,
    command: Sequence[str],
    cwd: Path,
    env_overrides: Mapping[str, str] | None = None,
    url: str | None = None,
) -> ManagedProcess:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
    log_path = RUNTIME_DIR / f"{safe_label}_{int(time.time())}.log"
    env = os.environ.copy()
    overrides = dict(env_overrides or {})
    env.update(overrides)
    log_file = log_path.open("ab")
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_file.close()
    return ManagedProcess(
        label=label,
        command=list(command),
        cwd=cwd,
        env_overrides=overrides,
        log_path=log_path,
        process=process,
        url=url,
        started_at=time.time(),
    )


def stop_process(proc: ManagedProcess, timeout: float = 5.0) -> None:
    if not proc.is_running:
        return
    try:
        os.killpg(proc.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.process.terminate()
    try:
        proc.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.process.pid, signal.SIGKILL)
        except OSError:
            proc.process.kill()
        proc.process.wait(timeout=timeout)


def read_recent_output(path: Path, max_bytes: int = 8000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, os.SEEK_END)
            data = f.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")
