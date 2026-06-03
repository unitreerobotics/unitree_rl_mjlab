"""Small process helpers for the Streamlit train log manager."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
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
    process: subprocess.Popen | None
    url: str | None = None
    started_at: float = 0.0
    pid_value: int | None = None

    @property
    def pid(self) -> int:
        if self.process is not None:
            return self.process.pid
        if self.pid_value is None:
            raise RuntimeError("ManagedProcess has neither process nor pid_value")
        return self.pid_value

    @property
    def is_running(self) -> bool:
        if self.process is not None:
            return self.process.poll() is None
        return is_pid_running(self.pid)

    @property
    def returncode(self) -> int | None:
        if self.process is not None:
            return self.process.poll()
        return None if is_pid_running(self.pid) else 0


def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


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


def _is_url_reachable(url: str, host: str, port: int, timeout: float = 0.5) -> bool:
    if not url.startswith(("http://", "https://")):
        return is_port_listening(host, port, timeout=timeout)

    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(1)
        return True
    except urllib.error.HTTPError:
        return True
    except OSError:
        return False


def announce_url_when_listening(
    *,
    proc: ManagedProcess,
    host: str,
    port: int,
    url: str,
    label: str,
    timeout: float = 300.0,
    poll_interval: float = 0.5,
) -> None:
    """Print a URL when a child process starts listening on the expected port.

    VS Code Remote watches integrated terminal output and auto-forwards URLs it
    sees there. Background child output is redirected to log files, so the parent
    Streamlit process needs to print the viewer URL itself.
    """

    def _worker() -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not proc.is_running:
                return
            if _is_url_reachable(url, host, port):
                print(f"{label}: {url}", flush=True)
                return
            time.sleep(poll_interval)

    thread = threading.Thread(
        target=_worker,
        name=f"announce-url-{proc.pid}-{port}",
        daemon=True,
    )
    thread.start()


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


def _terminate_pid(pid: int, sig: signal.Signals) -> None:
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return
    try:
        if pgid == pid:
            os.killpg(pgid, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        return


def stop_process(proc: ManagedProcess, timeout: float = 5.0) -> None:
    if not proc.is_running:
        return
    pid = proc.pid
    _terminate_pid(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not proc.is_running:
            return
        time.sleep(0.1)
    _terminate_pid(pid, signal.SIGKILL)
    if proc.process is not None:
        try:
            proc.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass


def _read_proc_cmdline(pid: int) -> list[str]:
    try:
        data = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", errors="replace") for part in data.split(b"\0") if part]


def _read_proc_environ(pid: int) -> dict[str, str]:
    try:
        data = (Path("/proc") / str(pid) / "environ").read_bytes()
    except OSError:
        return {}
    env: dict[str, str] = {}
    for part in data.split(b"\0"):
        if not part or b"=" not in part:
            continue
        key, value = part.split(b"=", 1)
        env[key.decode("utf-8", errors="replace")] = value.decode("utf-8", errors="replace")
    return env


def _read_proc_cwd(pid: int) -> Path | None:
    try:
        return Path(os.readlink(Path("/proc") / str(pid) / "cwd"))
    except OSError:
        return None


def _listening_tcp_ports() -> dict[str, int]:
    ports: dict[str, int] = {}
    for path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = path.read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10 or parts[3] != "0A":
                continue
            try:
                port = int(parts[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            ports[parts[9]] = port
    return ports


def _proc_listening_ports(pid: int) -> list[int]:
    inode_to_port = _listening_tcp_ports()
    fd_dir = Path("/proc") / str(pid) / "fd"
    ports: list[int] = []
    try:
        fd_paths = list(fd_dir.iterdir())
    except OSError:
        return ports
    for fd_path in fd_paths:
        try:
            target = os.readlink(fd_path)
        except OSError:
            continue
        if not (target.startswith("socket:[") and target.endswith("]")):
            continue
        inode = target.removeprefix("socket:[").removesuffix("]")
        port = inode_to_port.get(inode)
        if port is not None:
            ports.append(port)
    return sorted(set(ports))


def discover_play_processes(repo_root: Path) -> list[ManagedProcess]:
    repo_root = repo_root.resolve()
    found: list[ManagedProcess] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        command = _read_proc_cmdline(pid)
        if not command or not any(arg.endswith("scripts/play.py") for arg in command):
            continue
        cwd = _read_proc_cwd(pid)
        command_text = " ".join(command)
        if cwd != repo_root and str(repo_root) not in command_text:
            continue
        env = _read_proc_environ(pid)
        viewer_ports = [port for port in _proc_listening_ports(pid) if 8080 <= port <= 8099]
        env_port = env.get("_VISER_PORT_OVERRIDE")
        port = str(viewer_ports[0]) if viewer_ports else env_port
        url = f"http://localhost:{port}" if port and port.isdigit() else None
        found.append(
            ManagedProcess(
                label=f"external_play_{pid}",
                command=command,
                cwd=cwd or repo_root,
                env_overrides={},
                log_path=RUNTIME_DIR / f"external_play_{pid}.log",
                process=None,
                url=url,
                started_at=0.0,
                pid_value=pid,
            )
        )
    found.sort(key=lambda proc: proc.pid)
    return found


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
