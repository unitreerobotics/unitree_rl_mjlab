"""Streamlit UI for browsing rsl_rl training runs.

Run with:

    streamlit run tools/train_log_manager/app.py -- --logs-root logs/rsl_rl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow ``streamlit run`` to resolve sibling modules.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import columns as col_mod  # noqa: E402
import process_manager as proc_mgr  # noqa: E402
from diff import dict_diff, dot_get  # noqa: E402
from scanner import Run, checkpoint_iter, scan_runs  # noqa: E402


REPO_ROOT = _HERE.parents[1]
DEFAULT_LOGS_ROOT = REPO_ROOT / "logs" / "rsl_rl"
CONFIG_PATH = _HERE / ".columns.json"
_TERRAIN_VIEWER_SCRIPT = REPO_ROOT / "scripts" / "visualize_terrain.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=DEFAULT_LOGS_ROOT,
        help="Directory containing <experiment>/<run_id>/ subdirs.",
    )
    parser.add_argument(
        "--tb-url",
        default=None,
        help="TensorBoard base URL. Defaults to http://localhost:<tb-port>.",
    )
    parser.add_argument(
        "--tb-host",
        default="localhost",
        help="Host passed to TensorBoard when auto-starting it.",
    )
    parser.add_argument(
        "--tb-port",
        type=int,
        default=6006,
        help="Port used for TensorBoard auto-start and default links.",
    )
    parser.add_argument(
        "--viewer-url-host",
        default="localhost",
        help="Host used when building links to launched Viser viewers.",
    )
    parser.add_argument(
        "--viewer-port-start",
        type=int,
        default=8080,
        help="First Viser port the app may allocate for play launches.",
    )
    parser.add_argument(
        "--viewer-port-end",
        type=int,
        default=8099,
        help="Last Viser port the app may allocate for play launches.",
    )
    # Streamlit passes its own flags before "--"; argparse only sees what's after.
    return parser.parse_args()


@st.cache_data(show_spinner=False)
def _cached_scan(logs_root_str: str) -> list[Run]:
    return scan_runs(Path(logs_root_str))


def _load_columns_file() -> list[dict] | None:
    if not CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(CONFIG_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, list) else None


def _save_columns_file(cols: list[dict]) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cols, indent=2))
    except OSError:
        pass


_COLS_KEY = "run_columns"


def _sync_columns() -> None:
    cols = st.session_state[_COLS_KEY]
    st.query_params["cols"] = col_mod.encode(cols)
    _save_columns_file(cols)


def _init_columns() -> list[dict]:
    """Load or initialize the column state and return it.

    Priority: ?cols= URL (shareable) > persisted file > defaults. Always
    returns a usable list even on first run, and persists back to disk +
    query params whenever it had to synthesize or repair state.
    """
    existing = st.session_state.get(_COLS_KEY)
    if existing is not None:
        return existing

    token = st.query_params.get("cols")
    cols = col_mod.decode(token) if token else None
    if not cols:
        cols = _load_columns_file()
    cols = cols or col_mod.default_columns()
    # Guarantee protected built-in columns — a stale ?cols= URL may have omitted
    # them, or encoded them without the protected flag.
    _required = [
        {"name": "run_id",           "source": "builtin", "kind": "builtin", "path": "run_id",           "protected": True},
        {"name": "task_id",          "source": "builtin", "kind": "builtin", "path": "task_id",          "protected": True},
        {"name": "max_iter",         "source": "builtin", "kind": "builtin", "path": "max_iter",         "protected": True},
        {"name": "max_mean_reward",  "source": "builtin", "kind": "builtin", "path": "max_mean_reward",  "protected": True},
    ]
    for idx, required in enumerate(_required):
        for c in cols:
            if c.get("name") == required["name"]:
                c["protected"] = True
                break
        else:
            cols.insert(idx, dict(required))
    st.session_state[_COLS_KEY] = cols
    _save_columns_file(cols)
    return cols


def _tb_run_regex(run_ids: list[str]) -> str:
    """Build a TB run-filter regex anchored to end-of-run-name.

    TB run names are ``<experiment>/<run_id>``; anchoring with ``$`` ensures a
    timestamp like ``2026-04-16_14-54-05`` matches only that exact run rather
    than any run that happens to contain it as a substring. Run-ids are
    timestamps, so they don't need regex-escaping (and skipping escape keeps
    the URL readable in the cell display).
    """
    body = run_ids[0] if len(run_ids) == 1 else f"({'|'.join(run_ids)})"
    return f"{body}$"


def _tb_run_url(tb_url: str, run_id: str) -> str:
    return f"{tb_url.rstrip('/')}/#timeseries&regexInput={_tb_run_regex([run_id])}"


def _build_table(
    runs: list[Run], cols: list[dict], tb_url: str
) -> tuple[pd.DataFrame, list[str]]:
    """Return the display dataframe and a parallel list of plain run_ids.

    Protected run_id cells are rewritten to full TensorBoard URLs so they can
    render as clickable links via ``st.column_config.LinkColumn``.
    """
    rows = []
    for run in runs:
        row = {}
        for c in cols:
            val = col_mod.extract(run, c)
            if c.get("protected") and c.get("path") == "run_id":
                val = _tb_run_url(tb_url, val)
            row[c["name"]] = val
        rows.append(row)
    run_ids = [r.run_id for r in runs]
    if not rows:
        return pd.DataFrame(columns=[c["name"] for c in cols]), run_ids
    df = pd.DataFrame(rows, columns=[c["name"] for c in cols])
    return df, run_ids


def _filter_mask(df: pd.DataFrame, query: str) -> pd.Series | None:
    """Return a boolean mask for rows that match ``query`` (or None for all)."""
    if not query:
        return None
    q = query.lower()
    return df.apply(
        lambda row: any(q in str(v).lower() for v in row.values),
        axis=1,
    )


def _render_column_manager(runs: list[Run]) -> None:
    with st.expander("Columns", expanded=False):
        cols = st.session_state[_COLS_KEY]

        st.caption("Active columns")
        if not cols:
            st.write("_(none)_")
        for i, col in enumerate(list(cols)):
            c1, c2, c_up, c_dn, c_rm = st.columns([3, 5, 0.5, 0.5, 0.5])
            desc = _describe(col)
            if col.get("protected"):
                # Name is read-only and remove is disabled — the column's
                # literal name is load-bearing (LinkColumn keys on it).
                c1.markdown(f"**{col['name']}**")
                c2.markdown(f"`{desc}`")
                c_rm.markdown("🔒")
                continue
            # Key widgets by column name, not index — otherwise deleting a
            # column leaves stale cached text in `colname_{i}` when later
            # columns shift up to fill the hole.
            wkey = f"col_{col['name']}"
            new_name = c1.text_input(
                "name",
                value=col["name"],
                key=f"{wkey}_name",
                label_visibility="collapsed",
            )
            c2.markdown(f"`{desc}`")
            if c_up.button("▲", key=f"{wkey}_up", help="Move up",
                           disabled=not _can_move(cols, i, -1)):
                _move_col(cols, i, -1)
                _sync_columns()
                st.rerun()
            if c_dn.button("▼", key=f"{wkey}_dn", help="Move down",
                           disabled=not _can_move(cols, i, +1)):
                _move_col(cols, i, +1)
                _sync_columns()
                st.rerun()
            if c_rm.button("✕", key=f"{wkey}_rm", help="Remove"):
                cols.pop(i)
                _sync_columns()
                st.rerun()
            if new_name != col["name"]:
                col["name"] = new_name
                _sync_columns()

        st.divider()

        st.caption("Add column")
        with st.form("add_col", clear_on_submit=True):
            a1, a2, a3 = st.columns([2, 2, 2])
            name = a1.text_input("Name", placeholder="my_column")
            source = a2.selectbox("Source", col_mod.SOURCES, index=0)
            kind = a3.selectbox("Kind", col_mod.KINDS, index=1)
            path_or_pattern = st.text_input(
                "Path (dot.notation) or regex pattern",
                placeholder="observations.policy.height_scan",
            )
            if st.form_submit_button("Add"):
                if name and path_or_pattern:
                    new = {"name": name, "source": source, "kind": kind}
                    if kind == "regex":
                        new["pattern"] = path_or_pattern
                    else:
                        new["path"] = path_or_pattern
                    cols.append(new)
                    _sync_columns()
                    st.rerun()
                else:
                    st.warning("Name and path/pattern are both required.")

        st.divider()

        st.caption("Auto-add columns by scanning a group across every run")
        g1, g2 = st.columns(2)
        if g1.button("+ rewards (weight per term)"):
            _append_unique(cols, col_mod.expand_group(
                runs, "env", "rewards", "value",
                name_prefix="r:", child_suffix=".weight",
                require_key="func",
            ))
            _sync_columns()
            st.rerun()
        if g2.button("+ observations.policy (presence)"):
            _append_unique(cols, col_mod.expand_group(
                runs, "env", "observations.policy", "exists",
                name_prefix="o:",
                require_key="func",
            ))
            _sync_columns()
            st.rerun()
        g3, g4 = st.columns(2)
        if g3.button("+ terminations (presence)"):
            _append_unique(cols, col_mod.expand_group(
                runs, "env", "terminations", "exists",
                name_prefix="t:",
                require_key="func",
            ))
            _sync_columns()
            st.rerun()
        if g4.button("+ events (presence)"):
            _append_unique(cols, col_mod.expand_group(
                runs, "env", "events", "exists",
                name_prefix="e:",
                require_key="func",
            ))
            _sync_columns()
            st.rerun()

        with st.expander("Custom group expand", expanded=False):
            with st.form("expand_group", clear_on_submit=False):
                e1, e2 = st.columns([2, 1])
                grp_source = e1.selectbox(
                    "Source", ("env", "agent"), key="grp_source"
                )
                grp_kind = e2.selectbox(
                    "Kind", ("exists", "value"), key="grp_kind"
                )
                grp_path = st.text_input(
                    "Group path",
                    placeholder="rewards",
                    key="grp_path",
                )
                grp_suffix = st.text_input(
                    "Child suffix (e.g. .weight, leave blank for the child itself)",
                    placeholder=".weight",
                    key="grp_suffix",
                )
                grp_prefix = st.text_input(
                    "Column name prefix",
                    placeholder="r:",
                    key="grp_prefix",
                )
                if st.form_submit_button("Expand"):
                    if grp_path:
                        new_cols = col_mod.expand_group(
                            runs, grp_source, grp_path, grp_kind,
                            name_prefix=grp_prefix,
                            child_suffix=grp_suffix,
                        )
                        if not new_cols:
                            st.warning(
                                f"No child keys found under {grp_source}:{grp_path}"
                            )
                        else:
                            _append_unique(cols, new_cols)
                            _sync_columns()
                            st.rerun()

        st.divider()

        st.caption("Peek a value (to discover paths before adding a column)")
        run_labels = [f"{r.experiment}/{r.run_id}" for r in runs]
        if run_labels:
            p1, p2, p3 = st.columns([3, 2, 5])
            peek_run_label = p1.selectbox("Run", run_labels, key="peek_run")
            peek_source = p2.selectbox(
                "Source", ("env", "agent"), key="peek_source"
            )
            peek_path = p3.text_input(
                "Path", key="peek_path", placeholder="scene.num_envs"
            )
            if peek_path:
                run = runs[run_labels.index(peek_run_label)]
                root = run.env if peek_source == "env" else run.agent
                val = dot_get(root, peek_path)
                st.code(repr(val), language="python")

        st.divider()

        if st.button("Reset to defaults"):
            st.session_state[_COLS_KEY] = col_mod.default_columns()
            _sync_columns()
            st.rerun()


def _describe(col: dict) -> str:
    source = col.get("source", "?")
    kind = col.get("kind", "?")
    tail = col.get("pattern") if kind == "regex" else col.get("path", "")
    return f"{source}:{kind}  {tail}"


def _can_move(cols: list[dict], i: int, delta: int) -> bool:
    """A non-protected column can swap with an adjacent non-protected one.

    Protected columns are load-bearing (their names key into column_config)
    and stay pinned at the top in a fixed order, so we never cross that
    boundary.
    """
    j = i + delta
    if j < 0 or j >= len(cols):
        return False
    return not cols[i].get("protected") and not cols[j].get("protected")


def _move_col(cols: list[dict], i: int, delta: int) -> None:
    if _can_move(cols, i, delta):
        j = i + delta
        cols[i], cols[j] = cols[j], cols[i]


def _append_unique(existing: list[dict], new: list[dict]) -> None:
    """Append ``new`` columns to ``existing`` in place, skipping duplicates by name."""
    names = {c["name"] for c in existing}
    for col in new:
        if col["name"] in names:
            continue
        existing.append(col)
        names.add(col["name"])


def _cellstr(v) -> str:
    """Render any yaml value as a short string for a dataframe cell."""
    if v is None:
        return ""
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return repr(v)
    if isinstance(v, (dict, list, tuple)):
        return repr(v)
    return str(v)


_PLAY_PROCS_KEY = "play_processes"
_TB_PROC_KEY = "tensorboard_process"
_TB_MESSAGE_KEY = "tensorboard_message"
_LAST_TB_RUN_URL_KEY = "last_tensorboard_run_url"
_TERRAIN_PROC_KEY = "terrain_viewer_process"

_ATTRIBUTION_METHODS = (
    "integrated_gradients",
    "gradient_saliency",
    "gradient_input",
    "deep_lift_rescale",
    "deep_shap",
)


@st.cache_data(show_spinner=False)
def _go2_task_ids() -> list[str]:
    try:
        import mjlab.tasks  # noqa: F401
        import src.tasks  # noqa: F401
        from mjlab.tasks.registry import list_tasks
    except Exception:
        return ["Unitree-Go2-Test"]
    # Match the whole Go2 family: "Unitree-Go2-*" (legged) and "Unitree-Go2W-*"
    # (wheeled). The prefix deliberately omits the trailing hyphen so "Go2W"
    # is included.
    tasks = [task for task in list_tasks() if task.startswith("Unitree-Go2")]
    return tasks or ["Unitree-Go2-Test"]


def _load_terrain_viewer_module():
    spec = importlib.util.spec_from_file_location(
        "unitree_train_log_terrain_viewer", _TERRAIN_VIEWER_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {_TERRAIN_VIEWER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@st.cache_data(show_spinner=False)
def _terrain_viewer_options() -> tuple[dict[str, list[str]], str | None]:
    try:
        module = _load_terrain_viewer_module()
    except Exception as exc:
        return {"Built-in Presets": ["All Terrains"]}, str(exc)

    source_builtin = getattr(module, "TERRAIN_SOURCE_BUILTIN", "Built-in Presets")
    source_eval = getattr(module, "TERRAIN_SOURCE_EVALUATION", "Evaluation Terrains")
    source_env = getattr(module, "TERRAIN_SOURCE_ENVIRONMENT", "Environment Terrains")
    options: dict[str, list[str]] = {
        source_builtin: list(module.builtin_terrain_option_names()),
        source_eval: list(module.discover_evaluation_terrain_options().keys()),
        source_env: list(module.discover_environment_terrain_options().keys()),
    }
    return options, None


def _default_index(options: list[str], preferred: str) -> int:
    try:
        return options.index(preferred)
    except ValueError:
        return 0


def _effective_tb_url(args: argparse.Namespace) -> str:
    return args.tb_url or f"http://localhost:{args.tb_port}"


def _connect_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "localhost"} else host


def _checkpoint_label(path: Path) -> str:
    iteration = checkpoint_iter(path)
    if iteration is None:
        return path.name
    return f"{path.name} (iter {iteration})"


def _play_processes() -> dict[str, proc_mgr.ManagedProcess]:
    procs = st.session_state.setdefault(_PLAY_PROCS_KEY, {})
    return procs


def _start_play_process(
    *,
    run: Run,
    task_id: str,
    checkpoint: Path,
    attribution: bool,
    attribution_method: str,
    enable_terminations: bool,
    args: argparse.Namespace,
) -> proc_mgr.ManagedProcess:
    port = proc_mgr.find_free_port(args.viewer_port_start, args.viewer_port_end)
    url = f"http://{args.viewer_url_host}:{port}"
    command = [
        sys.executable,
        "scripts/play.py",
        task_id,
        "--checkpoint-file",
        str(checkpoint.resolve()),
        "--viewer",
        "viser",
    ]
    if not enable_terminations:
        command.extend(["--no-terminations", "True"])
    if attribution:
        command.extend([
            "--attribution",
            "True",
            "--attribution-method",
            attribution_method,
        ])
    proc = proc_mgr.start_process(
        label=f"play_{run.experiment}_{run.run_id}_{task_id}_{checkpoint.stem}",
        command=command,
        cwd=REPO_ROOT,
        env_overrides={"_VISER_PORT_OVERRIDE": str(port)},
        url=url,
    )
    proc_mgr.announce_url_when_listening(
        proc=proc,
        host="127.0.0.1",
        port=port,
        url=f"http://localhost:{port}",
        label="Train Log Manager Play viewer",
    )
    key = f"{proc.pid}:{run.experiment}/{run.run_id}:{task_id}:{checkpoint.name}"
    _play_processes()[key] = proc
    return proc


def _start_terrain_viewer(
    args: argparse.Namespace,
    *,
    terrain_source: str,
    terrain_name: str,
) -> proc_mgr.ManagedProcess:
    existing = st.session_state.get(_TERRAIN_PROC_KEY)
    if existing is not None and existing.is_running:
        return existing

    port = proc_mgr.find_free_port(args.viewer_port_start, args.viewer_port_end)
    url = f"http://{args.viewer_url_host}:{port}"
    command = [
        sys.executable,
        "scripts/visualize_terrain.py",
        "--port",
        str(port),
        "--no-robots",
        "--terrain-source",
        terrain_source,
        "--terrain",
        terrain_name,
    ]
    proc = proc_mgr.start_process(
        label=f"terrain_viewer_{terrain_source}_{terrain_name}",
        command=command,
        cwd=REPO_ROOT,
        url=url,
    )
    proc_mgr.announce_url_when_listening(
        proc=proc,
        host="127.0.0.1",
        port=port,
        url=f"http://localhost:{port}",
        label="Train Log Manager Terrain viewer",
    )
    st.session_state[_TERRAIN_PROC_KEY] = proc
    return proc


def _ensure_tensorboard(args: argparse.Namespace, tb_url: str) -> str:
    existing = st.session_state.get(_TB_PROC_KEY)
    if existing is not None and existing.is_running:
        st.session_state[_TB_MESSAGE_KEY] = f"TensorBoard is running as PID {existing.pid}."
        return tb_url

    if proc_mgr.is_port_listening(_connect_host(args.tb_host), args.tb_port):
        st.session_state[_TB_MESSAGE_KEY] = (
            f"Port {args.tb_port} is already listening; using existing TensorBoard URL."
        )
        return tb_url

    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(args.logs_root.resolve()),
        "--host",
        args.tb_host,
        "--port",
        str(args.tb_port),
    ]
    proc = proc_mgr.start_process(
        label="tensorboard",
        command=command,
        cwd=REPO_ROOT,
        url=tb_url,
    )
    proc_mgr.announce_url_when_listening(
        proc=proc,
        host=_connect_host(args.tb_host),
        port=args.tb_port,
        url=f"http://localhost:{args.tb_port}",
        label="Train Log Manager TensorBoard",
    )
    st.session_state[_TB_PROC_KEY] = proc
    st.session_state[_TB_MESSAGE_KEY] = f"Started TensorBoard as PID {proc.pid}."
    return tb_url


def _render_managed_process(proc: proc_mgr.ManagedProcess, *, key_prefix: str) -> None:
    status = "running" if proc.is_running else f"exited ({proc.returncode})"
    st.caption(f"{proc.label} | PID {proc.pid} | {status}")
    cols = st.columns([1, 1, 5])
    if proc.url:
        cols[0].link_button("Open", proc.url)
    if cols[1].button("Stop", key=f"{key_prefix}_stop_{proc.pid}", disabled=not proc.is_running):
        proc_mgr.stop_process(proc)
        st.rerun()
    st.code(shlex.join(proc.command), language="bash")
    output = proc_mgr.read_recent_output(proc.log_path)
    if output:
        with st.expander("Recent output", expanded=False):
            st.code(output)


def _render_processes() -> None:
    procs = _play_processes()
    stale = [key for key, proc in procs.items() if not proc.is_running and proc.returncode == 0]
    for key in stale[:-5]:
        procs.pop(key, None)

    tracked_pids = {proc.pid for proc in procs.values()}
    external_procs = [
        proc
        for proc in proc_mgr.discover_play_processes(REPO_ROOT)
        if proc.pid not in tracked_pids
    ]

    tb_proc = st.session_state.get(_TB_PROC_KEY)
    terrain_proc = st.session_state.get(_TERRAIN_PROC_KEY)
    if not procs and not external_procs and tb_proc is None and terrain_proc is None:
        return

    with st.expander(
        "Launched processes",
        expanded=bool(procs or external_procs or terrain_proc),
    ):
        if tb_proc is not None:
            _render_managed_process(tb_proc, key_prefix="tb")
        if terrain_proc is not None:
            _render_managed_process(terrain_proc, key_prefix="terrain")
        for key, proc in list(procs.items()):
            _render_managed_process(proc, key_prefix=f"play_{key}")
        for proc in external_procs:
            _render_managed_process(proc, key_prefix=f"external_play_{proc.pid}")


def _render_terrain_viewer_action(args: argparse.Namespace) -> None:
    with st.expander("Terrain Viewer", expanded=False):
        st.caption("Choose a terrain here, then launch it in Viser without spawning robots.")
        existing = st.session_state.get(_TERRAIN_PROC_KEY)
        if existing is not None and existing.is_running:
            st.info(f"Terrain viewer is running as PID {existing.pid}.")
            if existing.url:
                st.link_button("Open Terrain Viewer", existing.url)
            st.code(shlex.join(existing.command), language="bash")
            return

        options_by_source, options_error = _terrain_viewer_options()
        if options_error:
            st.warning(f"Could not load full terrain options: {options_error}")
        source_options = [
            source for source, terrain_names in options_by_source.items() if terrain_names
        ]
        if not source_options:
            st.error("No terrain options are available.")
            return

        preferred_source = (
            "Evaluation Terrains"
            if "Evaluation Terrains" in source_options
            else source_options[0]
        )
        c1, c2 = st.columns([1, 2])
        terrain_source = c1.selectbox(
            "Terrain source",
            source_options,
            index=_default_index(source_options, preferred_source),
        )
        terrain_names = options_by_source[terrain_source]
        preferred_terrain = (
            "rough_curriculum_corridor"
            if terrain_source == "Evaluation Terrains"
            and "rough_curriculum_corridor" in terrain_names
            else terrain_names[0]
        )
        terrain_name = c2.selectbox(
            "Terrain",
            terrain_names,
            index=_default_index(terrain_names, preferred_terrain),
        )

        if st.button("Launch Terrain Viewer", type="primary"):
            try:
                proc = _start_terrain_viewer(
                    args, terrain_source=terrain_source, terrain_name=terrain_name
                )
            except Exception as exc:
                st.error(f"Could not start terrain viewer: {exc}")
            else:
                st.success(f"Started terrain viewer PID {proc.pid}.")
                if proc.url:
                    st.link_button("Open Terrain Viewer", proc.url)
                st.code(shlex.join(proc.command), language="bash")


def _render_run_actions(
    run: Run,
    *,
    args: argparse.Namespace,
    tb_url: str,
    task_ids: list[str],
) -> None:
    st.subheader("Actions")
    st.caption(f"Selected `{run.experiment}/{run.run_id}`")

    play_tab, tb_tab = st.tabs(["Play", "TensorBoard"])

    with play_tab:
        if not run.checkpoints:
            st.warning("This run has no `model_<iter>.pt` checkpoints.")
        checkpoint_labels = [_checkpoint_label(path) for path in run.checkpoints]
        c1, c2 = st.columns([2, 2])
        preferred_task = run.task_id if run.task_id in task_ids else "Unitree-Go2-Test"
        task_id = c1.selectbox(
            "Environment",
            task_ids,
            index=_default_index(task_ids, preferred_task),
        )
        checkpoint_index = max(len(run.checkpoints) - 1, 0)
        checkpoint_label = c2.selectbox(
            "Checkpoint",
            checkpoint_labels or ["No checkpoints"],
            index=checkpoint_index,
            disabled=not run.checkpoints,
        )
        c3, c4, c5 = st.columns([1, 2, 1])
        attribution = c3.checkbox("Attribution", value=True)
        attribution_method = c4.selectbox(
            "Attribution method",
            _ATTRIBUTION_METHODS,
            index=_default_index(list(_ATTRIBUTION_METHODS), "deep_shap"),
            disabled=not attribution,
        )
        enable_terminations = c5.checkbox(
            "Terminations",
            value=True,
            help="Stop episodes on configured termination conditions such as falls or illegal contact.",
        )
        selected_checkpoint = (
            run.checkpoints[checkpoint_labels.index(checkpoint_label)]
            if run.checkpoints
            else None
        )
        disabled = selected_checkpoint is None
        if st.button("Play", disabled=disabled, type="primary") and selected_checkpoint is not None:
            try:
                proc = _start_play_process(
                    run=run,
                    task_id=task_id,
                    checkpoint=selected_checkpoint,
                    attribution=attribution,
                    attribution_method=attribution_method,
                    enable_terminations=enable_terminations,
                    args=args,
                )
            except Exception as exc:
                st.error(f"Could not start play process: {exc}")
            else:
                st.success(f"Started play process PID {proc.pid}.")
                if proc.url:
                    st.link_button("Open Viser", proc.url)
                st.code(shlex.join(proc.command), language="bash")

    with tb_tab:
        run_url = _tb_run_url(tb_url, run.run_id)
        if st.button("View in TensorBoard", type="primary"):
            try:
                _ensure_tensorboard(args, tb_url)
            except Exception as exc:
                st.error(f"Could not start TensorBoard: {exc}")
            else:
                st.session_state[_LAST_TB_RUN_URL_KEY] = run_url
        message = st.session_state.get(_TB_MESSAGE_KEY)
        if message:
            st.info(message)
        last_url = st.session_state.get(_LAST_TB_RUN_URL_KEY)
        if last_url:
            st.link_button("Open TensorBoard", last_url)
        st.caption("TensorBoard is filtered to the selected run.")


def _render_detail(base: Run, other: Run) -> None:
    st.subheader("Detail / compare")
    st.caption(
        f"Comparing `{base.experiment}/{base.run_id}` against "
        f"`{other.experiment}/{other.run_id}`"
    )

    tab_yaml, tab_git = st.tabs(["YAML diff", "Git diff"])

    with tab_yaml:
        for label, a, b in (
            ("run.yaml", base.run, other.run),
            ("env.yaml", base.env, other.env),
            ("agent.yaml", base.agent, other.agent),
        ):
            st.markdown(f"**{label}**")
            rows = dict_diff(a, b)
            if not rows:
                st.write("_(identical)_")
                continue
            # Stringify values — columns mix ints, tuples, bytes, dicts, None,
            # which pyarrow can't serialize together.
            str_rows = [(p, _cellstr(av), _cellstr(bv)) for p, av, bv in rows]
            df = pd.DataFrame(
                str_rows,
                columns=["path", f"base ({base.run_id})", f"other ({other.run_id})"],
            )
            st.dataframe(df, width="stretch", hide_index=True)

    with tab_git:
        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown(f"**base — {base.run_id}**")
            st.code(base.git_diff or "(empty)", language="diff")
        with gc2:
            st.markdown(f"**other — {other.run_id}**")
            st.code(other.git_diff or "(empty)", language="diff")


def main() -> None:
    st.set_page_config(page_title="Train Log Manager", layout="wide")
    args = _parse_args()

    cols_state = _init_columns()
    tb_url = _effective_tb_url(args)
    task_ids = _go2_task_ids()

    st.title("Train Log Manager")
    st.caption(f"Scanning `{args.logs_root}`")

    _render_terrain_viewer_action(args)

    runs = _cached_scan(str(args.logs_root))
    if not runs:
        st.warning(f"No runs found under {args.logs_root}.")
        _render_processes()
        return

    col_l, col_r = st.columns([4, 1])
    query = col_l.text_input(
        "Filter", placeholder="substring — matches any visible cell"
    )
    if col_r.button("↻ Rescan"):
        _cached_scan.clear()
        st.rerun()

    df, _ = _build_table(runs, cols_state, tb_url)
    display_runs = runs
    mask = _filter_mask(df, query)
    if mask is not None:
        keep_flags = mask.tolist()
        df = df[mask].reset_index(drop=True)
        display_runs = [run for run, keep in zip(runs, keep_flags) if keep]

    event = st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        column_config={
            "run_id": st.column_config.LinkColumn(
                "run_id",
                help="Open this run in TensorBoard",
                # URL ends in "…regexInput=<escaped_run_id>$". Strip the
                # trailing $ anchor and unescape '\-' so the cell shows a
                # clean timestamp instead of the raw regex.
                display_text=r"regexInput=(.+)\$$",
                pinned=True,
            ),
            "task_id": st.column_config.Column("task_id", pinned=True),
            "max_iter": st.column_config.Column("max_iter", pinned=True),
            "max_mean_reward": st.column_config.NumberColumn(
                "max_mean_reward", pinned=True, format="%.3f"
            ),
        },
    )
    st.caption(f"{len(df)} / {len(runs)} runs shown")

    selected_rows = list(getattr(event.selection, "rows", []) or [])
    selected_runs = [
        display_runs[index]
        for index in selected_rows
        if 0 <= index < len(display_runs)
    ]
    if len(selected_runs) == 1:
        _render_run_actions(
            selected_runs[0],
            args=args,
            tb_url=tb_url,
            task_ids=task_ids,
        )
    elif len(selected_runs) == 2:
        _render_detail(selected_runs[0], selected_runs[1])
    elif len(selected_runs) > 2:
        st.info("Select one run for Actions, or exactly two runs for Detail / compare.")
    else:
        st.info("Select one run for Actions, or two runs for Detail / compare.")

    _render_processes()

    _render_column_manager(runs)


if __name__ == "__main__":
    main()
