"""Streamlit UI for browsing rsl_rl training runs.

Run with:

    streamlit run tools/train_log_manager/app.py -- --logs-root logs/rsl_rl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow ``streamlit run`` to resolve sibling modules.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import columns as col_mod  # noqa: E402
from diff import dict_diff, dot_get  # noqa: E402
from scanner import Run, scan_runs  # noqa: E402


DEFAULT_LOGS_ROOT = Path(__file__).resolve().parents[1] / "logs" / "rsl_rl"
CONFIG_PATH = _HERE / ".columns.json"


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
        default="http://localhost:6006",
        help="TensorBoard base URL — run_id cells link into it with #timeseries&regexInput=<run_id>.",
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


def _render_detail(runs: list[Run]) -> None:
    st.subheader("Detail / compare")
    if len(runs) < 1:
        st.info("No runs found.")
        return

    labels = [f"{r.experiment}/{r.run_id}" for r in runs]
    c1, c2 = st.columns(2)
    base_label = c1.selectbox("Base run", labels, index=0, key="base_run")
    default_other = 1 if len(labels) > 1 else 0
    other_label = c2.selectbox(
        "Compare against", labels, index=default_other, key="other_run"
    )
    base = runs[labels.index(base_label)]
    other = runs[labels.index(other_label)]

    tab_yaml, tab_git = st.tabs(["YAML diff", "Git diff"])

    with tab_yaml:
        for label, a, b in (("env.yaml", base.env, other.env),
                            ("agent.yaml", base.agent, other.agent)):
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

    st.title("Train Log Manager")
    st.caption(f"Scanning `{args.logs_root}`")

    runs = _cached_scan(str(args.logs_root))
    if not runs:
        st.warning(f"No runs found under {args.logs_root}.")
        return

    col_l, col_r = st.columns([4, 1])
    query = col_l.text_input(
        "Filter", placeholder="substring — matches any visible cell"
    )
    if col_r.button("↻ Rescan"):
        _cached_scan.clear()
        st.rerun()

    df, run_ids = _build_table(runs, cols_state, args.tb_url)
    mask = _filter_mask(df, query)
    if mask is not None:
        df = df[mask].reset_index(drop=True)
        run_ids = [rid for rid, keep in zip(run_ids, mask.tolist()) if keep]

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
            "max_iter": st.column_config.Column("max_iter", pinned=True),
            "max_mean_reward": st.column_config.NumberColumn(
                "max_mean_reward", pinned=True, format="%.3f"
            ),
        },
    )
    st.caption(f"{len(df)} / {len(runs)} runs shown")

    selected_rows = list(getattr(event.selection, "rows", []) or [])
    if selected_rows:
        selected_ids = [run_ids[i] for i in selected_rows if 0 <= i < len(run_ids)]
        if selected_ids:
            combined = (
                f"{args.tb_url.rstrip('/')}/#timeseries&regexInput="
                f"{_tb_run_regex(selected_ids)}"
            )
            st.link_button(
                f"Open {len(selected_ids)} selected run"
                f"{'s' if len(selected_ids) != 1 else ''} in TensorBoard",
                combined,
            )

    _render_column_manager(runs)

    st.divider()
    _render_detail(runs)


if __name__ == "__main__":
    main()
