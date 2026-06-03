# Train Log Manager

A small Streamlit app that lists every rsl_rl training run under `logs/rsl_rl/`
in a sortable table, with user-defined columns that surface run metadata from
`params/run.yaml`, any parameter from `params/agent.yaml`, `params/env.yaml`,
or the captured `git/*.diff` — e.g.
"height_scan on/off", "num_envs", "learning_rate", "was `rewards.py` patched".

## Install & run

```bash
pip install streamlit pyyaml tensorboard

# from the repo root
streamlit run tools/train_log_manager/app.py -- --logs-root logs/rsl_rl
```

Open http://localhost:8501 (port-forward it if running over SSH).

Flags:

- `--logs-root PATH` - directory containing `<experiment>/<run_id>/` subdirs.
  Defaults to `<repo>/logs/rsl_rl`.
- `--tb-host HOST` / `--tb-port PORT` - host and port used when the app auto-starts TensorBoard.
- `--tb-url URL` - public TensorBoard base URL for links. Defaults to `http://localhost:<tb-port>`.
- `--viewer-url-host HOST` - host used for Viser links after Play launches. Defaults to `localhost`.
- `--viewer-port-start PORT` / `--viewer-port-end PORT` - port range allocated to background Viser play processes.

## UI

1. **Table view** - one row per run. The filter box does a case-insensitive
   substring match across visible cells. Click a column header to sort.
   `↻ Rescan` drops the cache and re-reads the filesystem. Select one row to
   enable run actions, or select two rows to compare those runs.

2. **Actions** - the selected run gets Play and TensorBoard tabs.
   - Play defaults to the latest `model_<iter>.pt`, lets you choose a Go2 env,
     launches `scripts/play.py` in the background with `--viewer viser`, and
     shows an `Open Viser` link plus process output.
   - TensorBoard starts `python -m tensorboard.main --logdir <logs-root>` if
     the configured port is not already listening, then links to the selected
     run filter.

3. **Columns expander** — manage columns live:
   - Rename inline, remove with `✕`.
   - Add a new one with `source` (`env` / `agent` / `git_diff` / `builtin`),
     `kind` (`exists` / `value` / `regex` / `builtin`), and a path or pattern.
   - "Peek a value" lets you test a dot-path against a specific run before
     committing it as a column.
   - "Reset to defaults" restores the built-in column set.

4. **Detail / compare** — select two rows in the table; the `YAML diff` tab
   shows a flattened per-key diff of `run.yaml`, `env.yaml`, and
   `agent.yaml`, and the `Git diff` tab shows both runs' captured diffs side by side.

## Column kinds

| kind      | source     | meaning                                                  | example path / pattern                        |
|-----------|------------|----------------------------------------------------------|-----------------------------------------------|
| `value`   | env/agent  | value at a dot-path (`None` if missing)                  | `scene.num_envs`, `algorithm.learning_rate`   |
| `exists`  | env/agent  | `True`/`False` — is the dot-path present at all?         | `observations.policy.height_scan`             |
| `regex`   | git_diff   | `True`/`False` — `re.search(pattern, git_diff_text)`     | `mdp/rewards\.py`                             |
| `builtin` | builtin    | field on the `Run` record                                | `run_id`, `task_id`, `max_iter`               |

Paths support list indexing: `scene.robot.actuators[0].effort_limit`.

## URL-based persistence

The active column set is encoded into the URL as `?cols=<base64>`. Reloading
the page restores the same columns; copying the URL shares them with someone
else. No config file is written. "Reset to defaults" clears the customization.

## Layout

```
tools/train_log_manager/
├── app.py       # Streamlit entrypoint
├── scanner.py          # find runs, load metadata, discover checkpoints
├── process_manager.py  # background Play / TensorBoard process helpers
├── columns.py          # column definitions + value extraction + URL codec
└── diff.py             # flatten, dict_diff, dot_get / dot_has
```

## Notes

- `env.yaml` is loaded with `yaml.unsafe_load` because Isaac Lab emits
  `!!python/tuple` tags. Only your own training logs are read. Logs with stale
  Python class references are still listed, but unavailable YAML metadata is
  shown as missing.
- A run directory is considered valid iff it contains a `params/` subdir.
- `params/run.yaml` is optional; when present, `task_id` is shown in the table
  and used as the default Play environment if it is a Go2 task.
- Results are cached per `logs_root`; use `↻ Rescan` after new runs finish.
