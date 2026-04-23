# Train Log Manager

A small Streamlit app that lists every rsl_rl training run under `logs/rsl_rl/`
in a sortable table, with user-defined columns that surface any parameter from
`params/agent.yaml`, `params/env.yaml`, or the captured `git/*.diff` — e.g.
"height_scan on/off", "num_envs", "learning_rate", "was `rewards.py` patched".

## Install & run

```bash
pip install streamlit pyyaml

# from the repo root
streamlit run tools/train_log_manager/app.py -- --logs-root logs/rsl_rl
```

Open http://localhost:8501 (port-forward it if running over SSH).

Flags:

- `--logs-root PATH` — directory containing `<experiment>/<run_id>/` subdirs.
  Defaults to `<repo>/logs/rsl_rl`.

## UI

1. **Table view** — one row per run. The filter box does a case-insensitive
   substring match across visible cells. Click a column header to sort.
   `↻ Rescan` drops the cache and re-reads the filesystem.

2. **Columns expander** — manage columns live:
   - Rename inline, remove with `✕`.
   - Add a new one with `source` (`env` / `agent` / `git_diff` / `builtin`),
     `kind` (`exists` / `value` / `regex` / `builtin`), and a path or pattern.
   - "Peek a value" lets you test a dot-path against a specific run before
     committing it as a column.
   - "Reset to defaults" restores the built-in column set.

3. **Detail / compare** — pick any two runs; the `YAML diff` tab shows a
   flattened per-key diff of `env.yaml` and `agent.yaml`, and the `Git diff`
   tab shows both runs' captured diffs side by side.

## Column kinds

| kind      | source     | meaning                                                  | example path / pattern                        |
|-----------|------------|----------------------------------------------------------|-----------------------------------------------|
| `value`   | env/agent  | value at a dot-path (`None` if missing)                  | `scene.num_envs`, `algorithm.learning_rate`   |
| `exists`  | env/agent  | `True`/`False` — is the dot-path present at all?         | `observations.policy.height_scan`             |
| `regex`   | git_diff   | `True`/`False` — `re.search(pattern, git_diff_text)`     | `mdp/rewards\.py`                             |
| `builtin` | builtin    | field on the `Run` record                                | `run_id`, `max_iter`, `experiment`            |

Paths support list indexing: `scene.robot.actuators[0].effort_limit`.

## URL-based persistence

The active column set is encoded into the URL as `?cols=<base64>`. Reloading
the page restores the same columns; copying the URL shares them with someone
else. No config file is written. "Reset to defaults" clears the customization.

## Layout

```
tools/train_log_manager/
├── app.py       # Streamlit entrypoint
├── scanner.py   # find runs, load agent.yaml / env.yaml / git diff
├── columns.py   # column definitions + value extraction + URL codec
└── diff.py      # flatten, dict_diff, dot_get / dot_has
```

## Notes

- `env.yaml` is loaded with `yaml.unsafe_load` because Isaac Lab emits
  `!!python/tuple` tags. Only your own training logs are read.
- A run directory is considered valid iff it contains a `params/` subdir.
- Results are cached per `logs_root`; use `↻ Rescan` after new runs finish.
