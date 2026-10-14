# Direct replay buttons in datasets and reports

Start one local service from the repository folder:

```bash
.venv/bin/python -m analysis.replay_launcher
```

Keep it running, open `WATCH_TRAJECTORIES.md`, `REPORT.md`, or `RUN_REPORT.md`
in Markdown Preview, and click **▶ Replay** or **▶ Action Replay**.
Both open MuJoCo and begin immediately; no Python file or Space-key start is
needed. Close the current window before opening another. Space pauses/resumes;
Escape closes the viewer.

Replay displays saved motion. Action Replay executes the demonstration's saved
actions through the current controller and physics; it requires initial state,
metadata and action records. It saves its own results under
`outputs/simulator_runs/action_replays/`. Simulator inference trials that lack an
expert action sequence offer only recorded-motion replay. No cloud pod is needed.

The dataset index, sweep report and individual-run report generators create
direct links automatically. Archive restoration and sweep fetching rebind links
to the Mac's local launcher. Starting the service refreshes existing dataset
indexes, sweep reports and individual simulator reports; recordings and dataset
manifests are preserved. You can refresh just one set with `--dataset PATH` or
`--sweep PATH`.

Links use a private workspace token stored in
`outputs/diagnostics/replay/launcher.json`, so ordinary service restarts do not
expire them. The service listens only on loopback, validates the token, restricts
paths to recordings under `datasets/` and `outputs/`, and launches fixed replay
modules rather than executing linked Python files. If the service is stopped,
restart it with the command above. If port 8765 is busy, use `--port 0`, then
reload Markdown Preview for refreshed links. Links target this Mac's service;
they do not operate another computer's viewer.

## Restore downloaded sweep recordings

After downloading a complete Hugging Face sweep archive, restore it locally:

```bash
.venv/bin/python -m data_tools.restore_sweep_archive \
  outputs/archives/hf-sweeps/hf-sweep-archives/scripted_multi_color_models_20261013_164548_eeda379b
```

This verifies downloaded checksums and restores the images and recorded simulator
states under `outputs/sweeps/scripted_multi_color_models_20261013_164548_eeda379b`.
Completed archive parts are remembered if restoration is interrupted.

Then use the service above. Reload the local report preview after restoration.
