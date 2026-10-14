# Replay simulator trials from REPORT.md

After downloading a complete Hugging Face sweep archive, restore it locally:

```bash
.venv/bin/python -m data_tools.restore_sweep_archive \
  outputs/archives/hf-sweeps/hf-sweep-archives/scripted_multi_color_models_20261013_164548_eeda379b
```

This verifies downloaded checksums and restores the images and recorded simulator
states under `outputs/sweeps/scripted_multi_color_models_20261013_164548_eeda379b`.
Completed archive parts are remembered if restoration is interrupted.

Start the local replay launcher from the repository folder:

```bash
.venv/bin/python -m analysis.replay_launcher \
  --sweep outputs/sweeps/scripted_multi_color_models_20261013_164548_eeda379b
```

Keep that terminal running. Open the restored `REPORT.md` in VS Code Markdown
Preview (`Cmd+Shift+V`). Every trial row has a **▶ Replay** link, which opens
MuJoCo on your Mac and starts playback. Your browser may open a confirmation
page. Close the MuJoCo window before selecting another trial. Space pauses or
resumes playback; Escape closes it. No model weights or cloud pod are required.

The viewer plays recorded simulator poses, omitting cloud prediction waits.
It does not rerun model inference or recompute actions.

Ctrl+C stops the launcher. Restarting it refreshes the links in `REPORT.md`;
reload Markdown Preview to use the new links. The launcher listens only on your
Mac's loopback address and accepts only recorded trial names with its current
session token. If port 8765 is busy, add `--port 0` to select a free port.

Replay links are local to this launcher session and do not work from GitHub's
preview or another computer. Other sweep report formats are not yet supported.
