# Keep the model loaded between simulator trials

Requested and implemented on 2026-10-12 after the checkpoint-2,000 bowl trials
finished and the user stopped the pod.

Previously, a shell loop launched a separate `cloud.cloud_session` for each scene
seed, and Ctrl+C stopped the model server. The next seed required another load.

The launcher now allows multiple trials with different scene seeds in one cloud session. Keep
the server and loaded model alive between those trials, and save each trial's
seed, outputs, timing, and replay separately. Distinguish ending a local trial
from stopping the cloud model server. Local session exit leaves the model loaded;
the server runtime limit remains effective. Loading a different checkpoint
requires replacing the loaded model.

Use `--task bowl --scene-seeds 0 1 2` to run a sequence using a single model load.
Each window starts paused. Close the window or press Esc to finish that trial;
the next queued seed starts automatically. Once the queue finishes, Enter repeats
the last seed, an integer selects another seed, and `q` disconnects locally.
Ctrl+C also disconnects locally without stopping the model server. Launch the same
command again to reconnect. A ready server is reused only when its base/adapter
identity and recorded live process match. The adapter identity includes its path,
configuration/statistics hashes and weight-file metadata; no repeated full weight
read is required. Missing provenance or a different checkpoint causes a restart.

Use the same SSH argument with `--stop-server` to explicitly unload the model:

```bash
.venv/bin/python -m cloud.cloud_session \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519' --stop-server
```

The default server lifetime remains 1,800 seconds (30 minutes) after readiness.
For a longer working session, specify `--server-runtime-seconds 21600` at the
initial launch (six hours). Reconnecting reuses the existing remaining lifetime;
it does not extend that timer. Expiry, server failure, or stopping/terminating the
pod can still release VRAM. Disconnecting does not stop GPU billing.

Verified with a launcher test covering queued seeds, invalid input, selecting a
new seed, repeating, and quitting: one bootstrap and no shutdown at local exit.
Separate tests verify explicit stop, matching checkpoint reuse, and rejection of
stale processes, changed checkpoints and servers that are not ready. The prior
bowl-scene test verifies recorded replay geometry and camera. Model-server shutdown
does not stop the rented pod; stop it in RunPod afterward.
