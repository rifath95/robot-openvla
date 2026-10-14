# Scripted touch evaluation

This is a separate, explicitly invoked workflow. The usual `cloud.cloud_session` interactive simulator continues to start paused, accept Space presses, and keep the model loaded when you disconnect.

The scripted launcher uses the same `inference.remote_loop`, camera, action decoding, IK, physics and contact checks. It runs without a viewer or keyboard prompts. Replay the saved trials afterward to watch them.

Use a dedicated inference pod with the training template, RunPod API secret and the network volume containing the checkpoints. Keep the Mac awake and connected throughout. End any previous local simulator session to free port 8000 first.

Run from the repository in a **local Mac terminal**, substituting the pod's current SSH address:

```bash
caffeinate -i .venv/bin/python -m cloud.scripted_touch_session \
  --ssh 'ssh root@HOST -p PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --training-session panda-train-20261013-034928-1786
```

Defaults:

- Checkpoints 50, 100, 300 and 500, with sampling off.
- The same 20 predetermined unseen seeds, 1000–1019, for every checkpoint: 80 trials total.
- Fixed initial arm pose and camera; only cube XY varies within the existing dataset bounds.
- Instruction: `touch the red cube.`
- Up to 50 actions and 180 active seconds per trial. Stop on contact or an existing controller/connection failure.
- One model load per checkpoint; the model stays loaded for all 20 positions.
- Continue after individual trial failures while the server remains healthy; abort if the server becomes unavailable.

Optional `--checkpoints 40 50 100 300 500` includes checkpoint 40. `--scene-seeds` accepts unique seeds >=50; the script checks positions against the local dataset manifest. `--steps`, `--trial-timeout-seconds` and `--startup-timeout-seconds` set limits. The local dataset manifest must remain available for the unseen-position check.

## Saved results and shutdown

Each launch creates `outputs/sweeps/scripted_touch_<timestamp>_<id>/` with a `REPORT.md`, `results.json`, `console.log`, checkpoint startup metadata, and one directory per checkpoint/seed. Each trial retains the standard images, predictions, executed actions, simulator states and replay launcher. The report distinguishes positive-force finger–cube contact from clean completion: contact can coincide with a tracking error.

The summary JSON and Markdown are also copied to `/workspace/panda-training/scripted-evaluations/<session>/` before stopping. Detailed trial images and actions stay on the Mac.

After completion, a launcher error or Ctrl+C, the script attempts to stop the **pod addressed by the SSH command**, using that pod's environment/API identity. It does not stop another training pod. Its stop status is recorded on the volume when writable.

This is a connected Mac workflow, not a detached cloud supervisor. Keep it running: lid closure, a Mac shutdown, forced process termination, a second interrupt during cleanup, or a lost connection can prevent cleanup. A network/API failure can prevent the stop request from succeeding. Confirm **Stopped** in RunPod afterward. The cloud model-server runtime limit alone does not stop pod billing.

To watch a recorded trial later:

```bash
.venv/bin/python outputs/sweeps/scripted_touch_<timestamp>_<id>/checkpoint_000100_seed_1000/replay.py
```
