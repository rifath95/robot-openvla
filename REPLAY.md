# Offline run playback

Each new `remote_loop_*` output folder contains `replay.py`. Open it in VS Code,
select the repository's `.venv/bin/python` interpreter, and choose **Run Python
File**. It opens the local simulator paused. Space starts/pauses/resumes;
Esc or closing the window exits. At the end, the final scene stays visible.
No pod, SSH tunnel, weights, inference, IK solve, or physics execution is needed.

From the repository terminal, for example:

```bash
.venv/bin/python outputs/remote_loop_20261010_051422_5c23417b/replay.py
```

For any saved loop folder, including one without a launcher:

```bash
.venv/bin/python replay_run.py outputs/<RUN_FOLDER>
.venv/bin/python replay_run.py outputs/<RUN_FOLDER> --speed 0.5
.venv/bin/python replay_run.py outputs/<RUN_FOLDER> --check
```

Replace `<RUN_FOLDER>` with the directory name under `outputs/`. Speed 0.5 is
half speed; 2 is double speed. Playback omits prediction waits and original
pauses, showing movement at its simulation duration rather than reproducing
original wall-clock timing.

New runs save `step_*/motion_frames.npz` with the initial pose and a pose every
20 physics ticks (normally every 0.04 s), including gripper and object positions.
The viewer displays these recorded poses without recomputing commands. This is
sampled motion playback, not a recording of every physics tick or a video.
Rejected actions hold their saved pose; incomplete observation-only steps show
only the available state. Interrupted movements retain their recorded frames.

Older runs saved before/after states only. Their playback interpolates between
those endpoints, including quaternion-aware interpolation for free objects.
Intermediate motion is approximate and cannot reconstruct contact events or
paths that were not recorded. The console explicitly identifies this mode.
Launchers have been added to the existing local loop folders, including the
latest 100-step run.

Playback requires this repository, its local simulation dependencies, and the
scene/MuJoCo Menagerie assets. Use the same scene/assets as the original run;
folders are not standalone robot model bundles. Generated launchers and motion
files remain under ignored `outputs/`; the replay implementation is tracked.
