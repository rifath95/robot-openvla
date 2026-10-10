# One cloud-predicted robot movement

`remote_single_action.py` runs on the **Mac**. It captures the settled scene,
requests one real cloud OpenVLA prediction, applies bounded movement to that
same simulation, and captures the resulting scene. Physics does not advance
while waiting for the response. Each invocation starts a new home scene.

## Run

First start the real model server on the pod and the SSH tunnel on the Mac,
following steps 1–8 of [RUNPOD_CONNECTION_GUIDE.md](RUNPOD_CONNECTION_GUIDE.md).
Those commands are unchanged. In **Terminal 3 on the Mac**, replace the
display-only `connection_client.py` command with:

```bash
cd "/Users/rifath95/Desktop/Extra work/Robotics/robot-openvla"
.venv/bin/python remote_single_action.py --instruction "pick up the red cube"
```

It makes exactly one request and one movement, then opens the standard MuJoCo
viewer showing the **final** scene. Close the viewer to exit. It does not show a
live viewer during inference/motion and does not request a second action. Do not
launch `launch_sim.py` first; this script owns its simulation.

To save results without opening the final viewer:

```bash
.venv/bin/python remote_single_action.py --no-view
```

The default endpoint is `http://127.0.0.1:8000` through the tunnel, and the response
timeout is 180 seconds. Change them with `--server-url` and `--timeout-seconds`.
A timeout fails the local run without executing the requested action; it does
not cancel a cloud inference that is already running or stop pod billing.

## Action checks

- Requires an `openvla` response with a matching request ID, `model_loaded: true`,
  seven finite values, and `bridge_orig` action statistics. Fixed test mode is rejected.
- Reuses `adapt_bridge_action()` from the existing single-action demo. The total
  translation-vector magnitude is limited to **0.01 m**; the rotation-vector
  magnitude to **0.05 radians**, preserving their directions.
- Converts the Bridge gripper convention (0 closed / 1 open) to the Panda
  controller convention (-1 closed / +1 open) using `2 * gripper - 1`.
- Uses the shared Panda grasp point and inverse kinematics, with 200 physics
  ticks (0.4 s) per action. Historical runs used hand origin and 1,000 ticks.
  IK must converge before physics advances for the returned action.
- Executes 1,000 physics steps, then checks finite positions/velocities and
  position tracking error at most **5 mm**. A failed tracking check is reported
  after movement; it is not a rollback. The initial 500 settling steps happen
  before the input capture, including runs that later reject a response.

This is the existing **provisional** Bridge world XYZ/Euler to Panda mapping,
not calibrated transfer from the training robot. Bounds and position tracking
checks do not guarantee collision avoidance or grasping. Rotation components
are commanded but this experiment's final tracking threshold checks position.

## Results on the Mac

Each run saves `outputs/remote_single_action_<timestamp>_<id>/`:

| File | Contents |
| --- | --- |
| `before.png` | Image actually sent to cloud inference |
| `prediction.json` | Raw model response, action statistics, server and request timings |
| `scene_state.npz` | State corresponding to the input image |
| `after.png` | Image after executing the bounded command |
| `after_state.npz` | Resulting simulated state |
| `execution.json` | Raw/bounded actions, actual displacement, pose, tracking checks, physics time, local execution/capture timings |
| `status.json` | Current phase, elapsed time, motion-started flag, and failure details or `Completed` |

The prediction file measures the HTTP request only. The execution report adds
initial capture, IK, **wall time** for simulation execution, simulated physics
seconds, the next image capture/state save, and total time to the next image.
Viewer time is excluded. The simulation executes as fast as it can; two seconds
of simulated physics are not necessarily two wall-clock seconds.

If the request/response is invalid, no returned action is executed and no after
state is generated. `status.json` distinguishes failure before motion from
failure after motion started. The simulation stays in memory during the request;
there is no recreation of a new scene between the before and after captures.

## Verification and next milestone

Local tests use real MuJoCo and a local HTTP server with a **fixture response**
replaying the previously observed cloud action. They do not load OpenVLA:

```bash
.venv/bin/python -m unittest test_remote_single_action -v
```

The movement test passed with approximately **0.079 mm** position tracking error,
bounded translation, changed before/after images, and the expected physics-time
advance. Wrong server mode, unknown action statistics, and out-of-range gripper
values were rejected before action execution. Existing inference contracts also
passed after sharing the HTTP request helper.

The real cloud-to-movement test subsequently passed with **0.082 mm** tracking
error. The next experiment repeats observe/predict/move with a live viewer using
`remote_loop.py`; see [REMOTE_LOOP.md](REMOTE_LOOP.md).
Keep the pod stopped while preparing code; stop/terminate it in RunPod after
paid tests. Closing terminals or the viewer alone does not stop billing.
