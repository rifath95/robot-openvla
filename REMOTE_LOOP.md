# Repeated cloud control with a live local simulator

`remote_loop.py` runs on the **Mac**. The existing cloud server keeps OpenVLA
loaded and needs no code changes for this step. No new clone or dependency
installation in the pod is needed if that server is still running.

The real single-action cloud test succeeded with **0.082 mm** position tracking
error. This repeated loop has passed local tests with real MuJoCo and HTTP
fixture predictions; its next check is a run against the real cloud model.
Tracking error measures execution of a requested movement, not task success.

## Run on the Mac

Keep the cloud server and the Mac SSH tunnel running as described in
[RUNPOD_CONNECTION_GUIDE.md](RUNPOD_CONNECTION_GUIDE.md). In another **local Mac
terminal**, in this repository, check readiness:

```bash
curl --fail http://127.0.0.1:8000/health
```

The response must identify `openvla` mode and a loaded model. If it cannot
connect, check the tunnel and server terminal first. The server's runtime limit
may have expired; restarting it reloads the model. Do not reinstall dependencies
or download weights again in the same running pod.

Start a small experiment:

```bash
.venv/bin/python remote_loop.py --instruction "pick up the red cube" --steps 100 --max-runtime-seconds 900
```

The simulator opens **paused**. Click inside its window and press **Space** to
start. Space also pauses/resumes; **Esc** or closing the window stops the loop.
Each movement is paced to approximately two seconds of simulation time so you
can see it. The arm stays still while waiting for the next prediction.

After 100 completed actions, the viewer closes and the script exits. The local
900-second limit counts active work, including predictions and motion, but excludes
time paused. These are also the defaults when no step/time options are supplied.
Pause/resume preserves the current scene and step count: after 15 completed actions,
the next action after resuming is action 16. Pausing during a movement resumes the
remaining portion of that same action before advancing to the next one.
Use `--start-immediately` to skip the initial pause, or `--no-view` to run without
the viewer. Headless execution advances physics without real-time pacing.

## What repeats

1. Capture the current scene and send the image plus instruction to the cloud.
2. Wait for OpenVLA to return seven numbers, keeping the same simulation state.
3. Validate the response, adapt the command, and solve inverse kinematics.
4. Execute the bounded motion, capture the resulting scene, and repeat from it.

There is one outstanding request at a time. Translation is bounded to **1 cm**
per action and rotation to **0.05 radians**. Invalid responses, failed IK,
nonfinite state, or tracking error above **5 mm** stop the run. Predictions
received after a stop are discarded. Pausing during a request holds its result
until resume; pausing during movement stops further simulation steps until resume.
Stopping midway through movement saves the partial execution separately and
does not count it as a completed action.

The Bridge-to-Panda coordinate, Euler rotation, gripper, and control-point
mapping remains provisional. The pretrained model may move without successfully
picking up the cube. Task completion and fine-tuning are later milestones.

## Results and timings

The terminal prints `outputs/remote_loop_<timestamp>_<id>/`:

- `status.json`: current phase, step counts, elapsed time, and any failure.
- `summary.json`: completed steps and final stop reason/status.
- `run.log`: phase transitions with elapsed times.
- `step_001/`, `step_002/`, etc.: `before.png`, `after.png`, simulation states,
  `prediction.json`, and `execution.json`.

Each step records client encoding, connection, upload send, response-header wait,
response-body read, and round-trip timings. Server timings separate image
decoding/preprocessing, processor/GPU transfer, and model prediction. Execution
timings include IK, robot movement, and preparation of the next image. Network
measurements include buffering/overlap; they are not exact one-way transit times.
The response-header wait includes server work and network travel.

**The local limit and stop controls do not stop the cloud server, terminate the
pod, or stop billing.** An already submitted cloud prediction may finish after
the local run stops. Stop or terminate the pod in RunPod when finished testing.
The independent Global volume retains the exported model weights. Pausing locally
does not pause GPU billing or the cloud server's own runtime limit. If the server
expires during a long pause, a subsequent request will fail; allow sufficient
server runtime for the experiment. An in-flight HTTP request also retains its
network timeout while the local viewer is paused.

## Local verification

```bash
.venv/bin/python -m unittest test_remote_loop test_remote_single_action -v
```

Tests use real MuJoCo with fixture responses, without loading model weights.
They check repeated updated observations, bounded motion, step limits, stopping
while waiting, runtime limits, and preservation of completed steps after failure.
