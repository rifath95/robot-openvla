# Repeated cloud control with a live local simulator

New execution uses `panda_grasp_v1`: the grasp point 0.103 m along hand-local Z,
200 physics ticks (0.4 s) per action, and a saved `action_contract.json`. Earlier
trials used hand origin and 1,000 ticks. Pretrained predictions still use
`bridge_orig`; Panda statistics belong to a future adapted model.
See [PANDA_DEMONSTRATIONS.md](PANDA_DEMONSTRATIONS.md).

`remote_loop.py` runs on the **Mac**. The existing cloud server keeps OpenVLA
loaded and needs no code changes for this step. No new clone or dependency
installation in the pod is needed if that server is still running.

The real single-action cloud test succeeded with **0.082 mm** position tracking
error. This repeated loop has passed local tests with real MuJoCo and HTTP
fixture predictions; a real-cloud trial on an A40 subsequently completed 95 actions before IK rejected
action 96. The next cloud trial completed 96 actions and rejected action 97.
Both saved failures now recover locally, and exhausted retries hold the pose
and stop cleanly after repeated rejections. See [CONTROL_FINDINGS.md](CONTROL_FINDINGS.md).
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

After 100 observation/prediction cycles, the viewer closes and the script exits.
Rejected cycles count toward this budget but not toward completed movements. The local
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
per action and rotation to **0.05 radians**. If IK rejects the bounded command,
the controller retries **half**, **quarter**, then **one-eighth** of its translation and rotation.
The gripper command remains unchanged. No new cloud request is made for retries;
all attempts start from the same physical pose without advancing simulation.
Rejected trials restore actuator targets, including the gripper. If none solves,
the loop records a rejection, holds the current pose without advancing physics,
and requests another prediction from a newly captured observation. After **three
consecutive rejections**, it stops cleanly; successful execution resets the counter.
Set `--max-consecutive-rejections` to change that bound. An unchanged scene may
produce the same prediction repeatedly. Invalid responses, nonfinite state, or
tracking error above **5 mm** still stop the run as errors. Predictions
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
- `summary.json`: successful movements, rejected cycles, prediction count, and final stop reason/status.
- `run.log`: phase transitions with elapsed times.
- `step_001/`, `step_002/`, etc.: `before.png`, `after.png`, simulation states,
  `prediction.json`, `ik_attempts.json`, and `execution.json`.

`ik_attempts.json` records attempted scales, solver errors, convergence, and
timings even when every attempt fails. `execution.json` distinguishes the raw
model action, original bounded action, actual executed action, and accepted scale.
A successful size retry is still one prediction cycle. A completely rejected
action has `executed: false`, no executed controller action, and zero movement
time. It is stored separately from completed movements in the summary. Pose
diagnostics include joint-limit margins, hand/cube distance, and Jacobian conditioning.

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
.venv/bin/python -m unittest test_ik_retries test_remote_loop test_cloud_session -v
```

Tests use real MuJoCo with fixture responses, without loading model weights.
They check repeated updated observations, bounded motion, step limits, stopping
while waiting, runtime limits, and preservation of completed steps after failure.

The saved real step-96 regression selects half scale and executes with **0.056 mm**
position tracking error. Tests also cover quarter-scale selection and restoring
all actuator/pose targets when every size fails. The step-97 regression selects
one-eighth size and tracks with **0.059 mm** error. Recovery tests verify holding
and requesting new predictions, and clean stopping after repeated rejection. The new
retry path has been checked locally; a fresh cloud trial is still pending.

## Replay a saved run without a pod

Open the run folder’s `replay.py` using `.venv/bin/python`, or run
`.venv/bin/python replay_run.py outputs/<RUN_FOLDER>`. Space starts/pauses/resumes.
New runs include sampled motion frames; older runs interpolate saved endpoints.
See [REPLAY.md](REPLAY.md) for speed controls and playback limitations.

## Saved startup measurements

The one-command `cloud_session.py` launcher saves `startup_info.json` in an
`outputs/cloud_session_<timestamp>_<id>/` folder when the server is ready and
copies it into each `remote_loop_*` trial folder it launches. Subsequent trials
reuse the session measurements; they do not reload the model.

The file records model loading seconds (processor setup, checkpoint reading,
model construction and synchronized GPU transfer), checkpoint preparation seconds
(validation or download/export), whether a completed export existed before setup,
mount source/filesystem, GPU name, total/free/used VRAM after loading, PyTorch
allocated/reserved/peak loading memory, device/dtype, and software versions.
Memory figures are bytes; divide by 1024**3 for GiB. Visible CPU/RAM figures may
reflect the host rather than the pod's allocated limits. GPU-wide usage can
include allocations outside PyTorch. These are startup measurements, not peak
inference memory or isolated volume-to-VRAM transfer timings. A marker present
before setup does not guarantee the export is valid: preparation validates it.

The mount information distinguishes filesystems and volume sources; consult the
RunPod console for the actual volume name/type. Loading comparisons should use
the same GPU/model settings and note whether preparation just downloaded files.
Direct `remote_loop.py` runs can supply `--startup-info <PATH>`; without it no
cloud startup measurements are available. This update is pulled as source by the
launcher after pushing it to GitHub; no dependency Docker rebuild is needed.

## Read the run summary

Each finalized loop creates **`RUN_REPORT.md`** in its output folder, including
completed, stopped, and failed trials. Open it for startup/loading, GPU/VRAM and
storage information, then the full-cycle overview before remote-request and local
simulation timing breakdowns. The remote request includes server processing;
its remaining overhead is not an isolated upload/download measurement. Finalized
cycle averages include recorded rejected holds and interrupted movements; cycles
without timing records are excluded. Missing measurements say “Not recorded.”

To regenerate a report from existing saved files:

```bash
.venv/bin/python run_report.py outputs/<RUN_FOLDER>
```

No cloud connection or model loading is needed. Existing local remote-loop folders
have been backfilled. Reports require the recorded startup metadata to identify
GPU/storage/loading; historical missing data is not guessed. As usual, push source
updates before launching the next session; no Docker dependency rebuild is needed.
