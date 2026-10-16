# Interactive bowl checkpoint testing

Use a separate A40 pod with the training image and the same network volume.
60 GB container disk is sufficient when at least 22 GiB is actually free. There
is no dataset download or global-volume requirement for manual testing.
The running automated sweep uses its own pod and checkout.

## First fetch the training reports

From the Mac repository folder, substitute the new pod's SSH command:

```bash
caffeinate -i .venv/bin/python -m cloud.interactive_bowl_session \
  --ssh 'ssh root@HOST -p PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --reports-only
```

This finds the exact 600-demonstration colored bowl training session from its
dataset provenance, downloads reports to `outputs/training/phase3_training/`,
and leaves the pod running. It does not stage weights or load a GPU model.
Use `--training-session panda-train-color-bowl-...` if multiple matching runs exist.

## Then open the simulator controls

Open **`playground.py` in the repository root** in your Mac editor and press
**Run Python File / Play** (in the terminal). Paste the full pod SSH command
at the prompt and press Enter. It uses `.venv/bin/python` automatically, prepares
the existing cloud host and opens the browser controls. On macOS it also keeps
the Mac awake while the launcher runs; keep the lid open and the Mac connected.
Keep its terminal open while testing. Closing it leaves the cloud model and pod
running. The pasted generic key path uses the existing personal-key fallback.

The equivalent command is `.venv/bin/python playground.py`. You can optionally
pass `--training-session panda-train-color-bowl-...` to select a particular run.
The existing command remains available:

```bash
caffeinate -i .venv/bin/python -m cloud.interactive_bowl_session \
  --ssh 'ssh root@HOST -p PORT -i ~/.ssh/id_ed25519_maxalderone'
```

An ordinary local browser window shows the live MuJoCo camera and clickable
controls together. MuJoCo and physics run on the Mac; prediction runs on the GPU.
It does not require a separate native MuJoCo viewer window. The terminal also
prints the localhost URL so the page can be reopened.

- Checkpoint buttons: **0, 1,000, 2,000, 4,000, 6,000**. Checkpoint zero is the
  freshly initialized adapter baseline, with this dataset's action normalization.
- Choose **1, 2, 3 or 4 cubes** and distinct colors. Red/blue/green/yellow were
  trained; black/purple/orange/white/pink/brown/cyan were not.
- Choose the target color; the instruction box initially fills the corresponding
  bowl-placement instruction. Edit it freely: the exact text is sent to OpenVLA.
  Outcome scoring follows the target selector, even if the text asks otherwise.
- **Apply / reset same positions** applies scene/color/target selections, restoring
  the same seed's initial arm, cubes and bowl. **New positions** chooses a new seed.
  Count/color selections must be applied before running the new scene.
- **Play** runs up to the editable **Action limit** (1–10,000; default 120) predictions from the current pose, stopping on placement,
  tracking failure, three rejected actions or your stop request.
- **Act once** predicts and executes one action, retaining the resulting pose.
- **Stop** cancels the trial, discards late predictions, and holds the current pose.
  Play/Act can continue; Reset starts over. Changing checkpoint resets the same
  layout so the next model starts from its initial pose.
- Optional sampling uses temperature 1 and top-p 1; greedy decoding is the default.

Instruction and controls are HTML outside the unmodified 640×480 model image.
Only the existing MuJoCo camera observation is sent to the prediction endpoint.
The current grasp-point controller, IK retries, 1 cm / .05 radian bounds and 200
physics steps per action are reused. Simulation stays frozen during inference.

## Model staging and lifetime

On first startup, the base and all five checkpoints are copied and checked on
**container disk**, under `/root/panda-interactive-bowl`. GPU loads read these
copies. Adapters merge into the base in VRAM using the existing inference backend;
no merged model is written to the network volume. Switching checkpoint first
releases the old model and loads the new one. All trials with the same checkpoint
reuse its loaded model. Weight staging is reused when reopening local controls
on the same container. Stop/redeployment can erase container disk.

Closing the browser requests a local trial stop. Ctrl+C ends local controls and
the tunnel; both leave the cloud model resident. **This manual workflow does not
auto-stop the pod. Stop it in RunPod when finished.** A model server has a 12-hour
readiness runtime; selecting the checkpoint again reloads an expired server.
Keep the Mac awake and connected while actively controlling it.

Cloud host log: `/root/panda-interactive-bowl/host.log`.
Per-checkpoint logs: `/root/panda-interactive-bowl/checkpoint_NNNNNN.log`.
Local recordings: `outputs/simulator_runs/interactive/bowl_DATE_TIME/trial_.../`.
Each contains the typed instruction, checkpoint provenance, before/after images,
actions, physics states, motion frames and a `RUN_REPORT.md` with direct Replay
and Action Replay buttons through the shared local replay launcher.

Local checks cover all four scene counts rendering and initializing, real
controller execution, instruction preservation, cancelled-response rejection,
same-model reuse and checkpoint identity checks. A real GPU switch and the full
Mac-to-cloud path still need their first check on the new pod.

## Optional automatic recovery

In the instruction panel, set the action limit (for example 1,000), choose a
**Recovery method**, then press Play:

- **Off**: no automatic intervention.
- **Switch to sampling**: temporarily sample 5–10 predictions (default 8), then
  return to greedy prediction.
- **Move upward (up to 10 cm)**: use the normal controller to move vertically
  upward, then return to greedy prediction. It never enables sampling.
- **Manual — use Move up button**: automatic detection is off; you decide when
  to intervene with the **↑ Move up 10 cm** button.

The manual button is also available alongside either automatic recovery method
or Off. It pauses the current trial, discards any pending model response, and
records a separate manual upward-movement trial. It uses the same 10 cm move,
55 cm world-Z ceiling, controller bounds, fixed horizontal target and gripper
setting as automatic upward recovery. It does not query or require a loaded
model. After the move the arm holds; press Play or Act once to resume with the
settings currently selected. Stop cancels the movement. Repeated clicks during
an upward recovery are rejected. A background prediction already sent to the
GPU may finish, but its discarded response cannot move the arm.

Manual trials record `manual_nudge=true` in their episode and summary, a
`manual_nudge_started` event, and the usual replay/action-replay recordings.

Settings are captured at trial start. Manual **Sample action tokens throughout**
remains separate and disables automatic recovery. Act once executes only one
model prediction; recovery history resets with each Play/Act trial, scene or
checkpoint. Automated sweeps retain their existing behavior.

Both recovery methods use **robot feedback only**: seven arm joints, the
forward-kinematic grasp-point position and actual finger opening. Cube positions,
contact information and grasp detection are not inputs to recovery. Simulator
object state is still used separately for outcome scoring and recordings.
Repeated model outputs alone do not trigger recovery. The default stall window
is 8 action attempts (selectable 4–50), with tolerances of 0.015 radians per
joint, 3 mm per hand coordinate and 2 mm for gripper opening. Cycle detection
checks periods 2, 3 and 4 with three repeated returns and a real excursion.
These are heuristics; an intentional stationary hold can also trigger recovery.

The calibrated starting grasp point is at world Z=0.45 m and the tabletop is at
Z=0.25 m: 20 cm clearance. Upward recovery requests **half that clearance, 10 cm**,
from the current hand position, capped at world Z=0.55 m (30 cm above the table).
The target X/Y and commanded gripper setting stay fixed; rotational increments
are zero. The normal 1 cm translation bound, IK retries and physics execute the
move over several recorded actions, without teleporting or predicting actions
from the model. Actual poses can have normal controller tracking error. A move
has at most 20 action attempts; inability to reach it or already being at the
height ceiling stops the trial. This prevents repeated nudges climbing forever.

After either recovery, greedy assessment lasts at least 12 predictions (or the
longer chosen stall window). If stalling/cycling persists, the selected method
tries again up to **Maximum recovery attempts** (default 3). The count is bounded
across the whole trial. Successful assessment means no detected stall/cycle,
not task success. The detector history resets after a completed nudge.

Placement, manual Stop, action limit, nonfinite state, tracking failure and the
existing three-rejection stop retain priority. The action limit includes model
predictions and individual nudge increments. The live status distinguishes
`nudge`, `sampling`, `assessing`, `greedy`, `disabled` and `exhausted` modes.
Camera input remains the plain image without HTML controls.

Each trial records:

- `episode.json`: chosen action limit, manual sampling, recovery settings and nudge calibration.
- `step_NNN/generation_request.json`: requested decoding mode and its source; nudges explicitly mark `model_queried=false`.
- `step_NNN/execution.json`: physical state after execution when recovery is
  enabled, plus whether that prediction was greedy/manual/recovery sampling.
- `recovery_events.jsonl`: detections, sampling/nudge starts/ends, progress assessments
  and exhausted retries, including step, simulation time and physical state.
- `summary.json` / `RUN_REPORT.md`: recovery events, counts (including nudge actions), limit and outcome,
  with the usual direct Replay and Action Replay links.

Local tests exercise stall/cycle detection, continued movement/lifting, retry
limits, decoding switches, invalid inputs and real MuJoCo control with a mocked
prediction endpoint. A real-model comparison is still needed to measure whether
recovery improves success. These controls apply only to interactive playground
trials; automated sweep policies retain their configured decoding behavior.

`nudge_attempt_N.json` records each upward target, held gripper command and bounds.
Recovery moves are included in the usual Replay and Action Replay recordings.
