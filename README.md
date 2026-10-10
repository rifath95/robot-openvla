# robot-openvla

A robotics workspace for connecting a simulated Franka Panda arm to OpenVLA. Development currently runs on a MacBook using Python and MuJoCo 3.14.0, without ROS.

Last updated: 2026-10-10. Update this document as milestones, run instructions, and model locations change.

**Phases 1 and 2 are complete. Phase 3 is the current focus:** collect and
validate varied pick-and-place demonstrations, fine-tune OpenVLA for our Panda
setup, and evaluate task success on unseen scenes. Phase 4 will explore additional
tasks and transfer beyond the fine-tuning demonstrations.

The Phase 3 foundation already includes the shared **`panda_grasp_v1`** action
convention, expert recorder, sequential replay, and Panda dataset statistics.
One 72-action scripted contact pick/place passed recording/replay (13.7 cm lift,
2.0 mm placement error). This verifies data collection; OpenVLA has not yet been
fine-tuned or demonstrated successful pick-and-place in this setup.
See [PANDA_DEMONSTRATIONS.md](PANDA_DEMONSTRATIONS.md).

## Goal

Build this repeated control loop:

```text
simulated camera image + task instruction
    → OpenVLA
    → seven action values
    → robot controller
    → simulated arm moves
    → new camera image
    → repeat
```

The seven values are `[dx, dy, dz, droll, dpitch, dyaw, gripper]`: three position changes, three orientation changes, and a gripper command. The local controller uses world-frame position changes in metres and rotation changes in radians; its gripper convention is -1 closed and +1 open. Model output conventions and dataset statistics must be matched to this interface before deployment.

## Project phases

| Phase | Goal | Status |
| --- | --- | --- |
| 1 | Establish local simulation, seven-value control, and OpenVLA inference. | **Completed** |
| 2 | Connect local simulation to cloud OpenVLA and demonstrate a live repeated control loop. | **Completed** |
| 3 | Adapt OpenVLA to our Panda setup for pick-and-place using validated demonstrations and held-out evaluation. | **Current — data pipeline foundation implemented; collection and training remain** |
| 4 | Expand to other tasks and measure transfer to tasks excluded from fine-tuning. | **Planned** |

### Phase 1 — completed: local foundation

- Set up the Panda arm, table, red cube, and camera in MuJoCo on the Mac.
- Verified keyboard position/rotation/gripper control and inverse kinematics.
- Demonstrated scripted pick-and-place with contact physics.
- Downloaded OpenVLA, supplied an image and instruction, and obtained seven action values.
- Executed model-predicted movements and a short feedback loop locally.

Local inference took minutes per prediction, motivating the cloud GPU pipeline.
Scripted task success and model-predicted movement were separate demonstrations.

### Phase 2 — completed: local robot, cloud model

- Kept the simulator, camera capture, robot controller, and live viewer on the Mac.
- Ran resident OpenVLA on RunPod GPUs, accepting image/instruction requests and returning seven action values through an SSH tunnel.
- Demonstrated repeated fresh-image prediction and visible arm movement. Long trials completed 95 and 96 movements before IK rejection; a later short trial completed 26 movements and stopped cleanly on viewer close.
- Added Space start/pause/resume, viewer-close/stop handling, action/runtime limits, and per-step images, states, actions, status, and timings.
- Added bounded action retries, hold/re-predict behavior, and clean stopping after consecutive rejections; saved failure cases and recovery behavior were tested locally.
- Simplified startup with a Docker dependency image, reusable Global-volume weights, and the one-command cloud session launcher.
- Checked local axes, rotations, gripper behavior, and camera processing; defined the shared Panda grasp-point action convention for future data collection.

A short cloud trial measured mean prediction time of **0.34 s** and full cycle
of **3.40 s**. These are historical measurements using the earlier 2 s movement
interval; current commands use a 0.4 s interval. Model loading is a separate cost.
See [CONTROL_FINDINGS.md](CONTROL_FINDINGS.md) and
[CONTROL_CALIBRATION.md](CONTROL_CALIBRATION.md) for evidence and limitations.

**Completion means the remote control pipeline works.** Model-driven pickup has
not been demonstrated. The latest grasp-point/interval changes and exhausted-retry
recovery still need verification together in a long real cloud trial. The Panda
convention defines our interface; it does not establish pretrained Bridge-to-Panda
policy calibration. Session/server limits do not terminate the rented pod or stop billing.

### Phase 3 — current: pick-and-place adaptation

**Goal:** fine-tune the pretrained OpenVLA model to perform pick-and-place reliably
in our Panda environment, and measure performance on scenes excluded from training.

Already implemented:

- A versioned action contract covering world-frame metres, rotation order, grasp point, gripper values, bounds, and a fixed 0.4 s execution interval.
- An expert recorder that pairs each observation/instruction with the accepted bounded command actually executed, plus measured motion and simulator state.
- Sequential replay verification, dataset eligibility checks, and Panda-specific action statistics.
- One successful 72-transition scripted episode and a verified staging dataset; this is a pipeline check, not sufficient training coverage.

Phase 3 collection has started with a reproducible ten-scene pilot: six training,
two validation and two test scenes, assigned before recording. This checks spatial
variation and split handling; all ten expert episodes passed recording/replay
(425 training / 142 validation / 145 test transitions). It is not yet a sufficient
fine-tuning dataset or a measurement of OpenVLA task success.

Remaining goals:

1. Collect varied successful pick-and-place demonstrations with different object starting positions, destinations, approaches, orientations, and instructions.
2. Reserve separate validation/test scenes and demonstrations before training; exclude rejected, unsuccessful, or unverified episodes from expert training data.
3. Finalize normalization, package the dataset for OpenVLA training (including RLDS integration), and verify image/action alignment.
4. Run a small, budgeted LoRA fine-tuning experiment, then expand only after measuring runtime, memory, and cost.
5. Deploy the adapted checkpoint through the existing cloud pipeline and compare it with the pretrained baseline on held-out pick-and-place trials.

Record success rate, grasp/lift/placement outcomes, IK rejections, latency, and
cloud cost. Phase 3 is complete when the dataset and training path are reproducible
and held-out task performance is measured against an agreed success target.
No custom-domain fine-tuning has run yet. See
[PANDA_DEMONSTRATIONS.md](PANDA_DEMONSTRATIONS.md) for the current data workflow.

### Phase 4 — planned: broader tasks and transfer

Expand beyond pick-and-place, for example to pushing, stacking, or opening drawers.
First measure which pretrained skills transfer after adaptation without new task
demonstrations. Add mixed-task demonstrations where needed and evaluate both new
skills and retention of pick-and-place performance. This does not require assuming
a separate model or fine-tuning run for every task.

## What works so far

- Loaded the Franka Panda model from Google DeepMind's MuJoCo Menagerie.
- Created a scene with a table, a movable red cube, and a fixed camera.
- Captured camera images from the simulation.
- Implemented end-effector control using inverse kinematics: requested hand movements are converted into joint targets. This controller requires no model training.
- Added keyboard controls for position, rotation, and the gripper.
- Successfully ran scripted pick-and-place using simulated contact physics.
- Downloaded the pretrained `openvla/openvla-7b` checkpoint and ran inference on a camera image with the instruction “pick up the red cube.”
- Fixed an inference input mismatch by supplying a required prompt token together with its attention-mask entry.
- Confirmed that inference produced seven finite action values on the Mac's Apple GPU.
- Completed a fresh-image single-action test: the predicted command moved the simulated arm and produced a new camera image, with position tracking error below 0.1 mm. That run took approximately 579 seconds for inference.

The successful inference took approximately **516 seconds (8 minutes 36 seconds)**, excluding model loading. This proves local inference works, but its measured speed is too slow for practical interactive control.

## Important files

| File | Purpose |
| --- | --- |
| `scene.xml` | Panda scene, floor, table, cube, camera, and `scene_home` starting state. |
| `panda_actions.py` | Shared Panda action bounds, rotation rule, grasp point, gripper conversion and fixed interval. |
| `collect_panda_pilot.py` | Records a predeclared ten-scene pick-and-place pilot with 6 training / 2 validation / 2 test episodes; only training episodes enter staging normalization. |
| `panda_demonstrations.py` | Records expert image/action transitions and verifies sequential contact-physics replay. |
| `prepare_panda_dataset.py` | Validates replayed expert episodes and exports JSONL plus Panda statistics. |
| `test_panda_demonstrations.py` / `PANDA_DEMONSTRATIONS.md` | Recording/replay, dataset eligibility tests and usage guide. |
| `launch_sim.py` | Opens the scene in the standard MuJoCo viewer with actuator controls. |
| `end_effector_demo.py` | Opens a simulator with keyboard control of the seven action components. |
| `panda_controller.py` | Implements the action interface, inverse kinematics, joint targets, and physics stepping. |
| `panda_env.py` | Reusable simulation wrapper with reset, camera observation, and action/step methods. |
| `move_joint_demo.py` | Demonstrates a small movement of one joint. |
| `pick_place_demo.py` | Scripted cube pick-and-place; the movements are not predicted by OpenVLA. |
| `capture_camera.py` | Saves the fixed-camera view to `camera_image.png`; also provides a PNG writer. |
| `try_openvla_mac.py` | Runs the GPU check, checkpoint download/cache check, and one action prediction; updates status and logs. |
| `openvla_mac.py` | Local inference entry point: loads the processor/model, processes image and instruction, calls `predict_action()`, and saves the result. |
| `openvla_single_action.py` | Captures a fresh scene, predicts one action, executes a bounded command, and saves before/after images and a movement report. |
| `openvla_loop.py` | Runs three successive observation/action cycles while loading OpenVLA once; saves each image, prediction, movement, and timing. |
| `requirements-openvla-mac.txt` | Dependencies for the isolated OpenVLA environment. |
| `requirements-openvla.txt` | Shared pinned inference dependencies for Mac and Linux. |
| `requirements-sim-linux.txt` | Linux/Python 3.11 simulation dependencies for cloud runs. |
| `CLOUD_SETUP.md` | Linux NVIDIA GPU setup, headless rendering checks, and the CUDA loop benchmark. |
| `cloud_startup_metrics.py` | Records checkpoint preparation, mount and host information for session startup reports. |
| `cloud_session.py` | One Mac command to prepare a GPU host, open the SSH tunnel, wait for model readiness, and launch repeated local trials. |
| `scripts/cloud_start.sh` / `scripts/cloud_stop.sh` | Linux setup and recorded server lifecycle; reuse preinstalled Docker dependencies or install the tested environment. |
| `Dockerfile` / `.github/workflows/build-cloud-image.yml` | Dependency image and manual GitHub Container Registry build/publication workflow; weights remain on the external volume. |
| `CLOUD_QUICKSTART.md` | Simplified session instructions and one-time Docker publication/template setup. |
| `test_cloud_session.py` | SSH input, readiness, and lifecycle orchestration tests without a GPU or remote connection. |
| `connection_server.py` | HTTP server with fixed-action test mode and explicit resident OpenVLA inference mode. |
| `connection_client.py` | Captures a MuJoCo image, sends it with an instruction, and saves returned test/model actions and timings; never moves the arm. |
| `requirements-connection-server.txt` | Pillow dependency for the lightweight connection server. |
| `CONNECTION_TEST.md` | Local connection-test commands, later SSH-tunnel setup, output files, and timing interpretation. |
| `openvla_backend.py` | Loads OpenVLA once and predicts actions with processor and synchronized GPU inference timings. |
| `prepare_cloud_model.py` | Downloads the pinned model locally and exports regular files to Global storage for reuse. |
| `REMOTE_INFERENCE.md` | Cloud model setup and one real prediction through an SSH tunnel, without robot movement. |
| `RUNPOD_CONNECTION_GUIDE.md` | Copyable session checklist for pod setup, SSH login, three terminal roles, model reuse, prediction, and shutdown. |
| `run_report.py` | Automatically writes each trial’s RUN_REPORT.md with startup/GPU/storage details and full-cycle timing before detailed breakdowns; can regenerate reports for existing runs. |
| `replay_run.py` / `REPLAY.md` | Offline saved-run viewer with pause/resume and speed control; new run folders include `replay.py` and sampled motion frames. |
| `remote_loop.py` | Repeats cloud predictions and bounded local movements with a live viewer, Space pause/resume, a default 100-action budget, per-step artifacts, and a runtime limit excluding pauses, IK size retries, and bounded hold/re-predict recovery. |
| `test_ik_retries.py` / `tests/fixtures/ik_step*.json` | Saved real step-96/97 regressions: size recovery, target restoration, and offline solver analysis. |
| `control_diagnostics.py` / `analyze_remote_run.py` | Read-only pose metrics and offline trajectory/IK investigation without a GPU. |
| `CONTROL_FINDINGS.md` | Measured drift, solver findings, bounded recovery design, checks, and next calibration work. |
| `calibrate_controls.py` / `CONTROL_CALIBRATION.md` | Local known-command movement, rotation, gripper and camera checks with before/after images; comparison with upstream conventions. |
| `preview_model_input.py` | Saves the actual cached image processor's backbone inputs without loading weights or predicting actions. |
| `test_remote_loop.py` | Real MuJoCo and HTTP fixture checks for repeated observations, stop/runtime limits, and failure handling. |
| `remote_single_action.py` | Requests one cloud prediction from a preserved local scene, executes a bounded command, saves before/after images and movement timings, and optionally shows the final scene. |
| `REMOTE_SINGLE_ACTION.md` | Cloud-to-movement run instructions, output files, limits, and verification status. |
| `test_remote_single_action.py` | Real MuJoCo movement and rejection checks against local HTTP fixture responses. |
| `test_remote_inference.py` | Contract tests using substituted model objects; does not run the real 7B checkpoint. |
| `robot_controls.py` | Earlier browser-based control panel; the current keyboard demo does not use it. |
| `mujoco_menagerie/` | Cloned upstream robot models and mesh assets. |
| `.venv/` | Simulation Python environment. |
| `.venv-openvla/` | Separate Python environment containing PyTorch, Transformers, and OpenVLA inference dependencies. |
| `datasets/` / `datasets/README.md` | Verified expert trajectories and fine-tuning exports; each trajectory includes replay.py. Generated data is excluded from GitHub. |
| `outputs/` | Experiment images, model-driven action JSON, status, and logs. |

## What to run

For cloud sessions, use `.venv/bin/python cloud_session.py` on the Mac and paste
the current RunPod SSH command when prompted. It prepares the server/tunnel and
opens the paused 100-step simulator. See [CLOUD_QUICKSTART.md](CLOUD_QUICKSTART.md)
for the one-time Docker build and session shutdown. The Docker image was built and tested on an A40; set `PUBLIC_KEY` in the
custom template so its startup script enables SSH.

### Setup after cloning

This repository tracks source code and documentation. Python environments, downloaded OpenVLA weights, camera images, and generated outputs are excluded. MuJoCo Menagerie is recorded as a Git submodule pinned to the robot assets used here.

Clone with `git clone --recurse-submodules <REPOSITORY_URL>`, then enter the cloned directory. If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

For simulation, install Python 3.14 and create the local environment:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-sim.txt
.venv/bin/python launch_sim.py
```

For the current Mac OpenVLA test, install Python 3.11 separately:

```bash
python3.11 -m venv .venv-openvla
.venv-openvla/bin/python -m pip install -r requirements-openvla-mac.txt
.venv/bin/python capture_camera.py
.venv-openvla/bin/python try_openvla_mac.py
```

The first OpenVLA run needs internet access and downloads approximately 15 GB of weights from Hugging Face into `.cache-openvla/`. It then attempts inference using the Mac's MPS GPU. Subsequent runs reuse downloaded checkpoint files. Model loading and inference also require substantial memory. These setup commands describe the tested Mac path; use [CLOUD_SETUP.md](CLOUD_SETUP.md) for Linux/CUDA. The Mac inference wrapper does not move the robot.

Missing environments must be installed before running scripts, and missing submodule assets will prevent the Panda scene from loading. A fresh clone will not contain the original machine's result files; generate your own camera image and inference result with the commands above.

### Run individual demos

Open the desired script in VS Code and choose **Run Python File**. Select `.venv/bin/python` for simulation scripts. The OpenVLA wrapper `try_openvla_mac.py` selects `.venv-openvla/bin/python` itself.

Each viewer script launches its own window; **do not run `launch_sim.py` first** when using another demo.

From a terminal in this workspace:

```bash
.venv/bin/python launch_sim.py
.venv/bin/python end_effector_demo.py
.venv/bin/python pick_place_demo.py
.venv/bin/python capture_camera.py
.venv-openvla/bin/python try_openvla_mac.py
```

These are separate options, not a sequence that must all be run. To run inference directly from the already downloaded checkpoint:

```bash
.venv-openvla/bin/python openvla_mac.py --device mps
```

The direct command saves the action but does not update the wrapper's status file.

### One model-predicted movement

Run `openvla_single_action.py` in VS Code, or use:

```bash
.venv/bin/python openvla_single_action.py
```

It launches the simulation and inference stages in their respective environments. After fresh inference finishes, it restores the scene state that produced the input image, executes one bounded command, and opens the final scene in the standard viewer. No simulation advances during inference. Use `--no-view` to save results without opening the final viewer. You do not need to launch another simulator first.

Each run creates `outputs/single_action_<timestamp>/` containing `before.png`, `after.png`, `scene_state.npz`, `prediction.json`, `execution.json`, `status.json`, and `run.log`. Check `status.json` for progress or failures. `execution.json` records the raw action, converted command, measured movement, and tracking error.

This is a provisional plumbing test using `bridge_orig` statistics and an identity mapping of world XYZ/Euler axes, not a calibrated WidowX-to-Panda transfer. It limits the translation vector to 1 cm and the rotation vector to 0.05 radians, preserving their directions. The gripper changes from model 0 closed / 1 open to controller -1 closed / +1 open. New execution uses the shared Panda grasp point and 0.4 s interval; historical runs used hand origin and 2 s. A successful movement does not demonstrate model-driven grasping or task success.

### Three-action feedback loop

Run `openvla_loop.py` in VS Code, or:

```bash
.venv/bin/python openvla_loop.py
```

It loads OpenVLA once, predicts an action from the first image, advances the simulation, and feeds the resulting image into the next prediction. It stops after three actions. No final viewer is opened; inspect the saved images. Physics stays frozen during inference, and the same provisional mapping and per-action bounds apply. The simulated state is carried forward rather than reset between actions.

Results are saved in `outputs/openvla_loop_<timestamp>/`:

- `status.json`: phase, current step, completed steps, UTC update time, most recent inference time when available, and total elapsed time when finished.
- `summary.json`: completed movement reports and per-step/total inference times; updated after each completed action.
- `run.log`: model loading, prediction, execution, and failure details.
- `step_01/`, `step_02/`, `step_03/`: each step's `before.png`, `after.png`, `prediction.json`, `execution.json`, and simulation states. Step 2's input image is step 1's output image, and so on.

`status.json` becomes `Completed` after all requested actions pass execution checks, or `Failed` with the failing phase. `--steps 1` or `--steps 2` provides a shorter run; the initial experiment is limited to three actions. Keeping weights loaded avoids repeated loading, but does not guarantee faster prediction. Allow substantial time on the Mac.

### Keyboard controls

In `end_effector_demo.py`, click inside the simulator window to give it keyboard focus.

| Key | Move mode | Rotate mode |
| --- | --- | --- |
| Up / Down | +X / -X | +Roll / -Roll |
| Left / Right | +Y / -Y | +Pitch / -Pitch |
| Home / End | +Z / -Z | +Yaw / -Yaw |

- **Space:** switch Move/Rotate mode.
- **Enter:** toggle gripper open/closed.
- **Fn + Left / Fn + Right:** Home/End on a Mac keyboard.
- **Esc:** return to the free camera.
- **Close the window:** quit.

The camera can also be switched using `]`. The passive-viewer demos step physics in Python, so the viewer's standard Run/Pause buttons are disabled.

## Where inference code lives

Our entry point is **`openvla_mac.py`**. The neural-network implementation is official OpenVLA code downloaded from Hugging Face using `trust_remote_code=True`; the OpenVLA GitHub repository has not been cloned into this workspace.

For remote serving, **`openvla_backend.py`** loads the same official model implementation and processor. **`connection_server.py --mode openvla`** keeps them loaded between HTTP requests. Cloud weights can be stored as regular files in `/workspace/openvla-7b` using `prepare_cloud_model.py`; the original Mac cache paths below remain unchanged.

The pinned model snapshot is:

```text
.cache-openvla/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f/
```

Important files inside it:

- `modeling_prismatic.py`: OpenVLA network, `forward()`, and `predict_action()` implementation.
- `processing_prismatic.py`: image/text processor.
- `configuration_prismatic.py`: model configuration classes.
- `config.json`: architecture settings and action normalization statistics.

Transformers also copies the imported custom Python modules into `.cache-openvla/modules/transformers_modules/`. The underlying Llama implementation is installed at:

```text
.venv-openvla/lib/python3.11/site-packages/transformers/models/llama/modeling_llama.py
```

## Where model weights live

The downloaded weights total approximately **15 GB**. The snapshot above contains these three checkpoint links:

```text
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
```

The actual files are stored under `.cache-openvla/hub/models--openvla--openvla-7b/blobs/`; the snapshot links give them readable names. `model.safetensors.index.json` maps model parameters to their checkpoint shards.

`.cache-openvla/model_revision.json` records the downloaded model ID, revision, and absolute snapshot path. These paths describe the current Mac workspace; relocating the workspace requires updating that recorded absolute path.

## Inference results and logs

- `camera_image.png`: saved simulator image used by the local test.
- `outputs/openvla_mac_action.json`: seven predicted values, instruction, image path, device, normalization key, and inference time.
- `outputs/openvla_mac_status.json`: status of the managed test and paths to its log/result.
- `outputs/openvla_mac_test.log`: original test history, including the initial failed inference.
- `outputs/openvla_mac_retry.log`: successful inference after fixing the input mismatch.
- `outputs/pick_place_*/`: camera snapshots from captured scripted demonstrations.

The successful test used `bridge_orig` action statistics. These statistics are from an existing training domain and have not been validated for our Panda scene. Producing seven numbers does not establish that the pretrained model can successfully complete our task.

## Current status and next intended work

**Phases 1 and 2 are complete; Phase 3 is current.** The Mac/cloud observation,
prediction, and movement loop has been demonstrated with live viewing and saved
per-step results. Local controller recovery and the shared Panda action/data
contract have been tested. OpenVLA task success and custom-domain fine-tuning
remain to be established.

**Next:** collect varied, replay-verified pick-and-place demonstrations and reserve
held-out evaluation scenes, then prepare the training dataset and run an initial
budgeted fine-tuning experiment. The successful scripted episode is the data
pipeline's first check. See [Project phases](#project-phases) for completed work,
remaining goals, and the planned Phase 4 expansion.

Use [CLOUD_QUICKSTART.md](CLOUD_QUICKSTART.md) for existing remote sessions and
[PANDA_DEMONSTRATIONS.md](PANDA_DEMONSTRATIONS.md) for local recording/replay.
Running both simulation and inference in the cloud remains an optional separate
path described in [CLOUD_SETUP.md](CLOUD_SETUP.md), rather than a Phase 2 requirement.

Upstream references: [OpenVLA](https://github.com/openvla/openvla), [checkpoint](https://huggingface.co/openvla/openvla-7b), [LoRA fine-tuning](https://github.com/openvla/openvla#fine-tuning-openvla-via-lora), [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
