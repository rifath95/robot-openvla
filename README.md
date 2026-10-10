# robot-openvla

A Phase 1 robotics workspace for connecting a simulated Franka Panda arm to OpenVLA. Development currently runs on a MacBook using Python and MuJoCo 3.14.0, without ROS.

Last updated: 2026-10-10. Update this document as milestones, run instructions, and model locations change.

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
| `remote_loop.py` | Repeats cloud predictions and bounded local movements with a live viewer, Space pause/resume, a default 100-action budget, per-step artifacts, and a runtime limit excluding pauses, and half/quarter-size IK retries. |
| `test_ik_retries.py` / `tests/fixtures/ik_step96.json` | Saved real step-96 regression: half-size recovery, quarter fallback, and unchanged targets after all attempts fail. |
| `test_remote_loop.py` | Real MuJoCo and HTTP fixture checks for repeated observations, stop/runtime limits, and failure handling. |
| `remote_single_action.py` | Requests one cloud prediction from a preserved local scene, executes a bounded command, saves before/after images and movement timings, and optionally shows the final scene. |
| `REMOTE_SINGLE_ACTION.md` | Cloud-to-movement run instructions, output files, limits, and verification status. |
| `test_remote_single_action.py` | Real MuJoCo movement and rejection checks against local HTTP fixture responses. |
| `test_remote_inference.py` | Contract tests using substituted model objects; does not run the real 7B checkpoint. |
| `robot_controls.py` | Earlier browser-based control panel; the current keyboard demo does not use it. |
| `mujoco_menagerie/` | Cloned upstream robot models and mesh assets. |
| `.venv/` | Simulation Python environment. |
| `.venv-openvla/` | Separate Python environment containing PyTorch, Transformers, and OpenVLA inference dependencies. |
| `outputs/` | Generated images, action JSON, status, and logs. |

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

This is a provisional plumbing test using `bridge_orig` statistics and an identity mapping of world XYZ/Euler axes, not a calibrated WidowX-to-Panda transfer. It limits the translation vector to 1 cm and the rotation vector to 0.05 radians, preserving their directions. The gripper changes from model 0 closed / 1 open to controller -1 closed / +1 open. The control point is the Panda hand origin. A successful movement does not demonstrate model-driven grasping or task success.

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

**Current status:** simulation, keyboard end-effector control, scripted pick-and-place, camera capture, a fresh-image OpenVLA single-action movement, and the live three-action feedback loop work on the Mac. The three local predictions took approximately 462, 601, and 554 seconds, with position tracking error below 0.1 mm. Real remote CUDA inference has now passed on an RTX A6000; the separate cloud-hosted simulator/three-action benchmark remains untested. No custom-domain fine-tuning has been performed, and model-driven pick-and-place has not been demonstrated.

**Phase 2 status:** basic SSH and GPU computation passed on a RunPod A40. The architecture keeps MuJoCo on the Mac and serves OpenVLA predictions from the cloud. Real remote inference then passed on a RunPod RTX A6000: model loading took 202.13 seconds, image decoding 0.031 seconds, processor/device transfer 0.072 seconds, complete prediction 0.702 seconds, and request round trip 1.689 seconds. It returned seven finite values without moving the robot. The real 100-step trial on an A40 completed **95 actions** with visible local motion and fresh camera feedback, then stopped when IK rejected action 96. Average prediction took **0.346 s**, round trip **1.171 s**, and cycle **3.541 s**. Docker dependencies and existing Global-volume weights were reused; model loading took **310.27 s**. The saved action-96 case now succeeds at half scale with **0.056 mm** tracking error using the new local retry implementation. No GPU was rented for the regression test.

The local connection test passed with a fresh 640x480 camera image and approximately 18 ms request round trip. Invalid requests, an unavailable server, and automatic server exit were also checked. Artifacts are in `outputs/connection_test_verified/`. The subsequent Mac-to-RunPod fixed-action test passed with approximately 1.58 seconds round trip. Real remote inference uses `connection_server.py --mode openvla`, with the model resident between requests and the Mac client explicitly expecting that mode. Model files are exported to the Global volume while caches and dependencies stay on the container disk. See [REMOTE_INFERENCE.md](REMOTE_INFERENCE.md) and [RUNPOD_CONNECTION_GUIDE.md](RUNPOD_CONNECTION_GUIDE.md).

The new `remote_single_action.py` connects a cloud prediction to bounded local robot movement and an after image. Local real-MuJoCo tests passed using an HTTP fixture replaying the observed cloud action, with approximately 0.079 mm tracking error; invalid mode/statistics/gripper responses were rejected before execution. The real Mac-to-cloud-to-MuJoCo single-action test subsequently passed with **0.082 mm** tracking error. `remote_loop.py` now repeats this pipeline with a live viewer; local tests passed for updated observations across three actions, stop/runtime limits, and failure handling. See [REMOTE_LOOP.md](REMOTE_LOOP.md). See [REMOTE_SINGLE_ACTION.md](REMOTE_SINGLE_ACTION.md).

**Next intended milestone:** run another 100-action real-cloud trial with the tested half/quarter-size IK retries, inspect recovery counts and timing/images, and assess the provisional action mapping. The retry logic is a Mac controller change and does not require rebuilding the dependency Docker image. Before task evaluation, validate the coordinate, rotation, control-point, and normalization conventions for our Panda domain and plan demonstration collection/fine-tuning. [CLOUD_SETUP.md](CLOUD_SETUP.md) remains a separate guide for running both simulation and inference on a cloud machine.

For future training, a single **A100 80GB** is a documented starting point for OpenVLA LoRA fine-tuning, which adapts a small set of parameters rather than retraining the entire model. Custom-domain adaptation will also require collecting suitable demonstrations, preparing a dataset, and measuring success on unseen trials. Running both simulation and inference on cloud compute is available via `.venv/bin/python openvla_loop.py --device cuda`; the Mac default remains `--device mps`. Remote model inference has passed on an RTX A6000 using the separate server/client path above. Training is not implemented yet.

Upstream references: [OpenVLA](https://github.com/openvla/openvla), [checkpoint](https://huggingface.co/openvla/openvla-7b), [LoRA fine-tuning](https://github.com/openvla/openvla#fine-tuning-openvla-via-lora), [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
