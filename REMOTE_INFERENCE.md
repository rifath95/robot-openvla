# One cloud OpenVLA prediction, displayed on the Mac

Prepared 2026-10-09. The image-only connection test passed both locally and from
Mac to RunPod (approximately 1.58 seconds cloud round trip). The real-model server
is implemented, but **has not yet been validated with OpenVLA on a cloud GPU**.
No robot movement or continuous control loop is performed by these commands.

## Files and architecture

- `connection_server.py --mode openvla`: starts a resident model server; it loads
  the model once before announcing readiness, then handles image/instruction
  requests with one prediction at a time. An overlapping request gets HTTP 503.
- `openvla_backend.py`: official HF model/processor loading, prompt preparation,
  token/mask alignment, and deterministic `predict_action()` calls with eager
  attention. CUDA prediction timing is synchronized with the GPU.
- `prepare_cloud_model.py`: downloads the **same pinned revision** used by the
  successful Mac test on the container disk, then exports regular files to the
  Global volume. It does not copy Mac environments or upload Mac weights.
- `connection_client.py --expected-mode openvla`: captures a fresh home-scene
  image on the Mac, sends it and the instruction, validates and displays seven
  finite predicted numbers, and saves them without executing any movement.

The fixed test mode remains the default, so an old connection test cannot silently
start model inference. The client must explicitly expect `openvla` mode.

## Prepare before launching a pod

Commit and push these changes so the cloud copy can clone/pull them. Keep the
MuJoCo client on the Mac. Only the model server runs on the pod, so it does not
need MuJoCo or the Menagerie submodule. A single 48 GB A40/A6000 is the initial
inference target; speed and actual memory usage must be measured.

## On the pod: isolated dependencies and CUDA check

Use a Python 3.11 template, clone the repository to `/root/robot-openvla`, or pull
the new changes in an existing copy. Run the following **on the pod**:

```bash
cd /root/robot-openvla
python3.11 -m venv .venv-openvla
.venv-openvla/bin/python -m pip install --upgrade pip
.venv-openvla/bin/python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
.venv-openvla/bin/python -m pip install -r requirements-openvla.txt
.venv-openvla/bin/python -m pip check
.venv-openvla/bin/python openvla_mac.py --preflight --device cuda
```

This deliberately uses the existing project dependency pins in a separate
environment, even if the template includes PyTorch 2.4.1 or 2.8.0. Those template
versions have not yet been tested with our OpenVLA stack. Driver compatibility
and the preflight must pass before downloading the model. Do not modify the
template's global Python installation to resolve model dependency issues.

## Download and persist the weights

Your Global volume is mounted at `/workspace`; confirm this in RunPod's Volumes
tab. Keep environments, HF caches, and imported custom Python modules under
`/root/robot-openvla` on the container disk, not on that Global volume.

```bash
.venv-openvla/bin/python prepare_cloud_model.py --export-dir /workspace/openvla-7b
```

The approximately 15 GB download uses local
`.cache-openvla-server/download`. The export dereferences cache links and copies
ordinary files into `/workspace/openvla-7b`. It checks copied sizes and writes
`export_complete.json` after all copies succeed; this is a completion record, not
a full checksum validation. Wait for completion before loading. Use only one pod
to write this directory. This avoids requiring file locks or atomic rename on
Global storage. Completed exports with matching sizes are reused on later pods.

The local download cache remains on disk. After verifying the saved export, it
can be removed to reclaim space. Future pods load the Global-volume files
directly and only need to reinstall dependencies, not download the model again.
An existing Mac HF manifest with an absolute Mac path is not used by this server.

## Start the model server on the pod

```bash
.venv-openvla/bin/python connection_server.py --mode openvla --model-dir /workspace/openvla-7b --device cuda --max-runtime-seconds 900
```

Wait for `openvla server ready` after successful model loading. Startup prints
the load duration. The model stays resident between requests. `/health` reports
the mode, loaded status, and load duration once ready. Local-only loading means
missing model files cause an error rather than a silent 15 GB download.

The 900-second limit starts **after model loading**, and is a server lifecycle
limit, not a cloud spending cap or a per-inference deadline. It does not terminate
the pod, and must not be relied upon to stop GPU billing. Stop/terminate the pod
in RunPod when finished. Loading or prediction may fail or take longer than
expected; inspect the pod terminal. A client timeout does not cancel GPU work.

## On the Mac: tunnel and one prediction

Open an SSH tunnel using the **current pod's** IP and port. Stop any local server
on port 8000 first. Replace the placeholders below with the Connect-tab values:

```bash
ssh -N -F /dev/null -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -L 127.0.0.1:8000:127.0.0.1:8000 -i ~/.ssh/id_ed25519_maxalderone -p POD_SSH_PORT root@POD_IP
```

In another **Mac terminal**, from the local repository:

```bash
.venv/bin/python connection_client.py --expected-mode openvla --timeout-seconds 180 --instruction "pick up the red cube"
```

The 180-second response timeout is an initial experimental allowance, not a
latency promise. Inspect `outputs/remote_inference_*/input.png`, `result.json`, and
`status.json`. Completion confirms receipt of a real seven-value model prediction
only, not grasping or movement. The returned values use `bridge_orig` dataset
statistics: they are raw model actions, not yet converted/clipped for the Panda.
The real gripper convention is 0 closed / 1 open; the fixed test response uses
the controller convention of -1 closed / +1 open. Do not execute these raw values
without the existing bounded mapping and tracking checks.

## Timings and local checks

Server timings distinguish:

- `receive_body`: server-side request-body read.
- `preprocess`: JSON/base64/PNG decode and RGB conversion.
- `processor_and_device_transfer`: image processing, text tokenization, extra
  prompt-token/mask handling, device transfer, and synchronization.
- `model_inference`: synchronized `predict_action()` including autoregressive
  generation of the action and decoding. It is not necessarily a single neural
  network forward call.
- `server_processing`: total before response serialization.

Client timings retain capture/encode, connection setup, send, header wait,
response-body read, round trip, and total elapsed time. Send/read durations are
not isolated one-way network latencies. Separate model-load timing appears in
server startup and health. Movement remains null because this client does not
move the arm.

Run contract tests without loading the 7B model:

```bash
.venv-openvla/bin/python -m unittest test_remote_inference -v
```

These use substituted model/processor objects to check resident loading,
per-request prompt/image handling, aligned token masks, response modes, busy
handling, and exporting linked cache files as regular persistent files. They do
not establish CUDA compatibility or real model correctness. The unchanged fixed
mode can still be checked with [CONNECTION_TEST.md](CONNECTION_TEST.md).

Next milestone: validate one real GPU prediction through the tunnel, review its
timings, then connect bounded actions to the simulation in a separate change.
