# RunPod connection guide: Mac camera → cloud OpenVLA → Mac action

Last updated: 2026-10-09. This is the repeatable procedure used for our successful
RTX A6000 test. It returns a real seven-number prediction and displays it on the
Mac; it does **not** move the robot.

## What runs where

Use three terminal tabs in your **local VS Code window**:

| Terminal | Role | Where commands run |
| --- | --- | --- |
| 1 — Pod/server | SSH login, setup, model server | Starts on your Mac; after SSH login, runs on the pod |
| 2 — Tunnel | Forwards Mac port 8000 to pod port 8000 | Mac; leave the command running |
| 3 — Client | Captures MuJoCo image and receives prediction | Mac, in the local `robot-openvla` folder |

An SSH terminal does not automatically turn the editor into a remote editor.
The prompt `root@...` means the terminal is on the pod. Your usual
`PHY-JWC6YKMW96 ... robot-openvla` prompt means it is on your Mac.

## Values to replace

Replace the uppercase placeholders in commands before running them:

| Placeholder | Where to get it |
| --- | --- |
| `POD_IP` | RunPod → Pods → your running pod → Connect → **SSH over exposed TCP**, after `root@` |
| `POD_SSH_PORT` | The number after `-p` in that same command |
| `PRIVATE_KEY_PATH` | Path on your Mac to the private key matching your registered public key. For our setup: `/Users/rifath95/.ssh/id_ed25519_maxalderone` |

RunPod may show `-i ~/.ssh/id_ed25519` as a default. Replace that with **your
actual private-key path**; do not use the `.pub` suffix for SSH login. The
private key stays on your Mac. The matching public key is registered under
RunPod → Credentials → SSH Public Keys as **MacBook VS Code**.

Check IP/port every session; they can change. The successful session used
`194.68.245.167` and port `22103`, but these are historical values, not permanent.

## 1. Launch or resume a pod

In RunPod, select one GPU with enough VRAM. Our real inference test used
**1× RTX A6000, 48 GB VRAM, 50 GB RAM, 9 vCPUs**, with a **60 GB container disk**.
Attach **`robot-openvla-models`**, the existing Global volume, at `/workspace`.
Check the displayed price and that the pod starts successfully before continuing.

Push any local code changes to GitHub **before** this step so the cloud copy can
receive them. This guide does not launch resources automatically.

## 2. Terminal 1: log into the pod

Run on your Mac:

```bash
ssh -F /dev/null -o IdentitiesOnly=yes root@POD_IP -p POD_SSH_PORT -i PRIVATE_KEY_PATH
```

`-F /dev/null` bypasses saved SSH settings; `IdentitiesOnly` selects the supplied
key. On the first connection, SSH asks to confirm the server identity. Check the
connection details match your intended pod, type `yes`, then press Enter.
After the Ubuntu welcome message, expect `root@...:~#`.

If asked for the **key passphrase**, enter your key's passphrase locally. If asked
for **`root@...`'s password**, cancel with Control+C: key authentication has failed.
Check the private-key path and registered key rather than guessing a root password.

## 3. Terminal 1: check GPU and storage

These now run **on the pod**:

```bash
nvidia-smi
findmnt -T /workspace
```

Expect your GPU and the Global-volume mount. Our successful mount used filesystem
type `fuse.gee`. CUDA displayed by `nvidia-smi` describes driver compatibility,
not the version of PyTorch installed in our environment.

## 4. Terminal 1: obtain the code

For a fresh pod:

```bash
git clone https://github.com/rifath95/robot-openvla.git /root/robot-openvla
cd /root/robot-openvla
```

The URL is the source; `/root/robot-openvla` is the destination **on the pod's
container disk**, not your Mac or Global volume. The server does not need
Menagerie, so this clone does not need `--recurse-submodules`.

If this folder already exists, use this instead of cloning again:

```bash
cd /root/robot-openvla
git pull --ff-only
```

## 5. Terminal 1: install isolated dependencies

On a fresh container, run:

```bash
python3.11 -m venv .venv-openvla
.venv-openvla/bin/python --version
.venv-openvla/bin/python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
.venv-openvla/bin/python -m pip install -r requirements-openvla.txt
.venv-openvla/bin/python -m pip check
.venv-openvla/bin/python openvla_mac.py --preflight --device cuda
```

Wait for each command to finish; stop and inspect any error before continuing.
Expect `No broken requirements found` and a successful CUDA matrix test. Our
A6000 used `torch.bfloat16`. These commands install software and check the GPU;
they do not download model weights. The explicit environment path means you do
not need to activate the environment or change the template's global Python.

## 6. Terminal 1: download or reuse persistent weights

```bash
.venv-openvla/bin/python prepare_cloud_model.py --export-dir /workspace/openvla-7b
```

First time: downloads approximately 15 GB from **Hugging Face directly to the
pod**, then copies regular model files to the Global volume. Wait for
`Completed export: /workspace/openvla-7b`.

Later sessions: the same command checks the export's completion record and file
sizes. If complete, it prints `Completed export already exists` and skips the
download. It does not upload anything from your Mac. Keep the model directory on
the Global volume; keep dependencies and Hugging Face caches on the container disk.

## 7. Terminal 1: start OpenVLA server

```bash
.venv-openvla/bin/python connection_server.py --mode openvla --model-dir /workspace/openvla-7b --device cuda --max-runtime-seconds 900
```

Wait for **`openvla server ready at http://127.0.0.1:8000`**. Loading moves the
weights into GPU memory; it took approximately **202 seconds** in our first test.
Keep Terminal 1 running. The model stays loaded between requests.

The 900-second limit begins **after readiness** and only limits the server.
It does **not stop the pod or billing**. If it expires, restart this command;
that reloads the model. A client timeout does not cancel ongoing GPU work.

## 8. Terminal 2: start SSH tunnel on the Mac

Open a **new local terminal tab**, replace the same placeholders, and run:

```bash
ssh -N -F /dev/null -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -L 127.0.0.1:8000:127.0.0.1:8000 -i PRIVATE_KEY_PATH -p POD_SSH_PORT root@POD_IP
```

Normal behavior: no output, and the command stays running. `-N` means no remote
shell; `-L` maps Mac port 8000 to port 8000 on the pod. Keep Terminal 2 running.
Stop any local test server using port 8000 first, or the tunnel cannot bind it.
The model server remains private on the pod; the SSH tunnel carries the traffic.

## 9. Terminal 3: request one prediction from the Mac

Open another **local terminal** in your Mac's project folder:

```bash
cd "/Users/rifath95/Desktop/Extra work/Robotics/robot-openvla"
.venv/bin/python connection_client.py --expected-mode openvla --timeout-seconds 180 --instruction "pick up the red cube"
```

The `cd` above is your current Mac path; update it if you move the repository.
Your existing local simulation `.venv` and Menagerie assets must be present.
The client captures the 640×480 home-scene image, sends it through Terminal 2's
tunnel, and prints the real model response. Each invocation captures a fresh
home-scene image; this is not yet a moving-robot feedback loop.

Expected output includes `Received OpenVLA action`, seven finite numbers, server
timings, and an output-folder path. **The robot is not moved.** You may run the
client again while the same server/tunnel remains ready; model loading is not
repeated between requests.

## 10. Inspect saved output on the Mac

Each request creates `outputs/remote_inference_<timestamp>_<id>/`:

| File | Contents |
| --- | --- |
| `input.png` | Image sent to the server |
| `result.json` | Seven action values, instruction, image details, server/client timings |
| `status.json` | Progress, `Completed` or `Failed`, elapsed time, and any error |

The seven values are `[dx, dy, dz, droll, dpitch, dyaw, gripper]`, using
`bridge_orig` dataset statistics. They are not yet converted/clipped for our
Panda controller. Real model gripper convention: 0 closed, 1 open.

Our first successful run measured:

| Measurement | Seconds |
| --- | ---: |
| Request-body read on server | 0.352 |
| JSON/image decode to RGB | 0.031 |
| Image processing, text tokenization, GPU transfer | 0.072 |
| Complete OpenVLA action prediction | 0.702 |
| Full request round trip | 1.689 |

Prediction includes sequential generation and decoding of action tokens, not
necessarily a single neural-network forward call. Round trip excludes camera
capture, request encoding, and initial model loading. Send/read measurements are
not isolated one-way network latencies. Future runs may have different timings.

## 11. End the session

1. In **RunPod**, Stop or Terminate the pod to stop GPU compute billing. Closing
   terminals, stopping Python, or letting the server timer expire is insufficient.
2. Press Control+C in Terminal 2 to close the tunnel if it is still running.
3. Close the remaining terminals. SSH sessions may already have disconnected
   automatically when you stopped the pod.

**Our storage layout:** Stop clears the container disk, and Terminate deletes it.
Plan to clone the code and install dependencies again in either case. The
separate Global volume survives both operations, so retain it to reuse the
exported weights; its storage/operation charges are separate. Model loading into
GPU memory happens again on every new server process. A future custom Docker
image can avoid repeated dependency installation.

For implementation details, see [REMOTE_INFERENCE.md](REMOTE_INFERENCE.md).

For the next single-movement test, keep the same server/tunnel setup and replace
Terminal 3's display-only command with `.venv/bin/python remote_single_action.py`.
This executes one bounded action and opens the final scene; read
[REMOTE_SINGLE_ACTION.md](REMOTE_SINGLE_ACTION.md) before using it.

For repeated predictions and a live simulator, use the same server and tunnel,
then run `.venv/bin/python remote_loop.py --steps 10 --max-runtime-seconds 120`
in Terminal 3 on the Mac. The window starts paused; press Space to start. See
[REMOTE_LOOP.md](REMOTE_LOOP.md) for controls, results, limits, and shutdown.
