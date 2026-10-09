# One-command cloud sessions

Run one command on your Mac instead of copying the setup, server, tunnel, and
client commands separately. The inference code remains independent of RunPod:
the launcher connects to a Linux GPU host through SSH. No API key is needed.

## Each session

1. Push the latest code to GitHub **before renting a GPU**.
2. In RunPod, launch one compatible GPU pod, enable SSH over exposed TCP, and
   attach your existing `robot-openvla-models` Global volume at `/workspace`.
   Keep using the tested 60 GB container disk and your registered SSH public key.
3. In the **Mac workspace terminal**, run:

```bash
.venv/bin/python cloud_session.py
```

It asks you to paste the **SSH over exposed TCP** command from RunPod → Pods →
your pod → Connect. For example, paste your current value in this format:

```text
ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519
```

The launcher uses this Mac's `~/.ssh/id_ed25519_maxalderone` when RunPod supplies
the generic key filename and the matching private key exists. It does not copy
your private key to the cloud. An explicit `--key` overrides this choice.
Normal SSH host-identity confirmation is retained. A passphrase-protected key
should be added to your local SSH agent so repeated SSH operations do not each
ask for its passphrase. Root password authentication is disabled.

You can also supply the pasted command directly, enclosed in quotes:

```bash
.venv/bin/python cloud_session.py --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519'
```

## What the command does

- Checks that local port 8000 is free and connects using your private key.
- Clones or updates `/root/robot-openvla` from GitHub on the pod.
- Checks the persistent volume and GPU.
- Uses the Docker image's preinstalled dependencies when available. On ordinary
  RunPod templates, installs the tested dependencies automatically in an isolated
  environment. Subsequent sessions in the same container reuse that environment.
- Checks the completed model export at `/workspace/openvla-7b`. Existing weights
  are reused; if missing, downloads them from Hugging Face and exports them.
  Only one pod should prepare/write this model directory at a time.
- Starts the private model server and its log on the container disk.
- Opens the SSH tunnel and waits for actual model readiness.
- Opens the **paused local simulator** for the 100-action experiment.

Click the simulator and press **Space** to start/pause/resume. **Esc** ends the
local trial. After a trial, the launcher stays open with the model loaded; press
**Enter in the terminal** to run another trial without loading again.

**Ctrl+C in the launcher terminal** ends the session, stops the project server,
and closes the tunnel. It does **not** stop or terminate the rented pod. Finish
by stopping/terminating it in RunPod. Model weights on the independent Global
volume remain. Local artifacts are under `outputs/remote_loop_*`; the cloud
server log is `/root/robot-openvla/outputs/cloud_server.log`.

Defaults: 100 actions, 900 active local seconds (pauses excluded), 1,800 server
wall seconds after readiness (pauses included), and 600 seconds to wait for model
loading after setup. The server deadline is not renewed between trials. Pause
and setup time still incur pod billing. A terminated pod loses container-side
dependencies; use the Docker option below to avoid reinstalling them.

Optional examples:

```bash
# Prepare the server/tunnel, then use a separate Mac terminal for experiments.
.venv/bin/python cloud_session.py --setup-only

# A shorter first trial.
.venv/bin/python cloud_session.py --steps 10
```

Changing only Mac loop code does not require rebuilding the dependency image.
Changing inference dependencies does require a new image build. Loading model
weights into GPU memory still happens each time the model server starts. Image
pull/startup time varies by host; Docker does not guarantee instant readiness.

## One-time Docker preparation through GitHub

The repository includes a `Dockerfile` and a **manually triggered** workflow;
merely pushing code does not publish an image. No model weights, SSH keys,
Mac virtual environments, or outputs are included in the Docker build context.

1. Push these files to GitHub.
2. Open GitHub → `rifath95/robot-openvla` → **Actions** → **Build cloud dependency
   image** → **Run workflow** on `main`.
3. Wait for a successful build. It publishes these images to GitHub Container
   Registry using GitHub's built-in workflow token:
   `ghcr.io/rifath95/robot-openvla-cloud:latest` and a commit-SHA tag.
4. In GitHub, open the resulting package's settings and make the package public
   if you want RunPod to pull without registry credentials. A private package
   requires configuring registry credentials in RunPod instead.
5. Create a RunPod pod template using that container image. Prefer the **commit-SHA
   tag shown by the workflow** for reproducibility. Enable SSH and expose **TCP
   port 22**. Retain the image's default startup command; attach the Global volume
   at `/workspace` and use the same disk/GPU settings as before. Jupyter is optional.
6. Launch using that template. Use the same one-command launcher above.

The image extends the tested official RunPod PyTorch/CUDA image and installs
our isolated environment at `/opt/openvla`. It retains RunPod's SSH startup.
The server and model can still be launched on another provider using
`bash scripts/cloud_start.sh 1800` with a persistent mount at `/workspace`.
The inference modules themselves contain no RunPod-specific API calls.

The Docker image has **not yet been built or tested on a GPU**. This Mac has no
Docker executable available; the GitHub workflow is the prepared build path.
The first cloud session must validate image pulling, SSH startup, and inference.
The ordinary-template fallback is ready for use before publishing the image.

References: [RunPod official base-image startup](https://github.com/runpod/containers/blob/main/container-template/start.sh),
[RunPod template image examples](https://github.com/runpod-workers/pod-template).

## Local checks

```bash
.venv/bin/python -m unittest test_cloud_session -v
bash -n scripts/cloud_start.sh scripts/cloud_stop.sh
```

Tests cover SSH parsing/quoting, rejection of pasted extra shell commands,
real-model readiness checks, and cleanup orchestration with mocked subprocesses.
They do not rent a GPU or claim an end-to-end cloud setup has been executed.
