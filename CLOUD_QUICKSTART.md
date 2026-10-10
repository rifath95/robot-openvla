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

Defaults: 100 prediction cycles (including rejected ones), three consecutive
IK rejections before a clean stop, 900 active local seconds (pauses excluded), 1,800 server
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
   port 22**. Set the template environment variable `PUBLIC_KEY` to the complete
   contents of your Mac public-key file (`cat ~/.ssh/id_ed25519_maxalderone.pub`).
   This is the public key, never the private key. Our first custom-template test
   started without this value and skipped SSH startup, causing connection refused.
   Retain the image's default startup command; attach the Global volume
   at `/workspace` and use the same disk/GPU settings as before. Jupyter is optional.
6. Launch using that template. Use the same one-command launcher above.

The image extends the tested official RunPod PyTorch/CUDA image and installs
our isolated environment at `/opt/openvla`. It retains RunPod's SSH startup.
The server and model can still be launched on another provider using
`bash scripts/cloud_start.sh 1800` with a persistent mount at `/workspace`.
The inference modules themselves contain no RunPod-specific API calls.

The Docker image was built through GitHub Actions and tested on an **A40**.
Its dependencies were reused without pip installation and CUDA preflight passed.
The first custom-template deployment skipped SSH until the public key was added;
set `PUBLIC_KEY` in the template as described above for future sessions.
The live cloud loop completed 95 actions before an IK rejection. The ordinary
template fallback remains available. Docker is not installed on this Mac;
GitHub Actions remains the image build path.

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

## SSH connection refused after image startup

If container logs show `Pod Started` and environment export but no SSH setup,
check that `PUBLIC_KEY` is present. The inherited startup script skips SSH when
that variable is empty. Save the complete public key in the template environment
for future deployments; updating a template does not change an already running pod.
For the current pod, use its web terminal to add the matching public key to
`/root/.ssh/authorized_keys`, set directory/file modes to 700/600, run
`ssh-keygen -A`, and `service ssh start`. Then retry the Mac launcher.

Local recovery now records held actions and stops cleanly after repeated IK
rejection. Use `--max-consecutive-rejections 3` in the launcher to set this bound.
See [CONTROL_FINDINGS.md](CONTROL_FINDINGS.md) before interpreting repeated motion
as task success. No Docker rebuild is needed for this controller change.

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
