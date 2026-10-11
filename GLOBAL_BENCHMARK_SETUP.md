# Global-volume inputs for GPU benchmarks

For B300 and other Blackwell GPUs, build and test the separate candidate image
using [Blackwell training setup](BLACKWELL_TRAINING_SETUP.md).

The global volume is named **robot-openvla-global-volume**. It is separate from
the regional network volume used by the running 5,000-update fine-tuning job.
Do not modify or stop that training pod while preparing this new storage.

## Original 50-trajectory dataset on Hugging Face

The original bundle is now uploaded to the private dataset repository
`Maxalderone/robot-openvla-panda` under `training_v1`, pinned at
`f653bb505352c358fa7ea478e2338929d3222a68`. It contains 36 training trajectories
(2,595 training examples), 7 validation trajectories and 7 test trajectories.
Its 12 checksum entries were verified locally before uploading; the remote
checksum manifest matches and all referenced files are present.

The earlier upload under `training_v2` contains all 250 trajectories, including
the original 50 and the newer 200. Its existing pinned revision and downloads
remain unchanged. Both bundles use the same image/action format, so v1 is enough
for our short capacity/throughput tests. Such tests do not measure task success.

## Populate the global volume once

Launch a cheap preparation pod with the existing training image, attach the
global volume, and include the existing `HF_TOKEN` read secret. Keep the
`RUNPOD_API_KEY` stop secret configured for subsequent benchmark deployments.
Downloading does not require a large GPU or load the model into VRAM.

Connect using the pod's current SSH endpoint and your personal key. In the cloud
terminal, clone this repository (or pull its latest main if already cloned):

```bash
git clone https://github.com/rifath95/robot-openvla.git /root/robot-openvla
cd /root/robot-openvla
findmnt -T /workspace
findmnt -T /workspace-global
```

Check which path is actually the global-volume mount. With `/workspace` mounted:

```bash
/opt/openvla/bin/python prepare_global_benchmark.py --volume-root /workspace
```

If mounted at `/workspace-global`, use that path instead:

```bash
/opt/openvla/bin/python prepare_global_benchmark.py --volume-root /workspace-global
```

This command downloads the pinned original OpenVLA weights from Hugging Face
(approximately 15 GB) and the original dataset (approximately 425 MB) directly
to the pod, then copies verified regular files to the global volume. Hugging
Face caches and dataset staging stay on the pod's local container disk, avoiding
file locks and directory renames on global storage. Only one preparation pod
should write these destinations. Do not use them until preparation finishes.

The model completion record is `openvla-7b/export_complete.json`. Dataset hashes
are verified again after copying, and provenance is recorded at
`panda-training/training_v1_hf_source.json`. Existing complete datasets are
verified and reused without overwriting. If a global dataset copy is interrupted,
an incomplete destination is rejected; inspect it before retrying rather than
deleting arbitrary volume contents.

After inputs are verified, stop the preparation pod manually. This preparation
command does not stop a rented pod and does not start a GPU benchmark.

## Benchmarking after preparation

From the Mac workspace, use the current SSH command from RunPod → Pod → Connect:

```bash
caffeinate -i .venv/bin/python benchmark_cloud_session.py \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519' \
  --global-volume /workspace \
  --dataset /workspace/panda-training/training_v1
```

Use `/workspace-global` in both paths if that is the actual mount. The launcher
selects your configured personal SSH key. Include `RUNPOD_API_KEY` through the
existing stop secret in the template. No Docker rebuild is required.

Global mode verifies the mount, pod API access, model completion record and
all dataset checksums. It stages inputs on container disk under
`/root/robot-openvla/.benchmark-local`, where benchmark locks and working files
also live. On the preparation pod, the pinned local download cache can supply
the model copy; subsequent pods read the model from the global volume. The
recorded staging metadata distinguishes those cases. Training throughput uses
local inputs and excludes setup, staging and model loading.

The detached supervisor has a **60-minute outer limit**, beginning at supervisor
startup and including input staging and all candidate tests. Initial volume
population is a separate command outside that limit. Adapter checkpoint saving
is disabled. Reports and logs are copied to
`/workspace/panda-training/batch-benchmarks` before the supervisor requests a pod
stop after completion, failure or timeout. An API/network failure can prevent
stopping; confirm the stopped state in RunPod.

If the Mac stays connected, reports download automatically to
`outputs/gpu-benchmarks` and refresh the local `GPU_BATCH_COMPARISON.md`. If it
sleeps or disconnects, the cloud supervisor continues, saves reports and requests
a stop. Launch/resume a pod attached to the same volume and fetch later with:

```bash
.venv/bin/python benchmark_cloud_session.py \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519' \
  --global-volume /workspace --fetch-only --stop-after-fetch
```

Use one benchmark writer at a time on this global volume. Global storage does not
provide the locking and atomic rename behavior used on regional network volumes;
the global mode keeps those operations on container disk and publishes regular
report files. New GPU architectures may require a compatible CUDA/PyTorch image.
