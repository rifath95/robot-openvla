# Global-volume inputs for GPU benchmarks

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

Global storage does not provide the locking/atomic-renaming behavior expected
by the current regional-network benchmark supervisor. The current automatic
benchmark launcher also checks for a regional network-volume ID. **Do not use
that launcher unchanged on a global-volume-only pod.**

Before the first global benchmark, adapt and verify the launcher against the
preparation pod's actual mounts/API metadata. Run benchmark working files and
locks on container disk, and copy completed reports to the global volume before
stopping the pod. Stage the dataset locally so storage speed does not dominate
training throughput. Keep setup/model-loading time separate from measured update
throughput. Future benchmarks already disable adapter checkpoint saving.

The preparation command requires no Docker rebuild. New GPU architectures may
require a compatible CUDA/PyTorch image; verify that before benchmarking them.
