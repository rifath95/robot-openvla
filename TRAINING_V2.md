# Expanded dataset: training_v2

The expanded bundle is generated at `datasets/panda_pick_place/training_v2/`.
It combines the original 50 trajectories and the 200 broad-start trajectories.
The earlier `training_v1` package, uploaded data, base model and LoRA checkpoints
remain unchanged. Generated data is ignored by Git and must be uploaded separately.

| Split | Trajectories | Valid action examples |
|---|---:|---:|
| Training | 176 | 13,812 |
| Validation | 37 | 2,882 |
| Test | 37 | 2,931 |
| Total | 250 | 19,625 |

There are also 250 terminal observations with no outgoing training action.
TFRecords embed lossless RGB images, instruction strings and seven float32
action values. They do not store processor embeddings or tokenized inputs;
the official OpenVLA processor/tokenizer prepares those during training.

The combined manifest preserves cohort labels and fixed splits. New broad-scene
combinations are checked for cross-split leakage. Statistics are computed from
the 176 training trajectories only, and packaged images/actions/instructions
are checked against the source recordings, including terminal after-images.

## Local preparation

These commands have already been run locally; the builder refuses to overwrite
an existing output directory. Use a new bundle name when recollecting data.

```bash
.venv-training/bin/python prepare_training_dataset.py \
  --manifest datasets/panda_pick_place/panda_broad_20261011/combined_manifest.json \
  --output-dir datasets/panda_pick_place/training_v2

.venv-training/bin/python check_training_inputs.py \
  --dataset datasets/panda_pick_place/training_v2 \
  --source-manifest datasets/panda_pick_place/panda_broad_20261011/combined_manifest.json \
  --processor .cache-openvla/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f
```

The processor path is this Mac's cached OpenVLA snapshot, also recorded in
`.cache-openvla/model_revision.json`. Only the processor/tokenizer is loaded for
this check, not the seven-billion-parameter model. The bundle's `PACKAGE_REPORT.md`,
`verification.json`, `input_check.json` and `SHA256SUMS` record the results.

`training_v2` is the bundle name; its TFDS schema directory remains
`rlds/panda_pick_place/1.0.0`. No schema or model architecture change is needed.

## Download the completed Hugging Face upload (recommended)

The full 250-trajectory bundle has been uploaded to the private repository
`Maxalderone/robot-openvla-panda`, under `training_v2`. The verified repository
revision is `8ffd54b1dd27022102aba0866b65a7a9ac4eefc7`.

Before launching a training pod, create a Hugging Face **read** token in
Settings → Access Tokens. Save it in RunPod Secrets, then add `HF_TOKEN` to
the training template's environment variables using the secret picker. Do not
paste the token into commands or Git. Launch with the existing network volume.
No Docker rebuild is required.

Some RunPod images pass `HF_TOKEN` to the container but omit it from SSH shells.
The downloader automatically recovers only that variable from the container's
environment when needed. It never prints the token or saves it in the dataset.

After cloning/updating this repository in the cloud terminal, run:

```bash
cd /root/robot-openvla
/opt/openvla/bin/python download_training_dataset.py
```

The downloader pins the completed upload, downloads into a resumable staging
directory on the volume, verifies all 31 file hashes, and only then moves the
bundle into `/workspace/panda-training/training_v2`. Repeat the same command
after an interruption. Existing final bundles are verified and reused; mismatched
bundles are rejected without overwriting. Hugging Face provenance is recorded
outside the bundle in `/workspace/panda-training/training_v2_hf_source.json`.
Original v1, base weights and previous adapters remain unchanged. This download
does not start training or automatically stop the pod.

## Direct Mac upload after launching a pod (alternative)

Use the existing training template and attach **robot-openvla-network-volume-A40**
at `/workspace`. No Docker rebuild is needed. Once ready, obtain the SSH command
from **RunPod → Connect → SSH over exposed TCP**.

In the **local Mac terminal**, substitute the new IP and port:

```bash
.venv/bin/python upload_training_dataset.py \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519' \
  --dataset datasets/panda_pick_place/training_v2
```

The launcher selects this Mac's configured `id_ed25519_maxalderone` key. This
copies to `/workspace/panda-training/training_v2`, checks the mount and refuses
to overwrite any existing destination. After transfer, every upload checksum is
verified in the pod. The new bundle is approximately 2.1 GB, so allow for upload
time before training; upload time is outside the detached training timers.

## Select the dataset for a subsequent training run

The detached wrapper now accepts a second argument: the cloud dataset directory.
After updating the cloud checkout, it can be invoked as:

```bash
bash scripts/train_detached.sh CONFIG_JSON /workspace/panda-training/training_v2
```

`CONFIG_JSON` is the selected file under `training/`, such as `training/smoke.json`.
It controls update count, checkpoint spacing, validation and the inner timeout.
Choose the next training configuration before running this command. Packaging
does not launch training or decide that another 1,000 updates will be sufficient.
Omitting the dataset argument preserves the old `training_v1` default.

The wrapper checks the selected dataset, prints its path, and records it in the
persistent session command. The existing tmux, outer timeout and automatic pod
stop behavior remain. Confirm stopped status in RunPod after any cloud work.

## Reuse the existing A40 batch benchmark

No repeat benchmark or Docker rebuild is needed for v2. Both bundles use
processed images of shape `[6, 224, 224]` and eight supervised tokens (seven
action tokens plus the stop token). The existing reviewed A40 benchmark selects
microbatch **8**, accumulation **1**, when its original files are present on the
network volume and the GPU/training settings match. Selection is printed and
saved in `batch_selection.json`; a missing/incompatible result uses the
configured fallback and prints its reason.

For the next 1,000-update baseline on the expanded bundle, after updating the
cloud checkout and ensuring tmux is installed:

```bash
bash scripts/train_detached.sh training/pick_place_1000.json /workspace/panda-training/training_v2
```

This uses the expanded 250-trajectory bundle: **176 training**, **37 validation**,
and **37 held-out test** trajectories. Validation/test examples are not trained
on. At effective batch eight, 1,000 updates consume 8,000 sampled training steps;
this is an initial baseline, not a guarantee that every training step is seen.
Checkpoints and full validation occur every 250 updates. The existing one-hour
inner training limit and 90-minute outer wrapper limit still apply. The wrapper
requests a pod stop after it saves the results; confirm this in the portal.
