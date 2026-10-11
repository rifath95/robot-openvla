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

## Upload after launching a pod

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
