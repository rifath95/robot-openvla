# Prepared 600-demonstration colored bowl training

This is a training-only run. It uses the verified mixture in
[the dataset review](../reviews/COLOR_PLACE_BOWL_600.md), without launching a
simulator sweep. No paid run has been launched during local preparation.

| Setting | Value |
|---|---|
| Dataset | `training_color_place_bowl_v1_600`: 600 train / 43,344 action examples |
| Scene mixture | 200 one-cube + 400 two-cube; 150 requested targets per training color |
| Base | Fresh pretrained OpenVLA; no earlier adapter |
| GPU | NVIDIA A40, usual training template |
| Batch | Microbatch 8, accumulation 1, effective batch 8 |
| Updates | 6,000; 48,000 sampled examples, about 1.11 dataset passes worth |
| Checkpoints | 0, 1,000, 2,000, 4,000, 6,000 |
| Learning rate / LoRA | 0.0001 / rank 32, dropout 0; no quantization or image augmentation |
| Validation | Disabled; all 600 demonstrations are training data |
| Estimated complete job | About 3.5–4 hours, including setup and checkpoint backup |
| Inner deadline | 7 hours = 4-hour estimate + 3-hour buffer |
| Independent outer deadline | 12 hours = inner + 5-hour buffer |

Doubling updates relative to the 300-demo run preserves approximately the same
number of sampled examples per dataset example. The estimate uses its measured
roughly 2.06 seconds/update; host speed and HF transfers can vary. No claim of
behavioral success or the best checkpoint is made before a separate sweep.

## Inputs and persistent outputs

Use **200 GB container disk** (preflight requires 40 GiB actually free) and the
existing network volume at `/workspace` with **at least 5 GiB free**. A global
volume is not required for this training-only job.

The dataset downloads directly from HF to container disk; the existing base at
`/workspace/openvla-7b` is copied and verified once there. Training reads those
copies. All five checkpoints, plots, metrics, TensorBoard events, provenance,
status and timings are written to a unique network-volume session. No existing
sessions, datasets or checkpoints are deleted or overwritten.

The launcher requires the completed HF upload's prepared file sizes and checksum
manifest, resolves it to one immutable revision and records it in checkpoint
provenance. It refuses an incomplete/different upload before starting a job.
The guarded worker then checks every downloaded checksum before loading weights.
The expected inventory is stored in
`inference/training_profiles/color_place_bowl_600.json`; trainer settings are in
`training/color_place_bowl_600_6000.json`.

After all 6,000 updates and the complete five-checkpoint schedule are verified,
checkpoints and training reports are uploaded to the private model repository
`Maxalderone/robot-openvla-checkpoints` under `sessions/SESSION`. Pinned-commit
size/SHA-256 verification downloads one file at a time. Final stage reports and
completed timing snapshots are also verified under `training-reports/SESSION`.
Approximately 2.3 GiB of checkpoint backup is additional to the 5.04 GiB dataset.
Checkpoints remain on the network volume even if training/backup fails.

## Launch after the dataset upload finishes

Do not launch the pod merely to wait for the upload. No new image build is needed.
Have the saved HF **write** token ready (dataset read access and checkpoint-repo
write access). Keep the usual RunPod stop secret on the template.

From a Mac terminal in the repository, using the new pod's current SSH address:

```bash
caffeinate -i .venv/bin/python -m cloud.training_cloud_session \
  --ssh 'ssh root@HOST -p PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --prepared-color-bowl
```

The Mac prompts for the HF write token without displaying it. It is sent over
SSH stdin, then private subprocess pipes; never put it in command text, a file,
Git, or an environment variable in the generated job configuration. The remote
code checkout is frozen to its launch commit. Detachment occurs before bulk
downloads/staging/training/backup, so these stages are within the stated limits.
Pod boot/image pull, initial authentication and the small inventory/code checks
precede those clocks.

After `Detached cloud training confirmed`, the Mac may disconnect. When it stays
connected, the local watcher streams training progress, downloads reports to
`outputs/training/phase3_training/SESSION/`, and acknowledges successful download.
The supervisor has a bounded 120-second download window after the worker exits;
a disconnected watcher cannot keep the pod running indefinitely. Adapter weights
stay on the volume and are backed up to HF; local report synchronization imports
only checkpoint metadata.

Direct cloud-terminal launch is also available after cloning/pulling the repo:

```bash
cd /root/robot-openvla
/opt/openvla/bin/python -u -m cloud.color_bowl_training --launch
```

This version prompts in the cloud terminal, detaches, and prints the session and
log locations; it does not attach a Mac report watcher.

Later report-only fetch, using the current pod SSH address:

```bash
caffeinate -i .venv/bin/python -m cloud.training_cloud_session \
  --ssh 'ssh root@HOST -p PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --fetch-only --session SESSION
```

## Failure handling and timing

A detached supervisor owns stopping. An independently detached watchdog can kill
worker descendants (including separate Torch elastic process groups) and attempt
the API stop even if the supervisor/filesystem stalls. Inner timeout/failure also
leads to stop attempts; checkpoints/reports already saved remain on the volume.
API/network failure can prevent stopping, so confirm **Stopped** in RunPod.

Each stage has start/end timestamps, duration and error state in `TIMINGS.md` and
`stage_timings.jsonl`. Updates and checkpoint saves have separate wall times;
backup receipts separate copying/upload/verification time. Failed HF backup is
reported separately from completed training and cannot be presented as a fully
completed job. The final stop acknowledgment remains on the network volume.

Local checks cover upload rejection/pinning, actual-bundle prepare-only training,
worker ordering, secret handling, report synchronization, backup retention and the
shared inner/outer deadline behavior. The full paid GPU workflow remains untested
until deployment.
