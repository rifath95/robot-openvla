# Uninterrupted 12,000-update colored bowl experiment

Fresh base; same pinned 600-demo dataset (`training_color_place_bowl_v1_600`,
HF revision `0bf1faf4036cbfd5e3c7282111f1d9a54b2d020f`). Batch 8,
accumulation 1, LR 0.0001, LoRA rank 32, no augmentation or validation.
Checkpoints: 6,000, 8,000, 10,000, 12,000. No optimizer reset during training.
The earlier run did not record all initial RNG states; the first 6,000 updates
cannot be promised identical to that run. GPU nondeterminism also remains.

## First pod: training only

Use the existing A40 training image, same network volume with base weights,
container disk at least 80 GB (40 GiB actually free), and at least 8 GiB free on
the network volume for adapters, final optimizer/input state and reports.
Training is expected to take about 6.5–8 hours. Estimated budget 8 h, inner
supervisor limit 11 h, outer independent limit 16 h. The training loop's own
10 h guard leaves time for final saving and verified HF delivery.

Run in the pod terminal after cloning/pulling this repository:

```bash
cd /root/robot-openvla
/opt/openvla/bin/python -u -m cloud.color_bowl_training --launch \
  --profile inference/training_profiles/color_place_bowl_600_uninterrupted.json
```

Supply the HF write token at the hidden prompt. Dataset downloads directly to
container disk, and base weights are copied from volume to container disk.
Adapters and final training state go to the network volume. Checkpoints and
training reports are uploaded to the private model repository
`Maxalderone/robot-openvla-checkpoints` and verified before automatic pod stop.
Mac may disconnect. Keep the printed session name for the sweep.

## Final checkpoint and future full resume

Intermediate checkpoints remain adapter-only. The final checkpoint includes
`training_state/state.pt`, the TensorFlow input iterator checkpoint, and a
checksum manifest. It contains AdamW moments/step/parameter groups, Python,
NumPy, Torch and CUDA random states, cumulative update and batch configuration.
There is no LR scheduler in this trainer; constant-LR policy is recorded.
Final ready metadata is written after both adapter and training state succeed.
A graceful timeout also saves full state; hard process/pod failure cannot.
Resume requires matching dataset, base model, LoRA and batch settings.
TensorFlow iterator snapshots retain absolute record filenames. On a new pod,
the loader creates an alias from the old controlled container staging directory
to the verified current dataset. It never overwrites an existing directory; for
other input locations, supply the original path explicitly.

```bash
/opt/openvla/bin/python -m training.train_panda \
  --dataset /path/to/the/same/bundle --model /workspace/openvla-7b \
  --resume-checkpoint /path/to/checkpoint_012000 \
  --config /path/to/additional_updates_config.json --output-root /path/to/new/run
```

`max_steps` and checkpoint schedules in the new config count **additional**
updates; logs and output checkpoint names include the restored cumulative offset.
Use the same dependency stack/device count. Restored optimizer parameter groups
retain the original learning rate. This restores training state, not a guarantee
of bitwise GPU determinism. CPU tests compare the next AdamW update and next
shuffled input after a save/load with uninterrupted execution.

## Second pod: parallel 800-trial sweep

Start after checkpoint 6,000 is fully saved. Use the same network volume, an
inference-capable GPU with EGL (`NVIDIA_DRIVER_CAPABILITIES=all`), at least 25 GiB
free on container disk and **35 GiB free on network volume** for recordings.
The new training itself needs only the smaller 8 GiB reserve above.

Prepare locally, replacing the training session with the printed name:

```bash
.venv/bin/python -m data_tools.prepare_uninterrupted_bowl_sweep \
  --training-session panda-train-color-bowl-REPLACE
```

Then launch the prepared folder with the second pod's SSH:

```bash
caffeinate -i .venv/bin/python -m cloud.cloud_sweep_session \
  --ssh 'SSH_FROM_SECOND_POD' --resume outputs/sweeps/scripted_uninterrupted_bowl_REPLACE
```

800 trials: four checkpoints × 200 fixed trials. Same six groups and layouts as
previous sweeps: single red/black (20 each), two red+black/black+purple (40 each),
four red-present/red-absent (40 each). Greedy, maximum 120 actions, same placement
criteria. Later checkpoints are waited for (up to 12 h each, within the overall
budget); incomplete saves are never loaded. Staged inference copies exclude
optimizer/input state. Estimated budget 12 h, inner limit 15 h, outer limit 20 h.

**Default two-stage delivery:** simulator writes recordings to network volume
and auto-stops. Packing/upload stays off the simulator pod. Prepare the existing
HF write token before the cheaper transfer pod is launched; run there:

```bash
/opt/openvla/bin/python -u -m cloud.archive_sweep \
  --sweep scripted_uninterrupted_bowl_REPLACE \
  --repo Maxalderone/robot-openvla-simulator-results
```

This helper verifies the upload and auto-stops the transfer pod. Original volume
results remain unless deletion is explicitly selected. Do not delete checkpoints
while either pod may need them. Training pod shutdown does not stop the separate
sweep pod.
