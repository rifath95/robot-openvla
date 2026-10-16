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
Training is expected to take about 6.5–7.5 hours on a comparable A40. Estimated budget 8 h, inner
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

Start after checkpoint 6,000 is fully saved. Use the same network volume and
an inference-capable GPU with EGL (`NVIDIA_DRIVER_CAPABILITIES=all`). Set the
container disk to **200 GB**, with **110 GiB actually free** after image startup;
allow at least **1 GiB free on network volume** for compact reports, in addition
to the training checkpoint reserve. Raw recordings and packed archives stay on
container disk. Attach the existing global recovery volume at `/workspace-global`
for fallback if HF delivery fails, as in the previous same-pod sweeps.

On the sweep pod after cloning/pulling, replace the training session name:

```bash
/opt/openvla/bin/python -u -m cloud.color_bowl_sweep --launch \
  --profile inference/sweep_profiles/color_place_bowl_600_uninterrupted.json \
  --training-session panda-train-color-bowl-REPLACE
```

Supply the HF write token at the hidden prompt. This launches a detached worker;
Mac may disconnect. The source profile selects only this new uninterrupted run,
not the older 6,000-update or fresh-optimizer continuation runs.

800 trials: four checkpoints × 200 fixed trials. Same six groups and layouts as
previous sweeps: single red/black (20 each), two red+black/black+purple (40 each),
four red-present/red-absent (40 each). Greedy, maximum 120 actions, same placement
criteria. Later checkpoints are waited for (up to 12 h each, within the overall
budget); incomplete saves are never loaded. Staged inference copies exclude
optimizer/input state. Estimated budget **12 h including HF delivery**, inner
limit **15 h**, independent outer limit **20 h**.

**User-selected same-pod delivery:** finish trials → pack/upload to
`Maxalderone/robot-openvla-simulator-results` → verify archived files and final
reports → auto-stop the sweep pod. The existing supervisor handles failure,
timeout, bounded global recovery and shutdown. No transfer pod is required.
Training checkpoints remain on network volume and in their separate HF model
repository. Training pod shutdown does not stop the separate sweep pod.

The standard `cloud.cloud_sweep_session` transfer-pod route remains available for
other experiments; it is not the selected route for this one.

## Timing verification against completed runs (October 16, 2026)

These are actual recorded stage times, excluding provisioning/image download and
final shutdown latency. Nested checkpoint/model-loading stages are already inside
the simulator stage; do not sum them again.

| Completed work | Measured time |
|---|---:|
| Original 600-demo run: 6,000 updates, training stage | 3 h 23 min 36 s |
| Continuation: 6,000 further updates, training stage | 3 h 8 min 58 s |
| Combined training-stage reference for 12,000 updates | 6 h 32 min 33 s |
| Prior four-model, 800-trial sweep: simulator stage | 10 h 7 min 26 s |
| That sweep: packing, upload and verification | 22 min 3 s |
| That sweep: setup through final reports | 10 h 30 min 48 s |
| Continuation three-model, 600-trial sweep: simulator stage | 6 h 37 min 21 s |
| That sweep: packing, upload and verification | 13 min 49 s |

The exact upcoming checkpoint numbers have previous trial records:

| Checkpoint | Trials | Sum of trial times | Mean seconds/trial |
|---|---:|---:|---:|
| 6,000 | 200 | 2 h 33 min 26 s | 46.03 |
| 8,000 | 200 | 2 h 3 min 30 s | 37.05 |
| 10,000 | 200 | 2 h 16 min 21 s | 40.90 |
| 12,000 | 200 | 2 h 14 min 47 s | 40.43 |
| Combined | 800 | 9 h 8 min 4 s | 41.10 |

This combines the previous 6,000 checkpoint from the first run and the three
fresh-optimizer continuation checkpoints. The new uninterrupted run will have
different weights and may take different numbers of actions to finish or fail.
The historical sum is a timing reference, not a prediction for identical models.
Add checkpoint switching, trial/report overhead, staging, packing and verified
HF upload: **9.5–11 hours** is a reasonable expected sweep range on a comparable
A40. Budget **12 hours** remains appropriate, including HF delivery.

Likewise, the 6 h 33 min training reference supports **6.5–7.5 hours expected**,
including staging and delivery allowance; retain **8 hours** as the planning
budget. The new final optimizer/input-state save and its HF backup have not yet
been timed at full model scale, so the planning margin is retained. Training
backup stages previously took 25–41 seconds for adapters/reports only.

| Job | Expected duration | Planning estimate | Inner limit (+3 h) | Outer limit (+5 h beyond inner) |
|---|---|---:|---:|---:|
| Training, 12,000 updates | 6.5–7.5 h | 8 h | 11 h | 16 h |
| Sweep plus HF delivery, 800 trials | 9.5–11 h | 12 h | 15 h | 20 h |

No executable budget was reduced: the existing estimates are conservative and
consistent with observed times. Trial duration depends on behavior/early exits;
training duration depends on host/GPU contention. Starting the sweep after a
complete checkpoint 6,000 should normally avoid waits: previous 200-trial blocks
took about 2–2.6 hours while 2,000 training updates take about 1.0–1.1 hours.
The checkpoint wait guard remains in place for unusually slow training.

Sources: local `TIMINGS.md`, `stage_timings.jsonl` and `results.json` from
`panda-train-color-bowl-20261015-094416-4344d31f`,
`scripted_color_bowl_models_20261015_112315_46cfdb52` and
`scripted_color_bowl_continue_20261015_151409_140b3657`.
