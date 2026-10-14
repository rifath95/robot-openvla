# Detached bowl training, color-placement sweep, and HF backup

This is the user's explicitly selected single-pod alternative to the default
[two-stage delivery workflow](TWO_STAGE_SWEEP_WORKFLOW.md). Run all stages on
one A40, then request stopping. No second transfer pod or connected Mac is required.
Prepared locally; the paid GPU end-to-end run has not yet been launched.

## Configuration and verified input

- Usual OpenVLA training image; A40; `NVIDIA_DRIVER_CAPABILITIES=all`.
- Container disk: **200 GB**, with at least **110 GiB actually free** after startup.
- Existing network volume at `/workspace`: at least **5 GiB free** for retained
  checkpoints/reports. No need to enlarge it for the sweep recordings.
- Existing global volume at `/workspace-global`: recovery only. Startup checks
  that it is a separate mounted filesystem and can write/read a regular file.
- Attach both volumes when creating the pod. Global capacity is usage-based.
- Dataset: `Maxalderone/robot-openvla-panda-place-bowl-300` at pinned revision
  `dd69138a8afb9b0484cf90f5ceab037629811807`.
- Bundle: `training_place_bowl_v2_300`; **300 training episodes / 21,906 examples**;
  no validation/test split. Its uploaded inventory and checksum manifest were
  checked against the verified local package before staging this workflow.
- Training configuration: `training/place_bowl_300_3000.json`.
- HF source/destination configuration: `inference/chain_profiles/place_bowl_300_1000_trials.json`.
- Frozen sweep definition and validation: `cloud/bowl_chain_plan.py`.

The dataset downloads directly to container disk and all upload checksums are
verified before training. The base model is copied once from
`/workspace/openvla-7b` and checked on container disk. All inference loads reuse
that copy; adapters are copied from fully saved checkpoints and merged in GPU
memory, without full merged-weight files. The code checkout is frozen to the
launch commit. Simulator dependencies and the exact Panda asset commit are
prepared inside the bounded worker, before training/model loading.

## Training

Fresh pretrained base; LoRA rank 32, dropout 0; learning rate `0.0001`; batch 8,
accumulation 1; **3,000 updates**. This processes 24,000 randomly sampled examples
(about 1.1 passes worth, not a guarantee that every example appears once).
Checkpoint updates: **0, 500, 1,000, 2,000, 3,000**. There is no teacher-forced
held-out validation because all demonstrations are training data. Loss plots,
TensorBoard events, per-update wall times, checkpoint timing and provenance are
persisted. The coordinator refuses to proceed unless all 3,000 updates and all
five complete checkpoints and plots are present.

## Sweep: 1,000 fresh-reset trials

Evaluate **base, checkpoint 500, 1,000, 2,000 and 3,000**. The pretrained baseline
uses this dataset's normalization from checkpoint 500 without its adapter weights.

| Group | Visible cube colors | Layouts | Requests per layout | Trials/model |
|---|---|---:|---:|---:|
| Single seen | Red | 20 | 1 | 20 |
| Single unseen | Black | 20 | 1 | 20 |
| Two, red present | Red + black | 20 | 2 | 40 |
| Two, red absent | Black + purple | 20 | 2 | 40 |
| Four, red present | Red + green + black + purple | 10 | 4 | 40 |
| Four, red absent | Green + black + purple + orange | 10 | 4 | 40 |
| **Total** | | | | **200 × 5 = 1,000** |

Blue bowl present in every trial. Instruction: `pick up the COLOR cube and place
it in the blue bowl.` Sampling off; at most **120 actions** per trial; fixed open
starting gripper and the existing camera/pose-holding controller. Single-cube
layouts use the old 20 fixed seeds (including previously invalid 1003/1008).
Multi-cube layouts use a deterministic jittered lattice and mirror the bowl side;
cubes stay separated and each cube/bowl distance is in the trained 0.18–0.32 m
range. Colors rotate among physical slots. Every model gets the same layouts and
every requested target starts from a fresh reset. These are paired observations,
not 1,000 statistically independent scenes. Single-cube unseen-color success
measures task transfer; multi-cube counterfactual requests test color selection.

All **100 distinct group/layout combinations** initialize and render before model
loading. Local checks also exercised a physical expert grasp/placement of the
second cube with two and four cubes present, plus matching recorded action replay.
Model-driven success remains to be measured on the pod.

Success requires the requested cube previously at least 8 cm above initial height,
then settled within 4.5 cm of bowl center and 4 mm of the floor resting height,
with the empty hand at least 8 cm away. This preserves the prior bowl success
criterion. Wrong placement, target lift without placement, wrong-cube lift,
no lift and controller/infrastructure failure remain separately visible.
A server/request failure stops the sweep rather than fabricating hundreds of failed
policy trials. Saved reports have direct Replay and Action Replay buttons via the
local replay launcher. New recordings include the initial complete physics state
and executed action labels, so action replay starts once and then executes them.

## Persistence and verified backup

| Location | Contents |
|---|---|
| Container disk | Dataset, base/adapter copies, simulator environment, detailed recordings, packed archives, temporary verification |
| Network volume | Existing base; **all five new checkpoints**; training/summary reports, progress, timings, receipts |
| HF model repo | `Maxalderone/robot-openvla-checkpoints`, under `sessions/SESSION` and final metadata under `chains/SESSION` |
| HF dataset repo | `Maxalderone/robot-openvla-simulator-results`, under the printed `scripted_bowl_chain_...` name |
| Global volume | `panda-chain-recovery/SESSION/...`, only after a failed/unfinished backup |

Packing and uploading run on container disk. Each HF file is checked by size
and SHA-256 against its pinned commit, with one verification-download cache at a
time. **No volume checkpoints or datasets are deleted.** Root reports and receipts
are copied to HF and verified as well. Shutdown acknowledgement itself remains on
the volume because it is produced after those uploads.

If the worker fails or times out, checkpoints stay on the main volume. Unverified
sweep data is packaged on container disk and copied as a few regular archives to
the global volume, then read back and hashed. Already complete packing can be
reused; incomplete, regenerable container packing may be removed to make recovery
space. Original recordings are retained until shutdown. A full recovery receipt
is written only after every copy verifies. Partial recovery is not called complete.
Recovery has **at most five hours, including a two-minute shutdown reserve**.
A sudden pod/storage failure can still lose data present only on container disk.

## Limits and stage timing

Conservative total estimate: **25 hours** (usual range 18–25 hours).
Inner buffer: **3 hours** → **28 hours**; outer buffer beyond inner: **5 hours**
→ **33 hours**. Both clocks start with the detached supervisor, before dataset
and simulator downloads, staging, training, sweep, packing, upload and verification.
Pod deployment/image pull and the small initial CLI authentication/code-checkout
steps occur before these clocks start.

Only the chain supervisor requests pod stopping. Inner timeout kills the worker
process group, including model/trial children, then attempts bounded recovery.
An independently detached outer watchdog uses container-local PID/status records,
so a stalled network/global filesystem or supervisor cannot prevent its API stop
attempt. It reserves the final 90 seconds of the outer budget for kill/stop retries.
RunPod API failures can still prevent stopping; confirm status in the portal.

`TIMINGS.md` and `stage_timings.jsonl` include stage starts/ends, durations, errors,
model-loading and checkpoint-staging durations. Training metrics time each update;
checkpoint saves have `checkpoint_timings.jsonl`; trial results include timestamps
and wall time. Archive part records and receipts separate packing, uploading and
verification time. `RUN_CHAIN_REPORT.md` links timings and the sweep results.
Interrupted timing writes are tolerated so they cannot block recovery.

## Launch only after local preparation is pushed

Have the saved **HF write token** ready; do not paste it into command text or chat.
Use the existing RunPod stop secret and attach both volumes. From the cloud terminal:

```bash
if [ -d /root/robot-openvla/.git ]; then
  git -C /root/robot-openvla pull --ff-only
else
  git clone https://github.com/rifath95/robot-openvla.git /root/robot-openvla
fi
cd /root/robot-openvla
/opt/openvla/bin/python -u -m cloud.bowl_chain --launch
```

Paste the HF write token at the hidden prompt. It is passed through pipes/in-memory
and never written into the job configuration, commands, logs or Git. Startup checks
write access to both private HF destinations before expensive work.
After `Detached long bowl chain started`, the Mac may disconnect. The CLI prints
`SESSION` and container log paths. Confirm setup in `supervisor.log` if needed.

Watch in the web terminal or over SSH, substituting the printed session:

```bash
tail -F /root/panda-chain-work/SESSION/console.log
cat /workspace/panda-training/sessions/SESSION/status.json
tail -F /workspace/panda-training/sessions/SESSION/training.log
tail -F /workspace/panda-training/sessions/SESSION/sweep.log
```

Later, checkpoints/reports remain under
`/workspace/panda-training/sessions/SESSION/runs/training/`; stage summaries remain
at the session root. Detailed recordings are retrieved from the verified HF archive
using the usual HF download and `data_tools.restore_sweep_archive` workflow. The
restorer supports this new plan and the larger archive size; ensure sufficient Mac
storage before downloading/restoring the whole sweep.
