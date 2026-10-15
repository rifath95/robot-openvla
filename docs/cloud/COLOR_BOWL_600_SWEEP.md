# Prepared 800-trial colored bowl sweep and unattended HF delivery

This is the user's explicitly selected single-pod alternative to the default
[two-stage sweep workflow](TWO_STAGE_SWEEP_WORKFLOW.md). Run headless inference
and MuJoCo on the GPU pod, then pack/upload/verify on that pod and request stopping.
No connected Mac is required after detached startup. This specific GPU run has
not yet been deployed.

## Frozen comparison

Evaluate checkpoints **1,000, 2,000, 4,000 and 6,000** from the 600-demonstration
colored bowl training run, with **200 trials each: 800 new trials**.

| Scene group | Cubes | Layouts | Targets per reset layout | Trials/checkpoint |
|---|---|---:|---:|---:|
| Single red | Red | 20 | 1 | 20 |
| Single black | Black | 20 | 1 | 20 |
| Two, red present | Red + black | 20 | 2 | 40 |
| Two, red absent | Black + purple | 20 | 2 | 40 |
| Four, red present | Red + green + black + purple | 10 | 4 | 40 |
| Four, red absent | Green + black + purple + orange | 10 | 4 | 40 |

Each target request starts with a fresh reset. Sampling is **off**, with at most
**120 actions** and a 900-second active-trial runtime limit (1,200-second process
backstop). Targets, slot-color rotations, camera, controller, bowl and physical
success criteria match the previous 300-demo sweep. All **100 group/layout
combinations** are initialized and rendered before the first model load.
Coordinates are frozen from the actual previous cloud plan, including its exact
serialized float values, in `inference/sweep_profiles/color_place_bowl_600.json`.
A hash and independent seeded-layout validation reject changes.

Requested-cube success requires prior lift of at least 8 cm, settled placement
within 4.5 cm of bowl center and 4 mm of its resting height, followed by hand
separation of at least 8 cm. Wrong/both placements, partial lifts, no lift,
controller failures and infrastructure failures are reported separately. Measured
controller failures remain trials; server/unmeasured failures stop the sweep
rather than fabricating subsequent failures.

Training colors are red, blue, green and yellow; black, purple and orange are
unseen training colors. Group names preserve the old red-presence labels.
The old red-only dataset and the new mixture differ in size, content, updates
and normalization, so this compares complete training pipelines rather than
isolating a single causal change.

The **200 pretrained-base results are historical references**, not rerun here:
0/200 placements and no qualifying lifts, from
`scripted_bowl_chain_20261014_174530_4acde087`. They used that 300-demo dataset's
checkpoint-500 action normalization. Do not describe them as a fresh baseline
with the new 600-demo statistics. Their source-row hash and provenance are in
the frozen profile and generated report; base playback needs the old archive.

## Source checkpoint selection

On the mounted network volume, automatically find the single matching
`panda-train-color-bowl-*` session whose metadata identifies:

- `training/color_place_bowl_600_6000.json`, with `training_only=true`;
- `Maxalderone/robot-openvla-panda-color-place-bowl` at revision
  `0bf1faf4036cbfd5e3c7282111f1d9a54b2d020f`;
- 600 trajectories / 43,344 examples and the verified prepared checksum manifest.

Ambiguous matches require `--training-session NAME`; the launcher never guesses
between different training runs. The selection is recorded in
`source_training.json` before expensive model work.

Checkpoints are handled in order. Readiness checks require the completed adapter
header/size and the final training metadata. A missing/incomplete checkpoint can
wait **up to four hours**; a failed/finished source session without the checkpoint
ends that wait. Each checkpoint must also name the expected dataset revision,
checksum manifest and update number. Its copy is checked again and file hashes
are compared before loading. No source checkpoint is modified or removed.

The base is copied/verified once from `/workspace/openvla-7b` to container disk.
Each adapter is staged once from its complete volume checkpoint. The model stays
resident for all 200 trials of a checkpoint; LoRA merges in GPU memory, then the
server is closed before loading the next checkpoint. Evaluation is sequential,
batch 1. No training dataset download or training occurs on this pod.

## Pod and storage

Use the usual A40 **training template**, with:

- **200 GB container disk**, at least **110 GiB actually free** after startup;
- existing network volume mounted at `/workspace`, at least **1 GiB free** for
  compact reports and receipts;
- existing global recovery volume mounted at `/workspace-global`;
- `NVIDIA_DRIVER_CAPABILITIES=all` and the usual RunPod stop secret;
- saved **HF write token**, with access to the private simulator-results dataset.

Detailed images, actions, states, logs, packed parts and verification cache stay
on container disk. Compact `results.json`/`REPORT.md`, timings, source identity
and receipts stay on the network volume. Simulator dependencies/assets are set
up inside the guarded job; no image rebuild is required.

The HF destination is `Maxalderone/robot-openvla-simulator-results`, under the
printed `scripted_color_bowl_models_...` folder. Allow tens of GiB for results
(roughly **20–35 GiB**, depending on completed steps). Ensure capacity before
upload; the sweep may finish sooner than a day. Download/verify the preceding
archive before removing its HF copy, and confirm quota space was actually freed.
An upload failure cannot be treated as successful delivery.

## Limits, backup and failure handling

**20-hour estimate + 3-hour buffer = 23-hour inner deadline.**
**Inner + 5-hour buffer = 28-hour independent outer deadline.**

These include simulator setup, staging, waits for future checkpoints, all trials,
packing, upload and verification. Pod boot/image pull and initial small code/auth
checks precede detachment. The 20-hour estimate is a budget, not a promise that
upload will wait until tomorrow.

After 800 trials finish, archive parts are packed on container disk, uploaded and
verified against a pinned HF commit by size and SHA-256, one verification cache
at a time. Final source/job metadata and timings are added to the portable archive
manifest and separately verified; the existing archive parts are not reuploaded.
All timing stages and per-trial durations are recorded. Reports have direct
Replay/Action Replay buttons after local restoration.

The independently detached watchdog can kill compute and attempt pod stopping
if the main supervisor stalls. On failure/inner timeout, unverified partial or
complete sweep recordings are packed and copied as regular files to the global
volume, then read back and hashed, within a bounded recovery window of at most
five hours minus shutdown reserve. Compact receipts identify verified recovery.
Already verified HF recordings are not recopied after a small final-report error.
No checkpoints or existing volume datasets are deleted. A sudden pod/storage
failure can still lose data present only on container disk. API/network errors
can prevent stopping; confirm **Stopped** in RunPod.

## Launch when ready

Run in the new pod's **cloud terminal**:

```bash
if [ -d /root/robot-openvla/.git ]; then
  git -C /root/robot-openvla pull --ff-only
else
  git clone https://github.com/rifath95/robot-openvla.git /root/robot-openvla
fi
cd /root/robot-openvla
/opt/openvla/bin/python -u -m cloud.color_bowl_sweep --launch
```

Enter the saved HF write token at the hidden prompt. It travels only through
memory/private pipes, not command arguments, code, logs or saved configurations.
After **Detached 800-trial colored bowl sweep started**, the Mac may disconnect.
The code is frozen to the launch commit; it does not pull again mid-sweep.

The launcher prints `SESSION` and both paths. In the cloud/web terminal:

```bash
tail -F /root/panda-chain-work/SESSION/console.log
# After setup, detailed trial progress:
tail -F /workspace/panda-training/reports/sweep-sessions/SESSION/sweep.log
```

Successful delivery is recorded in `sweep_hf_receipt.json`. Later, download from
HF and use `data_tools.restore_sweep_archive`; the restorer validates the new
800-trial inventory and restores timings/replay links into `outputs/sweeps/`.
