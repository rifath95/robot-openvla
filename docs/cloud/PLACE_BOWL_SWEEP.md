# First Phase 4 bowl-placement sweep

Profile: `inference/sweep_profiles/place_bowl_50_all_models.json`.
Training session: `panda-train-20261014-104416-1195`, identified through the
previously authorized read-only volume connection. Its wrapper still reported
training at preparation time, so bounded checkpoint waiting is retained even
though the user reported completion.

The prepared local plan is
`outputs/sweeps/scripted_place_bowl_models_20261014_111236_f42bb9bf/`.
Run 100 trials: base and checkpoints 50, 100, 300, 500, each on twenty shared
unseen cube/bowl layouts, seeds 1000–1019. Both object positions were checked
against all fifty demonstrations. Starting pose, controller, blue interior,
bowl camera and instruction match training. Sampling is off. Allow 120 actions
per trial, stopping on success or existing controller/runtime guards.

Success uses the demonstration checks: prior cube lift of at least 8 cm,
settled cube within 4.5 cm of bowl center and 4 mm of the expected resting
floor height, and empty hand at least 8 cm from the cube. Grasping, lifting or
hovering above the bowl alone are failures. Success is checked in physics time,
not by giving prediction delays time to settle the simulator. The sweep report
includes placement rate, clean successes, failed trials, maximum lift, bowl
center distance and hand distance, plus direct local replay links after restore.

Estimate 1.5–2 hours on A40, extrapolated from the lift sweep and the longer
placement trajectory/action limit, not a measured bowl-model sweep duration.
The profile estimate is two hours; inner deadline three hours; outer deadline
four hours. A missing/partial checkpoint can wait up to 30 minutes, bounded
by these deadlines. Allow 4 GiB free on the volume and 60 GB container disk.

The base is staged once on container disk, with adapters added when needed.
Inference models load from those local copies and merge LoRA in memory.
MuJoCo physics/controller runs on CPU; EGL rendering and OpenVLA inference use
GPU. Launch a separate A40 training-template pod with the same network volume,
RunPod stop secret, and `NVIDIA_DRIVER_CAPABILITIES=all`.

From the local repository terminal, substitute its new SSH address:

```bash
caffeinate -i .venv/bin/python -m cloud.cloud_sweep_session \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --resume outputs/sweeps/scripted_place_bowl_models_20261014_111236_f42bb9bf
```

After the detached launch message, the Mac may disconnect. The simulator saves
results to the network volume and requests stopping on completion, failure or
timeout. Confirm stopped status. This separate pod and source snapshot do not
change the running lift sweep or stop another pod.

## Transfer after simulation

Keep packing/upload off the simulator GPU. Have a HF write token ready before
deployment. Later launch a cheaper transfer pod attached to the same volume,
clone/pull the repository, and run in its cloud terminal:

```bash
/opt/openvla/bin/python -u -m cloud.archive_sweep \
  --sweep scripted_place_bowl_models_20261014_111236_f42bb9bf \
  --repo Maxalderone/robot-openvla-simulator-results
```

The helper packs, uploads, downloads/verifies the pinned archive and requests
stop. Original results and packing cache remain until separately authorized
cleanup after successful verification. Download and restore on the Mac without
a running pod.

Local validation covered the 100-trial plan, unseen destination and cube
positions, cross-platform rounding consistency, lift-versus-placement success,
existing supervisor/checkpoint staging tests and real MuJoCo initialization
with matching replay geometry. GPU predictions for this task are evaluated in
the actual sweep. No simulator pod was launched as part of preparation.
