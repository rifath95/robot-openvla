# Two-cube color selection and lifting

Profile: `inference/sweep_profiles/color_lift_500_1000.json`.
Training session: `panda-train-20261014-095425-1136`.
Prepared sweep: `scripted_color_lift_models_20261014_144240_51b35ee5`.

Run **160 trials**: checkpoints 500 and 1000 × four pairs × ten predeclared
layouts × both target instructions. Pairs are red/blue, red/green, blue/yellow,
purple/orange. Half the trials have no red cube. Color-to-position assignments
alternate evenly across layouts; each requested color is tested in a fresh
reset of the same starting state. Both checkpoints see the same layouts.
The instruction is `pick up the COLOUR cube.` Sampling is off. There is no
fine-tuning on these colors, nor on two-cube scenes, in this experiment.

Geometry and camera reuse the two-cube scene. The arm starts in the fixed
successful lift pose, gripper open, using the same contact grip configuration
and unchanged pose-holding controller. Cubes keep their fixed orientation and
size. Success requires the requested cube at least 5 cm above its initial
height with bilateral finger contact near the grasp point for a continuous
simulator second. Prediction wait time does not count. A verified lift of the
other cube is a wrong-cube outcome and ends that trial. Contact alone is not
success. Allow 50 actions per trial.

Reports show correct-only lift rate, wrong/both lift counts, no-lift failures,
conditional correct choice among trials with lifts, and per-pair/per-target
results. The conditional choice metric must be interpreted alongside overall
success; it excludes no-lift cases. Outcomes support functional transfer and
retained color grounding, not a claim about exclusive provenance of all
knowledge. Multiple requests in the same layout are paired observations.

Budget 60–90 minutes on A40; profile estimate 90 minutes, inner limit 150
minutes, outer 210 minutes. Require 4 GiB free on the volume and 60 GB container
disk. Stage the base and both adapters once on container disk, then load and
merge from those copies for all trials. Keep the source checkpoints 500/1000
available until staging finishes; do not run checkpoint backup/deletion first
unless you restore those checkpoints from HF before the sweep.

Use a separate A40 training-template pod on the same network volume, with
RunPod stop secret and `NVIDIA_DRIVER_CAPABILITIES=all` for EGL rendering. From
the local repository terminal, substitute its SSH address:

```bash
caffeinate -i .venv/bin/python -m cloud.cloud_sweep_session \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --resume outputs/sweeps/scripted_color_lift_models_20261014_144240_51b35ee5
```

After detached launch, the Mac may disconnect. MuJoCo physics/controller runs
on CPU, rendering and model predictions on GPU. Images, states, actions, logs
and reports stay on the network volume. Automatic stop is requested on
completion, failure or timeout; confirm stopped status in the portal.

## Transfer after simulation

Use the agreed two-stage workflow: no packing/upload on the simulator pod.
Have a HF write token ready before deployment. On a later cheaper transfer pod
on the same volume, clone/pull the repository and run:

```bash
/opt/openvla/bin/python -u -m cloud.archive_sweep \
  --sweep scripted_color_lift_models_20261014_144240_51b35ee5 \
  --repo Maxalderone/robot-openvla-simulator-results \
  --delete-after-verification
```

The transfer uploads, downloads/verifies the pinned archive, retains reports
and receipts, deletes source sweep and packed cache, then requests stopping.
Download/restore on the Mac later, without a running pod.

Local checks covered all 160 trial specifications, balanced color assignment,
requested versus wrong lift, lost grip and prediction waits, an actual physics
grasp/lift of the second cube, and real initialization/replay scene selection.
Model-driven success is measured by the cloud sweep; no cloud pod was launched
while preparing it. The bowl initialization fix remains deferred in
[the follow-up list](../reviews/PHASE4_PENDING_WORK.md).
