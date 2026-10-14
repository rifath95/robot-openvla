# Lift pilot simulator sweep

Prepared for `panda-train-20261014-064135-2604`, trained on `training_lift_v1`.
The profile is `inference/sweep_profiles/lift_50_all_models.json`.

Run 50 trials: pretrained base plus checkpoints 50, 100, 300 and 500 on ten
predeclared positions absent from all 50 collected demonstrations. All models
share these positions and the unchanged current controller. The base model uses
checkpoint 50’s dataset action normalization, without its adapter. Sampling is
off. Each trial permits up to 50 predicted actions and stops on verified lift
success or existing controller/runtime limits.

Use the existing A40 training template with the same network volume, HF read
secret, RunPod stop secret, and `NVIDIA_DRIVER_CAPABILITIES=all` for EGL graphics.
Allow at least 2 GiB free on the volume. Estimated runtime is conservatively
60–90 minutes, depending on model loading and how many trials stop early. The
profile budgets 90 minutes plus one hour for the inner limit (150 minutes), then
another hour for the outer limit (210 minutes).

From the local repository terminal, substitute the current SSH address:

```bash
caffeinate -i .venv/bin/python -m cloud.cloud_sweep_session \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --profile inference/sweep_profiles/lift_50_all_models.json
```

The cloud supervisor runs MuJoCo and inference on the pod. After the detached
launch confirmation, the Mac may disconnect. Results remain on the volume,
and the supervisor requests stopping after completion, failure or timeout.
Confirm stopped status in RunPod.

Success uses the demonstration criterion: the cube stays at least 5 cm above
its initial height, with bilateral physical finger contact near the grasp point,
for one continuous second of simulation time. Continuing upward is allowed;
model prediction waits do not count. Reports include success, clean success,
maximum lift height, maximum held duration above the threshold, and failed
trials. Replay buttons use the common local playback launcher after restoration.

## Transfer after simulation

Keep packing/upload work off the simulator GPU. Before deployment, have a HF
write token ready for the later transfer pod, plus the RunPod stop secret.
After simulation stops, launch a cheaper transfer pod on the same volume.
Use the printed `scripted_lift_models_...` name in this cloud-terminal command:

```bash
if [ -d /root/robot-openvla/.git ]; then
  git -C /root/robot-openvla pull --ff-only
else
  git clone https://github.com/rifath95/robot-openvla.git /root/robot-openvla
fi
cd /root/robot-openvla
/opt/openvla/bin/python -u -m cloud.archive_sweep \
  --sweep YOUR_SCRIPTED_LIFT_MODELS_FOLDER \
  --repo Maxalderone/robot-openvla-simulator-results
```

Enter the HF write token at the hidden prompt. It packs, uploads, verifies, and
requests pod stop. Download and restore the archive on the Mac afterward;
see [the two-stage workflow](TWO_STAGE_SWEEP_WORKFLOW.md).

Local checks cover deterministic unseen positions, resume-plan consistency,
held-lift report metrics, the existing cloud worker/supervisor and archive path,
and real MuJoCo lift initialization with matching saved replay geometry. GPU
model inference and pod EGL are exercised when this sweep is launched.
