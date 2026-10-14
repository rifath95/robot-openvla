# Sweep of the 300-trajectory lift models

The prepared profile is `inference/sweep_profiles/lift_300_all_models.json`.
It selects training session `panda-train-20261014-095425-1136`, identified through
a user-authorized read-only check of the volume on October 14.

Run 140 trials: pretrained base plus checkpoints 50, 100, 300, 500, 750 and 1000,
each at twenty predeclared unseen positions (seeds 1000–1019). The first ten
match the previous pilot sweep, and the additional ten expand evaluation.
All positions were checked against the complete 300-demonstration manifest.
Sampling is off; each trial permits 50 actions. Lift success remains at least
5 cm of cube lift with bilateral physical contact near the grasp point for
one continuous simulator second; prediction wait time does not count.

Estimated runtime is 60–90 minutes on A40 before additional checkpoint waits.
The inner limit is 150 minutes, and the outer limit is 210 minutes. Allow at
least 4 GiB free for results and keep the existing 60 GB container disk setting.
GPU inference remains batch one. MuJoCo physics/controller runs on CPU and EGL
camera rendering on GPU; the Mac may disconnect after detached launch.

## Overlap with training

The simulator runs on its own pod, using the same network volume. The base
model and checkpoint 50 statistics are staged once to container disk. Each
later checkpoint is staged only when its model is next. It does not wait for
checkpoint 1000 before evaluating earlier models. Incremental staging preserves
the existing base copy; models load from container storage and LoRA merges in
memory through the existing connection server.

The worker verifies required JSON metadata and safetensors completeness before
copying any checkpoint. An incomplete save is retried for up to 30 minutes,
bounded by the cloud supervisor's process deadline. If training has ended and
the required checkpoint is absent or incomplete, the sweep fails explicitly.
It cannot load partially written weights. Completed trials stay saved on the
volume, and the sweep's pod is stopped on completion, failure or timeout.
The training pod is never stopped by the sweep supervisor.

Launch a separate A40 using the training template, same volume, RunPod stop
secret and `NVIDIA_DRIVER_CAPABILITIES=all`. From the local repository terminal:

```bash
caffeinate -i .venv/bin/python -m cloud.cloud_sweep_session \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --resume outputs/sweeps/scripted_lift_models_20261014_105736_941b29f8
```

A fresh plan can also be created with `--profile
inference/sweep_profiles/lift_300_all_models.json` instead of `--resume`; avoid
launching a duplicate experiment unintentionally.

## Delivery after simulation

Use the default two-stage delivery. The simulator only saves to the network
volume and stops; no packing/upload is performed on it. Have the HF write token
ready before deployment. After simulation, use a cheaper transfer pod on the
same volume, clone/pull the repository, and run:

```bash
/opt/openvla/bin/python -u -m cloud.archive_sweep \
  --sweep scripted_lift_models_20261014_105736_941b29f8 \
  --repo Maxalderone/robot-openvla-simulator-results
```

Enter the write token at the hidden prompt. The helper uploads, verifies the
pinned archive and requests stop. Source and packing-cache cleanup is separate
and should follow the verified receipt. Download and restore on the Mac later.

Local checks cover partial-save waiting, timeout and ended-training rejection,
incremental staging without recopying the base, existing worker/supervisor
behavior, and the 140-trial unseen-position plan. No sweep pod has been launched
while preparing this profile.
