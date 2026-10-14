# Scripted red/blue touch evaluation

Prepared locally while red/blue training runs. No cloud connection, model or controller change is needed to prepare this workflow.

Use the separate entry point `cloud.scripted_color_touch_session`. The original interactive `cloud.cloud_session` and single-cube scripted defaults are preserved.

## Default evaluation

- Checkpoints **50, 100, 300 and 500**.
- **10 unseen two-cube scenes**, seeds 1000–1009.
- Both `touch the red cube.` and `touch the blue cube.` from exactly the same initial scene, reset once per trial.
- **80 trials total**, with sampling off, no viewer and no Space presses.
- Up to 50 actions and 180 active seconds per trial.
- Same camera, fixed initial arm pose, closed gripper, cube orientations and position bounds as the red/blue dataset.
- Each proposed object position is checked against the local dataset manifest; collected positions are rejected. The scene generator fixes the positions before evaluation and never substitutes easier positions after failures.

Both instructions share one loaded model per checkpoint. Different checkpoints are merged and loaded separately. The new task uses the existing action decoding, IK and physics execution; privileged coordinates are used for scene setup and evaluation, not supplied as extra model inputs.

## Measurements

Contact is checked every 20 physics steps (0.04 simulated seconds), retaining brief correct- or wrong-cube contacts. Each action keeps the same 200-step physics duration. The trial ends after the action containing contact, or an existing tracking/IK/connection/time limit. This can detect contact earlier than the dataset's end-of-action check.

The summary records correct-only contact, wrong-cube contact, both-cube contact, no contact and unmeasured trials. It reports clean correct completion separately from contact accompanied by a tracking error. A clean process completion after touching the wrong cube is still a **task failure**. Wrong contact is never counted as success, even if correct contact also occurs.

The report includes per-checkpoint/per-color results and the number of scenes where **both instructions** selected their correct target. That paired measure helps detect always choosing one color, side or nearest object. Camera images do not include the viewer instruction overlay.

## Launch after checkpoints are ready

Use a dedicated inference pod with the training image, current network volume and RunPod stop secret. If training is still active, use a second pod. The launcher checks for active training processes before loading a model or arming automatic stop and rejects that pod without stopping it. Shared-volume reads may still slow concurrent training.

Run from the **Mac repository terminal**, substituting the inference pod's current SSH address:

```bash
caffeinate -i .venv/bin/python -m cloud.scripted_color_touch_session \
  --ssh 'ssh root@HOST -p PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --training-session panda-train-20261013-074359-896
```

For an early comparison while later checkpoints are still being trained, explicitly request only complete checkpoints, for example `--checkpoints 50 100`. The launcher does not wait for missing checkpoints. `--scene-seeds` accepts an explicit seed list >=50; using 20 seeds gives 160 trials across all four checkpoints. Keep the Mac awake and connected throughout.

## Reports and shutdown

Results are saved under `outputs/sweeps/scripted_color_touch_<timestamp>_<id>/`, including `REPORT.md`, `results.json`, elapsed evaluation time, logs, images, prediction settings, executed actions, states and replay launchers. Compact reports are copied to `/workspace/panda-training/scripted-evaluations/<session>/` before requesting the supplied inference pod's stop. Detailed images/actions remain on the Mac.

Cleanup also runs after ordinary errors or Ctrl+C. This connected workflow cannot guarantee shutdown after lost connectivity, forced termination or API failure; confirm Stopped in RunPod. The active-training guard is a launch-time check, so do not start a new training job on the inference pod during the sweep.

An optional interactive trial uses the same task:

```bash
caffeinate -i .venv/bin/python -m cloud.cloud_session \
  --ssh 'ssh root@HOST -p PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --adapter /workspace/panda-training/sessions/panda-train-20261013-074359-896/runs/training/checkpoint_000300 \
  --task color_touch --target-color blue --scene-seed 1000 --steps 50
```

That interactive command starts paused, displays the instruction top-right, and keeps the model loaded when the local session ends. It does not automatically stop pod billing.

## Local verification

Both red- and blue-target expert sequences were fed through the new inference loop without a cloud model. Every captured input PNG matched the demonstration; maximum qpos difference was 3.109e-15. The evaluator correctly measured the respective targets. This checks the simulator pipeline, not the trained model's success rate. The verification artifacts are under `outputs/diagnostics/color_touch_inference_pipeline_check/`.
