# Compare teacher forcing and normal generation

Use the same 80 saved expert images as the previous quick evaluation, including
approach corrections and first-lift examples. The diagnostic predicts each
seven-component action in two ways:

- Normal generation sees the image/instruction and generates seven tokens itself.
  It runs first and never receives the expert action.
- Teacher forcing conditions each component on the preceding correct action
  tokens within this action. Future tokens remain causally masked. No history
  of earlier robot steps is supplied.

Neither updates weights or moves the robot. Both use the same resident merged
model, image, lowercase prompt prefix and Panda normalization. The server checks
that image tensors and prompt prefixes match before reporting paired metrics.

## Local preparation

From the repository root with the generated dataset present:

```bash
.venv/bin/python evaluate_expert_actions.py --matched --dry-run \
  --expected-adapter /workspace/panda-training/sessions/panda-train-20261010-233038-932/runs/training/checkpoint_001000
```

This verifies the source images/labels and saves the plan without contacting a
server. Tests use fake model outputs:

```bash
.venv-training/bin/python -m unittest test_matched_diagnostic test_expert_evaluation test_remote_inference -v
```

The real cached processor and full diagnostic input path also passed on all
80 selected images with synthetic oracle logits. This verifies preparation and
token alignment, not model performance. Record:
`outputs/matched_diagnostic_local_check/checks.json`.

## Start the cloud model once

Resume or launch the existing **training template**, with
`robot-openvla-network-volume-A40` mounted at `/workspace`. Existing base weights
and checkpoint are reused. No Docker rebuild or retraining is needed.

In the first **local Mac terminal**, run:

```bash
.venv/bin/python cloud_session.py \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519' \
  --adapter /workspace/panda-training/sessions/panda-train-20261010-233038-932/runs/training/checkpoint_001000 \
  --setup-only
```

Get `POD_IP` and `POD_PORT` from **RunPod → Connect → SSH over exposed TCP**.
The launcher uses this Mac's configured `id_ed25519_maxalderone` key, updates
cloud code, merges the adapter on container disk and loads the model. Wait for
**Server and tunnel ready** and keep this terminal running. Use a newly started
server so the diagnostic code is loaded.

In the second **local Mac terminal**, run:

```bash
.venv/bin/python evaluate_expert_actions.py --matched \
  --expected-adapter /workspace/panda-training/sessions/panda-train-20261010-233038-932/runs/training/checkpoint_001000
```

Stale server code, incorrect adapter provenance or mismatched normalization are
rejected before evaluation. Completed records and partial reports survive a
later request failure or interruption.

## Saved results

Results are under `outputs/matched_evaluation_*`:

| File | Contents |
|---|---|
| `MATCHED_REPORT.md` | Generated versus teacher-forced comparison by group |
| `matched_summary.json` | Component errors, gripper/lift agreement, teacher accuracy and loss |
| `predictions.jsonl` | Both actions, teacher predicted/target tokens, hashes and timings |
| `server_health.json` | GPU/VRAM, loading, volume, adapter identity and software metadata |
| `plan.json` | Selected examples |
| `timing.json` | Completed diagnostic elapsed time |

`EVALUATION_REPORT.md` and `summary.json` still describe normal generation only.

Both methods' command errors use the same original continuous labels and
normalization bounds. Teacher `normalized_l1_token_targets` additionally follows
training's decoded-bin-center metric. Cross entropy includes the stop token;
accuracy covers seven action tokens only. Argmax uses the full vocabulary.
Non-action predictions are flagged because official decoding clips them to bins.

Good teacher forcing with poor generation suggests difficulty without correct
preceding action components. Errors in both show these expert inputs are not
reliably reproduced. Neither identifies the root cause or guarantees that more
training will solve the task. The balanced training-image subset is not a
held-out task-success rate. Merged checkpoint numerics can differ from unmerged
LoRA training; this does not exactly reproduce old sampled training metrics.

When done, **Ctrl+C in the first terminal**, then **stop the pod in RunPod**,
unless continuing another immediate check. Neither the client nor server timer
stops GPU billing.
