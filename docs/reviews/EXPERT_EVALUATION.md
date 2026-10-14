# Check expert images through the cloud inference pipeline

This diagnostic uses saved **before-action PNGs**, original demonstration
instructions, and recorded accepted expert commands. It sends them through the
same HTTP `/predict` endpoint used by the live simulator. It does not open a
viewer, advance physics, modify datasets, or perform training updates.

The first check targets checkpoint 500. It tests whether the model requests
upward movement on expert scenes where the next action should begin a lift.

## 1. Check the data locally before renting a pod

From a **Mac terminal in this repository**:

```bash
.venv/bin/python -m analysis.evaluate_expert_actions --dry-run
```

This verifies the source demonstrations against the prepared dataset manifest,
including their fingerprints. It saves a plan and checks the selected examples
without contacting a cloud server. The current dataset sources must be present
locally; GitHub does not contain these generated images.

The default selects eight examples from each diagnostic group, spread
deterministically across the source order. The first action of each lift is a
separate group. Stages are only used for selection/reporting; they are **not**
given to the model, and training sampling is unchanged.

## 2. Start the cloud server and tunnel

Push the code first. Deploy using the **training template**, with the existing
`robot-openvla-network-volume-A40` network volume mounted at `/workspace`.
No Docker rebuild or dataset re-upload is required. Wait until SSH is reachable.

Copy the pod's **Connect → SSH over exposed TCP** command. Substitute its IP
and port below, in the **first Mac terminal**:

```bash
.venv/bin/python cloud_session.py \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519' \
  --adapter /workspace/panda-training/sessions/panda-train-20261010-233038-932/runs/training/checkpoint_000500 \
  --setup-only
```

The launcher uses this Mac's actual private key, updates cloud code, merges the
adapter on container disk, loads the model, and opens an SSH tunnel on local
port 8000. Wait for **Server and tunnel ready** and keep this terminal running.
No simulator opens. Merging/loading still take time and the pod bills.

## 3. Evaluate in a second Mac terminal

From the same repository:

```bash
.venv/bin/python -m analysis.evaluate_expert_actions
```

The client verifies checkpoint provenance and action normalization before
predicting. The quick check currently makes 80 predictions. Each uses its own
saved image and original instruction; it does not execute the returned action.
Progress prints expert and predicted vertical movement.

The printed output folder contains:

- `EVALUATION_REPORT.md`: stage/group table and interpretation notes.
- `predictions.jsonl`: every evaluated label, prediction, instruction, image
  path/hash, error, and request/server timing.
- `summary.json`: aggregate and per-component errors.
- `server_health.json`: model loading, GPU, volume, normalization, and checkpoint.
- `plan.json`, `dataset_statistics.json`: sample selection and training bounds.
- `timing.json`: completed evaluation wall time.

Completed predictions and a partial report survive an interrupted/failed request.
These output files are saved on the Mac under `outputs/diagnostics/expert_evaluation_*`.

## Optional: complete split or another checkpoint

For all 2,595 training examples, rather than the diagnostic subset:

```bash
.venv/bin/python -m analysis.evaluate_expert_actions --selection all
```

Start the launcher with `--server-runtime-seconds 7200` for this longer check.
That timer starts after readiness; it does **not** stop the rented pod. Do not
leave the pod unattended assuming this evaluation will stop GPU billing.

Use `--split validation` to evaluate validation images instead of training
images. Reserve the test split for final evaluation.

To evaluate checkpoint 1000, change the launcher adapter path's final component
to `checkpoint_001000`, and pass that **same complete path** to this client:

```bash
.venv/bin/python -m analysis.evaluate_expert_actions \
  --expected-adapter /workspace/panda-training/sessions/panda-train-20261010-233038-932/runs/training/checkpoint_001000
```

## Interpretation and stopping

Use the latest server code: inference now lowercases instructions to match the
official training transform. Earlier evaluations preserved uppercase world X/Y
and need repeating. See [PIPELINE_AUDIT.md](PIPELINE_AUDIT.md) for the local checks.

These are **generated-action errors**, not the teacher-forced cross-entropy,
token accuracy, or normalized L1 printed during training. Teacher forcing gives
the model preceding expert action tokens; normal inference generates those
tokens itself. For the paired teacher-forced/generated-action diagnostic, use
[MATCHED_EVALUATION.md](MATCHED_EVALUATION.md). The default evaluation here is
generation only and does not claim to reproduce the training metrics.

The balanced diagnostic sample's overall average is not an estimate of the
full dataset average. Group results reveal specific difficulties; a full-split
check uses the actual frequency of examples. Neither proves rollout success.

If expert lift images produce upward predictions but a rollout does not, examine
the states reached during execution. If those images also fail, investigate
learning and preprocessing/normalization before assuming a history limitation.

When done, press **Ctrl+C in the first terminal** to stop the model server and
tunnel. Then **stop the pod in RunPod** to stop GPU billing. The network volume
retains the original base model and adapters.
