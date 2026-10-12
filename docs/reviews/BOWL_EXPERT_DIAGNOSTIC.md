# Compare the bowl policy with expert demonstration actions

Prepared locally on 2026-10-12 for checkpoint_002000 from session
`panda-train-20261012-072437-856`. No model predictions or cloud computation
have run yet for this test.

The existing matched diagnostic uses real before-images, original instructions,
and accepted action labels from the verified bowl training bundle. Four examples
from each of 13 stage groups give 52 samples. Selection includes positive and
negative X/Y movement, grasp, first lift, transfer, and drop; 20 targets have a
closed gripper. Stage information is used for selection/reporting, not supplied
to the model. This is a diagnostic selection, not a representative success rate.

Local preparation checked all 22,311 training source examples against bundle
provenance. Using the actual cached OpenVLA processor, all 52 training-transform
prompt prefixes and image tensors exactly matched deployment preprocessing.
Saved plan and preprocessing checks:
`outputs/matched_evaluation_20261012_125924_ffda249d/`.

## Start the cloud server

Launch the training-template GPU pod with the existing network volume containing
the untouched base model and adapter checkpoint. On the Mac, use its current SSH:

```bash
caffeinate -i .venv/bin/python -m cloud.cloud_session \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519' \
  --adapter /workspace/panda-training/sessions/panda-train-20261012-072437-856/runs/training/checkpoint_002000 \
  --setup-only --server-runtime-seconds 21600
```

Leave that terminal running to provide the SSH tunnel. No simulator opens.

## Run the comparison in another Mac terminal

```bash
caffeinate -i .venv/bin/python -m analysis.evaluate_expert_actions \
  --source-manifest datasets/panda_pick_bowl/manifest.json \
  --dataset datasets/panda_pick_bowl/training_bowl_v1 \
  --expected-adapter /workspace/panda-training/sessions/panda-train-20261012-072437-856/runs/training/checkpoint_002000 \
  --matched --per-group 4
```

For each input, the server first generates an action without receiving the expert
label as a model input. Then it evaluates the same example with teacher forcing,
verifying prompt/image equivalence on the deployed processor too. Reports compare
both actions to the expert in physical and normalized units, and record token
accuracy, timings, checkpoint provenance, image hashes, and per-stage results.

Outputs are saved directly on the Mac under `outputs/matched_evaluation_*`,
including `MATCHED_REPORT.md`, `EVALUATION_REPORT.md`, `predictions.jsonl` and
`summary.json`. Completed samples are retained on interruption. No arm moves and
no adapter training or modification occurs. Keep the Mac/tunnel connected for
the evaluation, then stop the pod when finished. Disconnecting the launcher alone
leaves the model server running.

Good teacher-forced results with poor generated actions suggest a gap between
conditional token prediction and generation. Poor results on both require further
investigation of learning and deployment. Neither result by itself identifies
the root cause. Model merging/weight loading still needs investigation if the
metadata and inputs match but predictions do not.
