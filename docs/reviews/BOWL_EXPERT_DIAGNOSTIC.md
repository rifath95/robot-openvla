# Compare the bowl policy with expert demonstration actions

Prepared and completed on 2026-10-12 for checkpoint_002000 from session
`panda-train-20261012-072437-856`. The live A40 evaluation completed all 52
examples. Reports were saved locally and archived on the network volume before
RunPod accepted stopping pod `dtzaz6pmxal2tg`; SSH then closed during shutdown.

## Completed results

| Metric | Independently generated action | Teacher-forced prediction |
| --- | ---: | ---: |
| Mean XYZ error against expert command | 2.983 mm | 1.201 mm |
| Mean normalized absolute error across seven components | 0.1037 | 0.0407 |
| Open/closed gripper agreement | 48/52 (92.3%) | 52/52 (100%) |

Both prompt-prefix and pixel-tensor equivalence checks passed on all 52 examples
on the deployed processor. Teacher-forced first-component X predictions matched
the generated X component in all 52 comparisons, as expected when neither has
earlier action tokens. No teacher-forced predictions used invalid action tokens.

The checkpoint produces nonzero horizontal movement on some demonstration
observations. Its predictions are not globally stuck at zero. On the four first-
lift examples, it generated upward movement and a closed gripper correctly;
three of four explicit close examples also commanded closure. Those results
contradict a claim that it cannot ever lift or close.

However, failures remain on familiar training inputs. Generated XYZ error is
7.948 mm during approach versus 3.595 mm under teacher forcing, and 6.989 mm
during transfer versus 2.975 mm. At an expert close step for train_001, generation
commands about 10 mm downward and an open gripper, while teacher forcing correctly
predicts closure. Rotational predictions in orient_for_grasp are poor under both
methods, including incorrect yaw direction. Thus the result supports both
imperfect conditional predictions and additional errors when generating without
expert preceding tokens; it is not solely a controller issue or a wholly broken
normalizer. It does not prove why the model learned those errors.

This is a balanced diagnostic subset of training images, not held-out accuracy,
and the sample count per stage is only four. Errors refer to requested commands,
not simulated motion or successful task completion. Do not treat the 92.3%
gripper agreement as a grasp success rate.

The 52 evaluations took 76.56 seconds, excluding merge/loading/source checking.
Local result directory: `outputs/matched_evaluation_20261012_131448_f2da1fde/`.
Cloud archive: `/workspace/panda-training/diagnostics/matched_evaluation_20261012_131448_f2da1fde/`.
The local directory also contains the server log and pod-stop receipt.

Next compare checkpoint_005000 on the same sample selection before another long
training run or changing the architecture. Separately address the demonstrated
collection/deployment controller mismatch. Neither more random trials nor the
teacher-forced loss alone can determine whether the newer checkpoint generates
better task-directed actions.

## Reproduce the test

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
