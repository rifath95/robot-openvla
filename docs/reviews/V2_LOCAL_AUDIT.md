# Local audit after Panda v2 fine-tuning

The 5,000-update run completed, but simulator trials using checkpoints 3,000 and
5,000 did not grasp or lift the cube. Both generated open-gripper commands for
all 100 actions. This audit inspects saved files without loading the 7B weights,
generating predictions, accessing a pod or moving a robot.

## Checks and findings

- All 250 source episode fingerprints match the prepared v2 bundle.
- Recomputed training-only action statistics match the bundle and the statistics
  recorded by both inference servers. Dataset/rollout action contracts match.
- All 250 source scene fingerprints match the current scene XML and Panda XML.
- Training and inference pixel tensors and instruction-prefix tokens match on
  72 images spanning nine stages and train/validation splits.
- All 151 distinct training instructions produce matching prompt tokenization.
- The controller mapping preserves recorded accepted-action labels to floating
  point precision. Closing/lifting examples are present: 6,225 of 13,812 training
  targets command the gripper closed; 176 are explicit close-stage transitions.
- The short inference instruction `pick up the red cube` is not an exact training
  instruction. Demonstrations use longer pick-and-place instructions specifying
  a destination. This difference is worth a controlled test, not proof of cause.

The script compares selected saved rollout poses with position-nearest training
approach/grasp observations relative to the cube. Orientation differences are
reported separately. These observations are not identical states or images;
their labels must not be treated as ground-truth actions for the rollout.

## Reproduce

Run in the local workspace with the training virtual environment. Use a new
output folder: the audit refuses to overwrite existing results. The processor
path below is the existing local cache; no weights are loaded or downloaded.

```bash
.venv-training/bin/python -m analysis.audit_panda_v2 \
  --processor .cache-openvla/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f \
  --rollout outputs/remote_loop_20261011_130030_90592976 \
  --rollout outputs/remote_loop_20261011_130522_3943abad \
  --output outputs/local_v2_audit_new
```

Outputs: `audit.json`, `AUDIT_REPORT.md` and `scene_comparison.png`. The actual
review is also saved beside the downloaded training results under
`outputs/phase3_training/panda-train-20261011-090041-1584/local_audit_v2/`.

## Next diagnostic

Before changing training, generate normal autoregressive predictions on exact
saved train/validation examples with their original instructions. Compare
approach, close and first-lift actions against their recorded targets. On the
same images, compare the short pickup instruction against the full recorded
instruction. This requires a GPU pod and has not been run for these v2 adapters.
Earlier matched evaluations used older checkpoints and the original v1 dataset.

These checks establish consistency in the inspected local pipeline and saved
metadata. They do not prove that the model predicts correct actions or isolate
the reason for the stalled simulator trials.
