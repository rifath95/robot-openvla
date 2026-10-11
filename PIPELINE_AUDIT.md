# Phase 3: training and inference consistency

The checkpoint-1000 rollout `remote_loop_20261011_021037_83d8c4a3` completed
45 actions before the user stopped it. It closed at action 27 while the grasp
point was approximately 42 mm horizontally from the cube center, then repeatedly
requested a closed gripper and almost zero movement. There were no IK rejections.

The local audit found an instruction-processing mismatch:

- The official training batch transform lowercases the instruction.
- Previous inference preserved uppercase `X` and `Y` in world-axis instructions.
- All 80 sampled expert-image prompts therefore had different tokens from training.

`openvla_prompt.py` now supplies the matching lowercase prompt to both the resident
cloud predictor and the standalone local predictor. Original request text is
retained in saved results. Server startup metadata records this convention.

After this correction, the audit verified exact prompt-token agreement for all
80 sampled images and all 11 distinct training instructions. Image tensors were
bit-for-bit identical across training and inference for all 80 samples. Saved
cloud action statistics also matched the training bundle. All 36 training episode
scene fingerprints matched the current scene.

Expert demonstrations closed with a median horizontal grasp-point offset of
0.110 mm and a maximum of 0.122 mm. Their correctly aligned close images differ
from the misaligned states reached in this rollout. These observations explain
why expert-image prediction results alone do not establish successful live grasping.

This audit does **not** establish that capitalization caused the failed rollout.
It loads only the cached processor/tokenizer, not the 7B weights or cloud adapter.
Previous generated-action evaluations used the old prompt and should be repeated
before treating them as final measurements. Training and its checkpoints need
no changes: the correction is to inference input preparation.

## Repeat the local audit

From the repository root, using the existing local training environment:

```bash
.venv-training/bin/python audit_openvla_pipeline.py \
  --rollout outputs/remote_loop_20261011_021037_83d8c4a3 \
  --processor .cache-openvla/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f \
  --output outputs/alignment_audit_repeat
```

The processor path is the locally cached snapshot recorded in
`.cache-openvla/model_revision.json`. Choose a new output directory each time.
The script writes `AUDIT_REPORT.md` and `audit.json`; it makes no cloud requests,
loads no model weights, and executes no robot movement. The completed initial
audit is under `outputs/alignment_audit_20261011_checkpoint1000_verified/`.

## Next cloud check

Push the code correction, then follow [EXPERT_EVALUATION.md](EXPERT_EVALUATION.md)
using the existing checkpoint `checkpoint_001000`. Repeat the same 80-image
diagnostic and the same live instruction and starting scene. Use a newly started
server so it loads the updated Python code. Reuse the training Docker image,
existing base weights, dataset and adapter; no Docker rebuild or retraining is
needed. Compare post-fix results with the saved pre-fix runs before choosing
additional training or dataset changes.
