# Uninterrupted colored bowl training: 12,000 updates

Reviewed October 16, 2026. Session:
`panda-train-color-bowl-20261016-084536-c17080be`.

Training completed all 12,000 updates. The final resumable state was verified
locally, and the checkpoint/report backup was uploaded and verified on HF.
The separate 800-trial simulator sweep is still pending review; training metrics
alone cannot establish that this run improves closed-loop task success.

## Setup and completion

- Same 600-demonstration colored bowl dataset: 43,344 training action examples,
  red/blue/green/yellow, 200 single-cube and 400 two-cube demonstrations.
- Exact original checkpoint-zero LoRA restored from pinned HF revision
  `80884aea4ccfb77277cedcbf9b1710ea09bb5a0d`; the source receipt confirms
  439 zero B matrices and adapter SHA256
  `8d987cb02c6cd7888bcb52993d68999a32c2ea543177e4e104b3d791cdadf277`.
- Batch 8, accumulation 1, constant learning rate 0.0001, LoRA rank 32.
  One continuous optimizer run from update zero; no reset at 6,000.
- Metrics contain exactly 12,000 consecutive updates, with no missing indices.
  Completion reason: `requested_updates_completed`; no validation split.
- Checkpoints: 6,000, 8,000, 10,000 and 12,000. Only the final checkpoint
  includes optimizer, RNG and input-iterator state.

## Training metrics

Each row averages the 1,000 updates ending at that checkpoint. Accuracy is
teacher-forced action-token accuracy, not robot task success. L1 is the logged
teacher-forced action error, not free-running simulator error.

| Checkpoint | Mean training loss | Action-token accuracy | Mean L1 error |
|---|---:|---:|---:|
| 6,000 | 0.9887 | 69.61% | 0.04134 |
| 8,000 | 0.8774 | 72.20% | 0.03493 |
| 10,000 | 0.7832 | 75.11% | 0.03006 |
| 12,000 | 0.6100 | 79.87% | 0.02338 |

The loss plot shows a steep initial reduction followed by continuing improvement
and ordinary fluctuations, without a sustained divergence. The first 1,000
updates average loss 1.7678 and accuracy 56.25%.

The previous adapter-only continuation with a fresh optimizer finished at
loss 0.6219, accuracy 80.06% and L1 0.02265 over its last 1,000 updates.
This uninterrupted run has slightly lower loss but slightly worse accuracy/L1;
these are broadly similar training metrics, not evidence yet of better robotic
performance or of a causal benefit from preserving optimizer state. Initial
LoRA matches the original run; unrecorded RNG/batch-order differences remain.

## Timings and memory

- Training stage: 6 h 42 min 35 s; update work: 6 h 40 min 55 s.
- Checkpoint saves: 31.46 s total.
- HF checkpoint backup and verification: 68.59 s, including 25.36 s upload
  and 29.84 s verification.
- Recorded stages through final report backup: about 6 h 45 min total.
  This is inside the 8-hour planning budget, 10-hour inner and 12-hour outer.
- Peak reserved VRAM: 41.861 GiB on a 44.432 GiB GPU;
  minimum logged free VRAM: 1.645 GiB. The run reached completion without a
  logged training failure.

## Verified backup and actual storage

HF model repository: `Maxalderone/robot-openvla-checkpoints`.
Checkpoint verification receipt revision:
`da8b4ef08c17b4f8cfb162e7535a2f968b150205`.
Downloaded report snapshot revision:
`9d3230d730351b858e56d5a4b72bc57ae18f1328`.
The final worker summary confirms `training_complete`, `work_complete`,
`checkpoint_backup_verified` and `delivery_verified`. The receipt confirms
`volume_checkpoints_retained=true`.

Actual verified backup: **6.498 GB** (decimal; about 6.05 GiB).
Intermediate checkpoint folders are about 0.487 GB each. The final checkpoint
folder is **5.016 GB**, including **4.092 GB input-stream state** and
**0.437 GB optimizer/RNG state**. The earlier 3–5 GiB estimate was low;
the 8 GiB training reservation covered the observed backup. Future full-state
storage estimates should account for this measured input-stream checkpoint size.

## Local files

Training logs, metrics and plots:
`outputs/archives/hf-training/sessions/panda-train-color-bowl-20261016-084536-c17080be/`.
The smoothed plot is `runs/training/loss_smoothed.png` below that folder.

Final HF delivery reports, receipt and completed timings:
`outputs/archives/hf-training/training-reports/panda-train-color-bowl-20261016-084536-c17080be/`.
The session subtree contains an earlier pre-upload timing snapshot; use the
`training-reports` subtree for delivery completion and final timings.
The reports-only download deliberately excludes checkpoint payloads.
