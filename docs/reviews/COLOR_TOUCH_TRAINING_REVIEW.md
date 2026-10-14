# Red/blue touch training review — October 13, 2026

Training session: `panda-train-20261013-074359-896`.
Reviewed downloaded local reports only; no connection to or modification of the active simulator pod.

## Completion and configuration

Completed all **500 updates**, exit code 0, with `training_complete: true`, no missing artifacts and a completed final full validation. Four checkpoint metadata folders are present locally: 50, 100, 300 and 500. Adapter weights remain on the network volume.

The run started from pretrained OpenVLA with a fresh adapter (`initial_adapter: null`, `continuation_mode: fresh_base`). Dataset path: `/workspace/panda-training/training_color_touch_v1`. Microbatch **8**, accumulation **1**, effective batch **8**, fixed configuration policy, learning rate **0.0001**, LoRA rank **32**.

Training data: 34 demonstrations, 609 action examples (303 red-target and 306 blue-target). Validation: 8 demonstrations, 136 action examples. Test: 8 demonstrations, 142 actions, not used during training/validation.

500 updates expose 4,000 sampled training examples, about 6.57 training-set passes worth. This is exposure count, not a guarantee of exactly equal sampling of every example or color.

## Losses

| Update | Full validation loss | Examples | Complete |
|---|---:|---:|---|
| 0 | 10.848279 | 136 | Yes |
| 50 | **2.572086** | 136 | Yes |
| 100 | 2.864061 | 136 | Yes |
| 300 | 3.432725 | 136 | Yes |
| 500 | 3.545015 | 136 | Yes |

Each validation processes all 136 held-out validation actions; five validations total 680 validation example evaluations, repeatedly using the same split.

Final training minibatch: loss **0.000672**, teacher-forced action-token accuracy **1.0**, L1 diagnostic **0.0**. These are minibatch/teacher-forced metrics, not evidence of perfect simulator control or correct color selection.

| Updates | Mean training minibatch loss |
|---|---:|
| 1–50 | 2.841692 |
| 51–100 | 0.459197 |
| 101–200 | 0.099290 |
| 201–300 | 0.020573 |
| 301–400 | 0.007758 |
| 401–500 | 0.005349 |

Training loss falls toward zero while held-out validation loss rises after update 50. This indicates overfitting on the teacher-forced action-token objective. It does not determine which checkpoint best performs the actual task. In the previous single-cube experiment, later checkpoints had better task success despite higher validation loss; the pending paired red/blue simulator sweep is needed to assess this run.

## Timing and memory

- Full supervisor run from recorded start to finish: **1,169.16 seconds = 19 minutes 29 seconds**, including setup, training, validation and result preparation; excludes deployment and the later report-download window.
- Typical update interval away from validation/checkpoint boundaries: median **1.967 seconds**, mean **1.955 seconds**. Approximately **16 minutes 17 seconds** for 500 update intervals at that measured mean; this is an estimate of update-loop work rather than an isolated GPU kernel timer.
- Validation durations: approximately **18–19 seconds each**, about **93 seconds total**. Computed from update/validation elapsed-time records; baseline measured from loop start.
- A40 reported capacity: **44.432 GiB**.
- Peak PyTorch allocated: **38.516 GiB**; peak reserved: **40.906 GiB**.
- Logged free VRAM at final update: **2.857 GiB**. Batch 8 completed without a recorded out-of-memory failure.

The downloaded `status.json` is an `awaiting_sync` snapshot with stop not yet requested. It confirms training completion but does not independently prove the subsequent pod stop succeeded; verify that in RunPod separately.

## Plots and source reports

- [Original loss curve](../../outputs/training/phase3_training/panda-train-20261013-074359-896/runs/training/loss.png)
- [Smoothed loss curve](../../outputs/training/phase3_training/panda-train-20261013-074359-896/runs/training/loss_smoothed.png)
- [Per-update and validation measurements](../../outputs/training/phase3_training/panda-train-20261013-074359-896/runs/training/metrics.jsonl)
- [Completion snapshot](../../outputs/training/phase3_training/panda-train-20261013-074359-896/status.json)

All four trained checkpoints are included in the separately running scripted evaluation. Assess correct target selection, wrong-cube contact, clean control completion and paired-scene performance before selecting one for further work.
