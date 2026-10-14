# Touch-cube training and simulator review — 2026-10-13

Reviewed the downloaded 500-update training run and all 30 saved touch-scene trials from this session. Reviewed predicted actions, actual action execution, contact measurements, state continuity, generation settings, checkpoint provenance, and a start/middle/end image montage. No cloud access or code changes were made.

## Training

Session: `panda-train-20261013-034928-1786`. Completed 500 updates, batch 8, accumulation 1, learning rate 0.0001. Total setup/training/report preparation: 23 minutes 40 seconds. The final full validation covered all 148 validation examples. All 15 checkpoint metadata directories were downloaded; adapter weights remain on the network volume.

Final training loss: 0.001122. Lowest validation loss: 2.511286 at update 40; final validation loss: 4.052724. This divergence indicates overfitting on the teacher-forced action-token objective. It does not, by itself, establish which checkpoint has the best closed-loop task performance.

[Original curve](../../outputs/training/phase3_training/panda-train-20261013-034928-1786/runs/training/loss.png) · [Smoothed curve](../../outputs/training/phase3_training/panda-train-20261013-034928-1786/runs/training/loss_smoothed.png)

## Same-scene comparison

Instruction: `touch the red cube.`. Seeds 0, 1, 2 reproduce three training demonstration locations. These counts are small observed samples, not population success-rate estimates.

| Checkpoint | Sampling | Finger–cube contact | Clean completion | Actions to contact |
|---|---|---:|---:|---|
| 0 | Off | 0/3 | 0/3 | — |
| 10 | Off | 0/3 | 0/3 | — |
| 50 | Off | 2/3 | 0/3 | 22, 22 |
| 50 | On, temperature 1, top-p 1 | 1/3 | 1/3 | 18 |
| 100 | Off | 3/3 | 3/3 | 18, 19, 17 |
| 300 | Off | 3/3 | 3/3 | 19, 17, 17 |
| 500 | Off | 3/3 | 3/3 | 19, 17, 17 |

Later fine-tuned checkpoints learned this touch task in the tested scenes. Sampling is not necessary for these successes. The three sampling trials at checkpoint 50 do not show an improvement over greedy decoding; more trials would be needed for a reliable comparison.

## New cube positions

All following trials used greedy decoding and the trained instruction. Seeds are new reproducible positions, rather than exact collected demonstration locations.

| Checkpoint | Seeds | Clean successes | Actions to contact |
|---|---|---:|---|
| 50 | 3535680, 34519768 | 2/2 | 16, 18 |
| 300 | 64322, 999, 9847 | 3/3 | 19, 21, 15 |
| 500 | 999 | 1/1 | 21 |

A separate checkpoint-300 attempt at seed 999 was manually stopped after four completed steps; its completed retry is listed above. Do not count the interrupted attempt as evidence of model failure. These positive results support some generalization within the existing cube-position bounds, with the same starting arm pose, camera and scene. They do not establish robustness outside those conditions.

## Tracking failures and success reporting

Every Failed trial ended with `Action executed but tracking failed`, meaning the realized control-point position differed from the controller target by more than 5 mm. Failure checks run after physics execution. This is distinct from an API failure or a model emitting no motion.

Two checkpoint-50 greedy trials (seeds 1 and 2) have verified positive-force finger–cube contact in step 22, but that same action exceeds the tracking threshold (8.356 mm and 7.220 mm). The loop checks tracking before accepting task success, so their overall phase is Failed even though contact was achieved. Their summary completed-step count excludes the failed final action; the final execution JSON preserves its motion and contact. Thus contact attainment and clean completion must be reported separately.

The failed seed-0 trials visually descend beside the cube toward the table. Later checkpoints instead reach the cube. The tracking error alone does not identify which collision caused it; this review does not claim every tracking failure is a table collision.

Across all 30 trials, maximum difference between one action's saved ending qpos and the next action's saved starting qpos is exactly zero. The simulator is not resetting the arm between actions. Server-returned sampling modes match the requested mode in every recorded prediction.

## Other instructions

Checkpoint 300 with `grip the red cube.` touched the cube in 19 actions. Because this is still the touch scene, completion means finger contact, not successful gripping or lifting.

Checkpoint 300 with `move up` instead moved down and failed at action 22. The touch action statistics have negative-only vertical ranges (q01 −9.927 mm to q99 −6.109 mm), zero rotation ranges, and fixed closed-gripper labels. Under these statistics, normal action-bin decoding cannot express an upward movement. This single-task fine-tune and its action normalization should not be interpreted as a general instruction-following controller.

## Next experiment

Checkpoint 100 is a reasonable first candidate: it is the earliest tested checkpoint with clean success at all three common seeds. Compare it with 300 and 500 using the same larger, predetermined set of held-out cube positions. Keep sampling off initially. Include checkpoint 40 if comparing the best validation-loss checkpoint with actual task success. Avoid selecting only successful or visually easy scenes.

Before another evaluation, distinguish verified task contact from a coincident tracking failure in reporting; retain tracking diagnostics rather than simply relaxing the threshold. No code was edited in this review.

## Individual trials

| Run | Checkpoint | Seed | Sampling | Instruction | Phase | Contact | Completed actions |
|---|---:|---:|---|---|---|---|---:|
| [remote_loop_20261013_040917_acebfb07](../../outputs/simulator_runs/cloud/remote_loop_20261013_040917_acebfb07/RUN_REPORT.md) | 0 | 0 | Off | touch the red cube. | Failed | No | 20 |
| [remote_loop_20261013_040954_239e0172](../../outputs/simulator_runs/cloud/remote_loop_20261013_040954_239e0172/RUN_REPORT.md) | 0 | 1 | Off | touch the red cube. | Failed | No | 20 |
| [remote_loop_20261013_041030_f12b9235](../../outputs/simulator_runs/cloud/remote_loop_20261013_041030_f12b9235/RUN_REPORT.md) | 0 | 2 | Off | touch the red cube. | Failed | No | 21 |
| [remote_loop_20261013_041321_8805e94b](../../outputs/simulator_runs/cloud/remote_loop_20261013_041321_8805e94b/RUN_REPORT.md) | 10 | 0 | Off | touch the red cube. | Failed | No | 21 |
| [remote_loop_20261013_041412_8a739ca4](../../outputs/simulator_runs/cloud/remote_loop_20261013_041412_8a739ca4/RUN_REPORT.md) | 10 | 1 | Off | touch the red cube. | Failed | No | 22 |
| [remote_loop_20261013_041455_1cbdf9b8](../../outputs/simulator_runs/cloud/remote_loop_20261013_041455_1cbdf9b8/RUN_REPORT.md) | 10 | 2 | Off | touch the red cube. | Failed | No | 21 |
| [remote_loop_20261013_041734_b4164a41](../../outputs/simulator_runs/cloud/remote_loop_20261013_041734_b4164a41/RUN_REPORT.md) | 50 | 0 | Off | touch the red cube. | Failed | No | 21 |
| [remote_loop_20261013_041813_f370b0c6](../../outputs/simulator_runs/cloud/remote_loop_20261013_041813_f370b0c6/RUN_REPORT.md) | 50 | 1 | Off | touch the red cube. | Failed | Yes | 21 |
| [remote_loop_20261013_041850_e86626e6](../../outputs/simulator_runs/cloud/remote_loop_20261013_041850_e86626e6/RUN_REPORT.md) | 50 | 2 | Off | touch the red cube. | Failed | Yes | 21 |
| [remote_loop_20261013_042244_e5d85e46](../../outputs/simulator_runs/cloud/remote_loop_20261013_042244_e5d85e46/RUN_REPORT.md) | 50 | 0 | On | touch the red cube. | Failed | No | 21 |
| [remote_loop_20261013_042322_289a8f5f](../../outputs/simulator_runs/cloud/remote_loop_20261013_042322_289a8f5f/RUN_REPORT.md) | 50 | 1 | On | touch the red cube. | Failed | No | 20 |
| [remote_loop_20261013_042402_f71b51e0](../../outputs/simulator_runs/cloud/remote_loop_20261013_042402_f71b51e0/RUN_REPORT.md) | 50 | 2 | On | touch the red cube. | Completed | Yes | 18 |
| [remote_loop_20261013_042546_0a547490](../../outputs/simulator_runs/cloud/remote_loop_20261013_042546_0a547490/RUN_REPORT.md) | 50 | 3535680 | Off | touch the red cube. | Completed | Yes | 16 |
| [remote_loop_20261013_042618_7f5c0bb6](../../outputs/simulator_runs/cloud/remote_loop_20261013_042618_7f5c0bb6/RUN_REPORT.md) | 50 | 34519768 | Off | touch the red cube. | Completed | Yes | 18 |
| [remote_loop_20261013_042935_ac686ee3](../../outputs/simulator_runs/cloud/remote_loop_20261013_042935_ac686ee3/RUN_REPORT.md) | 100 | 0 | Off | touch the red cube. | Completed | Yes | 18 |
| [remote_loop_20261013_043008_c17a4b95](../../outputs/simulator_runs/cloud/remote_loop_20261013_043008_c17a4b95/RUN_REPORT.md) | 100 | 1 | Off | touch the red cube. | Completed | Yes | 19 |
| [remote_loop_20261013_043040_fa0cd102](../../outputs/simulator_runs/cloud/remote_loop_20261013_043040_fa0cd102/RUN_REPORT.md) | 100 | 2 | Off | touch the red cube. | Completed | Yes | 17 |
| [remote_loop_20261013_043307_12ea3445](../../outputs/simulator_runs/cloud/remote_loop_20261013_043307_12ea3445/RUN_REPORT.md) | 300 | 0 | Off | touch the red cube. | Completed | Yes | 19 |
| [remote_loop_20261013_043343_b2368bf0](../../outputs/simulator_runs/cloud/remote_loop_20261013_043343_b2368bf0/RUN_REPORT.md) | 300 | 1 | Off | touch the red cube. | Completed | Yes | 17 |
| [remote_loop_20261013_043412_4c8ece6c](../../outputs/simulator_runs/cloud/remote_loop_20261013_043412_4c8ece6c/RUN_REPORT.md) | 300 | 2 | Off | touch the red cube. | Completed | Yes | 17 |
| [remote_loop_20261013_043611_4395c010](../../outputs/simulator_runs/cloud/remote_loop_20261013_043611_4395c010/RUN_REPORT.md) | 300 | 64322 | Off | touch the red cube. | Completed | Yes | 19 |
| [remote_loop_20261013_043651_1d69c320](../../outputs/simulator_runs/cloud/remote_loop_20261013_043651_1d69c320/RUN_REPORT.md) | 300 | 999 | Off | touch the red cube. | Stopped | No | 4 |
| [remote_loop_20261013_043712_3ca44d08](../../outputs/simulator_runs/cloud/remote_loop_20261013_043712_3ca44d08/RUN_REPORT.md) | 300 | 999 | Off | touch the red cube. | Completed | Yes | 21 |
| [remote_loop_20261013_043816_dfd3e0a0](../../outputs/simulator_runs/cloud/remote_loop_20261013_043816_dfd3e0a0/RUN_REPORT.md) | 300 | 9847 | Off | touch the red cube. | Completed | Yes | 15 |
| [remote_loop_20261013_044101_933eca37](../../outputs/simulator_runs/cloud/remote_loop_20261013_044101_933eca37/RUN_REPORT.md) | 500 | 0 | Off | touch the red cube. | Completed | Yes | 19 |
| [remote_loop_20261013_044141_7fb563b3](../../outputs/simulator_runs/cloud/remote_loop_20261013_044141_7fb563b3/RUN_REPORT.md) | 500 | 1 | Off | touch the red cube. | Completed | Yes | 17 |
| [remote_loop_20261013_044211_e719a9ca](../../outputs/simulator_runs/cloud/remote_loop_20261013_044211_e719a9ca/RUN_REPORT.md) | 500 | 2 | Off | touch the red cube. | Completed | Yes | 17 |
| [remote_loop_20261013_044250_c01c5d9f](../../outputs/simulator_runs/cloud/remote_loop_20261013_044250_c01c5d9f/RUN_REPORT.md) | 500 | 999 | Off | touch the red cube. | Completed | Yes | 21 |
| [remote_loop_20261013_044525_7990fd2e](../../outputs/simulator_runs/cloud/remote_loop_20261013_044525_7990fd2e/RUN_REPORT.md) | 300 | 0 | Off | grip the red cube. | Completed | Yes | 19 |
| [remote_loop_20261013_044641_06df011b](../../outputs/simulator_runs/cloud/remote_loop_20261013_044641_06df011b/RUN_REPORT.md) | 300 | 0 | Off | move up | Failed | No | 21 |

[Image montage](../../outputs/diagnostics/touch_checkpoint_review/trial_montage.jpg) · [Machine-readable measurements](../../outputs/diagnostics/touch_checkpoint_review/review.json)
