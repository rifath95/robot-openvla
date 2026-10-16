# Colored bowl continuation: 600-trial review

Reviewed October 16, 2026. Source: `outputs/sweeps/scripted_color_bowl_continue_20261015_151409_140b3657/`.
Training session: `panda-train-color-bowl-cont-20261015-151409-140b3657`.
Continued checkpoint 6,000 on the same 600 demonstrations for another 6,000 updates, with a fresh optimizer, batch 8 and learning rate 0.0001. Only red, blue, green and yellow were fine-tuning colors.

All 600 frozen trial identities validate; all 600 restored summaries agree with saved final outcomes. There are no infrastructure failures or launcher error. Layout hashes match the previous 800-trial sweep, and requested cube initial positions match exactly on all 200 task instances per continuation checkpoint.
Computed data: [review aggregates](assets/color_bowl_continuation_sweep_review.json).

## Recommendation: retain checkpoint 6,000 as the best overall model

| Checkpoint | Requested cube lifted ≥8 cm | Correct placements | Success rate | Lifted but not placed | No lift | Controller failures |
|---|---:|---:|---:|---:|---:|---:|
| 6,000 (prior sweep) | 107/200 | 80/200 | 40.0% | 27 | 93 | 28 |
| 8,000 | 100/200 | 55/200 | 27.5% | 45 | 100 | 43 |
| 10,000 | 92/200 | 64/200 | 32.0% | 28 | 108 | 16 |
| 12,000 | 104/200 | 64/200 | 32.0% | 40 | 96 | 21 |

Controller failures overlap task outcomes and remain in denominators. There are no wrong-cube qualifying lifts or wrong/both placements in the continuation sweep.

Extra training reduced training loss but did not improve overall closed-loop task success on these matched scenes. That is consistent with a training-fit/generalization mismatch. These results do not identify whether overfitting, fresh optimizer state, learning rate, or action-generation dynamics caused the change; do not label it proven overfitting or color forgetting.

## Matched comparison by scene

| Scene group | Trials per model | 6,000 | 8,000 | 10,000 | 12,000 |
|---|---:|---:|---:|---:|---:|
| One red cube | 20 | 12 | 8 | 10 | 10 |
| One black cube | 20 | 11 | 10 | 13 | 5 |
| Two cubes, red present | 40 | 18 | 17 | 17 | 20 |
| Two cubes, red absent | 40 | 14 | 7 | 12 | 10 |
| Four cubes, red present | 40 | 11 | 6 | 5 | 13 |
| Four cubes, red absent | 40 | 14 | 7 | 7 | 6 |

Checkpoint 12,000 gains locally in red-present two/four-cube groups, but loses elsewhere. Do not conclude it is worse on every scene. Four-cube aggregate drops from 25/80 at 6,000 to 13/80, 12/80 and 19/80 respectively.

On paired task instances, relative to 6,000:

| Later checkpoint | New successes | Lost successes | Successes shared | Exact paired two-sided p |
|---|---:|---:|---:|---:|
| 8,000 | 20 | 45 | 35 | 0.0026 |
| 10,000 | 27 | 43 | 37 | 0.0722 |
| 12,000 | 31 | 47 | 33 | 0.0888 |

These exploratory comparisons are unadjusted and the same suite helped select checkpoints. The counts establish the best observed model on this suite; they do not establish universal dominance on future scenes. In particular, 40% versus 32% is an observed difference with remaining sampling uncertainty.

## Color retention remains evident

First qualifying lift among multi-cube trials: 8,000 correct 79/79, 10,000 correct 65/65, 12,000 correct 80/80. The other 81, 95 and 80 multi-cube trials respectively never achieved a qualifying lift. This is conditional selection accuracy, not overall task accuracy.

| Unseen requested color | Trials/model | 6,000 placements | 8,000 | 10,000 | 12,000 |
|---|---:|---:|---:|---:|---:|
| Black | 80 | 33 | 29 | 35 | 29 |
| Purple | 40 | 9 | 5 | 7 | 6 |
| Orange | 10 | 5 | 0 | 2 | 0 |
| All unseen targets | 130 | 47 | 34 | 44 | 35 |

Blue/yellow are training colors but were not requested in this evaluation. Orange sample size is small. Zero orange placements at 12,000 does not alone demonstrate loss of orange recognition: no model lifts a wrong cube to the qualifying threshold.

## What fails after lifting

At 12,000, 40 targets were lifted but not successfully placed. Overlapping final-condition failures: 37 outside bowl-center radius, 25 outside resting-height tolerance, 3 unsettled, 23 with hand too close. At 8,000: 42/45 outside radius; at 10,000: 27/28. Grasp/placement execution and clearance remain important; lower controller-failure count at 10,000 did not make it the most successful model.

Visually inspected before/lift/final images for the matched two-cube black/purple scene at seed 1001. Checkpoint 6,000 succeeds for purple where 12,000 leaves it at the bowl edge and fails the placement criterion. For black on the same layout, 12,000 succeeds where 6,000 fails. This illustrates changed trajectories and gains/losses, rather than a uniform collapse or color-selection reversal.

## Timing and next action

Training continuation: 3 h 8 min 58 s. Simulator stage: 6 h 37 min 21 s. Packing/upload/verification: 13 min 49 s. Total recorded launch-to-final-report span: about 10 h 3 min; provisioning and final shutdown delay are excluded.

Use checkpoint 6,000 as the reference. Before more updates at the same settings, inspect representative failed grasps and transport/release episodes, then consider targeted demonstrations or a smaller learning rate evaluated on fresh fixed scenes. Avoid assuming another 6,000 updates will improve task success.
