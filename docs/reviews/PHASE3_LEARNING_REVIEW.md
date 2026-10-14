# Phase 3: approach learning after 1,000 updates

This review uses the existing training logs, the post-fix 80-image diagnostic
`expert_evaluation_20261011_023518_fc155dbc`, and the post-fix live rollout
`remote_loop_20261011_023950_e87092ab`. No cloud requests, new training, or robot
movement were performed for this review. The diagnostic uses training images;
it does not measure held-out task success.

## Where the approach predictions fail

The eight sampled `approach` examples contain three initial approaches from
pilot episodes and five later alignment corrections from expansion episodes.
These subsets differ in episode type as well as timing, so this small sample
does not establish the accuracy of every early or late approach.

The three initial examples have translation command errors of 0.019–0.048 mm.
The five later examples have errors of 10.458–14.459 mm:

| Episode / step | Expert XYZ command (mm) | Predicted XYZ command (mm) | Command error (mm) |
|---|---|---|---:|
| train_01 / 1 | −4.334, −1.590, −8.871 | −4.313, −1.567, −8.863 | 0.033 |
| train_03 / 1 | −1.246, −2.577, −9.582 | −1.256, −2.586, −9.569 | 0.019 |
| train_05 / 1 | +2.155, +2.536, −9.430 | +2.192, +2.511, −9.412 | 0.048 |
| train_extra_01 / 11 | −10.000, +0.011, −0.054 | +0.233, +0.001, +9.961 | 14.318 |
| train_extra_08 / 13 | +2.245, −0.006, +0.542 | −0.002, +0.001, −9.961 | 10.741 |
| train_extra_16 / 11 | −0.010, +9.988, +0.495 | −0.002, +0.001, −9.961 | 14.459 |
| train_extra_23 / 14 | +1.945, +0.003, +0.421 | −0.002, +0.001, −9.961 | 10.563 |
| train_extra_30 / 15 | +0.125, +1.731, +0.352 | −0.002, +0.001, −9.961 | 10.458 |

In these later observations, the expert corrects horizontal alignment while
keeping approximately the same height. The model instead requests an almost
10 mm vertical move, often with essentially no horizontal correction. Some
offset-approach examples also have substantial errors; their mean translation
command error is 6.139 mm across eight examples. This is broader than just an
unreliable gripper-close command.

## Connection with the live failure

The live cube starts at approximately `(0.550, 0.000)` in world XY. The `train_05`
cube starts at `(0.580, 0.030)`. The live trial's first predicted action is exactly
the same seven-component prediction returned for the sampled `train_05` initial
image, even though the cube locations differ. The initial hand XY poses are
approximately the same. The instructions differ in placement distance (12 cm
in the live run, 10 cm in that demonstration).

The live hand moves toward approximately `(0.579, 0.030)` and closes about 42 mm
from the actual cube. This is consistent with insufficient adaptation of the
approach to the current scene. It does **not** prove the model memorized a
specific trajectory or reveal what it perceived internally. The later missed
expert alignment corrections independently show that some training observations
are not yet reproduced reliably.

The model then requests closed-gripper, almost-zero-motion actions. There are
no IK rejections. The cube remains on the table. First-lift expert images also
remain imperfect: 6/8 receive upward commands, while two receive effectively
zero translation. Correct closing on 8/8 aligned expert images is therefore
insufficient for end-to-end success.

## What the training curves support

| Update | Full validation loss (507 examples) |
|---:|---:|
| 250 | 1.5493 |
| 500 | 1.3447 |
| 750 | 1.4111 |
| 1000 | 1.3682 |

Mean training loss falls from 2.7445 over updates 1–100 to 0.9286 over updates
901–1000. However, validation loss does not improve monotonically beyond update
500. Lower training loss alone is not evidence of better live control.

At effective batch size four, 1,000 updates involve 4,000 training-example
presentations versus 2,595 distinct training transitions: about 1.54 dataset
equivalents. This is an exposure ratio, not a claim that every example was seen
equally often or that 1.54 shuffled epochs were completed. The loader shuffles
and repeats an iterable stream, and actual sampled episode/step IDs were not
recorded in the existing training logs.

## Recommendation

**Follow-up, 2026-10-11:** the matched diagnostic is now complete on the same
80 expert images. XYZ command error improves from 3.008 mm to 2.187 mm with
teacher forcing, but first-lift upward agreement remains 6/8 in both. Errors
remain even with correct preceding action tokens. This does not establish that
more training time is the solution. Given the validation plateau, the agreed
next experiment is broader demonstration coverage: 200 additional episodes
with substantially different initial hand positions/yaw and cube positions.
See [BROAD_PANDA_DATASET.md](../datasets/BROAD_PANDA_DATASET.md). The reasoning below is the
earlier proposal that motivated the diagnostic, not a decision to retrain now.

A controlled longer-training experiment is reasonable because the current model
still fails to reproduce several training examples. It is **not** established
that more updates alone will solve the task: validation has largely plateaued,
and live execution can reach states outside the expert paths.

Before committing to a long run, use the next short GPU diagnostic to measure
both teacher-forced predictions and ordinary autoregressive generated actions
on the **same** saved expert examples, including these alignment failures. This
comparison is still pending; the existing 80-image check measures generated
actions only, while the logged training metrics use teacher forcing and different
batches. Log component errors and action-token accuracy on the matched examples.

If both evaluations are poor, additional fitting is a reasonable experiment.
If teacher forcing succeeds but generated actions fail, investigate that gap
before assuming the solution is simply more updates. Either result helps choose
the next training configuration. During any longer experiment, retain periodic
checkpoints and full validation and compare generated actions and controlled
rollouts, rather than selecting solely by training loss.

Keep the current action convention, single-image OpenVLA model, and existing
dataset for this diagnostic. These results do not justify immediately adding
history, manually forcing task stages, or oversampling grasp/lift examples.

Local detailed artifacts are under `outputs/diagnostics/approach_learning_review_20261011/`:
`approach_examples.json` and `approach_comparison.png`.
