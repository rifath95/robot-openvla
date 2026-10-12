# Bowl checkpoint 2,000 after the holding-correction fix

Reviewed 2026-10-12 from saved input/output images, model predictions, actuator
states, and motion frames. All five trials used checkpoint_002000 from
`panda-train-20261012-072437-856`. No new cloud work was used for this review.

| Scene seed | Completed actions | Recorded behavior | End |
| --- | ---: | --- | --- |
| 0 | 14 | Same near-zero action every time; settles and stops moving | User stopped |
| 1 | 150 | Tiny translations, mostly upward; moves farther from cube | Step budget completed |
| 2 | 24 | Model alternates about 10 mm down / 10 mm up | User stopped |
| 676 | 20 | Initially commands upward motion, then repeats a near-zero action | User stopped |
| 7654345679976 | 35 | Tiny upward commands and changing rotations; cube untouched | User stopped |

None grasped, lifted, or placed the cube during the recorded trial. The cube
remained at its initial table position to numerical precision. All executed
gripper predictions were 0.996078 (nearly fully open). No IK rejections occurred.

## What the model commanded

The predicted X action is the same near-zero value, +0.0211 mm, across all
243 completed actions. Y is almost always -0.0140 mm, with one other quantized
value in seed 1. These are raw predictions, not changes caused by bounding or
IK retries. The policy does not provide the lateral movement needed to approach
the randomly placed cube in these observations.

Seed 0's entire seven-component action repeats exactly; translation norm is
0.0253 mm and requested Z change is effectively zero. This is below the IK
position tolerance of 0.1 mm. The solver can accept the existing pose instead
of resolving such a tiny displacement. After settling, the recorded physical
motion per step becomes effectively zero. The controller is not cancelling a
substantial movement command in this case.

In seed 2, the raw Z prediction alternates -9.9608 mm and +9.9608 mm from the
first action onward. The visible repeated up/down motion is therefore requested
by the model. Tracking has some overshoot (maximum final error 1.622 mm), but
that does not explain why successive requested directions reverse.

Seed 1 increases grasp-point distance from cube centre from about 300 to 324 mm
over 150 actions. Seed 676 starts with four roughly 10 mm upward commands and
then stalls. The final images show the hand separated from the cube in every run.

## Checks and limits of the diagnosis

The saved metadata identifies the intended adapter, bowl camera, Panda action
key, and exact bowl-task instruction. The deployed action statistics match the
verified local bowl bundle. The merge implementation loads the selected adapter
and attaches its normalization statistics. These checks rule out the specific
wrong-checkpoint-path, wrong-instruction and wrong-statistics explanations; they
do not certify every part of the training/inference pipeline or adapter weights.

The prior repeated 2–3 mm downward controller transient is absent from the
settled near-zero-command run. Endpoint errors remain small (maximum 0.026 mm
in seed 0 and 0.266 mm in seed 1). Remaining task failure is evident in the raw
policy commands. This does not prove whether the underlying cause is inadequate
learning, training/inference preprocessing, or another implementation issue.
Low teacher-forced validation loss alone cannot establish live task performance.

Before changing the dataset or paying for another long training run, compare
predictions on known demonstration observations against the saved expert actions
and verify the training/inference inputs match. A controlled checkpoint-5,000
comparison can then distinguish checkpoint selection from a shared pipeline or
learning failure. Further random simulator trials alone will not isolate that cause.

Source runs under `outputs/`:

- `remote_loop_20261012_122158_a28ecf48`
- `remote_loop_20261012_122230_1398de3f`
- `remote_loop_20261012_122634_3ccc119e`
- `remote_loop_20261012_122746_afdcf70c`
- `remote_loop_20261012_122831_de4972a4`

Offline measurements: `outputs/bowl_checkpoint_2000_corrected_controller_review.json`.
