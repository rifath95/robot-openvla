# Bowl checkpoint 5,000: same-seed simulator comparison

Reviewed 2026-10-13 using saved input/output images, raw actions, physical states,
and cloud model provenance. All runs used checkpoint_005000 from training session
`panda-train-20261012-072437-856`, the bowl instruction, and the corrected controller.

| Scene seed | Completed actions | Recorded behavior | End reason |
| --- | ---: | --- | --- |
| 0 | 10 | Repeats the same near-zero seven-component action from the first step | User stopped |
| 1 | 8 | Two larger rotation commands, then repeated near-zero actions | User stopped |
| 2 | 6 | First commands about 10 mm downward, then repeated near-zero actions | User stopped |

The saved server evaluation provenance identifies checkpoint_005000, not only
the launcher's requested adapter field. All three trials reused the same merged
model/server. Their first before-images are byte-for-byte identical to those of
the corrected-controller checkpoint_002000 trials on seeds 0, 1 and 2.

No recorded trial contacted, grasped, lifted or placed the cube. It remained at
its initial table position to numerical precision. All executed gripper actions
were 0.996078, nearly fully open. No IK rejections occurred. Maximum final
tracking errors were 0.026 mm, 0.083 mm and 1.576 mm respectively.

The repeated hold action has X +0.0211 mm, Y -0.0140 mm, effectively zero Z,
tiny rotation deltas, and an open gripper. Translation norm is about 0.0253 mm,
below the 0.1 mm IK tolerance. This is a raw policy output, not a large command
being reduced by retries or cancelled by the controller.

Checkpoint 2,000 already stalled on seed 0. Its seed 1 drifted slowly upward and
seed 2 repeatedly commanded down/up alternation. Checkpoint 5,000 changes those
predictions but still does not show useful task progress in these recordings.
The trials were stopped early, so they are not full 150-action evaluations or a
statistically reliable checkpoint success-rate comparison. There is no observed
evidence here that 5,000 fixes the failure to start approaching the cube.

The prior 52-example diagnostic was on checkpoint 2,000. Its numerical scores
must not be presented as checkpoint-5,000 diagnostic results. If another cloud
test is run, compare 5,000 on the same demonstration inputs rather than repeating
random live trials. Before another training run, inspect the early demonstration
labels/observations for redundant hold/open steps and determine whether state-only
inputs distinguish them from the subsequent movement phase. This is a hypothesis
to audit, not an established cause of these failures. The collection/current
controller mismatch remains a separate issue established by action replay.

Source runs:

- `outputs/simulator_runs/cloud/remote_loop_20261013_004429_0ac8a961`
- `outputs/simulator_runs/cloud/remote_loop_20261013_004457_d1ae30e2`
- `outputs/simulator_runs/cloud/remote_loop_20261013_004516_7b5da887`

Offline measurements: `outputs/diagnostics/bowl_checkpoint_5000_review.json`.
