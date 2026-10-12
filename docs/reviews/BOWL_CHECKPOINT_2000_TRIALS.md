# Bowl checkpoint 2,000: first three simulator trials

Reviewed 2026-10-12. All trials used checkpoint_002000 from training session
`panda-train-20261012-072437-856`, the bowl scene/camera, randomized starts, and
instruction "pick up the red cube and drop it in the blue bowl."

| Scene seed | Completed actions | End reason | Closest grasp control point to cube centre | Largest tracking error | Recorded cube lift |
| --- | ---: | --- | ---: | ---: | --- |
| 0 | 113 | User stopped | 360 mm | 0.211 mm | None |
| 1 | 41 | User stopped | 300 mm | 0.200 mm | None |
| 2 | 24 | User stopped | 184 mm | 1.923 mm | None |

There were no IK rejections in these trials. The cube remained at its initial
table position to numerical precision in all recorded before/after states.
None of these partial trials demonstrated a grasp, lift, or placement. They
are not full-length success-rate evaluations because the user ended them early.

The predicted gripper action was 0.996078 (nearly fully open) in every executed
step. Seeds 0 and 1 produced very small translations: mean commanded translation
over the last ten actions was 0.090 mm and 0.237 mm respectively. Their final
images show the hand still well above the table. Seed 2 produced substantial
movement (last-ten mean 9.961 mm per action); its final image shows the open
gripper above the bowl while the cube remains outside it.

The saved data support poor task-directed predictions in these observations,
with accurate execution of the bounded targets. They do not establish why the
model behaved this way or the user's reason for ending each trial. Obtain those
observations before choosing a training/data change. Compare checkpoint 5,000
using the same seeds and instruction before attributing the result to checkpoint
selection or making a new dataset.

Source runs under `outputs/`:

- Seed 0: `remote_loop_20261012_114419_a61c5550`
- Seed 1: `remote_loop_20261012_115022_efd46fc5`
- Seed 2: `remote_loop_20261012_115331_fcdda8c6`

Offline measurements are saved in `outputs/bowl_checkpoint_2000_review.json`.
Distances use the Panda grasp control offset from the action convention; they
are geometric diagnostics, not a contact-based grasp success classifier.

## Follow-up: repeated dip and return near the end

The user reported oscillation and apparent return to the preceding position in
seeds 0 and 1. Offline comparison confirms the visible dip-and-return within
individual actions, rather than a state reset between actions.

For the final 20 transitions in each of these two runs, `after.png` is byte-for-byte
identical to the next `before.png`. Saved joint positions and simulation time
also match exactly at that boundary. Consecutive input images differ slightly;
they do not alternate back to identical images from two steps earlier.

Saved motion frames in seed 0, steps 108–113, show about 2.75 mm maximum grasp-point
excursion per action, chiefly downward, followed by a return leaving about 0.21 mm
net displacement. The model repeats a roughly 0.025 mm translation command with
essentially zero requested vertical motion. In seed 1, steps 36–41, the excursion
is about 2.31 mm, leaving only 0.072 mm net displacement. Its repeated command is
about 0.237 mm, including an upward component of 0.235 mm.

This separates two observations: the policy produces tiny repeated commands, and
the physical controller introduces a larger transient excursion before settling.
The action application replaces actuator targets with the ideal IK joint targets;
the subsequent physics loop adds accumulated tracking/gravity corrections. That
target replacement is a plausible source of the repeated transient, but this
review has not experimentally isolated it. Low final tracking error does not
rule out substantial movement during an action. Do not interpret the prior
accurate-endpoint finding as a claim that the entire motion is smooth.

The policy therefore receives very similar images again, consistent with repeated
predictions. These observations alone do not prove whether removing the controller
transient would make the learned policy approach the cube successfully.

## Controller correction verified and fixed

An isolated replay of saved seed-0 step 108 and seed-1 step 36 reproduced the
bounce when new actions discard the holding offset. Preserving the actuator
target minus actual joint position when adding a new ideal IK target reduced
peak grasp-point excursion from 2.752 to 0.430 mm and from 2.336 to 0.614 mm,
respectively. Endpoint errors improved from 0.207 to 0.025 mm and from 0.183
to 0.064 mm. Residual motion remains; this fixes the repeated loss of the holding
correction, not every source of physical oscillation.

The controller now preserves that offset on accepted actions and clips the new
actuator targets to joint limits. Rejected actions retain existing retry/restore
behavior. Local regressions also exercise 5 mm commands in both directions on
all three translation axes from both poses, requiring endpoint error below
1.5 mm. These saved-state tests need no cloud GPU or model inference.

The update does not change model predictions or recorded trajectories. New live
trials are still needed to determine whether the learned policy progresses on
the bowl task with the corrected physical controller.
