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
