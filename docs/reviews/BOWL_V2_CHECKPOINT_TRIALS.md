# Bowl pilot simulator checkpoint review

Reviewed 2026-10-13. Training session `panda-train-20261013-013939-2113`.

Fourteen nonempty trials were saved across checkpoints 0, 50, 100, 200 and 400. Checkpoint 0 seed 2 was closed before any prediction; it is excluded. Checkpoint 500 has not been tested. These are three fixed scene seeds, with different user stop lengths; they do not estimate a general success rate.

All fourteen trials left the cube on the table, with no measurable lift or displacement. More visible movement is present than in the earlier stationary policy, but none of these trials completed a grasp or bowl placement. The untrained baseline also moves and sometimes approaches closer than a fine-tuned checkpoint, so motion alone does not establish improved task performance.

## Per-trial measurements

| Checkpoint | Seed | Executed steps | Rotation commands | Near-zero commands | Initial / closest cube distance (mm) | Still tail (steps) | Rejections |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 100 | 99 | 0 | 376.0 / 213.7 | 0 | 0 |
| 0 | 1 | 35 | 34 | 0 | 299.9 / 199.8 | 0 | 3 |
| 50 | 0 | 100 | 75 | 25 | 376.0 / 357.4 | 25 | 0 |
| 50 | 1 | 26 | 5 | 21 | 299.9 / 299.6 | 20 | 0 |
| 50 | 2 | 90 | 80 | 0 | 194.7 / 193.8 | 0 | 0 |
| 100 | 0 | 46 | 29 | 17 | 376.0 / 365.0 | 16 | 0 |
| 100 | 1 | 36 | 29 | 7 | 299.9 / 299.2 | 6 | 0 |
| 100 | 2 | 100 | 87 | 0 | 194.7 / 175.6 | 0 | 0 |
| 200 | 0 | 74 | 68 | 4 | 376.0 / 376.0 | 0 | 0 |
| 200 | 1 | 91 | 82 | 9 | 299.9 / 298.1 | 8 | 0 |
| 200 | 2 | 13 | 0 | 10 | 194.7 / 175.1 | 8 | 0 |
| 400 | 0 | 22 | 8 | 0 | 376.0 / 376.0 | 0 | 3 |
| 400 | 1 | 34 | 12 | 22 | 299.9 / 299.5 | 21 | 0 |
| 400 | 2 | 60 | 0 | 0 | 194.7 / 169.0 | 0 | 0 |

Rotation command: predicted rotational norm >0.005 rad. Near-zero command: translation norm <0.1 mm and rotation norm <0.001 rad. Still tail: consecutive final executed steps with actual translation <0.1 mm and actual rotation <0.05°. Cube distance uses the grasp control point, not the visible wrist centre.

## Rotation and phase transition

The expert first clears the starting height while opening the gripper, then aligns the hand with the scripted grasp orientation, then approaches and lowers toward the cube. Those transitions use privileged simulator coordinates and explicit position/orientation tolerances. The model receives the current image and fixed instruction, and has to learn when to change behavior.

Checkpoint 200 seed 1 reproduces the reported overshoot: scripted-reference orientation error starts at 64.53°, reaches 8.19° at step 24, then finishes at 155.60°. The arm performs 266° of total rotation while travelling only 5.14 mm; the final eight steps are nearly stationary. Source images at steps 24 and 82 visually confirm different orientations.

A square cube and parallel-jaw gripper admit equivalent yaw orientations. The reference-angle number alone is therefore not a universal grasp-quality metric. Accounting for 90° yaw symmetry, this trial reaches 8.19° error and ends at 25.62°: it still loses the closest alignment. Eight degrees also exceeds the expert orientation tolerance, so it was visually close, rather than demonstrably finished with orientation.

Checkpoint 400 seed 1 improves scripted-reference alignment from 64.53° to 42.49°, then produces 22 near-zero predictions. Checkpoint 200 seed 2 issues three meaningful downward commands followed by ten near-zero predictions, without rotating into the grasp pose. Seed 0 at checkpoint 400 encounters three IK rejections and ends; this is a different failure from a stationary policy.

## Controller continuity

For every adjacent executed step in all fourteen trials, the full saved simulator qpos after one action exactly matches the state before the next action (maximum difference 0). The simulator does not reset the arm between these predictions. Rotational commands create actual rotation; late stationary stretches coincide with tiny model outputs. This continuity check does not assert perfect tracking within every movement, but rules out between-request pose resets in these records.

## Interpretation and next step

The policy emits meaningful movement, but has not learned reliable direction, stopping, and transition from orientation to approach. Twenty training trajectories covering broad arm and object configurations may provide sparse coverage. Off-demonstration states and autoregressive prediction errors are also plausible contributors. These trials cannot isolate a single cause or prove that training longer will solve it.

Test checkpoint 500 on the same seeds if a final comparison is wanted. Before choosing a larger dataset or longer run, a focused comparison of generated actions against expert actions around the orientation-to-approach boundary would be more informative than assuming the loss curve measures closed-loop success. No new collection, training, or cloud diagnostic has been started.

Local measurement artifact: `outputs/bowl_v2_checkpoint_review/review.json`.
Plot: `outputs/bowl_v2_checkpoint_review/orientation_overshoot.png`.

## Source runs

- Checkpoint 0, seed 0: `outputs/remote_loop_20261013_021421_598d0987`.
- Checkpoint 0, seed 1: `outputs/remote_loop_20261013_021707_d8a114c5`.
- Checkpoint 50, seed 0: `outputs/remote_loop_20261013_022040_8ed20480`.
- Checkpoint 50, seed 1: `outputs/remote_loop_20261013_022319_a6d8b143`.
- Checkpoint 50, seed 2: `outputs/remote_loop_20261013_022403_5dc12f75`.
- Checkpoint 100, seed 0: `outputs/remote_loop_20261013_022854_75e43126`.
- Checkpoint 100, seed 1: `outputs/remote_loop_20261013_023013_9609f73d`.
- Checkpoint 100, seed 2: `outputs/remote_loop_20261013_023115_96815de3`.
- Checkpoint 200, seed 0: `outputs/remote_loop_20261013_024046_7faaeb28`.
- Checkpoint 200, seed 1: `outputs/remote_loop_20261013_024249_f3af7673`.
- Checkpoint 200, seed 2: `outputs/remote_loop_20261013_024522_2e573d8d`.
- Checkpoint 400, seed 0: `outputs/remote_loop_20261013_020816_48c2152f`.
- Checkpoint 400, seed 1: `outputs/remote_loop_20261013_020901_f64b721f`.
- Checkpoint 400, seed 2: `outputs/remote_loop_20261013_021001_2a2f89e2`.
