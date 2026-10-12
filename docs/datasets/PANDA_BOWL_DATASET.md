# Pick up the red cube and drop it in the blue bowl

The new collection lives in `datasets/panda_pick_bowl/`, separate from the older
pick/place dataset. Every episode uses exactly:

> pick up the red cube and drop it in the blue bowl.

The bowl is a visible destination. The instruction does not require remembering
the cube's initial position or interpreting a relative placement offset.
All observations use the approved fixed `bowl_camera` view.

## Review the first ten

Ten successful trajectories have been recorded: eight train, one validation,
and one test. Each contains RGB images, instructions, executed seven-component
action labels, physics states, stage labels, motion frames, and its own replay
launcher. No fine-tuning or dataset upload has started for this task.

```bash
.venv/bin/python datasets/panda_pick_bowl/train/train_001/replay.py
.venv/bin/python datasets/panda_pick_bowl/train/train_002/replay.py
.venv/bin/python datasets/panda_pick_bowl/validation/validation_001/replay.py
.venv/bin/python datasets/panda_pick_bowl/test/test_001/replay.py
```

Space starts/pauses/resumes; Esc or closing the viewer exits. Replay uses saved
motion frames and restores that episode's bowl location. It needs no cloud model.
The full index is `datasets/panda_pick_bowl/WATCH_TRAJECTORIES.md`.

## Additional cube-rotation examples

The original ten are preserved. Ten additional successful trajectories use
random cube yaw sampled over ±180°, keeping the cube flat on the table.
Their replay launchers are `train_009`–`train_014`, `validation_002`–`validation_003`,
and `test_002`–`test_003`. For example:

```bash
.venv/bin/python datasets/panda_pick_bowl/train/train_009/replay.py
.venv/bin/python datasets/panda_pick_bowl/validation/validation_002/replay.py
```

The grasp is aligned with a rotated cube edge. A cube has four equivalent yaw
orientations, so the expert selects the equivalent edge-aligned grasp requiring
the smallest yaw change from the home orientation. The requested cube yaw,
settled cube quaternion, and grasp orientation are stored in each new episode.
Initialization and action/state recordings preserve cube rotation in physics
replay. Roll/pitch are not randomized for the cube; it starts resting flat.

There are now 20 accepted trajectories (14 train / 3 validation / 3 test).
Further generation is paused for review. New trajectories after the initial ten
will also randomize cube yaw. To create/resume only these first twenty:

```bash
caffeinate -i .venv/bin/python -m data_tools.collect_panda_bowl --limit 20
```

These episodes belong to the final dataset; there are no separate pilot or
expansion cohorts. After the user approves the examples, resume collection with:

```bash
caffeinate -i .venv/bin/python -m data_tools.collect_panda_bowl --limit 300
```

The planned complete layout is:

```text
datasets/panda_pick_bowl/
├── train/train_001/ ... train_200/
├── validation/validation_001/ ... validation_050/
├── test/test_001/ ... test_050/
├── collection_plan.json
├── collection_report.json
├── manifest.json
└── WATCH_TRAJECTORIES.md
```

Only successful episodes enter these folders. Failed attempts, if any, remain
under `outputs/bowl_collection_checks/`, outside the accepted dataset. A fresh
clone does not contain generated images/states: they are excluded from GitHub.

## Variations and checks

The cube and bowl are independently sampled across the reachable table interior
with a clearance between them. Initial grasp-point XYZ, roll, pitch, yaw, and
gripper opening vary. The first eight training episodes include closed, open,
and intermediate gripper starts. Roll and pitch are sampled within ±0.85 radians,
yaw within ±2.2 radians relative to the downward home orientation. Initial grasp
points span X 0.36–0.70 m, Y ±0.24 m, and Z 0.38–0.60 m. Rejected unreachable,
colliding, or unsettled starts are resampled; this does not cover every conceivable
pose or every physical-table point.

The script opens the gripper, raises it if needed, reorients above the table,
approaches and grasps the cube, lifts it, carries it over the bowl, lowers above
the rim, releases it, and retreats. Grip is supported by contact forces and
friction; no artificial weld or attachment carries the cube.

Accepted trajectories must lift the cube at least 8 cm, leave its center within
4.5 cm of the bowl center, settle it at the bowl floor, and retreat the empty
gripper. Sampled motion contacts, bounded-action IK, finite states, and tracking
are checked. A separate sequential physics replay must reproduce recorded joint
and cube states within 1e-6 and pass the same task checks. Each replay fingerprint
binds its scenario, instruction, action convention, images, and physics states.

The first ten contain **1,102 recorded steps: 906 train, 88 validation, and 108 test**, with
88–126 steps per trajectory. Maximum replay-state difference was 8.3e-12.
The actual initial hand-position spans were about 25 cm X, 43 cm Y, and 20 cm Z;
relative starting-angle spans were about 93° roll, 71° pitch, and 234° yaw.
Cube XY spans were about 28 cm × 42 cm; bowl XY spans were about 25 cm × 35 cm.
These values describe the first ten, not guaranteed complete-dataset coverage.

The seven action components keep `panda_grasp_v1` units and bounds. The observation
description identifies `bowl_camera`; dataset preparation refuses to mix different
camera conventions in one export. RLDS packaging and fine-tuning setup follow
the user's review of these trajectories.
