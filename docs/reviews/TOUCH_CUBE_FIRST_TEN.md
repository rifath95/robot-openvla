# Touch-cube demonstrations for review

Created locally on 2026-10-13 in `datasets/panda_touch_cube`.
The pick-and-place and bowl datasets are unchanged. This collection is separate;
its first ten already use their final numbered training/validation/test folders.
No model training or Hugging Face upload has been started for this dataset.

## Task

Instruction: **touch the red cube.**

The scene has the robot, table, floor and one red cube, with no bowl or plate.
The arm starts with its grasp point at approximately `(0.50, 0, 0.45)` metres,
pointing down. The fingers start closed and remain closed. Every demonstration
starts with exactly the same saved robot joint positions, velocities and
controller targets. Only the cube's X/Y coordinates vary; its initial quaternion
is always `(1, 0, 0, 0)`.

The first ten cover five X positions and two Y rows with fixed seeded jitter:
X spans 0.418–0.665 m and Y spans −0.124–0.125 m. Later collection can append more
locations without renaming or replacing these ten. Failed locations are recorded
and cause collection to stop; they are not silently replaced by easier examples.

The expert makes continuous translations toward the cube's top. All rotation
components and the gripper component of every dataset action are exactly zero
(gripper zero means closed in the Panda dataset convention). Translations are
bounded by the existing 1 cm/action contract. Each episode ends on the first
action with a verified finger–cube contact and positive normal force (>0.01 N).
There is no opening phase, grasp, lift, retreat, or extra idle action.

A contact is detected after a bounded action's 0.4-second physics interval. The
cube may therefore be nudged during that final interval: displacement is
0.39–7.08 mm in this collection. This is actual physical contact, not a simulated
success based only on hand proximity, a weld, or resetting the robot to saved poses.

## Collection and verification

| Split | Trajectories | Action examples |
| --- | ---: | ---: |
| Training | 8 | 146 |
| Validation | 1 | 22 |
| Test | 1 | 18 |
| Total | 10 | 186 |

Episodes contain 17–22 actions (6.8–8.8 simulated seconds). All ten succeeded on
their predeclared locations, with no failed-location replacement. Initial robot
states are exactly equal; every action is a nonidle translation with no rotation
or gripper change. Each episode was independently replayed continuously through
the current IK/controller/physics from its initial state. Maximum replay qpos
difference over the ten episodes is below 4e-15.

All ten saved-pose launchers were checked. The shared action-replay launcher was
also run headlessly on `train_001`: 19/19 actions, zero recorded-pose difference,
and successful final finger contact. Regression tests cover the touch scene,
fixed starts, contact-based success, and existing bowl/action/pose replay behavior.

Audit: `datasets/panda_touch_cube/REVIEW_AUDIT.json`.
Collection: `datasets/panda_touch_cube/collection_report.json`.
Manifest: `datasets/panda_touch_cube/manifest.json`.
Each episode contains RGB before/after images, the instruction and accepted
seven-component action, initial/step physics states, motion frames, task/contact
metadata and the independent replay report.

## Review

`datasets/panda_touch_cube/WATCH_TRAJECTORIES.md` lists every trajectory in
Training, Validation, Test order. It links both replay modes for each episode.

Start with a physical action replay:

```bash
.venv/bin/python datasets/panda_touch_cube/train/train_001/replay_actions.py
```

Then try a cube on the other side of the table:

```bash
.venv/bin/python datasets/panda_touch_cube/train/train_008/replay_actions.py
```

Space starts/pauses/resumes; Esc or closing the window exits. Action replay
restores the initial state once and executes the saved actions in order through
physics. For saved-pose playback, run `replay.py` in the same episode folder.

Reproduce/check the current collection:

```bash
.venv/bin/python -m data_tools.collect_panda_touch --limit 10
```

This rechecks retained episode fingerprints and leaves accepted episodes intact.
After review, a larger `--limit` appends episodes under the same collection plan.
The next stage is expansion and a small training experiment only after the user
has reviewed these demonstrations.
