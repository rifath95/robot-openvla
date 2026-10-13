# Revised bowl demonstrations: first ten for review

Generated locally on 2026-10-13 under `datasets/panda_pick_bowl_v2`.
The previous 300-trajectory dataset is unchanged. No new fine-tuning or upload
has occurred. These ten belong to the new numbered collection, not a separate
pilot subset that would need renaming later.

The pilot contains eight training trajectories, one validation trajectory, and
one test trajectory (914 / 111 / 148 action examples respectively).
All ten successfully grasped, lifted, and placed the red cube in the blue bowl.
Their saved actions were replayed continuously through the revised physics
controller and reproduced successful placement to numerical precision.

Changes:

- No unconditional opening hold at the beginning. Opening occurs alongside the
  first necessary clearance or orientation movement, including closed starts.
- All ten begin with an arm translation of 10 mm or rotation norm of 0.05 radians.
  Five start with clearance and five with orientation; none starts with an
  `open_for_pickup` hold.
- Current holding-correction controller used for collection and replay.
- Cube yaw varies from the first trajectory; arm XYZ/roll/pitch/yaw/gripper,
  cube positions and bowl positions vary as before.
- New controller-code fingerprints bind replay verification to control code.
  This avoids silently certifying a trajectory after another controller change.

## Verification

All first actions create measured motion and changed camera images. The clearance
commands produce roughly 11.6 mm initial motion (including normal tracking
overshoot); orientation commands produce roughly 0.058 radians of rotation.
Initial placement errors after full physics replay are 0.083–1.835 mm from bowl
centre. Maximum replay difference over the ten accepted trajectories is under
1e-10 in saved simulator position values.

Two orientation configurations failed IK on their first attempt and succeeded
on the second. Their attempts are recorded in the collection report; no broad
quota coverage claim is made for this small pilot. The accepted starts span
0.267 / 0.429 / 0.207 m in arm XYZ, about 1.24 / 1.57 / 4.18 radians in requested
roll/pitch/yaw, and include closed, open and intermediate gripper starts.

These checks establish successful physical demonstrations and removal of the
redundant startup label. They do not yet establish that a model fine-tuned on
them will learn a reliable policy. Review first before expanding or training.

## Watch

`datasets/panda_pick_bowl_v2/WATCH_TRAJECTORIES.md` lists all ten in Training,
Validation, Test order with pose replay and action replay links.

Start with the already-open-gripper example:

```bash
.venv/bin/python datasets/panda_pick_bowl_v2/train/train_002/replay_actions.py
```

Then review a closed start and a rotation-first start:

```bash
.venv/bin/python datasets/panda_pick_bowl_v2/train/train_001/replay_actions.py
.venv/bin/python datasets/panda_pick_bowl_v2/train/train_003/replay_actions.py
```

Space starts/pauses/resumes; Esc or close exits. The new trajectories use the
default current controller; do not select the old collection-mode target assignment.
For saved-pose playback, use each folder's `replay.py` instead.

Measurements: `datasets/panda_pick_bowl_v2/STARTUP_AUDIT.json`.
Collection and failure details: `datasets/panda_pick_bowl_v2/collection_report.json`.

Reproduce/continue the collection only after review:

```bash
.venv/bin/python -m data_tools.collect_panda_bowl \
  --output-dir datasets/panda_pick_bowl_v2 --limit 10 \
  --startup-policy state_dependent --workers 2
```

The collection plan refuses incompatible policy changes in an existing directory.
