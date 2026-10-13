# Revised bowl demonstrations: 30-trajectory short training test

The first ten were reviewed and approved. On 2026-10-13, another twenty were
added to the same numbered collection; the original ten were retained unchanged.
All thirty pass continuous physics action replay and meaningful-start checks.

| Split | Trajectories | Training examples (action steps) |
| --- | ---: | ---: |
| Training | 20 | 2,279 |
| Validation | 5 | 539 |
| Test | 5 | 590 |

The checked portable bundle is `datasets/panda_pick_bowl_v2/training_bowl_v2_30`
(about 423 MB). All exported source images, instructions and actions match, and
the official processor checks pass. `WATCH_TRAJECTORIES.md` now lists all thirty
in Training / Validation / Test order. This small collection makes no exhaustive
coverage claim. The old 300-trajectory dataset remains unchanged.

## Short training configuration

`training/pick_bowl_v2_30_500.json` starts from the unchanged pretrained model
with fresh LoRA adapters. It selects the fastest compatible safe A40 benchmark
recommendation automatically; it does not resume the earlier bowl adapter.

- Maximum 500 optimizer updates, accumulation 1.
- Adapter checkpoints: 0, 10, 20, ..., 100, 200, 300, 400, 500.
- Update 0 saves the initialized adapter before learning, after baseline validation.
- Full validation of all 539 validation examples at 0, 100, 200, 300, 400, 500.
- Test examples are held out from training and validation.
- Inner limit 1 hour; outer supervisor limit 90 minutes.
- 9 GiB minimum free volume space: fifteen adapters occupy roughly 7 GB.
- Checkpoints, metrics, plots and session status remain on the volume before
  the supervisor requests a pod stop; connected Macs fetch reports automatically.
- If a runtime cap ends training early, the loop also attempts to save that
  final update even if it is outside the requested schedule.

No cloud training has been launched for this collection yet.

Upload from your local terminal (the command creates the private repository if
needed and supports resuming; retain the bundle's `.cache` directory):

```bash
caffeinate -i .venv-training/bin/hf upload-large-folder \
  Maxalderone/robot-openvla-panda-bowl-v2 \
  datasets/panda_pick_bowl_v2/training_bowl_v2_30 \
  --repo-type dataset --private --num-workers 4
```

After upload completes, copy its full Hugging Face commit SHA. On a preparation
pod with the training template, network volume and HF read secret:

```bash
cd /root/robot-openvla
git pull --ff-only
/opt/openvla/bin/python -m cloud.download_training_dataset \
  --bundle training_bowl_v2_30 --revision UPLOAD_COMMIT_SHA
```

Replace `UPLOAD_COMMIT_SHA` with the completed upload's actual revision. After
checksum verification, launch from the local Mac terminal:

```bash
caffeinate -i .venv/bin/python -m cloud.training_cloud_session \
  --ssh 'CURRENT_POD_SSH_COMMAND' \
  --config training/pick_bowl_v2_30_500.json
```

Use the pod's current SSH command in place of `CURRENT_POD_SSH_COMMAND`.
There is no Docker rebuild required for these repository-level changes.

## Original first-ten review record

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
