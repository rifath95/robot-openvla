# Expanded single-cube bowl placement dataset

The expanded dataset retains the original 50 reviewed demonstrations and adds
250 successful demonstrations under `datasets/panda_place_bowl/`. The portable
bundle is `datasets/panda_place_bowl/training_place_bowl_v2_300/`.

Per the user's October 14 decision, **all 300 episodes are training data**.
There are no held-out validation or test episodes in this bundle. The original
recordings, identities, and source split labels are retained as provenance;
`--training-only` explicitly maps them into the exported training split.
Future simulator sweeps provide separate behavioral evaluation. Do not describe
training loss or evaluation on these same demonstrations as held-out performance.

Instruction: `pick up the red cube and place it in the blue bowl.` The arm
starts in the fixed lift pose with an open gripper. Cube and bowl locations
vary within the same declared workspace; cube color, camera, orientation,
geometry, grip configuration and pose-holding controller remain unchanged.
The expert approaches, grasps through physical contact, lifts, transports,
releases above the bowl and retreats. Acceptance requires successful placement
and deterministic execution of the saved actions through MuJoCo physics.

The bowl initializer now places both objects before settling the robot.
Previously a relocated bowl could overlap the temporary home cube; contact
inspection confirmed this for the two previously failing sweep layouts.
All 301 candidate layouts and all 20 old sweep layouts passed reset
preflight. Future bowl sweeps initialize and render every layout before model
loading. Recorded-state replay preserves compatibility with the original 50.
See [the initialization audit](assets/place_bowl_initialization_audit.json).

Collection preserves rejected-layout diagnostics and samples reserve layouts
from the same seeded position distribution. It does not weaken collision checks.
The [collection audit](assets/place_bowl_300_audit.json) records accepted slots,
rejected attempts, replay errors, counts and the original 50's preserved hashes.
Acceptance-conditioned sampling can still bias the dataset toward layouts the
expert can solve; simulator test positions should remain independently fixed.

Watch everything from `datasets/panda_place_bowl/WATCH_TRAJECTORIES.md` using
the direct Replay and Action Replay buttons, with the local replay launcher on.
The source index groups all 300 into Training for this export; old episode
names such as `validation_001` remain only to preserve their original identity.

## Verified collection

300 successful trajectories contain **21,906 valid action examples**, all
assigned to training. Original 50 hashes are unchanged. All 300 saved-action
replays passed; one of 301 candidate layouts was rejected for hand/bowl contact.

Portable export and all-source image/action/instruction comparison passed.
The official OpenVLA processor batch check passed (6×224×224 pixels, seven
action tokens plus stop). The upload bundle is approximately 2.5 GiB.
No HF upload or training has been performed for this expanded bundle.

## Rebuild and upload

```bash
.venv/bin/python -m data_tools.collect_panda_place_bowl --limit 300 --training-only
.venv-training/bin/python -m data_tools.prepare_training_dataset \
  --manifest datasets/panda_place_bowl/manifest.json \
  --output-dir datasets/panda_place_bowl/training_place_bowl_v2_300 \
  --training-only
```

The preparer requires a new output directory. After export/input checks pass,
user-run upload (no upload occurs during local collection):

```bash
caffeinate -i .venv-training/bin/hf upload-large-folder \
  Maxalderone/robot-openvla-panda-place-bowl-300 \
  datasets/panda_place_bowl/training_place_bowl_v2_300 \
  --repo-type dataset --private --num-workers 4
```

The cloud downloader recognizes `training_place_bowl_v2_300` and requires the
revision from the completed upload. A future training configuration must disable
held-out validation rather than asking for the absent validation split.
