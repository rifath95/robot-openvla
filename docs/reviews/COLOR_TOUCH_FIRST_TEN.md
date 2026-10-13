# First ten red/blue touch demonstrations

This is the historical review of the first ten. The dataset has since [expanded to 50 demonstrations](COLOR_TOUCH_50.md), preserving all ten.

Local dataset: `datasets/panda_color_touch/`. This is separate from `datasets/panda_touch_cube/`; existing data and interactive inference remain unchanged.

Both colored cubes are present. Each of five predeclared scenes has two demonstrations with identical starting images and robot states: `touch the red cube.` and `touch the blue cube.` Targets vary across sides of the workspace. This small review batch contains 6 training, 2 validation and 2 test trajectories, with both colors balanced in each split. Scene pairs remain in one split to avoid paired-scene leakage.

The first ten contain 185 actions: 112 training, 35 validation and 38 test. Actions use the existing translation-only, closed-gripper touch convention and stop after verified contact with the instructed cube. Any positive-force contact with the other cube during collection or verification rejects the demonstration. The expert uses known object positions; these are metadata, not extra model inputs.

The added scene has a second free cube body. Collision classification recognizes it as an environment object, while still checking arm contacts and inappropriate table penetration. Existing single-cube scenes use the same rules as before. Replay supports the new scene profile without changing prior replay behavior.

All ten demonstrations passed continuous action replay, and paired first observations are byte-identical. Standalone action replay was also checked for all ten. Fifteen scene and replay regression tests passed. Recorded targets, accepted actions, images, simulator states, contact forces and replay verification are saved per episode.

Review these files:

- [All trajectories, grouped by train/validation/test](../../datasets/panda_color_touch/WATCH_TRAJECTORIES.md)
- [Five paired start/red-end/blue-end images](../../datasets/panda_color_touch/REVIEW_OVERVIEW.jpg)
- [Local dataset details](../../datasets/panda_color_touch/README.md)

Run in a local Mac terminal:

```bash
.venv/bin/python datasets/panda_color_touch/train/train_001/replay_actions.py
```

Press Space to start. This executes recorded actions through physics. The matching blue-target demonstration in the same initial scene is:

```bash
.venv/bin/python datasets/panda_color_touch/train/train_002/replay_actions.py
```

`replay.py` instead plays the saved poses. No model or cloud pod is needed to review demonstrations.

Generation is intentionally stopped at ten for review. The dataset is not packaged or uploaded to Hugging Face yet, and no training has begun. Larger collection and cloud inference for this task will be prepared after reviewing these demonstrations.
