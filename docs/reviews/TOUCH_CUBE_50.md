# Touch-cube dataset expanded to 50 demonstrations

Expanded locally on 2026-10-13 in `datasets/panda_touch_cube`, after the first ten
were reviewed. Forty trajectories were added under the original collection plan;
fingerprints confirm that the first ten were retained unchanged.

| Split | Successful trajectories | Action examples |
| --- | ---: | ---: |
| Training | 34 | 604 |
| Validation | 8 | 148 |
| Test | 8 | 152 |
| Total | 50 | 904 |

The split follows the existing repeating four-training / one-validation / one-test
slot pattern. Every episode has a distinct predeclared cube position. The first
ten retain their spatial cells; the additional forty use independent seeded XY
samples inside X=0.40–0.68 m, Y=−0.16–0.16 m. All forty succeeded at their original
positions; no failed location was replaced by an easier one.

The task remains **touch the red cube.** Only cube position changes. The robot
starts from exactly identical saved joint positions, velocities and controller
targets, facing downward with a closed gripper. Cube orientation is fixed. The
scene contains no bowl or plate. All seven-component labels have zero rotation
and zero gripper opening; only nonidle translation components vary.

Every episode ends on the first action with positive-force physical finger–cube
contact. There is no grasp, lift, retreat or extra stationary label. Episodes use
14–22 actions (5.6–8.8 simulated seconds). The final action can nudge the cube
slightly during its 0.4-second physics interval: observed displacement ranges
from 0.28 to 7.08 mm.

## Verification

- All 50 independent continuous physics action replays succeeded.
- Maximum replay qpos difference is below 6e-15.
- All 50 initial robot states are exactly equal, with identical cube orientation.
- All action labels are translation-only, closed-gripper, and nonidle.
- No earlier action has final finger contact; the final action does.
- All 50 saved-pose replays load their recorded motion frames and correct camera.
- All source images, instructions, transition counts, scene/controller contracts,
  eligibility records and episode fingerprints pass the portable builder's
  source checks. No training bundle was built during this expansion.

Audit: `datasets/panda_touch_cube/REVIEW_AUDIT.json`.
Manifest: `datasets/panda_touch_cube/manifest.json`.
Collection/failure report: `datasets/panda_touch_cube/collection_report.json`.

## Review and continue

`datasets/panda_touch_cube/WATCH_TRAJECTORIES.md` includes all 50 trajectories
in Training, Validation, Test order, each with pose replay and action replay links.
The original `REVIEW_OVERVIEW.png` shows the first ten; `CUBE_POSITIONS.png` shows
all 50 starting positions, coloured by split.

Example action replay for a newly added trajectory:

```bash
.venv/bin/python datasets/panda_touch_cube/train/train_009/replay_actions.py
```

Space starts/pauses/resumes; Esc or closing the window exits. Each episode also
has `replay.py` for recorded-pose playback. Action replay initializes once and
executes recorded commands through the current controller and physics.

The collector can recheck/retain all 50 without replacing them:

```bash
.venv/bin/python -m data_tools.collect_panda_touch --limit 50
```

No training or Hugging Face upload has started for this dataset. The next step,
after review, is to build and verify its portable bundle and choose a short
training experiment. These verification checks establish demonstration validity;
they do not establish learned task success or generalization.
