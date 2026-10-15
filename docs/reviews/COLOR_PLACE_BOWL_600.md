# Mixed one/two-cube color-conditioned bowl placement

Dataset root: `datasets/panda_color_place_bowl/`.
Portable bundle: `datasets/panda_color_place_bowl/training_color_place_bowl_v1_600/`.

| Scene | Red requested | Blue requested | Green requested | Yellow requested | Total |
|---|---:|---:|---:|---:|---:|
| One cube | 50 | 50 | 50 | 50 | 200 |
| Two cubes | 100 | 100 | 100 | 100 | 400 |
| Total | 150 | 150 | 150 | 150 | 600 |

The first 50 numbered single-red training demonstrations are copied unchanged
from `panda_place_bowl`; the original files are retained. Their existing placement
and replay certificates are checked against the source fingerprints. The remaining
550 demonstrations are newly simulated, rendered and independently action-replayed.
All 600 are training data, continuing the explicit all-training choice for the
bowl-placement task. Independent simulator sweeps must assess generalization;
training loss cannot serve as held-out evaluation.

Training colors are red, blue, green and yellow. Black, purple, orange and white
remain reserved. Instruction: `pick up the COLOR cube and place it in the blue bowl.`

## Coverage and execution

Single-cube scenes repeat the same 50 cube/bowl layouts across the four colors.
Two-cube scenes use 200 different seeded layouts in the declared workspace. Each
layout is recorded twice, requesting one cube and then the other after a full
reset. Both targets share physical positions, colors, arm pose and initial image.
Color assignments alternate between physical slots. Red/blue and green/yellow
have 34 layouts each; the other four pairs have 33 each, balancing every target
color exactly. Cube/bowl separation is 0.18–0.32 m; cube separation is at least
9.5 cm. Seeds are separate from the existing evaluation sweeps.

The arm starts in the same fixed downward-facing pose with an open gripper.
Cube geometry/orientation, camera, bowl and holding controller are preserved.
Both cubes and bowl are positioned before settling. The expert approaches,
lowers, grasps through contact, lifts, transports, releases above the bowl and
retreats. No unnecessary stationary holds are appended.

Acceptance requires at least 8 cm of lift, settled placement within 4.5 cm of
bowl center and 4 mm of resting height, hand at least 8 cm away, no wrong-cube
placement, and no distractor displacement above 3 mm during the demonstration.
Robot collisions, nonfinite states and action tracking errors above 5 mm reject
the recording. A saved-action replay must match recorded joint positions to
less than 1e-6 and independently satisfy the placement criterion.

The entire plan is fixed before collection. Execution failures do not silently
substitute easier scenes. Partial recordings and errors are retained for inspection;
accepted trajectories are never overwritten. Up to four local processes can
collect separate episodes, while only the parent updates the ordered manifest.

`collection_audit.json` and `COLLECTION_SUMMARY.md` record measured totals,
replay/tracking errors and paired-state/image checks. The tracked audit snapshot
is in [assets/color_place_bowl_600_audit.json](assets/color_place_bowl_600_audit.json).

## Completed collection audit

All **600 demonstrations passed**, comprising **43,344 valid action
examples**. Exactly 50 existing red recordings are unchanged; 550 are new. All
200 matched two-target layouts have identical initial physics states and first
images. The maximum action-replay joint-position difference was
`1.38e-10`; maximum tracking error was
**1.662 mm**. No execution-failure layout
substitutions were made. The direct user Action Replay entry point was also
checked on a full two-cube green-target demonstration.

The completed portable export and full image/action/instruction comparison passed.
The pinned official processor produces `6 × 224 × 224` input pixels and eight
supervised tokens (seven action dimensions plus stop). All upload checksums were
verified after adding the input report. Bundle size: **5.04 GiB**.
**No HF upload or model training has been performed for this new dataset.**

## Review, rebuild and package

Open `datasets/panda_color_place_bowl/WATCH_TRAJECTORIES.md` for all 600 numbered
recordings. Replay and Action Replay buttons directly launch local playback via
`analysis.replay_launcher`; they do not link to Python source files. The original
manual simulator/inference launch commands remain available.

```bash
.venv/bin/python -m data_tools.collect_panda_color_place_bowl --workers 4
.venv/bin/python -m data_tools.audit_color_place_bowl
.venv-training/bin/python -m data_tools.prepare_training_dataset \
  --manifest datasets/panda_color_place_bowl/manifest.json \
  --output-dir datasets/panda_color_place_bowl/training_color_place_bowl_v1_600 \
  --training-only
.venv-training/bin/python -m data_tools.check_training_inputs \
  --dataset datasets/panda_color_place_bowl/training_color_place_bowl_v1_600 \
  --source-manifest datasets/panda_color_place_bowl/manifest.json \
  --processor .cache-openvla/processor-only
```

Packaging requires a fresh output directory. The processor check loads only the
pinned processor/tokenizer, never the 7B model. No HF upload or cloud training is
performed by these commands.
