# Broad starting-pose demonstrations

The new collection targets **200 additional successful pick-and-place episodes**:
140 train / 30 validation / 30 test. The original 50 episodes and `training_v1`
remain unchanged. Together they provide 250 demonstrations, but the new broad
cohort is indexed separately so we can measure its generalization separately.

## Completed collection

All **200 new episodes** passed task execution, deterministic physics replay,
and the recorded-state coverage/contact audit. They contain **16,016 transitions**;
the combined 250 episodes contain **19,625 transitions**. Ten unsuccessful or
unreachable candidates were excluded and replaced within their preassigned groups.

| Actual measured variation across the 200 | Value |
|---|---|
| Initial hand X / Y / Z spans | 28.5 / 43.6 / 20.8 cm |
| Initial yaw range | −65.7° to +65.7° |
| Initial hand-to-cube distance | 13.5–45.1 cm |
| Saved motion poses audited | 176,176 |

Local evidence: `datasets/panda_pick_place/panda_broad_20261011/COLLECTION_SUMMARY.md`
and `coverage_audit.json`. Combined splits: **176 train / 37 validation / 37 test**.
The expanded RLDS package is now generated as `training_v2`; upload and another
training run are separate steps. See [TRAINING_V2.md](TRAINING_V2.md).

The first 20 episodes count toward the 200. Their recorded initial states passed
task success, deterministic replay and sampled path-contact checks. Measured
hand-position spans were 24.4 cm X / 40.7 cm Y / 17.1 cm Z, with starting yaw
−64.8° to +58.6° and hand-to-cube distances 16.4–43.1 cm. These are actual settled
states, not just requested random numbers.

## What varies

| Variable | Sampling region |
|---|---|
| Initial grasp point X | 0.38–0.67 m, in left/right regions |
| Initial grasp point Y | −0.22–+0.22 m, excluding a small central band |
| Initial grasp point Z | 0.38–0.59 m above the world origin |
| Initial yaw relative to home | ±26° to ±66° |
| Cube X / Y | 0.46–0.64 m / −0.14–+0.14 m, in four regions |
| Placement displacement | Independently sampled 6–16 cm |

Start positions, heights, yaw magnitudes, cube positions and placement destinations
vary within predeclared combination groups. There are 32 combinations of hand XY
quadrant, cube XY quadrant and yaw sign: 22 assigned to training, five to validation
and five to test. Failed candidates are replaced **within their original group**;
no neighboring frames or held-out episodes enter training.

This remains a downward-facing grasp family. Initial yaw changes substantially;
the expert corrects it toward the canonical aligned grasp before closing. Broad
roll/pitch changes and unconstrained random joint configurations are not included.

## How the expert works

IK finds joints for the initial tool pose. We reject unreachable, penetrating or
unstable starts and check the actual settled pose and joint limits. Setup sets
the initial state directly; it is not recorded as an expert action.

From that state, the expert approaches and aligns, descends, closes, lifts,
transfers, lowers, releases and retreats. The normal bounded action adapter and
contact physics execute the commands. The expert knows cube coordinates;
OpenVLA training inputs remain **RGB image + instruction → executed action**.
No new model, image history, cloud inference or training is used for collection.

Every accepted trajectory must lift the cube ≥8 cm, place it within 1.5 cm of
the destination at the right height, and pass full deterministic physics replay
within 1e-6 qpos. Saved frames also undergo joint-limit and contact checks.
These inspect 20-physics-step recording intervals; they are not a continuous
collision-aware planner or proof of safety for arbitrary poses.

## Collect and audit locally

From the repository root:

```bash
.venv/bin/python -m data_tools.collect_panda_broad --limit 20 \
  --output-dir datasets/panda_pick_place/panda_broad_20261011

.venv/bin/python -m analysis.audit_panda_broad \
  datasets/panda_pick_place/panda_broad_20261011

# Continue the same batch, preserving the verified pilot:
.venv/bin/python -m data_tools.collect_panda_broad --limit 200 --workers 4 \
  --output-dir datasets/panda_pick_place/panda_broad_20261011

.venv/bin/python -m analysis.audit_panda_broad \
  datasets/panda_pick_place/panda_broad_20261011
```

Use a new folder name for a new collection. Reusing this directory verifies
existing accepted episodes and resumes missing slots. Four workers use independent
local simulator/graphics contexts; only the parent writes collection manifests.

## Find and replay the results

The generated batch is under `datasets/panda_pick_place/panda_broad_20261011/`.

| File | Purpose |
|---|---|
| `WATCH_TRAJECTORIES.md` | Links to each accepted episode's `replay.py` |
| `COLLECTION_SUMMARY.md` | Measured coverage and verification summary |
| `coverage_audit.json` | Per-episode measured starts and sampled-path audit |
| `pilot_coverage_audit.json` | Archived audit of the first 20 |
| `STARTING_POSES.png` | Six actual starting observations from the pilot |
| `collection_plan.json` | Reproducible seed and withheld combinations |
| `collection_report.json` | Accepted results and excluded candidate failures |
| `manifest.json` | Accepted new episodes and split counts |
| `train_staging/` | New training-only JSONL and normalization statistics |
| `combined_manifest.json` | Original 50 plus new 200, with cohort labels and fixed splits |
| `combined_train_staging/` | All 176 training episodes; excludes 74 validation/test episodes |

Accepted retries may live under `attempts/`; use the manifest or trajectory index
to select them. Failed folders are retained but are not training-eligible.

For example:

```bash
.venv/bin/python datasets/panda_pick_place/panda_broad_20261011/train/train_broad_001/replay.py
```

Space starts/pauses/resumes; Esc or closing the viewer exits. Playback uses saved
motion frames, with no pod needed. Images, instructions, actual executed commands,
physics states and motion frames are saved for each action.

Generated images and data are ignored by Git. The code and this guide are pushed;
the actual dataset remains on this Mac. See [TRAINING_V2.md](TRAINING_V2.md) for
the expanded RLDS package. Upload and another training run are subsequent steps;
the original uploaded dataset/checkpoints are untouched.

Combined files are generated at completion when the original 50-episode manifest
is present locally. They contain 176 train / 37 validation / 37 test trajectories.
The new cohort's withheld-combination checks apply to the broad yaw/position
regions; report legacy and broad evaluation results separately.
The root `datasets/panda_pick_place/WATCH_TRAJECTORIES.md` then links all 250.
