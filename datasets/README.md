# Fine-tuning datasets

Generated expert demonstrations and training dataset exports belong here.
Experiment logs, model-driven rollouts and failed prototype runs remain in
`outputs/`. Large generated datasets are excluded from GitHub; this guide and
collection/preparation code are tracked. A fresh clone does not contain the data.
Keep a separate backup before moving to another machine.

Current layout:

```text
datasets/panda_pick_place/
├── panda_pilot_20261010_101733/
│   ├── train/          # 6 trajectories, 425 transitions
│   ├── validation/     # 2 trajectories, 142 transitions
│   ├── test/           # 2 trajectories, 145 transitions
│   ├── train_staging/  # Training-only JSONL and statistics
│   ├── collection_plan.json
│   ├── collection_report.json
│   └── WATCH_TRAJECTORIES.md
├── episodes/           # Earlier verified seed + future individual recordings
└── staging/            # Earlier verified seed export + future standalone exports
```

Every trajectory contains `replay.py`. Open it and select the repository
`.venv/bin/python` interpreter, then Run Python File. Or run from the repo:

```bash
.venv/bin/python datasets/panda_pick_place/panda_pilot_20261010_101733/train/train_01/replay.py
```

Space starts/pauses/resumes; Esc or closing the viewer exits. No cloud, weights or
model inference is needed. Existing expert data stores endpoints: playback
interpolates intermediate poses, preserving recorded endpoints. The physics
verification is a separate sequential replay check.

Future collection defaults:

```bash
.venv/bin/python collect_panda_pilot.py
.venv/bin/python panda_demonstrations.py
.venv/bin/python prepare_panda_dataset.py datasets/panda_pick_place/episodes/<EPISODE>
```

The pilot collector creates a timestamped batch under `datasets/panda_pick_place/`
and generates a trajectory index. Individual episodes go into `episodes/`;
standalone prepared exports go into `staging/`. Explicit `--output-dir` can override
the defaults. Keep held-out episodes out of training and training normalization.

The 10-trajectory pilot and separate seed are pipeline checks, not a complete
fine-tuning corpus. Training has not started. JSONL exports reference absolute
local image paths: after copying to another machine, regenerate exports there.
Replay launchers locate the repository among their parent directories; moving
an episode outside this repository requires the repository to remain available.
See [PANDA_DEMONSTRATIONS.md](../PANDA_DEMONSTRATIONS.md) for eligibility checks.

## Expanded collection

`collect_panda_expansion.py` adds a reproducible 40-scene plan (30 train / 5
validation / 5 test) under `panda_pick_place/panda_expansion_<timestamp>/`. It
combines successful training episodes with the pilot in `combined_train_staging/`
and lists all selected scenes in `combined_manifest.json`; the earlier seed is
separate. Failed attempts remain excluded. New expert episodes include sampled
motion frames; their launchers show recorded motion instead of interpolating
endpoints. See the expansion's `WATCH_TRAJECTORIES.md` to select a trajectory.

Current verified expansion: `panda_pick_place/panda_expansion_20261010_104720/`.
The original pilot plus 40 successful additions yields 50 trajectories:
36 train (2,595 transitions), 7 validation (507), and 7 test (507).
Open [the combined trajectory index](panda_pick_place/WATCH_TRAJECTORIES.md)
to watch all 50. Training JSONL/statistics are in the expansion's
`combined_train_staging/`; earlier seed data remains separate. No fine-tuning
has run. All generated data/index files are excluded from GitHub.

## Training-ready package

`panda_pick_place/training_v1/` contains the portable TFDS/RLDS export of the
50-trajectory experiment: 36 train / 7 validation / 7 test. Images are embedded
in TFRecords; no Mac paths are needed in the cloud. Training-only statistics,
split manifest, verification, input checks and upload checksums accompany it.
This package is generated and ignored by Git. See
[PHASE3_TRAINING.md](../PHASE3_TRAINING.md) for checks and cloud instructions.
