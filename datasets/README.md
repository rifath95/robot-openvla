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
