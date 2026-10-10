# Panda convention and expert demonstrations

Implemented and verified locally on 2026-10-10; no cloud GPU or model inference.
This starts a Panda dataset. It does not fine-tune OpenVLA.

## Shared contract: `panda_grasp_v1`

`panda_actions.py` defines the contract for simulation, local/remote execution,
recording and replay:

| Field | Meaning |
| --- | --- |
| XYZ | World-frame metres; delta vector norm limited to 0.01 m. |
| Rotation | Radians; `R_target = Rz(dyaw) @ Ry(dpitch) @ Rx(droll) @ R_current`; delta vector norm limited to 0.05 rad. Not subtraction of absolute Euler coordinates. |
| Gripper | Absolute: 0 closed, 1 open; internally `2*g-1`. |
| Control point | Approximate grasp point at hand-local `[0, 0, 0.103]` m. |
| Action duration | 200 physics ticks at current 0.002 s timestep: **0.4 simulated seconds**. Physics freezes during cloud prediction. |
| Observation | RGB `workspace_camera` immediately before command execution. |
| Label | Accepted command actually executed, including bounds/retry reduction. Original proposal and measured displacement are separate fields. |

Use `PandaEnv` or `make_controller()` for this interface. The base
`PandaController` still accepts internal -1/+1 commands and arbitrary explicit
tool offsets. Historical saved-state tests intentionally retain hand-origin IK.
New remote runs save `action_contract.json` and `executed_panda_action`; autonomous
rollouts remain `training_eligible: false`.

**Behavior change:** new runtime uses the grasp point and 0.4 s interval, replacing
hand-origin control and 2.0 s intervals. Historical timing/trajectory measurements
retain their original meaning. Pretrained Bridge output remains a provisional
mapping; defining Panda commands does not calibrate the pretrained policy.

## Record, replay, prepare

```bash
.venv/bin/python panda_demonstrations.py
```

This headless command records and automatically replays a scripted expert
pick/place 12 cm in positive world Y. Known cube coordinates guide the script;
MuJoCo contacts perform the grasp, without teleportation, welds or lifting forces.
Each action obeys the shared bounds and fixed duration. Rejected actions abort
the expert episode and are not executed labels. Successful retries record the
reduced command, keeping the original proposal separately.

The cube must rise at least 8 cm and end within 15 mm horizontally and 10 mm
vertically of the destination. Replay restores initial integration state and
controller targets/correction counter **once**, then executes saved commands
sequentially. It compares every resulting joint/free-body coordinate with the
recording and checks task success; later cube states are never restored.

Output: `datasets/panda_pick_place/episodes/panda_demo_<timestamp>/`:

- `episode.json`: instruction, source, contract, scene XML hash, MuJoCo version,
  success and training eligibility.
- `transitions.jsonl`: before image + instruction → accepted action → after image.
- `initial_state.npz`: full initial integration and controller state.
- `step_0001/`, etc.: before/after PNGs, states and `transition.json`.
- `replay.json`, `replay_final.png`: replay evidence and episode fingerprint.
- Failed episodes remain ineligible; `rejection.json` retains rejected proposals.

To replay again or export verified episodes:

```bash
.venv/bin/python panda_demonstrations.py --replay datasets/panda_pick_place/episodes/<EPISODE>
.venv/bin/python prepare_panda_dataset.py datasets/panda_pick_place/episodes/<EPISODE>
```

Replace `<EPISODE>` with the printed episode folder name. Standalone exports
default to a new directory in `datasets/panda_pick_place/staging/`.
Supply multiple episode paths to combine them. Replay requires the same MuJoCo
version, scene XML and pinned Menagerie assets; XML hashes do not cover meshes.
Eligibility resets when replay starts and becomes true only after success.

The exporter checks task/replay success, convention, bounds, execution flags,
duration and image availability. A replay fingerprint binds commands, initial
state, image pairs and resulting states; edited data requires fresh replay.
It writes `dataset.jsonl`, `normalization_stats.json`, and `dataset_report.json`.
Statistics use **`panda_grasp_v1`**, not `bridge_orig`; movement dimensions are
normalized and the absolute gripper is not. The JSONL references local images:
bundle episodes and regenerate paths when copying the dataset elsewhere.

**Do not apply Panda statistics to the unchanged pretrained model.** Its existing
Bridge normalization remains appropriate to its original training. A future
adapted checkpoint, training preprocessing and deployment must share the same
Panda statistics/conventions. Training/RLDS integration is not implemented yet.

## Verified locally

Saved episode: `datasets/panda_pick_place/episodes/panda_demo_20261010_091537_724389`:
**72 transitions**, **13.7 cm lift**, **2.0 mm placement error**. Sequential replay
maximum qpos discrepancy was **1.44e-14**. A staging dataset is in
`datasets/panda_pick_place/staging/panda_dataset_verified_v1`. An earlier 2 s action interval caused slipping and
failed placement; that episode is excluded.

```bash
.venv/bin/python -m unittest test_panda_demonstrations test_remote_loop test_remote_single_action test_ik_retries test_cloud_session -v
.venv/bin/python calibrate_controls.py
```

All 14 direction/gripper checks pass at the new grasp point. At the 0.4 s
boundary there is servo lag: translation errors up to about 1.9 mm and rotation
errors up to 0.0087 rad. Calibration separately adds 800 diagnostic settling
ticks to verify direction/settled accuracy (under 0.1 mm and 0.001 rad).
Those extra ticks are not part of dataset/runtime actions. Recorded labels
describe commands, while measured motion remains separate.

One scripted episode verifies the pipeline, not task coverage. It has no nonzero
rotation labels, so those statistics have zero variance. Next collect varied
cube positions, approaches, orientations and validation episodes, then finalize
normalization and implement RLDS/fine-tuning integration. Generated episodes live under `datasets/` and stay
out of GitHub; code and documentation are tracked. No Docker dependency rebuild
is needed for these local changes.

## Phase 3: first varied pilot batch

Run the local collector without launching a pod:

```bash
.venv/bin/python collect_panda_pilot.py
```

It writes a timestamped `datasets/panda_pick_place/panda_pilot_*` folder. Before recording any data,
`collection_plan.json` declares ten distinct cube starting scenes: six training,
two validation and two test. Placement distances vary from 8–13 cm along positive
world Y and instructions state the requested distance. Each episode is verified
for lift, placement and sequential replay. Failures remain documented and are
excluded; the collector never silently assigns replacement scenes to a split.

`collection_report.json` lists results and episode paths. Successful training
episodes alone enter `train_staging/dataset.jsonl` and its normalization statistics.
Validation/test episodes remain separate under their split folders; never mix
neighboring frames of an episode across splits or use held-out actions to compute
training normalization. Episode fingerprints bind the scenario/split metadata.

This is a ten-scene pipeline pilot, not a ready fine-tuning corpus. The camera,
object appearance, initial arm posture and grasp orientation remain fixed;
rotation labels remain zero and destination direction is positive Y. Before
training, broaden coverage within pick-and-place, define evaluation targets,
handle constant normalization dimensions, and add RLDS/training integration.
No OpenVLA predictions are used to label these expert demonstrations. The data
stays in ignored `datasets/`; code and collection plans generated by the collector
are reproducible from the repository.

First collected batch: `datasets/panda_pick_place/panda_pilot_20261010_101733`.
All **10/10 scripted expert episodes** passed task checks and sequential replay:
**425 training transitions**, **142 validation transitions**, and **145 test
transitions** (712 total). Maximum placement error was **2.19 mm**. Training staging
contains only the six training episodes. These results assess the expert data
collector, not OpenVLA performance. No GPU rental or fine-tuning was performed.

## Watch an expert trajectory in the simulator

Every newly recorded expert episode includes `replay.py`. Launch it with the
repository simulation interpreter, for example:

```bash
.venv/bin/python datasets/panda_pick_place/panda_pilot_20261010_101733/train/train_01/replay.py
```

Space starts/pauses/resumes; Esc or closing the window exits. Add `--speed 0.5`
for slower playback. To select another episode, use its `train/`, `validation/`
or `test/` directory. All ten existing pilot episodes now have launchers, listed
in the pilot folder's `WATCH_TRAJECTORIES.md`.

This is an offline visual replay, without a pod or model inference. Expert
recordings currently store before/after states: the viewer interpolates the
intermediate motion, preserving saved endpoints. Contact motion between those
endpoints is approximate. For the independent deterministic physics verification,
use `panda_demonstrations.py --replay <EPISODE_DIR>`; it checks the recorded
commands without opening a viewer. Watching an episode does not alter its data
or replay eligibility.
