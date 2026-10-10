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

Output: `outputs/panda_demo_<timestamp>/`:

- `episode.json`: instruction, source, contract, scene XML hash, MuJoCo version,
  success and training eligibility.
- `transitions.jsonl`: before image + instruction → accepted action → after image.
- `initial_state.npz`: full initial integration and controller state.
- `step_0001/`, etc.: before/after PNGs, states and `transition.json`.
- `replay.json`, `replay_final.png`: replay evidence and episode fingerprint.
- Failed episodes remain ineligible; `rejection.json` retains rejected proposals.

To replay again or export verified episodes:

```bash
.venv/bin/python panda_demonstrations.py --replay outputs/<episode-directory>
.venv/bin/python prepare_panda_dataset.py outputs/<episode-directory> --output-dir outputs/<new-dataset-directory>
```

Replace placeholders with the printed episode directory and a new output name.
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

Saved episode: `outputs/panda_demo_20261010_091537_724389`:
**72 transitions**, **13.7 cm lift**, **2.0 mm placement error**. Sequential replay
maximum qpos discrepancy was **1.44e-14**. A staging dataset is in
`outputs/panda_dataset_verified_v1`. An earlier 2 s action interval caused slipping and
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
normalization and implement RLDS/fine-tuning integration. Generated episodes stay
out of GitHub; code and documentation are tracked. No Docker dependency rebuild
is needed for these local changes.
