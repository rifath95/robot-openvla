# Action replay and pose replay

Pose replay (`replay.py`) renders saved simulator poses, without executing actions.
Action replay (`replay_actions.py`) restores the demonstration's complete initial
MuJoCo integration state and controller targets/correction counter, then executes
the accepted seven-component `action` labels in order through IK and physics.
It uses each recorded physics-step count. It never restores a later recorded
pose to force the result to match, and it does not call a model or cloud server.

Run a bowl demonstration locally:

```bash
.venv/bin/python -m simulation.action_replay datasets/panda_pick_bowl/train/train_001
```

The window starts paused. Space starts/pauses/resumes; Esc or closing the window
stops. `--speed 2` changes display pacing, not physics timesteps. `--no-view`
executes without a window; `--max-steps 10` limits the action sequence.

The default uses the **current controller**, including the holding-offset fix.
The existing bowl dataset was collected before that fix, so replayed motion can
depart from its original recording. To compare with the pre-fix actuator target
assignment used during collection:

```bash
.venv/bin/python -m simulation.action_replay \
  datasets/panda_pick_bowl/train/train_001 --controller collection
```

Collection mode reconstructs that particular controller behavior; it is not a
general historical-version loader. Different scene assets are rejected for bowl
episodes. Other changes in MuJoCo, control code, or assets can affect repeatability.

Each execution saves new states, motion frames and `action_replay_report.json`
under a separate `outputs/action_replay_*` folder. Reports include the executed
action count, task outcome, tracking errors and divergence from recorded poses.
The generated output `replay.py` can subsequently show this new executed motion.
The source dataset, labels, eligibility and existing replay verification remain
untouched. `completed` means the sequence executed, not that the task succeeded;
check `task_result.success` for bowl placement.

Initial local comparison on train_001 (126 actions):

| Controller mode | Lifted cube | Placement success | Final cube distance to bowl centre |
| --- | --- | --- | ---: |
| Collection target assignment | Yes | Yes | 1.55 mm |
| Current holding-offset fix | Yes | No | 113.77 mm |

Collection-mode maximum joint-position difference from the saved recording was
about 7e-13. This confirms the sequence reproduces its original demonstration in
that comparison. The current-controller result establishes a control mismatch
after the recent fix; it does not explain the model's near-zero X/Y predictions.
Do not silently reinterpret the existing successful-recording flags as proof of
success under the changed controller.
