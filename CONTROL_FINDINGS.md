# Local investigation and bounded recovery

The next data preparation step is now implemented: shared grasp-point commands,
expert recording, successful sequential replay and Panda statistics. See
[PANDA_DEMONSTRATIONS.md](PANDA_DEMONSTRATIONS.md). Historical trial metrics below
used hand-origin control with 2 s intervals; new trials use the grasp point and
0.4 s. Task performance of the unchanged pretrained model remains unproven.

Updated 2026-10-10. No model weights were loaded and no GPU was rented for
this investigation. Saved model predictions and MuJoCo states were replayed.

## What the latest cloud trial showed

Run: `outputs/remote_loop_20261010_024804_3098555d`. It completed 96 movements,
including a half-size retry on cycle 96, then rejected cycle 97 at full, half,
and quarter size. The earlier trial stopped at cycle 96 after 95 movements.

| Pose before cycle | Hand-origin distance to cube centre | Mixed-unit Jacobian condition number |
| --- | ---: | ---: |
| 1 | 0.355 m | 12.4 |
| 95 | 0.807 m | 62.0 |
| 96 | 0.816 m | 84.8 |
| 97 | 0.821 m | 145.9 |

83 of 97 predictions requested negative world-Y translation under the current
mapping. The hand drifted away from the cube while the Jacobian became more
poorly conditioned. This supports a harder IK problem in the later poses;
it does not prove physical unreachability. The condition number mixes metres
and rotation units and is only used to compare poses of this same model.

At cycle 97 the nearest current arm joint limit was still about **26.3 degrees**
away. We therefore have no basis for calling this simply a joint-limit violation.
For the full command, increasing the IK budget to 1,000 iterations and reducing
damping from 0.03 to 0.01 did not meet the position/orientation tolerances.
We kept those tolerances and solver settings unchanged.

The model is pretrained on another domain. The camera view, world-axis/Euler
interpretation, hand origin versus grasp point, and Bridge-to-Panda conventions
have not been calibrated. We cannot yet attribute the drift solely to the model
or solely to the mapping. Successful execution of a small command is not successful
pick-and-place. Merely increasing the loop length or reducing every command cannot
resolve that task-level problem.

## Latest short cloud trial review (2026-10-10)

Run: `outputs/remote_loop_20261010_032744_4fbbc1ff`. The instruction remained
`pick up the red cube`; the model was not instructed to move away.

- 26 executed movements, 27 request attempts. Closing the viewer stopped cycle
  27 before a prediction/execution record was saved. Final status is `Stopped`,
  with `stop_reason: user_stop`, not a failure.
- All 26 movements passed full-size IK; no reduced retries or rejections occurred.
  Maximum position tracking error was **0.095 mm**. This run does not exercise
  late-pose recovery or the consecutive-rejection stop.
- Hand-origin distance from the cube increased from **0.355 m to 0.400 m**.
  Distance increased in 21 of 26 movements. Net hand displacement was about
  (-8.74 cm, -13.70 cm, +1.22 cm) in world XYZ.
- 21 of 26 raw predictions requested negative world Y under our provisional
  mapping. Every saved gripper prediction was **0.9961**, near fully open.
  Cube coordinates were effectively unchanged; before/after images show no pickup.
- Mixed-unit Jacobian condition number changed from 12.4 to 11.0; the nearest
  current joint limit was still about 77 degrees away at the end. This short
  trial gives no evidence of the late poorly conditioned pose yet.

| Timing component | Mean over 26 completed cycles |
| --- | ---: |
| Server request-body receive | 0.0357 s |
| Server image decoding/preprocessing | 0.0109 s |
| Image/text processor and GPU transfer | 0.1654 s |
| Model prediction | 0.3363 s |
| Mac request/response round trip | 1.0936 s |
| Robot movement wall time | 2.2310 s |
| Movement plus next image/state save | 2.2657 s |
| Complete cycle | 3.3954 s |

Median model prediction was 0.3063 s; the first prediction took 0.7384 s.
The movements each advance 2.0 seconds of simulated physics. The live viewer
therefore accounts for much of the cycle duration. Local `upload_send` averaged
0.000569 s and `response_body_read` 0.000064 s, but these are socket write/read
measurements, **not** isolated one-way image/action transit times. The round trip
includes network, server work, and response delivery; do not add it to inference
as if they were independent components.

The trial lasted 101.2 seconds locally, including 89.4 active seconds; this does
not include pod startup/model loading and is not a billed-runtime measurement.
Offline replay saved `trajectory_analysis.json` in the run directory. The first
and final images are `step_001/before.png` and `step_026/after.png`.

Conclusion: the observation/cloud/action pipeline and manual stopping worked,
but the predicted behavior did not approach or grasp the cube. Accurate command
tracking does not imply task success. Early task drift exists well before the
later IK rejection. Next work should resolve the remaining orientation/tool-frame
conventions and prepare expert Panda demonstrations; another longer trial would
primarily test recovery, not establish task competence.

## Implemented recovery

1. Validate and bound the seven-number prediction as before: 1 cm translation,
   0.05 radians rotation, and the existing gripper mapping.
2. Solve the full command, then half, quarter, and one-eighth size. Scale only
   translation/rotation; retain the gripper command and starting physical pose.
3. Execute the first solution that meets the unchanged IK tolerances.
4. If every attempt fails, restore all actuator targets, advance no physics,
   save an explicit rejection and the unchanged scene, and request a new image-based
   prediction on the next cycle.
5. After **three consecutive** rejected predictions, stop cleanly with
   `stop_reason: consecutive_rejection_limit`. A successful execution resets the
   counter. Step and active-time budgets still apply.

With a frozen scene, identical instruction, and deterministic prediction, the
model can repeat the same rejected action. New observations are not guaranteed
to help; the finite rejection limit prevents endless requests. Recovery does not
automatically steer toward the cube, reset the scene, or change the instruction.

`--steps` now explicitly limits observation/prediction **cycles**, including
rejected ones. `completed_steps` counts executed movements, while
`rejected_steps` counts held actions. The default allows at most 100 predictions,
not 100 executed movements plus unlimited recovery requests.

Each cycle records `ik_attempts.json`, before/after images and states, and
`execution.json`. Rejected execution records have `executed: false`,
`controller_action: null`, and zero physics/movement time. The summary separates
successful executions from rejections. These are rollout logs, not an expert
demonstration dataset; rejected proposals must not become successful training labels.

IK respects joint limits. Collision avoidance and task-specific workspace bounds
are **not implemented**. Diagnostic metrics are recorded rather than converted
into arbitrary uncalibrated safety thresholds. This recovery handles the simulated
control loop; it is not a physical-robot safety controller.

## Local checks

- Saved cycle 96 selects half size and executes with **0.056 mm** tracking error.
- Saved cycle 97 selects one-eighth size and executes with **0.059 mm** tracking error.
- Tests cover all-rejected target restoration, fresh predictions after holds,
  zero state/image changes on rejected cycles, resetting rejection counts after
  success, and stopping without a traceback after repeated rejection.
- The observation capture now refreshes derived MuJoCo transforms before rendering
  so the saved image corresponds to the saved current joint state.

```bash
.venv/bin/python -m unittest test_ik_retries test_remote_loop test_cloud_session -v
.venv/bin/python analyze_remote_run.py outputs/remote_loop_20261010_024804_3098555d
```

The analysis command saves `trajectory_analysis.json` inside that run directory.
It measures the trajectory, records pose diagnostics, and probes the historical
rejected command with alternative IK iteration/damping settings. It reads saved
states without modifying them or running physics.

## Next intended work

The local known-command checks have now been performed: all 14 movement/gripper
checks passed and the cube remains visible after actual model preprocessing.
See [CONTROL_CALIBRATION.md](CONTROL_CALIBRATION.md) for measured results,
upstream sources, and remaining orientation/control-point/domain differences.
This establishes controller behavior, not calibrated pretrained task performance.

The shared Panda action convention and a replay-verified expert recording pipeline
are now implemented; see [PANDA_DEMONSTRATIONS.md](PANDA_DEMONSTRATIONS.md).
Phase 2 is complete as a remote control pipeline milestone. Phase 3 now focuses
on varied pick-and-place demonstrations, dataset preparation, fine-tuning, and
held-out task evaluation. The latest controller changes still need a combined
long cloud trial; successful local checks do not establish pretrained task success.
See [README.md](README.md#project-phases) for the authoritative phase roadmap.
