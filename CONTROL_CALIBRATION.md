# Local control and camera checks

The original measurements below describe the historical hand-origin/2 s setup.
New execution uses `panda_grasp_v1` with a grasp point and 0.4 s interval. The
calibrator now reports action-boundary errors separately from 800 additional
diagnostic settling ticks. See [PANDA_DEMONSTRATIONS.md](PANDA_DEMONSTRATIONS.md).

Verified locally on 2026-10-10. No rented GPU or model inference was used.
These checks establish the Panda controller's behavior; they do not establish
that pretrained Bridge predictions are appropriate for this Panda scene.

## Reproduce

```bash
.venv/bin/python calibrate_controls.py
```

Each translation/rotation case resets and settles the same scene, passes a known
seven-number command through the actual Bridge adapter, solves IK, executes
1,000 physics ticks, and measures the resulting position and orientation.
No size retries are used: all 12 movements succeeded at full requested size.
Tests use 1 cm translations and 0.05 rad rotations in each direction. Acceptance
requires the correct sign, translation error below 0.5 mm, and rotation error
below 0.003 rad, including unwanted movement of other axes.

Results and before/after images are written to a new
`outputs/control_calibration_<timestamp>/` directory. `calibration.json` contains
all measurements. The command exits unsuccessfully if a movement or gripper
check fails. It does not open a viewer or contact a server.

To preview the **actual cached image processor**, without loading the 7B model:

```bash
.venv-openvla/bin/python preview_model_input.py outputs/<calibration-directory>/camera_home.png
```

Replace `<calibration-directory>` with the directory printed by the first command.
The default model location comes from `.cache-openvla/model_revision.json`.
Alternatively supply `--model-dir /path/to/local/checkpoint`. The command saves
`model_input_preview.json` and two de-normalized backbone input PNGs next to the
source image. It uses local files only and does not download weights.

## Measured results

Recorded run: `outputs/control_calibration_20261010_084356_786085`.

- All **14** checks passed: 12 movement directions plus closed/open gripper.
- Position discrepancy across all cases: at most **0.100 mm** (rounded).
- Rotation discrepancy: at most **0.001071 rad**, approximately 0.061 degrees.
- Bridge gripper 0 maps to actuator 0 and closed fingers; 1 maps to actuator 255
  and open fingers. Finger-joint travel sums were approximately 0 and 80 mm.
  These sums are not a calibrated inner-pad clearance measurement.
- Camera image is RGB, 640 by 480. The cube appears near pixel (452, 243),
  confirmed by the rendered red pixels near (451, 243). Positive image Y is down.
- At the initial hand pose, positive 1 cm X projects about (+3.46, +0.74) pixels;
  Y projects (+2.22, -0.67); Z projects (+0.53, -3.97). These are geometric
  projections at this pose, not a constant camera-to-world conversion.
- The cached processor produces a tensor of shape **[1, 6, 224, 224]**:
  two normalized RGB backbone inputs. Its `resize-naive` strategy resizes the
  full rectangle to a square. The cube remains visible in both previews, with
  73 pixels meeting the scene-specific red threshold. Visibility does not prove
  model recognition or task understanding.

## Comparison with upstream conventions

OpenVLA's official zero-shot example uses `bridge_orig` for a **WidowX** robot
in Bridge environments. That statistics key does not calibrate camera extrinsics
or change a Panda into a WidowX. See the
[OpenVLA usage example](https://github.com/openvla/openvla#Getting-Started).

The [Bridge training relabeling code](https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/utils/data_utils.py)
computes the first six action coordinates as successive recorded state
differences. The [Bridge pose conversion code](https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_envs/utils/transformation_utils.py)
stores orientation as Euler coordinates relative to the robot's default
rotation. A difference of those Euler coordinates is not generally identical
to our controller's left-composed world rotation quaternion. Our single-axis
checks verify the local controller; they do not resolve that semantic difference
for an arbitrary starting orientation or combined rotations.

The [official Bridge evaluation](https://github.com/openvla/openvla/blob/main/experiments/robot/bridge/run_bridgev2_eval.py)
uses a 5 Hz control frequency and disables optional center cropping. Its
[image utility](https://github.com/openvla/openvla/blob/main/experiments/robot/bridge/bridgev2_utils.py)
applies JPEG encode/decode and antialiased Lanczos resizing before processing.
Our direct PNG-to-processor route instead uses the checkpoint's bicubic resize.
Both keep the full scene, but exact pixel preprocessing differs. We have not
measured whether that difference affects task performance.

Our scene uses a synthetic elevated camera, different robot geometry, a table
at world Z=0.25 m, and a different starting pose. The source's workspace bounds
belong to WidowX and should not be copied into Panda world coordinates.
There is no known rigid alignment between the two scenes that justifies changing
axis signs based solely on the previous drift.

## Control point and next decision

The remote controller tracks the **hand body origin**. The scripted pick/place
uses an approximate grasp point **0.103 m along hand-local Z**. At home these
are world Z=0.624 m and Z=0.521 m respectively. Translating either by the same
world delta is equivalent with fixed orientation, but rotating about different
points moves the grasp point differently. Choosing the control point for future
demonstrations must be explicit; these checks do not establish the WidowX tool
frame's alignment with Panda's hand.

Conclusion: no swapped axis, reversed gripper, or gross local units error was
found. The drift remains consistent with a pretrained model operating in an
unmatched camera/robot/task domain, with orientation and control-point semantics
also unresolved. Do not claim a calibrated model policy or flip Y to make one
trajectory look better.

The next cloud check can be limited to 20 predictions to assess the new rejection
handling and save fresh task-progress evidence:

```bash
.venv/bin/python cloud_session.py --steps 20
```

Paste the new pod's exposed-TCP SSH command when prompted. Use the existing
Docker template and Global model volume; these local utilities need no image
rebuild. Push the code first if the pod needs the latest server scripts. Review
raw and executed actions, cube distance, rejections, images, and timings; do not
count successful IK as successful picking. The command does not stop pod billing.
No cloud trial was launched as part of this calibration.
