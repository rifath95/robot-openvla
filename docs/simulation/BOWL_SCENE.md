# Manual cube-to-bowl scene

```bash
.venv/bin/python end_effector_demo.py
```

The task is **pick up the red cube and place it in the blue bowl**. Press **N**
to randomize the arm position/yaw, cube and bowl. Use `--seed 42` to reproduce
the start and reset sequence. The keyboard controls are listed in the README.
`launch_sim.py` uses the same scene with the standard actuator-control viewer.

The keyboard demo opens the fixed `bowl_camera` view: position
`[1.02, -0.85, 1.03]`, aimed at `[0.52, 0, 0.36]`, with a 45-degree vertical
field of view. This slightly more sideways view improves visibility beneath
the fingers while keeping the bowl interior visible. Camera settings are saved
in the scene initialization metadata and remain fixed when pressing **N**.
The inherited `workspace_camera` remains available; the original scene is unchanged.
In a future bowl-task recorder, use `bowl_camera` consistently for demonstrations
and inference.

The fixed bowl has a 15 cm interior diameter, a floor at 0.259 m and a rim at
0.314 m. Twenty-four overlapping convex wall segments create a hollow collision
bowl. Bring the held cube above the rim before lowering it inside; release it
over the interior. The walls contain ordinary sideways motion, but a sufficiently
large push or placement on the rim can still tip or eject the cube.

## Gripper settings

`bowl_scene.configure_grip()` applies the following settings before simulation:

| Setting | Original scene | New manual bowl scene |
|---|---:|---:|
| Gripper position gain | 100 | 400 |
| Velocity damping gain | 10 | 20 |
| Sliding friction on collision pads | 1 | 2 |
| Pad contact dimensions | 3 | 4 (includes torsional friction) |
| Pad contact time constant | 0.02 s | 0.005 s |
| Actuator control range | 0–255 | 0–255 |
| Actuator force limit | −100 to +100 | −100 to +100 |

The control scaling changes with the position gain so the opening range remains
the same. Fingertip pad contact priority is 1; this makes those pad friction
settings apply to cube contacts. See the official
[MuJoCo XML reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html)
for actuator gain/bias, contact dimensions and priority semantics.
The cube is held through contact forces, with no weld or artificial attachment.
Poor alignment or grasping at the fingertips can still cause slipping.

The original `scene.xml`, vendored Panda model, old datasets and checkpoints
are unchanged. `simulation/scenes/scene_plate.xml` retains the earlier flat plate. New recordings
must use both `simulation/scenes/scene_bowl.xml` and the new gripper settings consistently; existing
inference and expert-collection scripts still use the original scene.

## Local checks

```bash
.venv/bin/python -m unittest tests.test_bowl_scene tests.test_panda_broad
.venv/bin/python end_effector_demo.py --headless --seed 42
```

Checks cover 12 varied, settled starts, seed reproducibility, joint limits,
collisions, physical floor support, and containment under a sustained sideways
push in eight directions on a deliberately slippery floor. A contact grasp test
lifts the cube, transfers it 6 cm and holds for four simulated seconds.
In the aligned reference grasp, measured actuator force increased from about
1.99 to 7.98 force units (tendon force in this model). The revised grip held a
13.84 cm lift and moved less than 0.84 mm during the four-second hold. These
measurements are saved in `outputs/previews/bowl_scene_preview/grip_check.json`.
This is a simulation
setting change, not a claim about a physical Panda's gripping specifications.

Preview and initialization metadata: `outputs/previews/bowl_scene_preview/scene.png` and
`scene.json`. The new 250-trajectory dataset has not been generated yet.
