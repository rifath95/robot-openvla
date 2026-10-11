"""Reproducible, checked random starts for the visible cube-to-plate task."""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from panda_actions import make_controller
from panda_start_poses import initialize_start_pose

SCENE_PATH = Path(__file__).with_name('scene_plate.xml')
INSTRUCTION = 'pick up the red cube and place it on the blue plate'
PLATE_RADIUS = .07


def randomize_scene(model, data, rng, *, max_attempts=100):
    """Reset objects and sample a collision-checked arm position and yaw.

    The fixed plate's model body position is part of this scene's reset state.
    Returns the controller bound to the settled state and its specification.
    Sampling is restricted to the table's reachable interior, not every joint
    configuration or every point on the physical table.
    """
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home')
    plate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'plate')
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_joint')
    if min(home, plate, cube) < 0:
        raise ValueError('The cube-to-plate scene requires scene_home, plate, and cube_joint')
    address = int(model.jnt_qposadr[cube])
    for attempt in range(max_attempts):
        mujoco.mj_resetDataKeyframe(model, data, home)
        cube_xy = rng.uniform([.38, -.22], [.72, .22])
        plate_xy = rng.uniform([.40, -.21], [.70, .21])
        if not .14 <= np.linalg.norm(cube_xy-plate_xy) <= .36:
            continue
        model.body_pos[plate] = [*plate_xy, .254]
        data.qpos[address:address+3] = [*cube_xy, .271]
        mujoco.mj_forward(model, data)
        env = SimpleNamespace(model=model, data=data, controller=make_controller(model, data))
        reference = env.controller.end_effector_pose()[1].copy()
        # Both manipulation locations must have reachable downward-facing poses.
        if not all(env.controller._solve_ik(np.array([*xy, z]), reference,
                                           max_iterations=400).converged
                   for xy, z in ((cube_xy, .274), (plate_xy, .286))):
            continue
        position = rng.uniform([.36, -.24, .38], [.70, .24, .60])
        yaw = float(rng.uniform(-1.15, 1.15))
        try:
            start = initialize_start_pose(env, position, yaw, reference)
        except ValueError:
            continue
        cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        spec = dict(instruction=INSTRUCTION, sampling_attempt=attempt+1,
                    cube_position=data.xpos[cube_body].tolist(),
                    plate_position=data.xpos[plate].tolist(), plate_radius_metres=PLATE_RADIUS,
                    starting_pose=start)
        return env.controller, spec
    raise RuntimeError('No valid random scene found; try another seed')
