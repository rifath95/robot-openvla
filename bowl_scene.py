"""Checked random starts and contact-based grip for the cube-to-bowl task."""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from panda_actions import make_controller
from panda_start_poses import initialize_start_pose

SCENE_PATH = Path(__file__).with_name('scene_bowl.xml')
INSTRUCTION = 'pick up the red cube and place it in the blue bowl'
BOWL_RADIUS = .085


def configure_grip(model):
    """Increase closing pressure and pad friction only in the new task scene.

    Preserve the 0..255 control mapping and force limit. Contacts, not a weld
    or attachment, must support the cube. Call before stepping this model.
    """
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'actuator8')
    if actuator < 0:
        raise ValueError('Panda gripper actuator missing')
    model.actuator_gainprm[actuator, 0] = .04 * 400 / 255
    model.actuator_biasprm[actuator, :3] = [0, -400, -20]
    fingers = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
               for name in ('left_finger', 'right_finger')}
    pads = [g for g in range(model.ngeom) if model.geom_bodyid[g] in fingers
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    if len(pads) != 10:
        raise ValueError('Expected ten Panda fingertip collision pads')
    for geom in pads:
        model.geom_friction[geom] = [2, .02, .002]
        model.geom_condim[geom] = 4
        model.geom_priority[geom] = 1
        # Firmer pad contacts resist slow settling/sliding under the closing load.
        model.geom_solref[geom, :2] = [.005, 1]
        model.geom_solimp[geom, :3] = [.95, .99, .001]
    return dict(position_gain=400, velocity_gain=20,
                control_range=model.actuator_ctrlrange[actuator].tolist(),
                force_range=model.actuator_forcerange[actuator].tolist(),
                pad_friction=[2, .02, .002], pad_contact_dimension=4,
                pad_solref=[.005, 1], pad_solimp=[.95, .99, .001],
                simulated_contact_grip=True)


def randomize_scene(model, data, rng, *, max_attempts=100):
    """Reset objects and sample a collision-checked arm position and yaw.

    The fixed bowl's model body position is part of this scene's reset state.
    Returns the controller bound to the settled state and its specification.
    Sampling is restricted to the table's reachable interior, not every joint
    configuration or every point on the physical table.
    """
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home')
    bowl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bowl')
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_joint')
    if min(home, bowl, cube) < 0:
        raise ValueError('The cube-to-bowl scene requires scene_home, bowl, and cube_joint')
    grip = configure_grip(model)
    address = int(model.jnt_qposadr[cube])
    for attempt in range(max_attempts):
        mujoco.mj_resetDataKeyframe(model, data, home)
        cube_xy = rng.uniform([.38, -.22], [.72, .22])
        bowl_xy = rng.uniform([.40, -.21], [.70, .21])
        if not .16 <= np.linalg.norm(cube_xy-bowl_xy) <= .36:
            continue
        model.body_pos[bowl] = [*bowl_xy, .254]
        data.qpos[address:address+3] = [*cube_xy, .271]
        mujoco.mj_forward(model, data)
        env = SimpleNamespace(model=model, data=data, controller=make_controller(model, data))
        reference = env.controller.end_effector_pose()[1].copy()
        # Both manipulation locations must have reachable downward-facing poses.
        if not all(env.controller._solve_ik(np.array([*xy, z]), reference,
                                           max_iterations=400).converged
                   for xy, z in ((cube_xy, .274), (bowl_xy, .286))):
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
                    bowl_position=data.xpos[bowl].tolist(), bowl_radius_metres=BOWL_RADIUS,
                    bowl_floor_height_metres=.259, bowl_rim_height_metres=.314,
                    gripper_physics=grip, starting_pose=start)
        return env.controller, spec
    raise RuntimeError('No valid random scene found; try another seed')
