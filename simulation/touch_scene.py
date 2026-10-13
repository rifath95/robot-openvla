"""Fixed downward-facing closed hand; only cube XY varies in this task."""
import hashlib
import mujoco
import numpy as np
from common.paths import PROJECT_ROOT
from simulation.panda_start_poses import initialize_start_pose

SCENE_PATH = PROJECT_ROOT / 'simulation/scenes/scene_touch.xml'
CAMERA_NAME = 'touch_camera'
INSTRUCTION = 'touch the red cube.'
START_POSITION = (.50, 0., .45)
CUBE_X_BOUNDS = (.40, .68)
CUBE_Y_BOUNDS = (-.16, .16)


def scene_fingerprint():
    return hashlib.sha256(b''.join(p.read_bytes() for p in (
        SCENE_PATH, PROJECT_ROOT/'scene.xml',
        PROJECT_ROOT/'mujoco_menagerie/franka_emika_panda/panda.xml'))).hexdigest()


def initialize_scene(env, cube_xy):
    xy = np.asarray(cube_xy, dtype=float)
    if (xy.shape != (2,) or not np.isfinite(xy).all() or
        not CUBE_X_BOUNDS[0] <= xy[0] <= CUBE_X_BOUNDS[1] or
        not CUBE_Y_BOUNDS[0] <= xy[1] <= CUBE_Y_BOUNDS[1]):
        raise ValueError('Cube must be inside the declared touch-task region')
    reference = env.controller.end_effector_pose()[1].copy()
    # Always solve and settle the same scene before repositioning the cube.
    # This makes all robot initial positions, velocities and control targets equal.
    starting = initialize_start_pose(env, START_POSITION, 0, reference, gripper=0)
    joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_joint')
    address = int(env.model.jnt_qposadr[joint])
    env.data.qpos[address:address+2] = xy
    env.data.qpos[address+3:address+7] = (1, 0, 0, 0)
    env.data.qvel[env.model.jnt_dofadr[joint]:env.model.jnt_dofadr[joint]+6] = 0
    mujoco.mj_forward(env.model, env.data)
    return dict(starting_pose=starting, cube_xy=xy.tolist(),
                cube_orientation_wxyz=[1, 0, 0, 0], camera=CAMERA_NAME)


def finger_cube_contacts(model, data):
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
    fingers = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
               for name in ('left_finger', 'right_finger')}
    contacts = []
    for index, contact in enumerate(data.contact):
        bodies = {int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)}
        if cube in bodies and bodies.intersection(fingers) and contact.dist <= 1e-5:
            force = np.empty(6)
            mujoco.mj_contactForce(model, data, index, force)
            if force[0] > .01:
                contacts.append(dict(finger=mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, next(iter(bodies.intersection(fingers)))),
                    normal_force_newtons=float(force[0]), distance_metres=float(contact.dist)))
    return contacts
