"""Two visible cubes; the language instruction selects the contact target."""
import hashlib
import mujoco
import numpy as np
from common.paths import PROJECT_ROOT
from simulation.touch_scene import CAMERA_NAME, START_POSITION, CUBE_X_BOUNDS, CUBE_Y_BOUNDS
from simulation.panda_start_poses import initialize_start_pose

SCENE_PATH = PROJECT_ROOT / 'simulation/scenes/scene_color_touch.xml'
BODIES = {'red': 'cube', 'blue': 'blue_cube'}


def scene_fingerprint():
    return hashlib.sha256(b''.join(p.read_bytes() for p in (
        SCENE_PATH, PROJECT_ROOT/'mujoco_menagerie/franka_emika_panda/panda.xml'))).hexdigest()


def cube_position(env, color):
    return env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, BODIES[color])].copy()


def initialize_scene(env, positions, *, gripper=0):
    for xy in positions.values():
        if len(xy) != 2 or not np.isfinite(xy).all() or not (
                CUBE_X_BOUNDS[0] <= xy[0] <= CUBE_X_BOUNDS[1] and
                CUBE_Y_BOUNDS[0] <= xy[1] <= CUBE_Y_BOUNDS[1]):
            raise ValueError('Cube outside declared collection region')
    if set(positions) != set(BODIES) or np.linalg.norm(np.array(positions['red'])-positions['blue']) < .12:
        raise ValueError('Require separated red and blue cubes')
    reference = env.controller.end_effector_pose()[1].copy()
    starting = initialize_start_pose(env, START_POSITION, 0, reference, gripper=gripper)
    for color, body in BODIES.items():
        joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT,
                                 'cube_joint' if color == 'red' else 'blue_cube_joint')
        q = int(env.model.jnt_qposadr[joint]); v = int(env.model.jnt_dofadr[joint])
        env.data.qpos[q:q+3] = [*positions[color], .26989224457978134]
        env.data.qpos[q+3:q+7] = [1, 0, 0, 0]
        env.data.qvel[v:v+6] = 0
    mujoco.mj_forward(env.model, env.data)
    return dict(starting_pose=starting, cube_positions=positions, camera=CAMERA_NAME,
                cube_orientation_wxyz=[1, 0, 0, 0])


def finger_cube_contacts(model, data, color):
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODIES[color])
    fingers = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
               for name in ('left_finger', 'right_finger')}
    contacts = []
    for index, contact in enumerate(data.contact):
        bodies = {int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)}
        if cube in bodies and bodies.intersection(fingers) and contact.dist <= 1e-5:
            force = np.empty(6)
            mujoco.mj_contactForce(model, data, index, force)
            if force[0] > .01:
                contacts.append(dict(color=color, normal_force_newtons=float(force[0]),
                                     distance_metres=float(contact.dist)))
    return contacts
