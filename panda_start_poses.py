"""Checked, broad starting poses for offline expert collection."""
import mujoco
import numpy as np

from panda_actions import make_controller
from panda_controller import _euler_xyz_to_quat, _quat_multiply, _orientation_error


def robot_contacts(model, data):
    """Return penetrating robot contacts; cube/table contact is expected."""
    environment = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                   for name in ('world', 'table', 'cube', 'plate')}
    contacts = []
    for contact in data.contact:
        bodies = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
        if contact.dist < -1e-5 and any(body not in environment for body in bodies):
            contacts.append(dict(bodies=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                                        for b in bodies], distance=float(contact.dist)))
    return contacts


def initialize_start_pose(env, position, yaw, reference_quaternion):
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Starting position must have three finite coordinates')
    if not (.34 <= position[0] <= .72 and abs(position[1]) <= .26 and .35 <= position[2] <= .65):
        raise ValueError('Starting grasp point outside broad collection bounds')
    if not np.isfinite(yaw) or abs(yaw) > 1.3:
        raise ValueError('Starting yaw must be within 1.3 radians of the home orientation')
    quaternion = _quat_multiply(_euler_xyz_to_quat(0, 0, yaw), reference_quaternion)
    controller = env.controller
    result = controller._solve_ik(position, quaternion, max_iterations=400)
    if not result.converged:
        raise ValueError('Starting pose is unreachable by the current IK solver')
    env.data.qpos[controller.qpos_indices] = result.joint_targets
    env.data.qvel[:] = 0
    env.data.ctrl[controller.actuator_ids] = result.joint_targets
    mujoco.mj_forward(env.model, env.data)
    if robot_contacts(env.model, env.data):
        raise ValueError('Starting pose has a penetrating robot contact')
    env.controller = make_controller(env.model, env.data)
    env.controller.step(500)
    actual_position, actual_quaternion = env.controller.end_effector_pose()
    if not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all() or np.max(
            np.abs(env.data.qvel)) > .02:
        raise ValueError('Starting state is non-finite or still moving after settling')
    if robot_contacts(env.model, env.data):
        raise ValueError('Starting pose develops a penetrating robot contact while settling')
    if np.linalg.norm(actual_position - position) > .003 or np.linalg.norm(
            _orientation_error(quaternion, actual_quaternion)) > .03:
        raise ValueError('Starting pose did not settle within position/orientation tolerances')
    joints = env.data.qpos[env.controller.qpos_indices]
    if np.any(joints < controller.joint_limits[:, 0]) or np.any(joints > controller.joint_limits[:, 1]):
        raise ValueError('Starting joints violate their limits')
    return dict(requested_position=position.tolist(), requested_yaw_radians=float(yaw),
                actual_position=actual_position.tolist(), actual_quaternion_wxyz=actual_quaternion.tolist(),
                actual_joint_positions=joints.tolist(), penetrating_contacts=0)


def unsafe_motion_contacts(model, data):
    """Allow intended gripper contacts, reject arm collisions and deep table penetration."""
    gripper = {'hand', 'left_finger', 'right_finger'}
    unsafe = []
    for contact in robot_contacts(model, data):
        names = set(contact['bodies'])
        allowed = names <= gripper | {'cube', 'table'}
        if not allowed or ('table' in names and contact['distance'] < -.004):
            unsafe.append(contact)
    return unsafe
