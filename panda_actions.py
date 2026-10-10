"""Versioned Panda demonstration/execution contract (independent of Bridge)."""

import numpy as np

CONVENTION_ID = 'panda_grasp_v1'
GRASP_OFFSET = (0.0, 0.0, 0.103)
PHYSICS_STEPS = 200
TRANSLATION_LIMIT = 0.01
ROTATION_LIMIT = 0.05


def action_contract():
    return dict(id=CONVENTION_ID,
                components=['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper'],
                translation_frame='MuJoCo world', translation_units='metres',
                rotation_units='radians',
                rotation_rule='R_target = Rz(dyaw) @ Ry(dpitch) @ Rx(droll) @ R_current',
                control_body='hand', control_offset_body_metres=list(GRASP_OFFSET),
                gripper='absolute: 0 closed, 1 open; controller = 2*g-1',
                translation_norm_limit_metres=TRANSLATION_LIMIT,
                rotation_norm_limit_radians=ROTATION_LIMIT,
                physics_steps_per_action=PHYSICS_STEPS,
                label='bounded accepted command actually executed; measured motion is separate',
                observation='RGB workspace_camera immediately before command; frozen during prediction')


def to_controller_action(values):
    action = np.asarray(values, dtype=float).copy()
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise ValueError('Expected seven finite action values')
    if not 0 <= action[6] <= 1:
        raise ValueError('Gripper command must be between 0 and 1')
    for section, limit in ((slice(0, 3), TRANSLATION_LIMIT), (slice(3, 6), ROTATION_LIMIT)):
        magnitude = float(np.linalg.norm(action[section]))
        if magnitude > limit:
            action[section] *= limit / magnitude
    action[6] = 2 * action[6] - 1
    return action


def from_controller_action(values):
    """Convert the accepted, possibly reduced command into a dataset label."""
    action = np.asarray(values, dtype=float).copy()
    if action.shape != (7,) or not np.all(np.isfinite(action)) or not -1 <= action[6] <= 1:
        raise ValueError('Invalid executed controller command')
    action[6] = (action[6] + 1) / 2
    return action


def make_controller(model, data):
    from panda_controller import PandaController
    return PandaController(model, data, tool_offset=GRASP_OFFSET)
