"""Read-only pose diagnostics; no physics or model inference is performed."""

def pose_diagnostics(controller):
    import mujoco
    import numpy as np

    model, data = controller.model, controller.data
    position, quaternion = controller.end_effector_pose()
    joint_positions = data.qpos[controller.qpos_indices]
    margins = np.minimum(joint_positions - controller.joint_limits[:, 0],
                         controller.joint_limits[:, 1] - joint_positions)
    jp = np.zeros((3, model.nv))
    jr = np.zeros_like(jp)
    mujoco.mj_jac(model, data, jp, jr, position, controller.hand_id)
    singular = np.linalg.svd(np.vstack((jp[:, controller.dof_indices], jr[:, controller.dof_indices])), compute_uv=False)
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
    return {
        'hand_position_metres': data.xpos[controller.hand_id].tolist(),
        'control_point_position_metres': position.tolist(),
        'control_offset_body_metres': controller.tool_offset.tolist(),
        'hand_quaternion_wxyz': quaternion.tolist(),
        'hand_to_cube_distance_metres': float(np.linalg.norm(data.xpos[controller.hand_id] - data.xpos[cube_id])) if cube_id >= 0 else None,
        'control_point_to_cube_distance_metres': float(np.linalg.norm(position - data.xpos[cube_id])) if cube_id >= 0 else None,
        'joint_limit_margins_radians': margins.tolist(),
        'minimum_joint_limit_margin_degrees': float(np.min(margins) * 180 / np.pi),
        'jacobian_singular_values': singular.tolist(),
        'jacobian_condition_number': float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else None,
        'jacobian_note': 'Diagnostic mixes translational and rotational units. Compare poses of this same model; this is not a universal safety threshold.',
    }
