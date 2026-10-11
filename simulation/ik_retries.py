"""Bounded pose retries shared by simulation and policy loops."""
import time

def apply_with_ik_retries(controller, bounded_action):
    """Try full, half, quarter, then eighth pose deltas without advancing physics.

    Gripper command is unchanged. Failed trials restore actuator/pose targets,
    including the gripper target that apply_action updates even on failed IK.
    """
    import numpy as np
    original = np.asarray(bounded_action, dtype=float)
    if original.shape != (7,) or not np.all(np.isfinite(original)):
        raise ValueError('Expected seven finite controller action values')
    saved_ctrl = controller.data.ctrl.copy()
    saved_position = controller.target_position.copy()
    saved_quaternion = controller.target_quaternion.copy()

    def restore():
        controller.data.ctrl[:] = saved_ctrl
        controller.target_position = saved_position.copy()
        controller.target_quaternion = saved_quaternion.copy()

    attempts = []
    for scale in (1.0, 0.5, 0.25, 0.125):
        candidate = original.copy()
        candidate[:6] *= scale
        started = time.perf_counter()
        try:
            result = controller.apply_action(candidate)
        except Exception:
            restore()
            raise
        attempts.append({
            'scale': scale, 'controller_action': candidate.tolist(),
            'converged': bool(result.converged), 'iterations': int(result.iterations),
            'position_error_metres': float(result.position_error),
            'orientation_error_radians': float(result.orientation_error),
            'elapsed_seconds': time.perf_counter() - started,
        })
        if result.converged:
            return candidate, result, attempts
        restore()
    return None, result, attempts
