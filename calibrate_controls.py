"""Measure known Bridge-shaped commands locally, without a model or cloud.

Each case resets the scene so errors do not accumulate. Saves JSON and RGB
before/after images. This verifies Panda conventions, not cross-robot calibration.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np

from capture_camera import save_rgb_png
from openvla_single_action import adapt_bridge_action
from panda_env import PandaEnv

ROOT = Path(__file__).resolve().parent


def rotation_vector(matrix):
    """Measure small world rotation directly from relative rotation matrices."""
    angle = np.arccos(np.clip((np.trace(matrix) - 1) / 2, -1, 1))
    skew = np.array([matrix[2, 1] - matrix[1, 2],
                     matrix[0, 2] - matrix[2, 0],
                     matrix[1, 0] - matrix[0, 1]])
    return skew * (0.5 if angle < 1e-8 else angle / (2 * np.sin(angle)))


def project(env, point):
    """Pinhole projection: pixel X right, pixel Y down; no occlusion check."""
    camera = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, env.camera)
    local = env.data.cam_xmat[camera].reshape(3, 3).T @ (point - env.data.cam_xpos[camera])
    depth = -local[2]
    if depth <= 0:
        return None
    height, width = env.renderer.height, env.renderer.width
    focal = height / (2 * np.tan(np.deg2rad(env.model.cam_fovy[camera]) / 2))
    return np.array([width / 2 + focal * local[0] / depth,
                     height / 2 - focal * local[1] / depth])


def run_calibration(output_dir):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    cases = []
    with PandaEnv() as env:
        for axis, name in enumerate(('x', 'y', 'z', 'roll', 'pitch', 'yaw')):
            for sign in (1, -1):
                env.reset()
                env.controller.step(500)
                controller = env.controller
                before, _ = controller.end_effector_pose()
                matrix_before = env.data.xmat[controller.hand_id].reshape(3, 3).copy()
                raw = np.zeros(7)
                raw[axis] = sign * (0.01 if axis < 3 else 0.05)
                raw[6] = 1
                action = adapt_bridge_action(raw)
                name_signed = f'{name}_{"positive" if sign > 0 else "negative"}'
                save_rgb_png(folder / f'{name_signed}_before.png', env.observe())
                ik = controller.apply_action(action)
                if ik.converged:
                    controller.step(1000)
                after, _ = controller.end_effector_pose()
                matrix_after = env.data.xmat[controller.hand_id].reshape(3, 3).copy()
                measured = np.r_[after - before, rotation_vector(matrix_after @ matrix_before.T)]
                position_error = float(np.linalg.norm(measured[:3] - action[:3]))
                rotation_error = float(np.linalg.norm(measured[3:] - action[3:6]))
                passed = bool(ik.converged and position_error < 0.0005 and
                              rotation_error < 0.003 and sign * measured[axis] > 0)
                save_rgb_png(folder / f'{name_signed}_after.png', env.observe())
                cases.append(dict(name=name_signed, raw_action=raw.tolist(),
                                  controller_action=action.tolist(), measured_delta=measured.tolist(),
                                  position_error_metres=position_error,
                                  rotation_error_radians=rotation_error, passed=passed))
                print(f'{name_signed}: {"PASS" if passed else "FAIL"}; '
                      f'position error {position_error * 1000:.3f} mm; '
                      f'rotation error {rotation_error:.6f} rad', flush=True)

        env.reset()
        env.controller.step(500)
        fingers = [env.model.jnt_qposadr[mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, f'finger_joint{i}')] for i in (1, 2)]
        gripper = []
        for value, name in ((0, 'closed'), (1, 'open')):
            raw = [0, 0, 0, 0, 0, 0, value]
            ik = env.controller.apply_action(adapt_bridge_action(raw))
            if ik.converged:
                env.controller.step(1000)
            env.controller.end_effector_pose()
            opening = float(env.data.qpos[fingers].sum())
            passed = bool(ik.converged and (opening < 0.002 if value == 0 else opening > 0.075))
            gripper.append(dict(name=name, model_command=value,
                                actuator_command=float(env.data.ctrl[env.controller.gripper_actuator_id]),
                                finger_joint_sum_metres=opening, passed=passed))
            save_rgb_png(folder / f'gripper_{name}.png', env.observe())
            print(f'gripper_{name}: {"PASS" if passed else "FAIL"}; opening {opening * 1000:.2f} mm')

        env.reset()
        env.controller.step(500)
        hand, _ = env.controller.end_effector_pose()
        cube = env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')].copy()
        hand_pixel, cube_pixel = project(env, hand), project(env, cube)
        axes = {}
        for index, name in enumerate(('x', 'y', 'z')):
            delta = np.eye(3)[index] * 0.01
            axes[name] = (project(env, hand + delta) - hand_pixel).tolist()
        rgb = env.observe()
        red = (rgb[:, :, 0].astype(float) > 1.5 * rgb[:, :, 1]) & (rgb[:, :, 0] > 80)
        ys, xs = np.nonzero(red)
        save_rgb_png(folder / 'camera_home.png', rgb)
        # Same approximate grasp point used by the scripted pick/place demo.
        grasp = hand + env.data.xmat[env.controller.hand_id].reshape(3, 3) @ np.array([0, 0, 0.103])
        camera = dict(width=rgb.shape[1], height=rgb.shape[0],
                      hand_pixel=hand_pixel.tolist(), cube_pixel=cube_pixel.tolist(),
                      positive_1cm_axis_pixel_deltas=axes,
                      red_pixel_count=int(len(xs)),
                      red_pixel_centroid=None if not len(xs) else [float(xs.mean()), float(ys.mean())],
                      hand_origin_world=hand.tolist(), approximate_grasp_point_world=grasp.tolist(),
                      control_point_offset_metres=0.103,
                      note='Projection does not measure occlusion; red threshold is scene-specific. '
                           'Finger joint sum is travel, not calibrated inner pad clearance.')
    report = dict(passed=all(case['passed'] for case in cases + gripper),
                  movement_cases=cases, gripper_cases=gripper, camera=camera,
                  scope='Local identity mapping only; no OpenVLA prediction or Bridge-to-Panda extrinsic calibration.')
    (folder / 'calibration.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Results: {folder}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' /
                        datetime.now().strftime('control_calibration_%Y%m%d_%H%M%S_%f'))
    args = parser.parse_args()
    report = run_calibration(args.output_dir)
    if not report['passed']:
        raise SystemExit('Some control checks failed; see calibration.json')


if __name__ == '__main__':
    main()
