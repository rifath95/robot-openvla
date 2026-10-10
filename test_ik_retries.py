"""Reproduce rejected cloud action 96 using saved state and real MuJoCo IK."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from openvla_single_action import adapt_bridge_action
from panda_controller import IKResult, PandaController
from remote_loop import apply_with_ik_retries

ROOT = Path(__file__).resolve().parent


class RetryTests(unittest.TestCase):
    def scene(self):
        fixture = json.loads((ROOT / 'tests/fixtures/ik_step96.json').read_text())
        model = mujoco.MjModel.from_xml_path(str(ROOT / 'scene.xml'))
        data = mujoco.MjData(model)
        for key in ('qpos', 'qvel', 'ctrl', 'act'):
            getattr(data, key)[:] = fixture['state'][key]
        data.time = fixture['state']['time']
        return PandaController(model, data), adapt_bridge_action(fixture['action'])

    def test_saved_step96_recovers_at_half_scale_and_tracks(self):
        controller, original = self.scene()
        qpos = controller.data.qpos.copy()
        sim_time = controller.data.time
        action, ik, attempts = apply_with_ik_retries(controller, original)
        self.assertEqual([a['scale'] for a in attempts], [1, 0.5])
        self.assertFalse(attempts[0]['converged'])
        self.assertTrue(ik.converged)
        np.testing.assert_allclose(action[:6], original[:6] * 0.5)
        self.assertEqual(action[6], original[6])
        np.testing.assert_array_equal(qpos, controller.data.qpos)
        self.assertEqual(controller.data.time, sim_time)
        controller.step(1000)
        position, _ = controller.end_effector_pose()
        error = np.linalg.norm(controller.target_position - position)
        self.assertLess(error, 0.005)
        self.assertTrue(np.isfinite(controller.data.qpos).all())
        print(f'Recovered step 96 at half scale; tracking error {error * 1000:.3f} mm')

    def test_all_rejections_restore_targets_and_gripper(self):
        controller, action = self.scene()
        controller.data.ctrl[controller.gripper_actuator_id] = 0
        before_ctrl = controller.data.ctrl.copy()
        before_qpos = controller.data.qpos.copy()
        before_position = controller.target_position.copy()
        before_quaternion = controller.target_quaternion.copy()
        sim_time = controller.data.time
        failure = IKResult(controller.data.qpos[controller.qpos_indices].copy(), 0.01, 0.01, 150, False)
        with patch.object(controller, '_solve_ik', return_value=failure):
            accepted, _, attempts = apply_with_ik_retries(controller, action)
        self.assertIsNone(accepted)
        self.assertEqual([a['scale'] for a in attempts], [1, 0.5, 0.25])
        np.testing.assert_array_equal(controller.data.ctrl, before_ctrl)
        np.testing.assert_array_equal(controller.data.qpos, before_qpos)
        np.testing.assert_array_equal(controller.target_position, before_position)
        np.testing.assert_array_equal(controller.target_quaternion, before_quaternion)
        self.assertEqual(controller.data.time, sim_time)

    def test_quarter_fallback_does_not_accumulate_rejected_deltas(self):
        controller, original = self.scene()
        position, _ = controller.end_effector_pose()
        joints = controller.data.qpos[controller.qpos_indices].copy()
        failure = IKResult(joints, 0.01, 0.01, 150, False)
        success = IKResult(joints, 0, 0, 1, True)
        with patch.object(controller, '_solve_ik', side_effect=[failure, failure, success]):
            action, _, attempts = apply_with_ik_retries(controller, original)
        self.assertEqual(len(attempts), 3)
        np.testing.assert_allclose(action[:6], original[:6] * 0.25)
        np.testing.assert_allclose(controller.target_position, position + original[:3] * 0.25)
        self.assertEqual(action[6], original[6])


if __name__ == '__main__':
    unittest.main()
