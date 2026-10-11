"""Broad-pose physics checks and isolation of held-out scene combinations."""
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from data_tools.collect_panda_broad import combination_pools, scenario
from simulation.panda_actions import make_controller
from simulation.panda_start_poses import initialize_start_pose


class BroadTests(unittest.TestCase):
    def test_retries_preserve_disjoint_held_out_combinations(self):
        pools = {key: set(value) for key, value in combination_pools().items()}
        for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
            self.assertFalse(pools[left] & pools[right])
        counts = dict.fromkeys(pools, 0)
        for slot in range(200):
            original = scenario(slot, 0)
            counts[original['split']] += 1
            for attempt in (0, 1, 29):
                spec = scenario(slot, attempt)
                self.assertEqual(spec['combination'], original['combination'])
                self.assertIn(tuple(spec['combination']), pools[spec['split']])
                self.assertEqual(spec, scenario(slot, attempt))
        self.assertEqual(counts, dict(train=140, validation=30, test=30))

    def test_distant_starting_poses_really_settle_and_rotate(self):
        model = mujoco.MjModel.from_xml_path('scene.xml')
        positions, rotations = [], []
        for position, yaw in (([.40, -.18, .56], -.95), ([.64, .18, .40], .95)):
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home'))
            mujoco.mj_forward(model, data)
            env = SimpleNamespace(model=model, data=data, controller=make_controller(model, data))
            reference = env.controller.end_effector_pose()[1]
            measured = initialize_start_pose(env, position, yaw, reference)
            np.testing.assert_allclose(measured['actual_position'], position, atol=.003)
            positions.append(measured['actual_position'])
            rotations.append(measured['actual_quaternion_wxyz'])
        self.assertGreater(np.linalg.norm(np.subtract(*positions)), .4)
        self.assertLess(abs(np.dot(*rotations)), .65)

    def test_invalid_pose_cannot_modify_the_arm(self):
        model = mujoco.MjModel.from_xml_path('scene.xml')
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home'))
        mujoco.mj_forward(model, data)
        env = SimpleNamespace(model=model, data=data, controller=make_controller(model, data))
        before = data.qpos.copy()
        with self.assertRaises(ValueError):
            initialize_start_pose(env, [.5, 0, .1], 0, env.controller.end_effector_pose()[1])
        np.testing.assert_array_equal(data.qpos, before)

    def test_reachable_but_colliding_initial_pose_is_rejected(self):
        model = mujoco.MjModel.from_xml_path('scene.xml')
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home'))
        mujoco.mj_forward(model, data)
        controller = make_controller(model, data)
        position, quaternion = controller.end_effector_pose()
        cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_joint')
        address = int(model.jnt_qposadr[cube_joint])
        data.qpos[address:address+3] = data.xpos[controller.hand_id]
        env = SimpleNamespace(model=model, data=data, controller=controller)
        with self.assertRaisesRegex(ValueError, 'penetrating robot contact'):
            initialize_start_pose(env, position, 0, quaternion)


if __name__ == '__main__':
    unittest.main()
