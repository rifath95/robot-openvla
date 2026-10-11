"""Physics checks for randomized manual cube-to-plate scenes."""
import unittest

import mujoco
import numpy as np

from panda_start_poses import robot_contacts
from plate_scene import SCENE_PATH, randomize_scene


class PlateSceneTests(unittest.TestCase):
    def test_random_starts_are_broad_settled_and_repeatable(self):
        positions, cubes, plates, yaws = [], [], [], []
        for seed in range(12):
            model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
            data = mujoco.MjData(model)
            controller, spec = randomize_scene(model, data, np.random.default_rng(seed))
            self.assertFalse(robot_contacts(model, data))
            self.assertLess(np.max(np.abs(data.qvel)), .02)
            joints = data.qpos[controller.qpos_indices]
            self.assertTrue(np.all(joints >= controller.joint_limits[:, 0]))
            self.assertTrue(np.all(joints <= controller.joint_limits[:, 1]))
            cube, plate = np.array(spec['cube_position']), np.array(spec['plate_position'])
            self.assertGreater(np.linalg.norm(cube[:2]-plate[:2]), .14)
            self.assertGreater(cube[0]-.02, .30)
            self.assertLess(cube[0]+.02, .80)
            self.assertLess(abs(cube[1])+.02, .30)
            self.assertGreater(plate[0]-.07, .30)
            self.assertLess(plate[0]+.07, .80)
            self.assertLess(abs(plate[1])+.07, .30)
            positions.append(spec['starting_pose']['actual_position'])
            cubes.append(cube[:2]); plates.append(plate[:2])
            yaws.append(spec['starting_pose']['requested_yaw_radians'])
            if seed == 0:
                qpos = data.qpos.copy()
                _, repeated = randomize_scene(model, data, np.random.default_rng(seed))
                self.assertEqual(spec, repeated)
                np.testing.assert_allclose(data.qpos, qpos, atol=1e-12)
        self.assertGreater(np.ptp(positions, axis=0).min(), .12)
        self.assertGreater(np.ptp(cubes, axis=0).min(), .20)
        self.assertGreater(np.ptp(plates, axis=0).min(), .20)
        self.assertGreater(np.ptp(yaws), 1.5)

    def test_plate_physically_supports_the_cube(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        data = mujoco.MjData(model)
        controller, spec = randomize_scene(model, data, np.random.default_rng(42))
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_joint')
        address = int(model.jnt_qposadr[joint])
        data.qpos[address:address+3] = [*spec['plate_position'][:2], .31]
        data.qvel[:] = 0
        controller.step(1000)
        np.testing.assert_allclose(data.qpos[address:address+2], spec['plate_position'][:2], atol=.001)
        # White plate surface is at 0.259 m; cube half height is 0.020 m.
        self.assertAlmostEqual(data.qpos[address+2], .279, delta=.001)

    def test_old_scene_keeps_its_original_state_layout(self):
        original = mujoco.MjModel.from_xml_path(str(SCENE_PATH.with_name('scene.xml')))
        plate = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.assertEqual(original.nq, plate.nq)
        self.assertEqual(original.nu, plate.nu)
        self.assertEqual(mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_BODY, 'plate'), -1)


if __name__ == '__main__':
    unittest.main()
