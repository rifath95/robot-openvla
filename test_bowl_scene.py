"""Physics checks for randomized manual cube-to-bowl scenes."""
import unittest

import mujoco
import numpy as np

from panda_start_poses import robot_contacts
from bowl_scene import SCENE_PATH, configure_grip, randomize_scene
from panda_actions import make_controller


class BowlSceneTests(unittest.TestCase):
    def test_random_starts_are_broad_settled_and_repeatable(self):
        positions, cubes, bowls, yaws = [], [], [], []
        for seed in range(12):
            model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
            data = mujoco.MjData(model)
            controller, spec = randomize_scene(model, data, np.random.default_rng(seed))
            self.assertFalse(robot_contacts(model, data))
            self.assertLess(np.max(np.abs(data.qvel)), .02)
            joints = data.qpos[controller.qpos_indices]
            self.assertTrue(np.all(joints >= controller.joint_limits[:, 0]))
            self.assertTrue(np.all(joints <= controller.joint_limits[:, 1]))
            cube, bowl = np.array(spec['cube_position']), np.array(spec['bowl_position'])
            self.assertGreater(np.linalg.norm(cube[:2]-bowl[:2]), .16)
            self.assertGreater(cube[0]-.02, .30)
            self.assertLess(cube[0]+.02, .80)
            self.assertLess(abs(cube[1])+.02, .30)
            self.assertGreater(bowl[0]-.085, .30)
            self.assertLess(bowl[0]+.085, .80)
            self.assertLess(abs(bowl[1])+.085, .30)
            positions.append(spec['starting_pose']['actual_position'])
            cubes.append(cube[:2]); bowls.append(bowl[:2])
            yaws.append(spec['starting_pose']['requested_yaw_radians'])
            if seed == 0:
                qpos = data.qpos.copy()
                _, repeated = randomize_scene(model, data, np.random.default_rng(seed))
                self.assertEqual(spec, repeated)
                np.testing.assert_allclose(data.qpos, qpos, atol=1e-12)
        self.assertGreater(np.ptp(positions, axis=0).min(), .12)
        self.assertGreater(np.ptp(cubes, axis=0).min(), .20)
        self.assertGreater(np.ptp(bowls, axis=0).min(), .20)
        self.assertGreater(np.ptp(yaws), 1.5)

    def test_bowl_physically_supports_the_cube(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        data = mujoco.MjData(model)
        controller, spec = randomize_scene(model, data, np.random.default_rng(42))
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_joint')
        address = int(model.jnt_qposadr[joint])
        data.qpos[address:address+3] = [*spec['bowl_position'][:2], .36]
        data.qvel[:] = 0
        controller.step(1000)
        np.testing.assert_allclose(data.qpos[address:address+2], spec['bowl_position'][:2], atol=.001)
        # Bowl floor is at 0.259 m; cube half height is 0.020 m.
        self.assertAlmostEqual(data.qpos[address+2], .279, delta=.001)

    def test_old_scene_keeps_its_original_state_layout(self):
        original = mujoco.MjModel.from_xml_path(str(SCENE_PATH.with_name('scene.xml')))
        bowl = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.assertEqual(original.nq, bowl.nq)
        self.assertEqual(original.nu, bowl.nu)
        self.assertEqual(mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_BODY, 'bowl'), -1)

    def test_walls_contain_a_cube_pushed_in_eight_directions(self):
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
            data = mujoco.MjData(model)
            controller, spec = randomize_scene(model, data, np.random.default_rng(42))
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_joint')
            address = int(model.jnt_qposadr[joint])
            cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
            floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'bowl_floor')
            # Deliberately slippery floor: containment must come from the walls.
            model.geom_friction[floor, 0] = .01
            model.geom_priority[floor] = 2
            data.qpos[address:address+3] = [*spec['bowl_position'][:2], .279]
            data.qvel[:] = 0
            data.xfrc_applied[cube, :2] = .15*np.array([np.cos(angle), np.sin(angle)])
            controller.step(1500)
            offset = np.linalg.norm(data.qpos[address:address+2]-spec['bowl_position'][:2])
            self.assertGreater(offset, .03)  # It really moved into a wall.
            self.assertLess(offset, .06)
            self.assertLess(data.qpos[address+2], .285)
            data.xfrc_applied[:] = 0
            controller.step(500)
            self.assertLess(np.linalg.norm(data.qpos[address:address+2]-spec['bowl_position'][:2]), .06)

    def test_stronger_contact_grip_holds_through_lift_transfer_and_wait(self):
        forces = []
        for stronger in (False, True):
            model = mujoco.MjModel.from_xml_path(str(SCENE_PATH.with_name('scene.xml')))
            data = mujoco.MjData(model)
            home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home')
            mujoco.mj_resetDataKeyframe(model, data, home)
            if stronger:
                configure_grip(model)
            controller = make_controller(model, data)
            controller.step(500)
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_joint')
            address = int(model.jnt_qposadr[joint])
            initial = data.qpos[address:address+3].copy()
            def move(target, gripper):
                for _ in range(80):
                    delta = target-controller.end_effector_pose()[0]
                    distance = np.linalg.norm(delta)
                    if distance < .0015:
                        return
                    delta *= min(1, .01/distance)
                    self.assertTrue(controller.apply_action([*delta, 0, 0, 0, gripper]).converged)
                    controller.step(200)
                self.fail('Grasp/transfer waypoint did not settle')
            move(initial+[0, 0, .14], 1)
            move(initial+[0, 0, .003], 1)
            controller.apply_action([0, 0, 0, 0, 0, 0, -1])
            controller.step(200)
            forces.append(abs(data.actuator_force[controller.gripper_actuator_id]))
            move(initial+[0, 0, .14], -1)
            move(initial+[0, .06, .14], -1)
            before_wait = data.qpos[address:address+3].copy()
            controller.step(2000)
            self.assertGreater(data.qpos[address+2]-initial[2], .12)
            if stronger:
                self.assertLess(np.linalg.norm(data.qpos[address:address+3]-before_wait), .003)
        self.assertGreater(forces[1], 3*forces[0])


if __name__ == '__main__':
    unittest.main()
