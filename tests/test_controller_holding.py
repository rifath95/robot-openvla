"""Physical regression checks for actuator correction across action boundaries."""
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from common.paths import PROJECT_ROOT
from data_tools.bowl_demonstrations import configure_recorded_scene
from simulation.bowl_scene import SCENE_PATH
from simulation.panda_actions import make_controller, to_controller_action


class HoldingTests(unittest.TestCase):
    fixtures=json.loads((PROJECT_ROOT/'tests/fixtures/bowl_bounce_poses.json').read_text())

    def controller(self,fixture):
        model=mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        configure_recorded_scene(model,fixture['episode'])
        data=mujoco.MjData(model)
        for key in ('qpos','qvel','act','ctrl'):
            getattr(data,key)[:]=fixture['state'][key]
        data.time=fixture['state']['time']
        return make_controller(model,data)

    def motion(self,fixture,action,*,discard_offset=False):
        controller=self.controller(fixture)
        initial=controller.end_effector_pose()[0]
        offset=(controller.data.ctrl[controller.actuator_ids]
                -controller.data.qpos[controller.qpos_indices]).copy()
        result=controller.apply_action(action)
        self.assertTrue(result.converged)
        if discard_offset:
            # Reproduce the previous controller's ideal-target replacement.
            controller.data.ctrl[controller.actuator_ids]=result.joint_targets
        else:
            np.testing.assert_allclose(controller.data.ctrl[controller.actuator_ids],
                np.clip(result.joint_targets+offset,controller.joint_limits[:,0],controller.joint_limits[:,1]))
        positions=[initial]
        for _ in range(10):
            controller.step(20)
            positions.append(controller.end_effector_pose()[0])
        positions=np.asarray(positions)
        return (np.linalg.norm(positions-initial,axis=1).max(),
                np.linalg.norm(positions[-1]-controller.target_position))

    def test_saved_small_actions_no_longer_discard_holding_correction(self):
        for fixture in self.fixtures:
            with self.subTest(source=fixture['source']):
                action=to_controller_action(fixture['action'])
                old_peak,old_error=self.motion(fixture,action,discard_offset=True)
                peak,error=self.motion(fixture,action)
                self.assertGreater(old_peak,.002)
                self.assertLess(peak,.001)
                self.assertLess(peak,old_peak/2)
                self.assertLess(error,old_error)

    def test_five_mm_commands_still_reach_the_new_target(self):
        for fixture in self.fixtures:
            for axis in range(3):
                for sign in (-1,1):
                    with self.subTest(source=fixture['source'],axis=axis,sign=sign):
                        action=np.zeros(7);action[axis]=sign*.005;action[6]=1
                        _,error=self.motion(fixture,action)
                        self.assertLess(error,.0015)


if __name__=='__main__':unittest.main()
