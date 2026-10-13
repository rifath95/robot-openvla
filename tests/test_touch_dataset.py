import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from data_tools.collect_panda_touch import scenario
from simulation.panda_actions import make_controller
from simulation.replay_run import replay_model
from simulation.touch_scene import SCENE_PATH, CAMERA_NAME, initialize_scene, finger_cube_contacts


class TouchDatasetTests(unittest.TestCase):
    def test_predeclared_positions_and_splits(self):
        specs=[scenario(i) for i in range(10)]
        self.assertEqual([sum(s['split']==split for s in specs) for split in ('train','validation','test')], [8,1,1])
        self.assertEqual(specs,[scenario(i) for i in range(10)])
        self.assertEqual(len({tuple(s['cube_xy']) for s in specs}),10)
        self.assertEqual(scenario(10)['id'],'validation_002')

    def test_fixed_robot_start_and_no_bowl(self):
        model=mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.assertEqual(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'bowl'),-1)
        starts=[]
        for xy in ([.42,-.12],[.66,.12]):
            data=mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model,data,mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'scene_home'))
            mujoco.mj_forward(model,data)
            env=SimpleNamespace(model=model,data=data,controller=make_controller(model,data))
            initialize_scene(env,xy)
            starts.append((data.qpos[:9].copy(),data.qvel[:9].copy()))
            np.testing.assert_array_equal(data.qpos[-4:],[1,0,0,0])
            self.assertFalse(finger_cube_contacts(model,data))
        np.testing.assert_array_equal(starts[0][0],starts[1][0])
        np.testing.assert_array_equal(starts[0][1],starts[1][1])

    def test_contact_requires_finger_cube_and_positive_force(self):
        model=mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        cube=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'cube_geom')
        table=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'table_top')
        finger_body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'left_finger')
        finger=int(np.flatnonzero(model.geom_bodyid==finger_body)[0])
        contact=lambda other:SimpleNamespace(geom1=cube,geom2=other,dist=-.0001)
        data=SimpleNamespace(contact=[contact(table),contact(finger)])
        def force(m,d,index,result):result[:]=[.2,0,0,0,0,0]
        with patch('simulation.touch_scene.mujoco.mj_contactForce',side_effect=force):
            result=finger_cube_contacts(model,data)
            self.assertEqual(len(result),1)
            self.assertEqual(result[0]['finger'],'left_finger')
        def zero_force(m,d,index,result):result[:]=0
        with patch('simulation.touch_scene.mujoco.mj_contactForce',side_effect=zero_force):
            self.assertFalse(finger_cube_contacts(model,data))

    def test_pose_replay_selects_touch_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            (folder/'episode.json').write_text(json.dumps(dict(scene_profile='panda_touch_v1')))
            model,camera=replay_model(folder)
            self.assertEqual(camera,CAMERA_NAME)
            self.assertEqual(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'bowl'),-1)


if __name__=='__main__':unittest.main()
