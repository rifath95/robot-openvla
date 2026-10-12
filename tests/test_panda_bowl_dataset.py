"""Bowl dataset split, replay scene and seven-component start checks."""
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from data_tools.collect_panda_bowl import scenario
from simulation.bowl_scene import SCENE_PATH, CAMERA_NAME, randomize_scene
from simulation.panda_actions import action_contract
from simulation.replay_run import replay_model


class BowlDatasetTests(unittest.TestCase):
    def test_slots_have_exact_splits_unique_ids_and_reproducible_seeds(self):
        specs = [scenario(slot,0) for slot in range(300)]
        self.assertEqual({split:sum(s['split']==split for s in specs)
                          for split in ('train','validation','test')},
                         dict(train=200,validation=50,test=50))
        self.assertEqual(len({s['id'] for s in specs}),300)
        self.assertEqual(len({tuple(s['random_seed']) for s in specs}),300)
        self.assertEqual(scenario(0,0),scenario(0,0))
        self.assertNotEqual(scenario(0,0)['random_seed'],scenario(0,1)['random_seed'])
        self.assertEqual(specs[0]['id'],'train_001')
        self.assertEqual(specs[4]['id'],'validation_001')
        self.assertEqual(specs[5]['id'],'test_001')
        self.assertEqual({s['initial_gripper'] for s in specs[:2]}, {0,1})
        self.assertEqual(len({s['instruction'] for s in specs}),1)

    def test_full_start_varies_roll_pitch_yaw_and_settles_closed_gripper(self):
        values=[]
        for seed in range(6):
            model=mujoco.MjModel.from_xml_path(str(SCENE_PATH))
            data=mujoco.MjData(model)
            controller,spec=randomize_scene(model,data,np.random.default_rng(seed),
                                              full_pose=True,initial_gripper=0)
            start=spec['starting_pose']
            values.append([start['requested_roll_radians'],start['requested_pitch_radians'],
                           start['requested_yaw_radians']])
            self.assertEqual(spec['randomized_components'],['x','y','z','roll','pitch','yaw','gripper'])
            fingers=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)
                     for n in ('finger_joint1','finger_joint2')]
            self.assertLess(np.max(np.abs(data.qpos[model.jnt_qposadr[fingers]])),.001)
            self.assertLess(np.max(np.abs(data.qvel)),.02)
        self.assertTrue(np.all(np.ptp(values,axis=0) > .8))

    def test_replay_uses_saved_bowl_position_and_legacy_scene_stays_default(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            model,camera=replay_model(folder)
            self.assertEqual(camera,'workspace_camera')
            self.assertEqual(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'bowl'),-1)
            metadata=dict(scene_profile='panda_bowl_v1',scenario={'scene':{'bowl_position':[.62,-.12,.254]}})
            (folder/'episode.json').write_text(json.dumps(metadata))
            model,camera=replay_model(folder)
            self.assertEqual(camera,CAMERA_NAME)
            bowl=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'bowl')
            np.testing.assert_allclose(model.body_pos[bowl],[.62,-.12,.254])
            self.assertNotEqual(action_contract(camera=CAMERA_NAME),action_contract())


if __name__=='__main__':
    unittest.main()
