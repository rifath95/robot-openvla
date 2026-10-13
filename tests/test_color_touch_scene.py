"""Two-cube scene and replay selection without a rendered window."""
import json
from pathlib import Path
import tempfile
import unittest
import mujoco

from simulation.color_touch_scene import SCENE_PATH, CAMERA_NAME
from simulation.panda_start_poses import robot_contacts
from simulation.replay_run import replay_model


class ColorSceneTests(unittest.TestCase):
    def test_expansion_keeps_original_pairs_and_balances_pair_splits(self):
        from collections import Counter
        from data_tools.collect_panda_color_touch import pair_spec, PAIRS
        import numpy as np
        specs=[pair_spec(i) for i in range(25)]
        self.assertEqual(Counter(s['split'] for s in specs),
                         {'train':17,'validation':4,'test':4})
        self.assertEqual(specs,[pair_spec(i) for i in range(25)])
        for i,(red,blue) in enumerate(PAIRS):
            self.assertEqual(specs[i]['cube_positions'],{'red':list(red),'blue':list(blue)})
        for spec in specs:
            points=spec['cube_positions']
            self.assertGreaterEqual(np.linalg.norm(np.array(points['red'])-points['blue']),.12)
            for xy in points.values():
                self.assertTrue(.40<=xy[0]<=.68 and -.16<=xy[1]<=.16)

    def test_scene_has_two_distinct_colors_and_full_home_state(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.assertEqual(model.nq, 23)
        red = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'cube_geom')
        blue = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'blue_cube_geom')
        self.assertGreater(model.geom_rgba[red, 0], model.geom_rgba[red, 2])
        self.assertGreater(model.geom_rgba[blue, 2], model.geom_rgba[blue, 0])
        home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home')
        self.assertEqual(list(model.key_qpos[home, -4:]), [1, 0, 0, 0])

    def test_blue_cube_table_contact_is_not_robot_collision(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        data = mujoco.MjData(model)
        home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home')
        mujoco.mj_resetDataKeyframe(model, data, home)
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'blue_cube_joint')
        data.qpos[model.jnt_qposadr[joint]+2] = .269
        mujoco.mj_forward(model, data)
        self.assertGreater(data.ncon, 0)
        self.assertFalse(any('blue_cube' in r['bodies'] for r in robot_contacts(model, data)))

    def test_pose_replay_selects_color_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'episode.json').write_text(json.dumps({'scene_profile': 'panda_color_touch_v1'}))
            model, camera = replay_model(folder)
        self.assertEqual(camera, CAMERA_NAME)
        self.assertEqual(model.nq, 23)


if __name__ == '__main__':
    unittest.main()
