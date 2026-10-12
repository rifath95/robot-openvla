"""Fixed coverage targets balance splits and use actual orientation consistently."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_tools.bowl_coverage import BIN_COUNTS,SPLIT_COUNTS,build_plan,classify,sample
from data_tools.collect_panda_bowl import SPLITS,scenario
from simulation.panda_controller import _euler_xyz_to_quat,_quat_multiply

REFERENCE=np.array([0,2**-.5,2**-.5,0])


def scene(values):
    return dict(cube_position=[*values['cube_xy'],.27],bowl_position=[*values['bowl_xy'],.254],
                reference_quaternion_wxyz=REFERENCE.tolist(),
                requested_cube_yaw_radians=values['cube_yaw'],
                starting_pose=dict(actual_position=values['position'].tolist(),
                    actual_quaternion_wxyz=_quat_multiply(_euler_xyz_to_quat(
                        values['roll'],values['pitch'],values['yaw']),REFERENCE).tolist(),
                    requested_gripper_open_fraction=values['gripper']))


class CoverageTests(unittest.TestCase):
    def test_plan_requires_preserved_first_twenty(self):
        with self.assertRaisesRegex(ValueError,'first twenty'):
            build_plan([])

    def test_rotation_bins_follow_actual_rotation_not_its_inverse(self):
        values=dict(cube_xy=[.4,-.1],bowl_xy=[.6,.1],position=np.array([.6,-.1,.52]),
                    roll=-.6,pitch=.6,yaw=-.5,cube_yaw=0,gripper=0)
        result=classify(scene(values))
        self.assertEqual((result['roll'],result['pitch'],result['yaw']),(0,2,1))

    def test_categories_use_settled_cube_rotation_and_finger_opening(self):
        values=dict(cube_xy=[.4,-.1],bowl_xy=[.6,.1],position=np.array([.6,-.1,.52]),
                    roll=0,pitch=0,yaw=0,cube_yaw=0,gripper=0)
        recorded=scene(values)
        recorded['cube_quaternion_wxyz']=_euler_xyz_to_quat(0,0,-.5).tolist()
        recorded['starting_pose']['actual_gripper_open_fraction']=1
        result=classify(recorded)
        self.assertEqual(result['cube_rotation'],0)
        self.assertEqual(result['gripper'],2)

    def test_balanced_plan_accounts_for_existing_episodes_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            existing=[]
            for slot in range(20):
                spec=scenario(slot,0)
                folder=root/spec['id'];folder.mkdir()
                values=dict(cube_xy=[.4,-.1],bowl_xy=[.6,.1],position=np.array([.6,-.1,.52]),
                            roll=-.6,pitch=.6,yaw=-.5,cube_yaw=0,gripper=spec['initial_gripper'])
                (folder/'episode.json').write_text(json.dumps({'scenario':{'scene':scene(values)}}))
                existing.append(dict(slot=slot,split=spec['split'],episode=str(folder)))
            plan=build_plan(existing)
            self.assertEqual(plan,build_plan(existing))
            self.assertEqual(len(plan['assignments']),280)
            for split,total in SPLIT_COUNTS.items():
                for dimension,size in BIN_COUNTS.items():
                    counts=[0]*size
                    for row in existing:
                        if row['split']==split:
                            recorded=json.loads((Path(row['episode'])/'episode.json').read_text())['scenario']['scene']
                            counts[classify(recorded)[dimension]]+=1
                    for slot,target in plan['assignments'].items():
                        if SPLITS[int(slot)%6]==split:counts[target[dimension]]+=1
                    self.assertEqual(counts,plan['quotas_by_split'][split][dimension])
                    self.assertEqual(sum(counts),total)
                    self.assertLessEqual(max(counts)-min(counts),1)
            target=plan['assignments']['20']
            self.assertEqual(scenario(20,0,target)['coverage_target'],scenario(20,19,target)['coverage_target'])
            rng=np.random.default_rng(42)
            for _ in range(1000):
                values=sample(rng,target)
                if int(np.linalg.norm(values['position']-np.r_[values['cube_xy'],.27])>=.3)==target['hand_distance']:
                    self.assertEqual(classify(scene(values)),target)
                    break
            else:self.fail('No sample retained the assigned combination')


if __name__=='__main__':unittest.main()
