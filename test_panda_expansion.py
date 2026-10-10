"""Expanded scenes preserve split isolation and declared variation."""
import unittest
import numpy as np
from collect_panda_expansion import expansion_scenarios
from collect_panda_pilot import pilot_scenarios

class ExpansionTests(unittest.TestCase):
    def test_forty_predeclared_scenes_with_30_5_5_split(self):
        plan=expansion_scenarios()
        self.assertEqual(plan,expansion_scenarios())
        self.assertEqual(len(plan),40)
        self.assertEqual([sum(p['split']==s for p in plan) for s in ('train','validation','test')],[30,5,5])
        starts={tuple(p['cube_xy']) for p in plan}
        self.assertEqual(len(starts),40)
        self.assertFalse(starts & {tuple(p['cube_xy']) for p in pilot_scenarios()})
        self.assertEqual(len({p['id'] for p in plan}),40)

    def test_variation_and_table_bounds(self):
        plan=expansion_scenarios()
        self.assertEqual(len({tuple(p['destination_offset_xy']) for p in plan}),8)
        self.assertEqual(len({p['grasp_yaw_radians'] for p in plan}),4)
        self.assertEqual(len({tuple(p['approach_offset_xy']) for p in plan}),4)
        self.assertEqual(len({p['initial_joint1_radians'] for p in plan}),3)
        for p in plan:
            start=np.array(p['cube_xy']);end=start+p['destination_offset_xy']
            for point in (start,end):
                self.assertTrue(.45<=point[0]<=.65 and -.2<=point[1]<=.2)
            self.assertTrue(.04<=np.linalg.norm(p['destination_offset_xy'])<=.18)
            self.assertTrue(p['instruction'].startswith('pick up the red cube'))

if __name__=='__main__':unittest.main()
