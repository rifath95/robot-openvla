"""The new blue bowl preserves collision geometry and deterministic placement scenes."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from simulation.place_bowl_scene import SCENE_PATH,positions,INSTRUCTION
ROOT=Path(__file__).resolve().parents[1]

class PlaceBowlSceneTests(unittest.TestCase):
    def test_only_bowl_floor_color_changes_in_scene(self):
        old=ET.parse(ROOT/'simulation/scenes/scene_bowl.xml').getroot()
        new=ET.parse(SCENE_PATH).getroot()
        old_floor=old.find(".//geom[@name='bowl_floor']")
        new_floor=new.find(".//geom[@name='bowl_floor']")
        self.assertEqual(new_floor.get('rgba'),'0.22 0.48 0.95 1')
        old_floor.set('rgba',new_floor.get('rgba'))
        self.assertEqual(ET.tostring(old),ET.tostring(new))

    def test_predeclared_separated_positions(self):
        import numpy as np
        for slot in range(50):
            r=positions(slot);self.assertEqual(r,positions(slot))
            self.assertTrue(.18<=np.linalg.norm(np.array(r['cube_xy'])-r['bowl_xy'])<=.32)
        self.assertEqual(INSTRUCTION,'pick up the red cube and place it in the blue bowl.')

if __name__=='__main__':unittest.main()
