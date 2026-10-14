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

    def test_failed_sweep_layouts_settle_with_cube_in_declared_position(self):
        import mujoco
        import numpy as np
        from simulation.place_bowl_scene import initialize_scene
        model=mujoco.MjModel.from_xml_path(str(SCENE_PATH));data=mujoco.MjData(model)
        for slot in (1003,1008):
            controller,scene=initialize_scene(model,data,np.random.default_rng(0),slot=slot)
            cube=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'cube')
            np.testing.assert_allclose(data.xpos[cube][:2],positions(slot)['cube_xy'],atol=.002)
            self.assertLess(np.max(np.abs(data.qvel)),.02)
            self.assertEqual(scene['starting_pose']['requested_gripper_open_fraction'],1)
            for contact in data.contact:
                names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in (contact.geom1,contact.geom2)]
                if 'cube_geom' in names and any(n.startswith('bowl_') for n in names):
                    self.assertGreaterEqual(contact.dist,0)

    def test_predeclared_separated_positions(self):
        import numpy as np
        for slot in range(50):
            r=positions(slot);self.assertEqual(r,positions(slot))
            self.assertTrue(.18<=np.linalg.norm(np.array(r['cube_xy'])-r['bowl_xy'])<=.32)
        self.assertEqual(INSTRUCTION,'pick up the red cube and place it in the blue bowl.')

if __name__=='__main__':unittest.main()
