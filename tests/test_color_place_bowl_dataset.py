"""Balanced training mixture and matched target requests; no cloud/model required."""
import copy,json,tempfile,unittest
from collections import Counter
from pathlib import Path
from data_tools.collect_panda_color_place_bowl import build_plan,validate_plan,pair_layout
from simulation.color_place_bowl_scene import validate_layout
from inference.place_bowl_task import positions


class ColorBowlDatasetTests(unittest.TestCase):
    def plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for i in range(50):
                episode=root/f'train_{i+1:03d}';episode.mkdir()
                (episode/'episode.json').write_text(json.dumps(dict(instruction='pick up the red cube and place it in the blue bowl.')))
                layout=positions(i)
                rows.append(dict(id=episode.name,split='train',episode=str(episode),episode_sha256=str(i),cube_positions=dict(red=layout['cube_xy']),bowl_position=layout['bowl_xy']))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(episodes=rows)))
            return build_plan(manifest)

    def test_exact_balance_reuse_and_distinct_pairs(self):
        plan=self.plan();validate_plan(plan)
        self.assertEqual(len(plan),600)
        self.assertEqual(sum('reused_from' in r for r in plan),50)
        self.assertEqual(Counter(r['target_color'] for r in plan),dict.fromkeys(('red','blue','green','yellow'),150))
        self.assertEqual(Counter(tuple(sorted(r['cube_colors'].values())) for r in plan if r['cube_count']==2),
                         {('blue','red'):68,('green','yellow'):68,('green','red'):66,('red','yellow'):66,('blue','green'):66,('blue','yellow'):66})
        # Every single-cube layout is repeated across all four colors.
        for i in range(50):self.assertTrue(all(plan[i]['layout']==plan[i+50*k]['layout'] for k in range(4)))

    def test_unpaired_request_and_wrong_balance_rejected(self):
        plan=self.plan();broken=copy.deepcopy(plan);broken[201]['target_color']=broken[200]['target_color']
        with self.assertRaises(ValueError):validate_plan(broken)
        broken=copy.deepcopy(plan);broken[201]['layout']=copy.deepcopy(broken[201]['layout']);broken[201]['layout']['bowl'][0]+=.001
        with self.assertRaises(ValueError):validate_plan(broken)

    def test_layouts_are_fixed_separated_and_cover_workspace(self):
        layouts=[pair_layout(60000+i) for i in range(200)]
        for i,layout in enumerate(layouts):
            validate_layout(layout,2);self.assertEqual(layout,pair_layout(60000+i))
        pts=[xy for layout in layouts for xy in layout['cubes'].values()]
        self.assertLess(min(p[0] for p in pts),.42);self.assertGreater(max(p[0] for p in pts),.66)
        self.assertLess(min(p[1] for p in pts),-.14);self.assertGreater(max(p[1] for p in pts),.14)

if __name__=='__main__':unittest.main()
