"""Four-color collection coverage and replay appearance checks."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
import mujoco
import numpy as np
from data_tools.collect_panda_four_color_touch import expansion_plan
from data_tools.collect_panda_color_touch import pair_spec
from simulation.color_palette import PALETTE
from simulation.replay_run import replay_model


class FourColorTests(unittest.TestCase):
    def test_complete_plan_balanced_and_preserves_holdouts(self):
        plans=expansion_plan()
        old=[dict(pair_spec(i),cube_colors={'red':'red','blue':'blue'}) for i in range(25)]
        all_plans=old+plans
        self.assertEqual(len(all_plans),150)
        self.assertEqual(Counter(p['split'] for p in all_plans),{'train':100,'validation':25,'test':25})
        self.assertEqual(set(Counter(tuple(sorted(p['cube_colors'].values())) for p in all_plans).values()),{25})
        self.assertEqual(Counter(c for p in all_plans for c in p['cube_colors'].values()),dict.fromkeys(('red','blue','green','yellow'),75))
        self.assertEqual(plans,expansion_plan())
        for p in plans:
            self.assertLess(p['pair_id'],1000)
            self.assertGreaterEqual(np.linalg.norm(np.array(p['cube_positions']['red'])-p['cube_positions']['blue']),.12)

    def test_demo_replay_uses_saved_colors(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            (folder/'episode.json').write_text(json.dumps(dict(scene_profile='panda_color_touch_v1',cube_colors={'red':'green','blue':'yellow'})))
            model,_=replay_model(folder)
            for name,color in [('cube_geom','green'),('blue_cube_geom','yellow')]:
                geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
                np.testing.assert_allclose(model.geom_rgba[geom],PALETTE[color])


if __name__=='__main__':unittest.main()
