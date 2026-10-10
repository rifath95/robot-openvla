"""Collection plan keeps entire scenes separate before recording."""
import unittest
from collect_panda_pilot import pilot_scenarios
from dataset_paths import PANDA_DATASET_ROOT, write_trajectory_index
from pathlib import Path
import tempfile

class PilotTests(unittest.TestCase):
    def test_dataset_default_and_nested_replay_links(self):
        self.assertEqual(PANDA_DATASET_ROOT.parts[-2:], ('datasets', 'panda_pick_place'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / 'train' / 'train_01'
            episode.mkdir(parents=True)
            write_trajectory_index(root, [dict(passed=True, episode=str(episode), id='train_01', split='train')])
            self.assertIn('(train/train_01/replay.py)', (root/'WATCH_TRAJECTORIES.md').read_text())

    def test_scene_splits_are_disjoint_and_predeclared(self):
        plan = pilot_scenarios()
        self.assertEqual(len(plan), 10)
        self.assertEqual([sum(p['split']==split for p in plan) for split in ('train','validation','test')], [6,2,2])
        scenes = [tuple(p['cube_xy']) for p in plan]
        self.assertEqual(len(set(scenes)), 10)
        self.assertEqual(len(set(p['id'] for p in plan)), 10)
        self.assertTrue(all(str(round(p['destination_offset_xy'][1]*100)) in p['instruction'] for p in plan))

if __name__ == '__main__':
    unittest.main()
