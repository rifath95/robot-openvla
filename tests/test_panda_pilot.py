"""Collection plan keeps entire scenes separate before recording."""
import unittest
from data_tools.collect_panda_pilot import pilot_scenarios
from data_tools.dataset_paths import PANDA_DATASET_ROOT, write_trajectory_index
from pathlib import Path
import tempfile
from unittest.mock import patch

class PilotTests(unittest.TestCase):
    def test_dataset_default_and_nested_replay_links(self):
        self.assertEqual(PANDA_DATASET_ROOT.parts[-2:], ('datasets', 'panda_pick_place'))
        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory).resolve()
            root = workspace/'datasets/pilot'
            episode = root / 'train' / 'train_01'
            episode.mkdir(parents=True)
            with patch('common.replay_links.PROJECT_ROOT',workspace),patch('common.replay_links.CONFIG',workspace/'launcher.json'):
                write_trajectory_index(root, [dict(passed=True, episode=str(episode), id='train_01', split='train')])
            text=(root/'WATCH_TRAJECTORIES.md').read_text()
            self.assertIn('[▶ Replay](http://127.0.0.1:',text)
            self.assertIn('run=datasets%2Fpilot%2Ftrain%2Ftrain_01',text)
            self.assertNotIn('[replay.py]',text)

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
