"""Placement sweeps preserve unseen destinations and report release, not just lift."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cloud.create_sweep_plan import create
from cloud.sweep_plan import validate_saved
from cloud.sweep_worker import summarize
from cloud.scripted_touch_session import write_reports

ROOT=Path(__file__).resolve().parents[1]

class PlacementSweepTests(unittest.TestCase):
    def test_profile_has_100_trials_and_immutable_unseen_layouts(self):
        with tempfile.TemporaryDirectory() as tmp,patch('cloud.create_sweep_plan.output_path',return_value=Path(tmp)/'sweep'):
            folder=create(ROOT/'inference/sweep_profiles/place_bowl_50_all_models.json')
            payload=json.loads((folder/'results.json').read_text())
            self.assertEqual(len(validate_saved(payload)),100)
            self.assertEqual(payload['plan']['checkpoints'],[0,50,100,300,500])
            payload['plan']['cube_positions'][0]['bowl'][0]+=1e-16
            self.assertEqual(len(validate_saved(payload)),100)
            payload['plan']['cube_positions'][0]['bowl'][0]+=.01
            with self.assertRaises(ValueError):validate_saved(payload)

    def test_lift_without_release_is_not_a_placement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);trial=root/'trial';trial.mkdir()
            (trial/'summary.json').write_text(json.dumps(dict(phase='Completed',completed_steps=80,
                task_result=dict(success=False,cube_settled=False,maximum_cube_lift_metres=.15,
                    bowl_center_distance_metres=.01,empty_hand_cube_distance_metres=.02))))
            spec=dict(checkpoint=300,seed=1000,directory='trial',cube_colors=None)
            row=summarize(spec,trial,0)
            self.assertFalse(row['placement_success']);self.assertNotIn('lift_success',row)
            with patch('analysis.place_bowl_sweep_report.add_trial_replays',side_effect=lambda folder,rows,text:text):
                write_reports(root,dict(task='place_bowl'),[row])
            self.assertIn('0.0%',(root/'REPORT.md').read_text())

if __name__=='__main__':unittest.main()
