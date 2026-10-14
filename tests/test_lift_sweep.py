"""Lift sweeps must evaluate unseen scenes and report held lifting rather than touching."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cloud.create_sweep_plan import create
from cloud.sweep_plan import validate_saved,trial_specs
from cloud.sweep_worker import summarize
from cloud.scripted_touch_session import write_reports
from inference.lift_task import cube_position_for_seed

ROOT=Path(__file__).resolve().parents[1]

class LiftSweepTests(unittest.TestCase):
    def test_shared_unseen_positions_and_base_reference(self):
        with tempfile.TemporaryDirectory() as tmp, patch('cloud.create_sweep_plan.output_path',return_value=Path(tmp)/'sweep'):
            folder=create(ROOT/'inference/sweep_profiles/lift_50_all_models.json')
            payload=json.loads((folder/'results.json').read_text());plan=payload['plan']
            self.assertEqual(len(validate_saved(payload)),50)
            self.assertEqual(plan['checkpoints'],[0,50,100,300,500])
            self.assertTrue(plan['include_base_model'])
            self.assertEqual(plan['cube_positions'],[{'red':cube_position_for_seed(seed)} for seed in plan['scene_seeds']])
            self.assertEqual(len(list(trial_specs(plan))),50)
            self.assertEqual(len({s['directory'] for s in trial_specs(plan)}),50)
            plan['cube_positions'][0]['red'][0]+=.001
            with self.assertRaisesRegex(ValueError,'lift positions'):validate_saved(payload)

    def test_touch_does_not_count_as_lift_and_reports_show_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);trial=root/'trial';trial.mkdir()
            spec=dict(checkpoint=50,seed=1000,directory='trial',cube_colors=None)
            def measured(success,height,duration):
                (trial/'summary.json').write_text(json.dumps(dict(phase='Completed',completed_steps=32,
                    task_result=dict(success=success,maximum_cube_lift_metres=height,maximum_continuous_grasp_seconds=duration))))
                return summarize(spec,trial,0)
            failure=measured(False,.03,0)
            self.assertFalse(failure['lift_success'])
            success=measured(True,.074,1.12)
            self.assertTrue(success['lift_success'])
            self.assertEqual(success['maximum_lift_metres'],.074)
            with patch('analysis.lift_sweep_report.add_trial_replays',side_effect=lambda folder,rows,text:text):
                write_reports(root,dict(task='lift',include_base_model=True,elapsed_seconds=60),[success,failure])
            report=(root/'REPORT.md').read_text()
            self.assertIn('50.0%',report)
            self.assertIn('7.40',report)
            self.assertIn('1.12',report)
            self.assertIn('one continuous simulator second',report)

if __name__=='__main__':unittest.main()
