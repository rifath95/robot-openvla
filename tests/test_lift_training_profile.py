"""Expanded lift training must use the complete new bundle and explicit schedules."""
import json
from pathlib import Path
import unittest
from cloud.download_training_dataset import bundle_source
from training.checkpoint_schedule import validate_schedule
from training.dataset_profile import select_dataset
ROOT=Path(__file__).resolve().parents[1]

class ExpandedLiftTrainingTests(unittest.TestCase):
    def test_download_requires_completed_upload_revision(self):
        with self.assertRaisesRegex(ValueError,'requires --revision'):
            bundle_source('training_lift_v2_300')
        self.assertEqual(bundle_source('training_lift_v2_300',revision='uploaded-commit'),
            ('Maxalderone/robot-openvla-panda-lift-300','uploaded-commit',''))

    def test_complete_dataset_fixed_batch_and_schedule(self):
        c=json.loads((ROOT/'training/lift_cube_300_1000.json').read_text())
        self.assertEqual(select_dataset(c).name,'training_lift_v2_300')
        with self.assertRaises(ValueError):select_dataset(c,'/workspace/panda-training/training_lift_v1')
        self.assertEqual({k:v['episodes'] for k,v in c['expected_split_counts'].items()},dict(train=200,validation=50,test=50))
        self.assertEqual((c['batch_size'],c['grad_accumulation_steps'],c['batch_policy']),(8,1,'config'))
        for key in ('checkpoint_updates','validation_updates'):
            self.assertEqual(validate_schedule(c[key],c['max_steps']),[0,50,100,300,500,750,1000])
        self.assertEqual(c['hard_limit_seconds']-c['max_runtime_seconds'],3600)

if __name__=='__main__':unittest.main()
