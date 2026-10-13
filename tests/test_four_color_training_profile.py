"""Bind the expanded dataset to its download and fixed A40 training profile."""
import json
from pathlib import Path
import unittest
from cloud.download_training_dataset import bundle_source
from training.dataset_profile import select_dataset,validate_dataset
from training.checkpoint_schedule import validate_schedule

ROOT=Path(__file__).resolve().parents[1]

class FourColorTrainingTests(unittest.TestCase):
    def test_download_requires_explicit_upload_revision(self):
        with self.assertRaisesRegex(ValueError,'requires --revision'):
            bundle_source('training_four_color_touch_v1')
        self.assertEqual(bundle_source('training_four_color_touch_v1',revision='uploaded-sha'),
            ('Maxalderone/robot-openvla-panda-four-color-touch','uploaded-sha',''))

    def test_profile_schedule_and_fixed_batch(self):
        config=json.loads((ROOT/'training/four_color_touch_300_1000.json').read_text())
        self.assertEqual(config['max_steps'],1000)
        self.assertEqual(validate_schedule(config['checkpoint_updates'],1000),[50,100,300,500,1000])
        self.assertEqual(validate_schedule(config['validation_updates'],1000),[0,50,100,300,500,1000])
        self.assertEqual((config['batch_size'],config['grad_accumulation_steps'],config['batch_policy']),(8,1,'config'))
        self.assertEqual(config['expected_split_counts']['train'],dict(episodes=200,transitions=3544))
        self.assertGreater(config['hard_limit_seconds'],config['max_runtime_seconds'])
        self.assertEqual(select_dataset(config).name,'training_four_color_touch_v1')
        with self.assertRaises(ValueError):select_dataset(config,'/workspace/panda-training/training_color_touch_v1')

if __name__=='__main__':unittest.main()
