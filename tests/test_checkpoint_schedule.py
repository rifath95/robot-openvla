import json
from pathlib import Path
import os
import unittest
from unittest.mock import patch

from training.checkpoint_schedule import checkpoint_due, validate_schedule
from training.integrate_upstream import integrated_source


class CheckpointScheduleTests(unittest.TestCase):
    def test_requested_schedule(self):
        schedule = [0] + list(range(10, 101, 10)) + [200, 300, 400, 500]
        self.assertEqual(validate_schedule(schedule, 500), schedule)
        with patch.dict(os.environ, {'PANDA_CHECKPOINT_UPDATES': str(schedule)}):
            self.assertEqual([i for i in range(501) if checkpoint_due(i, 100)], schedule)

    def test_default_preserves_interval_without_initial_save(self):
        with patch.dict(os.environ, {'PANDA_CHECKPOINT_UPDATES': ''}):
            self.assertFalse(checkpoint_due(0, 10))
            self.assertTrue(checkpoint_due(10, 10))
            self.assertFalse(checkpoint_due(11, 10))

    def test_short_profile_schedule(self):
        config = json.loads((Path(__file__).resolve().parents[1] /
                             'training/pick_bowl_v2_30_500.json').read_text())
        self.assertEqual(config['max_steps'], 500)
        self.assertEqual(len(validate_schedule(config['checkpoint_updates'], 500)), 15)
        self.assertEqual(config['dataset_bundle'], 'training_bowl_v2_30')
        self.assertEqual(config['expected_split_counts']['train']['episodes'], 20)
        self.assertGreater(config['hard_limit_seconds'], config['max_runtime_seconds'])

    def test_pilot_download_requires_completed_revision(self):
        from cloud.download_training_dataset import bundle_source
        with self.assertRaisesRegex(ValueError, 'requires --revision'):
            bundle_source('training_bowl_v2_30')
        self.assertEqual(bundle_source('training_bowl_v2_30', revision='completed-sha'),
                         ('Maxalderone/robot-openvla-panda-bowl-v2', 'completed-sha', ''))

    def test_reject_invalid(self):
        for schedule in ([10, 0], [0, 10, 10], [-1], [501], [True], '10'):
            with self.assertRaises(ValueError):
                validate_schedule(schedule, 500)

    def test_generated_loop_compiles_and_saves_initial_after_baseline(self):
        source = integrated_source()
        compile(source, 'integrated', 'exec')
        self.assertLess(source.index('validate(vla, cfg, batch_transform, collator, device_id, 0)'),
                        source.index('save_adapter(vla, processor, cfg, vla_dataset, 0)'))
        self.assertIn('or finished:', source)
