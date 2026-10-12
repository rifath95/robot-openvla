"""Verify bowl-profile data binding and reject incompatible bundles before GPU work."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common.paths import PROJECT_ROOT
from simulation.panda_actions import action_contract
from training.dataset_profile import select_dataset,validate_dataset


class BowlTrainingProfileTests(unittest.TestCase):
    def config(self):
        return json.loads((PROJECT_ROOT/'training/pick_bowl_5000.json').read_text())

    def bundle(self,root):
        path=root/'training_bowl_v1';path.mkdir()
        counts=self.config()['expected_split_counts']
        (path/'manifest.json').write_text(json.dumps(dict(split_counts=counts,action_contract=action_contract(camera='bowl_camera'))))
        (path/'verification.json').write_text(json.dumps(dict(passed=True,split_counts=counts)))
        (path/'input_check.json').write_text(json.dumps(dict(passed=True,split_counts=counts,
                                                 all_source_images_actions_instructions_match=True)))
        return path

    def test_profile_selection_preserves_legacy_and_binds_new_dataset(self):
        self.assertEqual(select_dataset(self.config()),Path('/workspace/panda-training/training_bowl_v1'))
        self.assertEqual(select_dataset({}),Path('/workspace/panda-training/training_v1'))
        with self.assertRaisesRegex(ValueError,'requires training_bowl_v1'):
            select_dataset(self.config(),'/workspace/panda-training/training_v2')

    def test_bowl_data_matches_and_wrong_camera_or_counts_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.bundle(Path(tmp));validate_dataset(self.config(),path)
            for change in (dict(action_contract=action_contract()),dict(split_counts={})):
                manifest=json.loads((path/'manifest.json').read_text());manifest.update(change)
                (path/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):validate_dataset(self.config(),path)

    def test_profile_requires_source_input_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.bundle(Path(tmp));checks=json.loads((path/'input_check.json').read_text())
            checks['all_source_images_actions_instructions_match']=False
            (path/'input_check.json').write_text(json.dumps(checks))
            with self.assertRaisesRegex(ValueError,'source-input check'):validate_dataset(self.config(),path)

    def test_checkpoint_validation_and_wall_limits_are_kept(self):
        config=self.config()
        self.assertEqual(config['max_steps'],5000)
        self.assertEqual(config['save_steps'],1000)
        self.assertEqual(config['validation_every'],1000)
        self.assertEqual(config['validation_max_batches'],0)
        self.assertEqual((config['max_runtime_seconds'],config['hard_limit_seconds']),(14400,18000))
        self.assertTrue(config['require_benchmark'])
        self.assertEqual(config['expected_gpu'],'NVIDIA A40')

    def test_training_command_keeps_bowl_input_and_strips_project_only_flags(self):
        import contextlib
        import io
        from training.train_panda import main
        def selection(config,context,root):
            return dict(config,batch_size=5),dict(source='compatible_benchmark',microbatch=5)
        with tempfile.TemporaryDirectory() as tmp:
            dataset=self.bundle(Path(tmp));output=io.StringIO()
            with patch('sys.argv',['train','--dataset',str(dataset),'--config',
                                  str(PROJECT_ROOT/'training/pick_bowl_5000.json'),
                                  '--output-root',str(Path(tmp)/'runs'),'--prepare-only']), \
                 patch('training.batch_policy.benchmark_context',return_value=dict(gpu='NVIDIA A40')), \
                 patch('training.batch_policy.choose_batch',side_effect=selection), \
                 patch.object(Path,'write_text'),contextlib.redirect_stdout(output):
                main()
            command=output.getvalue()
            self.assertIn('--data_root_dir '+str(dataset.resolve()),command)
            self.assertIn('--max_steps 5000',command)
            self.assertIn('--batch_size 5',command)
            for key in ('dataset_bundle','expected_camera','expected_split_counts'):
                self.assertNotIn('--'+key,command)


if __name__=='__main__':unittest.main()
