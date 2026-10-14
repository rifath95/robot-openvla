"""Versioned upload and detached training select the intended bundle."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cloud.upload_training_dataset as upload_training_dataset
from training import detached_run
from data_tools.prepare_training_dataset import load_episodes, write_package_report, write_upload_checksums
from simulation.panda_actions import action_contract


class DatasetVersionTests(unittest.TestCase):
    def camera_manifest(self, root, cameras):
        entries = []
        for split, camera in zip(('train', 'validation', 'test'), cameras):
            episode = root / split; episode.mkdir()
            (episode/'episode.json').write_text(json.dumps(dict(
                camera=camera, action_contract=action_contract(camera=camera),
                success=True, training_eligible=True, transition_count=1,
                instruction='pick up the red cube and drop it in the blue bowl.')))
            (episode/'replay.json').write_text(json.dumps(dict(
                passed=True, episode_sha256='verified', replayed_transitions=1)))
            (episode/'transitions.jsonl').write_text(json.dumps(dict(
                index=1, executed=True, instruction='pick up the red cube and drop it in the blue bowl.',
                observation='image.png'))+'\n')
            entries.append(dict(id=split, split=split, episode=str(episode), passed=True))
        manifest = root/'manifest.json'
        manifest.write_text(json.dumps(dict(episodes=entries)))
        return manifest

    def test_bowl_camera_exports_and_preserves_contract(self):
        with tempfile.TemporaryDirectory() as folder, patch(
                'data_tools.prepare_training_dataset.fingerprint', return_value='verified'):
            episodes = load_episodes(self.camera_manifest(Path(folder), ['bowl_camera']*3),
                                     verify_images=False)
            self.assertEqual(episodes[0]['action_contract'], action_contract(camera='bowl_camera'))

    def test_training_only_is_explicit_and_keeps_source_splits(self):
        with tempfile.TemporaryDirectory() as folder, patch(
                'data_tools.prepare_training_dataset.fingerprint', return_value='verified'):
            manifest=self.camera_manifest(Path(folder), ['bowl_camera']*3)
            original=manifest.read_bytes()
            episodes=load_episodes(manifest,verify_images=False,training_only=True)
            self.assertEqual([e['split'] for e in episodes],['train']*3)
            self.assertEqual([e['source_split'] for e in episodes],['train','validation','test'])
            self.assertEqual(manifest.read_bytes(),original)

    def test_export_rejects_mixed_camera_contracts(self):
        with tempfile.TemporaryDirectory() as folder, patch(
                'data_tools.prepare_training_dataset.fingerprint', return_value='verified'):
            manifest = self.camera_manifest(Path(folder), ['bowl_camera', 'workspace_camera', 'bowl_camera'])
            with self.assertRaisesRegex(ValueError, 'same camera/action contract'):
                load_episodes(manifest, verify_images=False)

    def test_manifest_cannot_reassign_a_predeclared_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            episode = root / 'episode'; episode.mkdir()
            (episode/'episode.json').write_text(json.dumps(dict(scenario=dict(id='broad_001',split='test'))))
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(dict(episodes=[dict(id='broad_001',split='train',
                                                               episode=str(episode),passed=True)])))
            with self.assertRaisesRegex(ValueError, 'predeclared'):
                load_episodes(manifest,verify_images=False)

    def test_package_report_is_included_in_upload_checksums(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            counts={s:dict(episodes=1,transitions=3) for s in ('train','validation','test')}
            (root/'manifest.json').write_text(json.dumps(dict(dataset_name='panda_pick_place')))
            (root/'verification.json').write_text(json.dumps(dict(passed=True,split_counts=counts)))
            (root/'input_check.json').write_text(json.dumps(dict(
                all_source_images_actions_instructions_match=True,official_processor_batches=dict(train={}))))
            write_package_report(root);write_upload_checksums(root)
            self.assertIn('PACKAGE_REPORT.md',(root/'SHA256SUMS').read_text())
            self.assertIn('seven float32',(root/'PACKAGE_REPORT.md').read_text())

    def test_upload_uses_new_name_and_refuses_existing_remote_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / 'training_v2'
            dataset.mkdir()
            (dataset / 'verification.json').write_text('{"passed": true}')
            with patch('sys.argv', ['upload_training_dataset.py', '--ssh', 'fake', '--dataset', str(dataset)]), \
                 patch('cloud.upload_training_dataset.parse_ssh', return_value=('root@fake',22,Path('/tmp/fake-key'))), \
                 patch('cloud.upload_training_dataset.ssh_arguments', return_value=['ssh','root@fake']), \
                 patch('cloud.upload_training_dataset.subprocess.run') as run:
                upload_training_dataset.main()
            calls = run.call_args_list
            self.assertIn('test ! -e /workspace/panda-training/training_v2', calls[0].args[0][-1])
            self.assertEqual(calls[1].args[0][-1], 'root@fake:/workspace/panda-training/training_v2')
            self.assertIn('cd /workspace/panda-training/training_v2', calls[2].args[0][-1])

    def test_upload_rejects_untrusted_remote_paths(self):
        for destination in ('/root/data', '/workspace/panda-training/../training_v1',
                            '/workspace/panda-training/data;echo bad', '/workspace/panda-training/'):
            with self.assertRaises(ValueError):
                upload_training_dataset.remote_destination('training_v2',destination)

    def test_detached_cli_passes_selected_dataset_to_training(self):
        def read(path, *args, **kwargs):
            if path.name == 'verification.json':
                return '{"passed": true}'
            return json.dumps(dict(max_steps=1000,validation_max_batches=0,max_runtime_seconds=3600))
        with patch('sys.argv', ['detached_run', '--dataset', '/workspace/panda-training/training_v2',
                               '--session-dir', '/workspace/panda-training/sessions/fake']), \
             patch.object(Path, 'resolve', lambda self: self), \
             patch.object(Path, 'read_text', read), patch.object(Path, 'mkdir'), \
             patch('training.detached_run.preflight',return_value=('fake','test-secret')), \
             patch('training.detached_run.supervise',return_value=0) as supervisor:
            self.assertEqual(detached_run.main(),0)
        command = supervisor.call_args.args[0]
        self.assertEqual(command[command.index('--dataset')+1], '/workspace/panda-training/training_v2')
        self.assertNotIn('/workspace/panda-training/training_v1',command)


if __name__ == '__main__':
    unittest.main()
