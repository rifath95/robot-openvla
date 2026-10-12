"""Training report synchronization without paid cloud jobs."""
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from cloud import training_cloud_session as launcher
from training.detached_run import acknowledge_download, wait_for_download, watch_session


SESSION = 'panda-train-20261012-010203-123'


class TrainingSyncTests(unittest.TestCase):
    def archive(self, path, files):
        with tarfile.open(path, 'w:gz') as archive:
            for name, content in files.items():
                data = content.encode()
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

    def test_fetch_imports_final_reports_and_keeps_session_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'source.tar.gz'
            self.archive(archive, {'status.json': json.dumps({'training_complete': True}),
                                  'runs/training/metrics.jsonl': '{"update":20}\n',
                                  'runs/training/loss.png': 'plot',
                                  'runs/training/loss_smoothed.png': 'smooth',
                                  'runs/training/checkpoint_000020/adapter_config.json': '{}',
                                  'runs/training/tensorboard/events.out.tfevents.fake': 'event'})
            def download(command, stdout, **kwargs):
                stdout.write(archive.read_bytes())
            with patch.object(launcher.subprocess, 'run', side_effect=download):
                output = launcher.fetch_reports(['ssh', 'fake'], SESSION, root / 'reports')
            self.assertEqual(output.name, SESSION)
            self.assertTrue((output / 'runs/training/loss_smoothed.png').is_file())
            self.assertTrue((output / 'runs/training/checkpoint_000020/adapter_config.json').is_file())

    def test_rejects_traversal_weights_and_invalid_json_before_overwrite(self):
        for name, content in [('../escape.json', '{}'),
                              ('runs/training/adapter_model.safetensors', 'weight'),
                              ('metadata.json', 'invalid')]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                destination = root / 'reports'
                destination.mkdir()
                (destination / 'status.json').write_text('{"previous":true}')
                archive = root / 'source.tar.gz'
                self.archive(archive, {'status.json': '{}', name: content})
                with self.assertRaises(ValueError):
                    launcher.extract_reports(archive, destination)
                self.assertEqual((destination / 'status.json').read_text(), '{"previous":true}')

    def test_grace_expires_when_mac_disconnects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = [0.0]
            def sleep(seconds):
                clock[0] += seconds
            status = {'phase': 'completed'}
            with patch('training.detached_run.time.monotonic', side_effect=lambda: clock[0]), \
                 patch('training.detached_run.time.sleep', side_effect=sleep):
                wait_for_download(root, status, 100, 2)
            self.assertEqual(clock[0], 2)
            self.assertEqual(status['download_state'], 'window_expired')
            self.assertEqual(status['phase'], 'completed')

    def test_download_ack_ends_wait_and_does_not_change_training_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = [0.0]
            def sleep(seconds):
                clock[0] += seconds
                acknowledge_download(root)
            status = {'phase': 'training_failed'}
            with patch('training.detached_run.time.monotonic', side_effect=lambda: clock[0]), \
                 patch('training.detached_run.time.sleep', side_effect=sleep):
                wait_for_download(root, status, 100, 120)
            self.assertEqual(status['download_state'], 'acknowledged')
            self.assertEqual(status['phase'], 'training_failed')
            self.assertLess(clock[0], 1)

    def test_outer_limit_overrides_download_window_and_timeout_skips_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = [0.0]
            status = {'phase': 'completed'}
            with patch('training.detached_run.time.monotonic', side_effect=lambda: clock[0]), \
                 patch('training.detached_run.time.sleep', side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
                wait_for_download(root, status, 1, 120)
            self.assertEqual(clock[0], 1)
            self.assertEqual(status['download_state'], 'outer_limit_reached')
            status = {'phase': 'wall_limit_reached'}
            wait_for_download(root, status, 1, 120)
            self.assertNotIn('download_state', status)

    def test_ack_requires_finished_training_and_watch_returns_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'status.json').write_text(json.dumps({'phase': 'training'}))
            with self.assertRaises(ValueError):
                acknowledge_download(root)
            (root / 'status.json').write_text(json.dumps({'phase': 'awaiting_sync', 'training_complete': True}))
            self.assertEqual(watch_session(root), 0)

    def test_failed_download_is_not_acknowledged(self):
        with patch('sys.argv', ['training_cloud_session', '--ssh', 'fake']), \
             patch.object(launcher, 'parse_ssh', return_value=('root@fake', 22, Path('/tmp/key'))), \
             patch.object(launcher.subprocess, 'run', side_effect=[
                 subprocess.CompletedProcess([], 0, stdout=f'Cloud session started: {SESSION}\n'),
                 subprocess.CompletedProcess([], 0)]) as run, \
             patch.object(launcher, 'fetch_reports', side_effect=OSError('download lost')):
            with self.assertRaisesRegex(SystemExit, 'Saved results remain'):
                launcher.main()
        self.assertEqual(run.call_count, 2)

    def test_keyboard_interrupt_leaves_cloud_job_running(self):
        with patch('sys.argv', ['training_cloud_session', '--ssh', 'fake']), \
             patch.object(launcher, 'parse_ssh', return_value=('root@fake', 22, Path('/tmp/key'))), \
             patch.object(launcher.subprocess, 'run', side_effect=[
                 subprocess.CompletedProcess([], 0, stdout=f'Cloud session started: {SESSION}\n'),
                 KeyboardInterrupt]) as run:
            with self.assertRaisesRegex(SystemExit, 'it continues'):
                launcher.main()
        self.assertEqual(run.call_count, 2)

    def test_fetch_only_starts_no_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'status.json').write_text('{"training_complete":true}')
            with patch('sys.argv', ['training_cloud_session', '--ssh', 'fake', '--fetch-only', '--session', SESSION]), \
                 patch.object(launcher, 'parse_ssh', return_value=('root@fake', 22, Path('/tmp/key'))), \
                 patch.object(launcher, 'fetch_reports', return_value=root), \
                 patch.object(launcher.subprocess, 'run') as run:
                launcher.main()
            run.assert_not_called()

    def test_success_acknowledges_only_after_reports_are_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'status.json').write_text('{"phase":"awaiting_sync","training_complete":true}')
            order = []
            def run(command, **kwargs):
                if 'bash -s' in command:
                    order.append('launch')
                    return subprocess.CompletedProcess(command, 0, stdout=f'Cloud session started: {SESSION}\n')
                order.append('ack' if '--ack-download' in command[-1] else 'watch')
                return subprocess.CompletedProcess(command, 0)
            def fetch(*args):
                order.append('download')
                return root
            with patch('sys.argv', ['training_cloud_session', '--ssh', 'fake']), \
                 patch.object(launcher, 'parse_ssh', return_value=('root@fake', 22, Path('/tmp/key'))), \
                 patch.object(launcher.subprocess, 'run', side_effect=run), \
                 patch.object(launcher, 'fetch_reports', side_effect=fetch):
                launcher.main()
            self.assertEqual(order, ['launch', 'watch', 'download', 'ack'])

    def test_invalid_paths_cannot_become_shell_commands(self):
        script = launcher.launch_script('training/smoke.json', '/workspace/panda-training/training_v1')
        self.assertIn('PANDA_TRAIN_DOWNLOAD_GRACE_SECONDS=120', script)
        for value in ('training/a;echo.json', 'training/../bad.json'):
            with self.assertRaises(ValueError):
                launcher.launch_script(value, '/workspace/panda-training/training_v1')
        with self.assertRaises(ValueError):
            launcher.session_path('../escape')


if __name__ == '__main__':
    unittest.main()
