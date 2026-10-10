"""Supervisor checks with fake training processes and mocked RunPod API calls."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from training.detached_run import api_request, supervise, verify_results


class DetachedTrainingTests(unittest.TestCase):
    def artifacts(self, directory, update=1000):
        run = directory / 'runs/training'
        checkpoint = run / f'checkpoint_{update:06d}'
        checkpoint.mkdir(parents=True)
        for name in ('adapter_config.json', 'adapter_model.safetensors',
                     'dataset_statistics.json', 'training_config.json'):
            (checkpoint / name).write_text('fixture')
        for name in ('loss.png', 'experiment.json', 'model_info.json'):
            (run / name).write_text('fixture')
        (run / 'metrics.jsonl').write_text(json.dumps(dict(update=update, train_loss=2)) + '\n' +
                                          json.dumps(dict(update=update, validation_complete=True)) + '\n')
        return run

    def run_supervisor(self, directory, process, request):
        with patch('training.detached_run.subprocess.Popen', return_value=process) as launch, \
             patch('training.detached_run.api_request', request), \
             patch('training.detached_run.os.sync'), patch('training.detached_run.time.sleep'), \
             patch.dict('os.environ', {'RUNPOD_API_KEY': 'test-secret'}):
            result = supervise(['python', 'fake_training.py'], directory, 'fakepod', 'test-secret',
                               hard_limit=5400, expected_updates=1000)
        self.assertNotIn('RUNPOD_API_KEY', launch.call_args.kwargs['env'])
        self.assertNotIn('test-secret', (directory / 'status.json').read_text())
        return result, json.loads((directory / 'status.json').read_text())

    def test_success_verifies_artifacts_then_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.artifacts(directory)
            process = MagicMock(); process.wait.return_value = 0; process.poll.return_value = 0
            def request(*args, **kwargs):
                status = json.loads((directory / 'status.json').read_text())
                self.assertTrue(status['training_complete'])
                self.assertEqual(status['stop_state'], 'requested')
                return {}
            result, status = self.run_supervisor(directory, process, request)
            self.assertEqual(result, 0)
            self.assertEqual(status['phase'], 'completed')
            self.assertEqual(status['stop_state'], 'accepted')

    def test_failure_still_requests_stop_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as folder:
            process = MagicMock(); process.wait.return_value = 1; process.poll.return_value = 1
            request = MagicMock(return_value={})
            result, status = self.run_supervisor(Path(folder), process, request)
            self.assertEqual(result, 1)
            self.assertEqual(status['phase'], 'training_failed')
            request.assert_called_once_with('fakepod', 'test-secret', stop=True)
            self.assertTrue((Path(folder) / 'console.log').exists())

    def test_partial_run_is_not_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            run = self.artifacts(directory, update=250)
            self.assertFalse(verify_results(run, 1000)['training_complete'])
            (run / 'checkpoint_000250/adapter_model.safetensors').unlink()
            self.assertTrue(verify_results(run, 1000)['missing_artifacts'])

    def test_timeout_terminates_process_group_and_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            process = MagicMock(); process.pid = 1234
            process.wait.side_effect = [subprocess.TimeoutExpired('train', 5400), -15]
            process.poll.return_value = -15; process.returncode = -15
            request = MagicMock(return_value={})
            with patch('training.detached_run.os.killpg') as kill:
                result, status = self.run_supervisor(Path(folder), process, request)
            kill.assert_called_once()
            self.assertEqual(status['phase'], 'wall_limit_reached')
            self.assertEqual(status['stop_state'], 'accepted')
            self.assertEqual(result, 1)

    def test_stop_api_failure_is_retried_and_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            process = MagicMock(); process.wait.return_value = 1; process.poll.return_value = 1
            request = MagicMock(side_effect=RuntimeError('RunPod API returned HTTP 403'))
            _, status = self.run_supervisor(Path(folder), process, request)
            self.assertEqual(request.call_count, 3)
            self.assertEqual(status['stop_state'], 'failed')

    def test_api_uses_authorization_header_and_stop_endpoint(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{}'
        with patch('training.detached_run.urlopen', return_value=response) as opened:
            api_request('fakepod', 'test-secret', stop=True)
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, 'https://rest.runpod.io/v1/pods/fakepod/stop')
        self.assertEqual(request.method, 'POST')
        self.assertNotIn('test-secret', request.full_url)


if __name__ == '__main__':
    unittest.main()
