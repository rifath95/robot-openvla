"""Verify benchmark stop ordering and retries with no real pod operations."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.benchmark_stop import stop_pod
import benchmark_cloud_session as launcher


class BenchmarkStopTests(unittest.TestCase):
    def test_stop_saves_status_and_retries_without_recording_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'status.json'
            with patch('training.benchmark_stop.api_request',side_effect=[RuntimeError('HTTP 503'),{}]) as api, \
                 patch('training.benchmark_stop.time.sleep'),patch('training.benchmark_stop.os.sync'):
                self.assertEqual(stop_pod('pod123','test-secret',path),0)
            status=json.loads(path.read_text())
            self.assertEqual(status['stop_state'],'accepted')
            self.assertEqual(status['stop_attempt'],2)
            self.assertNotIn('test-secret',path.read_text())
            self.assertEqual(api.call_count,2)

    def test_stop_failure_is_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'status.json'
            with patch('training.benchmark_stop.api_request',side_effect=RuntimeError('HTTP 403')), \
                 patch('training.benchmark_stop.time.sleep'),patch('training.benchmark_stop.os.sync'):
                self.assertEqual(stop_pod('pod123','secret',path),1)
            self.assertEqual(json.loads(path.read_text())['stop_state'],'failed')

    def test_launcher_stops_only_after_download_finishes(self):
        events=[]
        def run(command,**kwargs):
            events.append('stop' if 'training.benchmark_stop' in command[-1] else 'benchmark')
            return type('Result',(),{'returncode':0})()
        with patch('sys.argv',['launcher','--ssh','fake']), \
             patch.object(launcher,'parse_ssh',return_value=('root@fake',22,Path('/tmp/key'))), \
             patch.object(launcher.subprocess,'run',side_effect=run), \
             patch.object(launcher,'fetch_reports',side_effect=lambda *a:events.append('download')):
            launcher.main()
        self.assertEqual(events,['benchmark','download','stop'])
        self.assertIn('training.benchmark_stop --check-only',launcher.remote_benchmark_script('/workspace/panda-training/training_v1',64,auto_stop=True))

    def test_failed_download_still_requests_stop(self):
        with patch('sys.argv',['launcher','--ssh','fake']), \
             patch.object(launcher,'parse_ssh',return_value=('root@fake',22,Path('/tmp/key'))), \
             patch.object(launcher.subprocess,'run',return_value=type('Result',(),{'returncode':0})()) as run, \
             patch.object(launcher,'fetch_reports',side_effect=ValueError('download interrupted')):
            with self.assertRaisesRegex(SystemExit,'Results were not fully synchronized'):
                launcher.main()
        self.assertEqual(run.call_count,2)

    def test_duplicate_benchmark_does_not_stop_existing_supervisor(self):
        with patch('sys.argv',['launcher','--ssh','fake']), \
             patch.object(launcher,'parse_ssh',return_value=('root@fake',22,Path('/tmp/key'))), \
             patch.object(launcher.subprocess,'run',return_value=type('Result',(),{'returncode':75})()) as run, \
             patch.object(launcher,'fetch_reports') as fetch:
            with self.assertRaisesRegex(SystemExit,'left running'):launcher.main()
        self.assertEqual(run.call_count,1)
        fetch.assert_not_called()

    def test_full_disk_does_not_prevent_api_stop(self):
        with patch('training.benchmark_stop.write_status',side_effect=OSError('disk full')), \
             patch('training.benchmark_stop.api_request',return_value={}) as api, \
             patch('training.benchmark_stop.os.sync'):
            self.assertEqual(stop_pod('pod123','secret',Path('/missing/status.json')),0)
        api.assert_called_once_with('pod123','secret',stop=True)

    def test_cloud_error_still_downloads_and_stops(self):
        events=[]
        def run(command,**kwargs):
            stop='training.benchmark_stop' in command[-1]
            events.append('stop' if stop else 'benchmark')
            return type('Result',(),{'returncode':0 if stop else 1})()
        with patch('sys.argv',['launcher','--ssh','fake']), \
             patch.object(launcher,'parse_ssh',return_value=('root@fake',22,Path('/tmp/key'))), \
             patch.object(launcher.subprocess,'run',side_effect=run), \
             patch.object(launcher,'fetch_reports',side_effect=lambda *a:events.append('download')):
            with self.assertRaises(SystemExit) as error:launcher.main()
        self.assertEqual(error.exception.code,1)
        self.assertEqual(events,['benchmark','download','stop'])


if __name__=='__main__':unittest.main()
