"""Verify automatic report fetching without contacting or training on a pod."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import cloud.benchmark_cloud_session as benchmark_cloud_session
from cloud.benchmark_cloud_session import extract_reports,fetch_reports,remote_benchmark_script,measure_mac_inference


class BenchmarkSyncTests(unittest.TestCase):
    def test_launcher_watches_training_after_inference_before_stopping(self):
        from types import SimpleNamespace
        session='/workspace/panda-training/batch-benchmarks/sessions/20261012_120000_123456'
        with patch('sys.argv',['launcher','--ssh','fake']), \
             patch('cloud.benchmark_cloud_session.parse_ssh',return_value=('root@fake',22,Path('/tmp/key'))), \
             patch('cloud.benchmark_cloud_session.fetch_reports') as fetch, \
             patch('cloud.benchmark_cloud_session.measure_mac_inference',return_value=session), \
             patch('cloud.benchmark_cloud_session.subprocess.run',return_value=SimpleNamespace(returncode=0)) as run:
            benchmark_cloud_session.main()
        commands=[call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands),3)
        self.assertIn('--watch-session',commands[1][-1])
        self.assertIn('training.benchmark_stop',commands[2][-1])
        self.assertEqual(fetch.call_count,2)

    def test_mac_probe_averages_twenty_after_five_warmups_and_persists_before_stop(self):
        run='20261012_120000_123456'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);session=root/'sessions'/run;session.mkdir(parents=True)
            report=dict(live_session_dir=f'/workspace/panda-training/batch-benchmarks/sessions/{run}',
                        server_port=1234,warmup_requests=5,measured_requests=20,
                        image_png_base64='aW1hZ2U=',instruction='task')
            path=session/'inference.json';path.write_text(json.dumps(report))
            (session/'status.json').write_text('{"phase":"inference_ready"}')
            response=dict(client_timings_seconds=dict(request_round_trip=1),timings_seconds=dict(model_inference=.3))
            with patch('cloud.benchmark_cloud_session.socket.create_connection'), \
                 patch('cloud.benchmark_cloud_session.subprocess.Popen') as launch, \
                 patch('cloud.benchmark_cloud_session.subprocess.run') as persist, \
                 patch('cloud.connection_client.request_prediction',return_value=response) as predict, \
                 patch('analysis.summarize_gpu_inference.comparison'):
                launch.return_value.poll.return_value=None
                measure_mac_inference(['ssh','host'],root)
            self.assertEqual(predict.call_count,25)
            saved=json.loads(path.read_text())['mac_requests']
            self.assertEqual(sum(not r['warmup'] for r in saved),20)
            self.assertIn('inference_client_done',persist.call_args.args[0][-1])
            self.assertEqual(len(json.loads(persist.call_args.kwargs['input'])['mac_requests']),25)
            launch.return_value.terminate.assert_called_once()

    def archive(self,path,files):
        with tarfile.open(path,'w:gz') as archive:
            for name,data in files.items():
                content=data.encode();member=tarfile.TarInfo(name);member.size=len(content)
                archive.addfile(member,io.BytesIO(content))

    def test_fetch_updates_workspace_from_full_measurements(self):
        run='20261011_054257_777818'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'input.tar.gz'
            row=dict(microbatch=8,accumulation=1,effective_batch=8,status='completed',
                headroom_screen_passed=True,seconds_per_update=2,examples_per_second=4,
                gpu_peak_allocated_gib=40,gpu_peak_reserved_gib=42)
            self.archive(archive,{
                f'{run}/benchmark.json':json.dumps([row]),
                f'{run}/benchmark_metadata.json':json.dumps(dict(mode='capacity',dataset='/workspace/panda-training/training_v1')),
                f'{run}/BATCH_BENCHMARK.md':'# Saved report\n',
                f'{run}/batch_8_accum_1/model_info.json':json.dumps(dict(gpu='NVIDIA A40',total_vram_gib=47.4))})
            def download(command,stdout,check):stdout.write(archive.read_bytes())
            with patch('cloud.benchmark_cloud_session.subprocess.run',side_effect=download):
                fetch_reports(['ssh','fake'],root/'outputs',root/'GPU_TRAINING_COMPARISON.md')
            self.assertIn('NVIDIA A40',(root/'GPU_TRAINING_COMPARISON.md').read_text())
            self.assertTrue((root/'outputs'/run/'benchmark.json').exists())
            self.assertTrue((root/'outputs/GPU_TRAINING_COMPARISON.md').exists())
            self.assertTrue((root/'outputs/A40_COMPARISON.md').exists())

    def test_rejects_traversal_and_weights_before_writing(self):
        for name in ('../escape.json','20261011_054257_777818/adapter.safetensors'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);archive=root/'bad.tar.gz';destination=root/'reports';destination.mkdir()
                self.archive(archive,{name:'bad'})
                with self.assertRaises(ValueError):extract_reports(archive,destination)
                self.assertEqual(list(destination.iterdir()),[])

    def test_launch_selects_dataset_and_refuses_shell_injection(self):
        script=remote_benchmark_script('/workspace/panda-training/training_v2',32)
        self.assertIn('--find-capacity --max-batch 32 --dataset /workspace/panda-training/training_v2',script)
        self.assertIn('pull --ff-only',script)
        supervised=remote_benchmark_script('/workspace/panda-training/training_v2',32,auto_stop=True,hard_limit=3600)
        self.assertIn('training.benchmark_supervisor --launch',supervised)
        self.assertIn('--hard-limit-seconds 3600',supervised)
        with self.assertRaises(ValueError):remote_benchmark_script('/workspace/panda-training/training_v2',32,hard_limit=0)
        with self.assertRaises(ValueError):
            remote_benchmark_script('/workspace/panda-training/a;echo bad',64)

    def test_fetch_only_does_not_start_training(self):
        with patch('sys.argv',['benchmark_cloud_session.py','--ssh','fake','--fetch-only']), \
             patch('cloud.benchmark_cloud_session.parse_ssh',return_value=('root@fake',22,Path('/tmp/key'))), \
             patch('cloud.benchmark_cloud_session.fetch_reports') as fetch, \
             patch('cloud.benchmark_cloud_session.subprocess.run') as run:
            benchmark_cloud_session.main()
        run.assert_not_called()
        fetch.assert_called_once()

    def test_failed_download_keeps_existing_workspace_report(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);report=root/'GPU_TRAINING_COMPARISON.md';report.write_text('previous results')
            with patch('cloud.benchmark_cloud_session.subprocess.run',side_effect=subprocess.CalledProcessError(255,['ssh'])):
                with self.assertRaises(subprocess.CalledProcessError):
                    fetch_reports(['ssh'],root/'outputs',report)
            self.assertEqual(report.read_text(),'previous results')


if __name__=='__main__':unittest.main()
