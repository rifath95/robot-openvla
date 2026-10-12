"""Exercise repeat sampling and inference handoff without loading model weights."""
import base64
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from analysis.summarize_gpu_inference import comparison, measurement
from training.benchmark_pipeline import inference, sample_requests, main


class InferenceBenchmarkTests(unittest.TestCase):
    def test_inference_failure_skips_training_and_propagates_to_watchdog(self):
        with patch('sys.argv',['pipeline','--session-dir','/tmp/run','--dataset','/workspace/panda-training/training_v1']), \
             patch('training.benchmark_pipeline.subprocess.run',return_value=SimpleNamespace(returncode=3)), \
             patch('training.benchmark_pipeline.inference') as predict:
            self.assertEqual(main(),3)
        predict.assert_not_called()

    def test_inference_child_finishes_before_fresh_training_child_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'status.json').write_text('{"phase":"inference_ready"}')
            with patch('sys.argv',['pipeline','--session-dir',str(root),'--dataset','/workspace/panda-training/training_v1']), \
                 patch('training.benchmark_pipeline.subprocess.run',return_value=SimpleNamespace(returncode=0)) as run:
                self.assertEqual(main(),0)
            commands=[call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands),2)
            self.assertIn('--inference-only',commands[0])
            self.assertIn('training.benchmark_training_batches',commands[1])
            self.assertIn('--find-capacity',commands[1])
            self.assertEqual(json.loads((root/'status.json').read_text())['phase'],'training')

    def report(self, mac=False):
        rows=[dict(warmup=i<5,request_timings_seconds=dict(request_round_trip=100 if i<5 else 1),
                   server_timings_seconds=dict(model_inference=.3,server_processing=.5)) for i in range(25)]
        return dict(startup_info=dict(gpu=dict(name='NVIDIA A40',total_vram_bytes=48*2**30),
                                      pytorch_version='test',dtype='bfloat16'),
                    model_identity='base-hash',image_sha256='image-hash',cloud_requests=rows,
                    mac_requests=rows if mac else [])

    def test_cloud_only_does_not_claim_mac_round_trip_or_simulator_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=measurement(Path(tmp),benchmark=self.report())
            self.assertEqual(m['measured'],20)
            self.assertAlmostEqual(m['remote']['model_inference'],.3)
            self.assertIsNone(m['request'])
            self.assertIsNone(m['cycle'])
            self.assertIsNone(m['local_total'])

    def test_mac_samples_exclude_warmup_and_summary_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i, latency in enumerate((1,3)):
                folder=root/'sessions'/str(i);folder.mkdir(parents=True)
                report=self.report(mac=True)
                for r in report['mac_requests'][5:]:r['request_timings_seconds']['request_round_trip']=latency
                (folder/'inference.json').write_text(json.dumps(report))
            output=root/'report.md';self.assertEqual(comparison(root,output),2)
            text=output.read_text()
            self.assertIn('2 | 40 | — | 2.000 | 0.300',text)
            self.assertLess(text.index('## Average by comparable'),text.index('## Full cycle'))

    def test_sampling_makes_five_warmups_and_twenty_measurements(self):
        response=dict(client_timings_seconds=dict(request_round_trip=1),timings_seconds=dict(model_inference=.3))
        with patch('training.benchmark_pipeline.request_prediction',return_value=response) as request:
            rows=sample_requests(Path('image.png'),'task','http://127.0.0.1:1234')
        self.assertEqual(request.call_count,25)
        self.assertEqual(sum(r['warmup'] for r in rows),5)

    def test_cloud_publishes_samples_before_ready_and_closes_server(self):
        from PIL import Image
        stream=BytesIO();Image.new('RGB',(16,16),'red').save(stream,format='PNG')
        predictor=SimpleNamespace(startup_info=dict(gpu=dict(name='NVIDIA A40')),
                                  torch=SimpleNamespace(version=SimpleNamespace(cuda='12.1')),dtype='bfloat16')
        def predict(image,instruction):
            return dict(action=[0.]*6+[1.],timings_seconds=dict(model_inference=.001,processor_and_device_transfer=.001))
        predictor.predict=predict
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'sessions'/'run';directory.mkdir(parents=True)
            (directory/'status.json').write_text('{"phase":"benchmarking"}')
            (root/'config.json').write_text('{}')
            with patch('training.benchmark_pipeline.fixture',return_value=(stream.getvalue(),'task')), \
                 patch('inference.openvla_backend.OpenVLAPredictor',return_value=predictor), \
                 patch('training.gpu_ecc.collect_ecc',return_value=dict(current='Disabled')), \
                 patch('training.benchmark_software.collect_software',return_value={}):
                self.assertEqual(inference(directory,root,root,wait_seconds=0),0)
            saved=json.loads((directory/'inference.json').read_text())
            self.assertEqual(len(saved['cloud_requests']),25)
            self.assertEqual(json.loads((directory/'status.json').read_text())['phase'],'inference_ready')
            self.assertEqual(base64.b64decode(saved['image_png_base64']),stream.getvalue())


if __name__=='__main__':unittest.main()
