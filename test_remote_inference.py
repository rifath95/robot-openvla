"""Contract checks without loading the 7B model or renting a GPU.

Run with .venv-openvla/bin/python -m unittest test_remote_inference -v.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from connection_server import Handler, ThreadingHTTPServer
from openvla_backend import OpenVLAPredictor
from prepare_cloud_model import main as export_model


class Inputs(dict):
    def to(self, device, dtype):
        return self


class Processor:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, image):
        self.calls.append((prompt, image.size))
        return Inputs(input_ids=torch.tensor([[1, 42]]), attention_mask=torch.ones((1, 2), dtype=torch.long))


class Model:
    norm_stats = {"bridge_orig": {"action": {"q01": [0]*7, "q99": [1]*7}}}

    def eval(self):
        return self

    def predict_action(self, **inputs):
        assert inputs['input_ids'].shape == inputs['attention_mask'].shape
        assert inputs['input_ids'][0, -1].item() == 29871
        assert inputs['attention_mask'][0, -1].item() == 1
        assert inputs['unnorm_key'] == 'bridge_orig'
        return np.array([0.01, 0, 0, 0, 0, 0, 0.5])


class RemoteInferenceTests(unittest.TestCase):
    def test_resident_model_processes_each_request_and_aligns_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'config.json').write_text('{}')
            processor = Processor()
            with patch('transformers.AutoProcessor.from_pretrained', return_value=processor) as load_processor, patch('transformers.AutoModelForVision2Seq.from_pretrained', return_value=Model()) as load_model:
                predictor = OpenVLAPredictor(folder, device='cpu')
                first = predictor.predict(Image.new('RGB', (20, 10)), 'pick up the cube')
                second = predictor.predict(Image.new('RGB', (30, 10)), 'place the cube')
            self.assertEqual(load_model.call_count, 1)
            self.assertEqual(load_processor.call_count, 1)
            self.assertEqual(len(processor.calls), 2)
            self.assertIn('place the cube', processor.calls[1][0])
            self.assertEqual(processor.calls[1][1], (30, 10))
            self.assertEqual(first['action'][0], 0.01)
            self.assertGreaterEqual(second['timings_seconds']['model_inference'], 0)
            with patch.object(predictor.model, 'predict_action', return_value=np.full(7, np.nan)):
                with self.assertRaisesRegex(RuntimeError, 'seven finite'):
                    predictor.predict(Image.new('RGB', (20, 10)), 'invalid output test')

    def test_cloud_export_has_regular_files_and_can_be_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            (root / 'blob').write_bytes(b'example weights')
            (snapshot / 'weights.safetensors').symlink_to(root / 'blob')
            (snapshot / 'config.json').write_text('{}')
            args = ['prepare_cloud_model.py', '--cache-dir', str(root / 'cache'), '--export-dir', str(root / 'export')]
            with patch.object(sys, 'argv', args), patch('huggingface_hub.snapshot_download', return_value=str(snapshot)) as download:
                export_model()
                export_model()
            self.assertEqual(download.call_count, 1)
            weights = root / 'export' / 'weights.safetensors'
            self.assertFalse(weights.is_symlink())
            self.assertEqual(weights.read_bytes(), b'example weights')
            self.assertTrue((root / 'export' / 'export_complete.json').exists())

    def test_client_accepts_real_mode_and_rejects_wrong_mode(self):
        class Predictor:
            load_seconds = 1
            def predict(self, image, instruction):
                return {'action': [0.01,0,0,0,0,0,0.5], 'unnorm_key': 'bridge_orig',
                        'timings_seconds': {'processor_and_device_transfer': 0.01, 'model_inference': 0.02}}
        with tempfile.TemporaryDirectory() as folder, ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            server.mode = 'openvla'
            server.predictor = Predictor()
            server.prediction_lock = threading.Lock()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            image = Path(folder) / 'image.png'
            Image.new('RGB', (20, 10)).save(image)
            base = [sys.executable, 'connection_client.py', '--server-url', f'http://127.0.0.1:{server.server_port}', '--image', str(image)]
            try:
                run = subprocess.run(base + ['--expected-mode', 'openvla', '--output-dir', str(Path(folder)/'ok')], capture_output=True, text=True, timeout=10)
                self.assertEqual(run.returncode, 0, run.stderr)
                result = json.loads((Path(folder)/'ok'/'result.json').read_text())
                self.assertTrue(result['model_loaded'])
                self.assertFalse(result['robot_moved'])
                self.assertEqual(result['timings_seconds']['model_inference'], 0.02)
                run = subprocess.run(base + ['--output-dir', str(Path(folder)/'wrong')], capture_output=True, text=True, timeout=10)
                self.assertNotEqual(run.returncode, 0)
                server.prediction_lock.acquire()
                try:
                    run = subprocess.run(base + ['--expected-mode','openvla','--output-dir',str(Path(folder)/'busy')], capture_output=True,text=True,timeout=10)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn('503',run.stderr)
                finally:
                    server.prediction_lock.release()
                with patch.object(server.predictor, 'predict', side_effect=RuntimeError('simulated GPU failure')):
                    run = subprocess.run(base + ['--expected-mode','openvla','--output-dir',str(Path(folder)/'failed')], capture_output=True,text=True,timeout=10)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn('500',run.stderr)
                    self.assertFalse(server.prediction_lock.locked())
            finally:
                server.shutdown()
                thread.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
