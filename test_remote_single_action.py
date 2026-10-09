"""Local HTTP + real MuJoCo checks; server responses are fixtures, not inference.

Run with .venv/bin/python -m unittest test_remote_single_action -v.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from panda_controller import PandaController
from remote_single_action import run_single_action


class SingleActionTests(unittest.TestCase):
    def run_case(self, *, mode="openvla", unnorm_key="bridge_orig", gripper=0.996078431372549):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                reply = {
                    'request_id': request['request_id'], 'mode': mode,
                    'model_loaded': mode == 'openvla', 'unnorm_key': unnorm_key,
                    # Previously observed real prediction, replayed solely as a fixture.
                    'action': [-0.004458597905495604,-0.031828825103885995,0.0036561353351263343,
                               -0.005247634638758394,-0.02891491554589834,0.004768682880728764,gripper],
                    'timings_seconds': {'model_inference': 0.01},
                }
                body = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        with tempfile.TemporaryDirectory() as folder, ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            output = Path(folder) / 'run'
            steps = []
            original_step = PandaController.step
            def record_step(controller, count):
                steps.append(count)
                return original_step(controller, count)
            try:
                with patch.object(PandaController, 'step', record_step):
                    if mode != 'openvla' or unnorm_key != 'bridge_orig' or not 0 <= gripper <= 1:
                        with self.assertRaises(ValueError):
                            run_single_action(server_url=f'http://127.0.0.1:{server.server_port}',output_dir=output,no_view=True)
                        status = json.loads((output/'status.json').read_text())
                        self.assertEqual(status['phase'],'Failed')
                        self.assertFalse(status['motion_started'])
                        self.assertEqual(steps,[500])  # Only initial settling, no returned-action execution.
                        self.assertFalse((output/'after_state.npz').exists())
                    else:
                        run_single_action(server_url=f'http://127.0.0.1:{server.server_port}',output_dir=output,no_view=True)
                        report = json.loads((output/'execution.json').read_text())
                        self.assertEqual(steps,[500,1000])
                        self.assertTrue(report['tracking_passed'])
                        self.assertLess(report['position_tracking_error_metres'],0.005)
                        self.assertLessEqual(np.linalg.norm(report['controller_action'][:3]),0.0100000001)
                        self.assertLessEqual(np.linalg.norm(report['controller_action'][3:6]),0.0500000001)
                        self.assertAlmostEqual(report['controller_action'][6],2*gripper-1)
                        self.assertGreater(np.linalg.norm(report['actual_translation']),0.001)
                        self.assertTrue(report['simulation_frozen_during_request'])
                        with np.load(output/'scene_state.npz') as before, np.load(output/'after_state.npz') as after:
                            self.assertAlmostEqual(float(after['time']-before['time']),report['physics_seconds'])
                            self.assertFalse(np.array_equal(before['qpos'],after['qpos']))
                        self.assertNotEqual((output/'before.png').read_bytes(),(output/'after.png').read_bytes())
                        self.assertEqual(json.loads((output/'status.json').read_text())['phase'],'Completed')
            finally:
                server.shutdown()
                thread.join(timeout=5)

    def test_prediction_executes_one_bounded_movement(self):
        self.run_case()

    def test_wrong_server_mode_does_not_move_robot(self):
        self.run_case(mode='connection_test')

    def test_unknown_normalization_does_not_move_robot(self):
        self.run_case(unnorm_key='unsupported')

    def test_invalid_gripper_does_not_move_robot(self):
        self.run_case(gripper=2)


if __name__ == '__main__':
    unittest.main()
