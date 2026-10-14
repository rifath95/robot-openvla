"""Real MuJoCo with HTTP fixtures; no model weights or paid cloud required."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from inference.remote_loop import ActiveTimer, run_loop
from simulation.panda_controller import PandaController, IKResult
from simulation.panda_actions import PHYSICS_STEPS


class LoopTests(unittest.TestCase):
    def test_two_cube_lift_start_uses_open_gripper_and_matching_replay_colors(self):
        from simulation.replay_run import replay_model
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run';stop=threading.Event();stop.set()
            run_loop(task='color_lift',scene_seed=1000,target_color='yellow',cube_colors=['blue','yellow'],
                steps=1,no_view=True,stop_event=stop,output_dir=output)
            metadata=json.loads((output/'episode.json').read_text())
            self.assertEqual(metadata['instruction'],'pick up the yellow cube.')
            self.assertEqual(metadata['scene_profile'],'panda_color_lift_v1')
            self.assertEqual(metadata['scenario']['scene']['starting_pose']['requested_gripper_open_fraction'],1)
            model,camera=replay_model(output);self.assertEqual(camera,'touch_camera')
            result=json.loads((output/'summary.json').read_text())['task_result']
            self.assertFalse(result['correct_cube_lift']);self.assertFalse(result['wrong_cube_lift'])

    def test_placement_start_records_matching_blue_scene(self):
        from simulation.replay_run import replay_model
        from inference.place_bowl_task import layout
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run';stop=threading.Event();stop.set()
            saved=Path(tmp)/'positions.json';saved.write_text(json.dumps(layout(1000)))
            run_loop(task='place_bowl',scene_seed=1000,steps=1,no_view=True,
                stop_event=stop,output_dir=output,scene_positions=saved)
            metadata=json.loads((output/'episode.json').read_text())
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(metadata['scene_profile'],'panda_place_bowl_v1')
            self.assertEqual(metadata['instruction'],'pick up the red cube and place it in the blue bowl.')
            self.assertFalse(summary['task_result']['success'])
            model,camera=replay_model(output);self.assertEqual(camera,'bowl_camera')

    def test_lift_start_records_open_gripper_task_and_replay_scene(self):
        from simulation.replay_run import replay_model
        from inference.lift_task import cube_position_for_seed
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run';stop=threading.Event();stop.set()
            run_loop(task='lift',scene_seed=1000,steps=1,no_view=True,
                stop_event=stop,output_dir=output)
            metadata=json.loads((output/'episode.json').read_text())
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(metadata['scene_profile'],'panda_lift_v1')
            self.assertEqual(metadata['instruction'],'pick up the red cube.')
            self.assertEqual(metadata['scenario']['scene']['cube_xy'],cube_position_for_seed(1000))
            self.assertFalse(summary['task_result']['success'])
            model,camera=replay_model(output)
            self.assertEqual(camera,'touch_camera')

    def test_bowl_start_records_matching_camera_and_replay_geometry(self):
        from simulation.replay_run import replay_model
        import mujoco
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            stop=threading.Event();stop.set()
            run_loop(task='bowl',scene_seed=0,steps=1,no_view=True,
                stop_event=stop,output_dir=output,
                instruction='pick up the red cube and drop it in the blue bowl.')
            metadata=json.loads((output/'episode.json').read_text())
            contract=json.loads((output/'action_contract.json').read_text())
            self.assertIn('bowl_camera',json.dumps(contract))
            model,camera=replay_model(output)
            self.assertEqual(camera,'bowl_camera')
            bowl=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'bowl')
            np.testing.assert_allclose(model.body_pos[bowl],metadata['scenario']['scene']['bowl_position'])

    def test_active_timer_excludes_pauses_and_resumes(self):
        now = [0.0]
        timer = ActiveTimer(lambda: now[0])
        now[0] = 15
        timer.set_paused(True)
        now[0] = 300
        self.assertEqual(timer.elapsed(), 15)
        timer.set_paused(False)
        now[0] = 310
        self.assertEqual(timer.elapsed(), 25)

    def test_viewer_starts_paused_and_resumes_mid_movement(self):
        import glfw
        import mujoco.viewer
        pauses = []
        class Viewer:
            def __init__(self, model, data, key_callback, **kwargs):
                self.data = data
                self.key = key_callback
                self.cam = SimpleNamespace()
                self.initial_frames = 0
                self.pause_frames = 0
                self.did_pause = False
                self.snapshot = None
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def is_running(self):
                return True
            def lock(self):
                return nullcontext()
            def set_texts(self, texts):
                pass
            def sync(self):
                if self.initial_frames < 3:
                    self.initial_frames += 1
                    if self.initial_frames == 3:
                        self.key(glfw.KEY_SPACE)
                elif self.snapshot is not None:
                    np.testing.assert_array_equal(self.data.qpos, self.snapshot)
                    self.pause_frames += 1
                    if self.pause_frames == 3:
                        self.snapshot = None
                        self.key(glfw.KEY_SPACE)
                elif not self.did_pause and self.data.time > 1.1:
                    self.did_pause = True
                    self.snapshot = self.data.qpos.copy()
                    pauses.append(float(self.data.time))
                    self.key(glfw.KEY_SPACE)
        prediction = dict(request_id='fixture', unnorm_key='bridge_orig',
                          action=[0, 0.005, 0, 0, 0, 0, 1], client_timings_seconds={})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            with patch.object(mujoco.viewer, 'launch_passive', Viewer), patch('inference.remote_loop.request_prediction', return_value=prediction):
                run_loop(steps=1, output_dir=output, max_runtime_seconds=10)
            self.assertEqual(len(pauses), 1)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['completed_steps'], 1)
            self.assertEqual(summary['steps'][0]['executed_physics_steps'], PHYSICS_STEPS)
            self.assertLess(summary['active_seconds'], summary['elapsed_seconds'])

    def test_touch_stops_when_actual_finger_contact_succeeds(self):
        import mujoco
        from unittest.mock import patch
        initial=dict(success=False,finger_cube_contacts=[],cube_displacement_metres=0.)
        contact=dict(success=True,finger_cube_contacts=[dict(finger='left_finger',normal_force_newtons=.2)],cube_displacement_metres=0.)
        prediction=dict(request_id='touch',unnorm_key='panda_grasp_v1',
                        action=[0,0,-.005,0,0,0,0],client_timings_seconds={})
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'run'
            with patch('inference.remote_loop.request_prediction',return_value=prediction) as request, \
                 patch('data_tools.touch_demonstrations.task_result',side_effect=[initial,contact]):
                run_loop(task='touch',scene_seed=0,steps=5,no_view=True,output_dir=folder)
            self.assertEqual(request.call_count,1)
            self.assertEqual(request.call_args.args[1],'touch the red cube.')
            summary=json.loads((folder/'summary.json').read_text())
            self.assertEqual(summary['completed_steps'],1)
            self.assertTrue(summary['task_result']['success'])
            self.assertEqual(summary['phase'],'Completed')
            metadata=json.loads((folder/'episode.json').read_text())
            self.assertEqual(metadata['scene_profile'],'panda_touch_v1')
            self.assertEqual(metadata['demonstration_id'],'train_001')
            self.assertEqual(metadata['scenario']['scene']['starting_pose']['requested_position'],[.5,0,.45])
            self.assertEqual(json.loads((folder/'action_contract.json').read_text())['observation'],
                             'RGB touch_camera immediately before command; frozen during prediction')
            from simulation.replay_run import replay_model
            model,camera=replay_model(folder)
            self.assertEqual(camera,'touch_camera')
            self.assertEqual(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'bowl'),-1)

    def test_sampling_is_sent_and_saved_for_every_cycle(self):
        self.run_case(sample_actions=True)

    def run_case(self, *, fail_second=False, stop_during_request=False, rejection_cycles=(), steps=3, unnorm_key='bridge_orig', sample_actions=False):
        requests = []
        stop = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(request)
                if stop_during_request:
                    stop.set()
                reply = {
                    'request_id': request['request_id'], 'mode': 'openvla',
                    'model_loaded': True, 'unnorm_key': unnorm_key,
                    'action': [-0.0044585979, -0.0318288251, 0.0036561353,
                               -0.0052476346, -0.0242258626, -0.0016785094, 0.9960784314],
                    'timings_seconds': {'model_inference': 0.01},
                }
                if request.get('generation_options'):
                    reply['generation_options']=dict(request['generation_options'],action_tokens_only=True)
                body = json.dumps(reply).encode()
                self.send_response(500 if fail_second and len(requests) == 2 else 200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as directory, ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            output = Path(directory) / 'run'
            startup_path = Path(directory) / 'startup.json'
            startup_path.write_text(json.dumps({'model_load_seconds': 44.64, 'gpu': {'name': 'fixture'}}))
            kwargs = dict(startup_info=startup_path, server_url=f'http://127.0.0.1:{server.server_port}', steps=steps,
                          max_runtime_seconds=30, no_view=True, output_dir=output, stop_event=stop,
                          sample_actions=sample_actions,temperature=.7,top_p=.95)
            original_solver = PandaController._solve_ik
            def solver(controller, *args, **kwargs):
                if len(requests) in rejection_cycles:
                    return IKResult(controller.data.qpos[controller.qpos_indices].copy(), .01, .01, 150, False)
                return original_solver(controller, *args, **kwargs)
            try:
                if fail_second:
                    with self.assertRaises(ValueError):
                        run_loop(**kwargs)
                else:
                    with patch.object(PandaController, '_solve_ik', solver):
                        run_loop(**kwargs)
                if sample_actions:
                    expected=dict(do_sample=True,temperature=.7,top_p=.95,top_k=0,num_beams=1)
                    self.assertEqual(json.loads((output/'generation_options.json').read_text()),expected)
                    self.assertTrue(all(r['generation_options']==expected for r in requests))
                summary = json.loads((output / 'summary.json').read_text())
                status = json.loads((output / 'status.json').read_text())
                self.assertEqual(json.loads((output / 'startup_info.json').read_text())['model_load_seconds'], 44.64)
                self.assertTrue((output / 'replay.py').exists())
                self.assertTrue((output / 'RUN_REPORT.md').exists())
                for record in summary['steps']:
                    step = output / f"step_{record['step']:03d}"
                    with np.load(step / 'motion_frames.npz') as frames, np.load(step / 'scene_state.npz') as before, np.load(step / 'after_state.npz') as after:
                        np.testing.assert_array_equal(frames['qpos'][0], before['qpos'])
                        np.testing.assert_array_equal(frames['qpos'][-1], after['qpos'])
                        self.assertEqual(len(frames['time']), PHYSICS_STEPS // 20 + 1)
                if rejection_cycles:
                    rejected = summary['rejections']
                    self.assertEqual(summary['rejected_steps'], len(rejected))
                    for record in rejected:
                        step = output / f"step_{record['step']:03d}"
                        self.assertFalse(record['executed'])
                        self.assertEqual(record['executed_physics_steps'], 0)
                        self.assertIsNone(record['controller_action'])
                        self.assertEqual(len(record['ik_attempts']), 4)
                        self.assertTrue((step / 'ik_attempts.json').exists())
                        self.assertEqual((step / 'before.png').read_bytes(), (step / 'after.png').read_bytes())
                        with np.load(step / 'scene_state.npz') as before, np.load(step / 'after_state.npz') as after:
                            for name in ('qpos', 'qvel', 'ctrl', 'time'):
                                np.testing.assert_array_equal(before[name], after[name])
                    if rejection_cycles == (1, 2, 3):
                        self.assertEqual(len(requests), 3)
                        self.assertEqual(summary['completed_steps'], 0)
                        self.assertEqual(summary['stop_reason'], 'consecutive_rejection_limit')
                        self.assertEqual(status['phase'], 'Stopped')
                    else:
                        self.assertEqual(len(requests), steps)
                        self.assertEqual(summary['completed_steps'], steps - len(rejection_cycles))
                        self.assertEqual(status['consecutive_rejections'], 0)
                        self.assertEqual(status['phase'], 'Completed')
                        self.assertEqual([r['consecutive_rejections'] for r in rejected], [1, 1, 2])
                elif stop_during_request:
                    self.assertEqual(summary['completed_steps'], 0)
                    self.assertEqual(summary['stop_reason'], 'user_stop')
                    self.assertEqual(len(requests), 1)
                    self.assertFalse((output / 'step_001' / 'after_state.npz').exists())
                elif fail_second:
                    self.assertEqual(summary['completed_steps'], 1)
                    self.assertEqual(status['phase'], 'Failed')
                    self.assertFalse((output / 'step_002' / 'after_state.npz').exists())
                else:
                    self.assertEqual(summary['completed_steps'], 3)
                    self.assertEqual(len(requests), 3)
                    self.assertEqual(status['phase'], 'Completed')
                    self.assertEqual(len(set(r['request_id'] for r in requests)), 3)
                    for report in summary['steps']:
                        self.assertEqual(report['unnorm_key'], unnorm_key)
                        self.assertTrue(report['finite_state'])
                        self.assertEqual(report['action_scale'], 1)
                        self.assertEqual(len(report['ik_attempts']), 1)
                        self.assertLess(report['position_tracking_error_metres'], 0.005)
                        self.assertEqual(report['executed_physics_steps'], PHYSICS_STEPS)
                        self.assertLessEqual(np.linalg.norm(report['controller_action'][:3]), 0.010000001)
                    for index in (1, 2):
                        previous = output / f'step_{index:03d}'
                        following = output / f'step_{index + 1:03d}'
                        self.assertEqual((previous / 'after.png').read_bytes(), (following / 'before.png').read_bytes())
                        with np.load(previous / 'after_state.npz') as after, np.load(following / 'scene_state.npz') as before:
                            np.testing.assert_array_equal(after['qpos'], before['qpos'])
                            self.assertEqual(float(after['time']), float(before['time']))
            finally:
                server.shutdown()
                thread.join(timeout=5)

    def test_three_steps_use_updated_scene(self):
        self.run_case()

    def test_panda_finetuned_actions_execute_and_record_convention(self):
        self.run_case(unnorm_key='panda_grasp_v1')

    def test_failure_preserves_previous_step_and_stops(self):
        self.run_case(fail_second=True)

    def test_stop_discards_inflight_prediction(self):
        self.run_case(stop_during_request=True)

    def test_rejections_hold_and_success_resets_rejection_counter(self):
        self.run_case(rejection_cycles=(2, 4, 5), steps=6)

    def test_repeated_rejections_stop_cleanly_before_budget_exhaustion(self):
        self.run_case(rejection_cycles=(1, 2, 3), steps=10)

    def test_runtime_limit_stops_while_waiting(self):
        release = threading.Event()
        entered = threading.Event()
        def slow_prediction(*args, **kwargs):
            entered.set()
            release.wait(5)
            raise TimeoutError('fixture released')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            try:
                with patch('inference.remote_loop.request_prediction', slow_prediction):
                    run_loop(steps=3, no_view=True, max_runtime_seconds=1, output_dir=output)
                self.assertTrue(entered.is_set())
                summary = json.loads((output / 'summary.json').read_text())
                self.assertEqual(summary['stop_reason'], 'runtime_limit')
                self.assertEqual(summary['completed_steps'], 0)
                self.assertFalse((output / 'step_001' / 'after_state.npz').exists())
            finally:
                release.set()


if __name__ == '__main__':
    unittest.main()
