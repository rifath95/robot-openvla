"""Manual selection, cancellation, and existing controller compatibility."""
from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from cloud.interactive_bowl_host import Models
from cloud.interactive_bowl_session import bootstrap
from inference.playground_config import CHECKPOINTS, PROFILE_RELATIVE,validate_host
from inference.interactive_bowl import Engine, validate_scene


class InteractiveControlsTests(unittest.TestCase):
    def test_default_family_and_reused_host_identity(self):
        self.assertEqual(CHECKPOINTS,(6000,8000,10000,12000))
        ready=dict(training_profile=PROFILE_RELATIVE,checkpoints=list(CHECKPOINTS),training_session='new-run')
        validate_host(ready,'new-run')
        for changed in (dict(training_profile='old-profile'),dict(checkpoints=[0,1000,2000,4000,6000]),dict(training_session='other-run')):
            with self.assertRaisesRegex(ValueError,'different'):
                validate_host(dict(ready,**changed),'new-run')
        self.assertIn('validate_host',bootstrap())

    def test_staging_excludes_final_optimizer_and_input_state(self):
        from cloud.interactive_bowl_host import stage,PROFILE
        from types import SimpleNamespace
        profile=json.loads(PROFILE.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'new-run'
            for cp in CHECKPOINTS:
                folder=source/f'runs/training/checkpoint_{cp:06d}';folder.mkdir(parents=True)
                (folder/'training_config.json').write_text(json.dumps(dict(checkpoint_update=cp,
                    dataset_provenance=dict(checksums_sha256=profile['source_dataset_sha256'],dataset_revision=profile['source_dataset_revision']))))
            state=source/'runs/training/checkpoint_012000/training_state';state.mkdir()
            (state/'state.pt').write_bytes(b'optimizer');(state/'input.data').write_bytes(b'iterator')
            target=root/'staging'
            with patch('cloud.scripted_eval_control.active_training',return_value=False), \
                    patch('cloud.color_bowl_sweep.resolve_training',return_value=source) as resolve, \
                    patch('cloud.checkpoint_ready.check_ready',side_effect=Path), \
                    patch('cloud.bowl_chain_worker.stage_base',side_effect=lambda _,p:p.mkdir()), \
                    patch('cloud.interactive_bowl_host.shutil.disk_usage',return_value=SimpleNamespace(free=30*1024**3)):
                identity=stage(target)
            self.assertEqual(identity['training_profile'],PROFILE_RELATIVE)
            self.assertEqual(resolve.call_args.args[0]['source_training_config'],'training/color_place_bowl_600_uninterrupted_12000.json')
            self.assertTrue((target/'adapters/checkpoint_012000/training_config.json').is_file())
            self.assertFalse((target/'adapters/checkpoint_012000/training_state').exists())

    def test_client_binds_prediction_to_selected_checkpoint(self):
        from cloud.connection_client import request_prediction
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/'before.png'
            image.write_bytes(b'image')
            response = Mock(status=200)
            response.read.return_value = json.dumps(dict(request_id='test', mode='openvla',
                model_loaded=True, checkpoint=2000, action=[0, 0, 0, 0, 0, 0, 1])).encode()
            with patch('cloud.connection_client.http.client.HTTPConnection') as connection:
                connection.return_value.getresponse.return_value = response
                result = request_prediction(image, 'typed text', checkpoint=2000, request_id='test')
                self.assertEqual(result['checkpoint'], 2000)
                body = json.loads(connection.return_value.request.call_args.kwargs['body'])
                self.assertEqual(body['instruction'], 'typed text')
                self.assertEqual(body['checkpoint'], 2000)
                with self.assertRaisesRegex(ValueError, 'different checkpoint'):
                    request_prediction(image, 'typed text', checkpoint=6000, request_id='test')

    def test_scene_validation_includes_three_cubes_and_reserved_colors(self):
        value = dict(count=3, colors=['red', 'purple', 'black'], seed=1234, target='purple')
        self.assertEqual(validate_scene(value), value)
        for replacement in (dict(colors=['red']*3), dict(target='orange'), dict(seed=-1), dict(count=5)):
            with self.assertRaises(ValueError):
                validate_scene(dict(value, **replacement))

    def test_stop_discards_response_without_executing(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine('http://127.0.0.1:1', tmp)
            engine.health_future = Future()
            engine.active = True
            engine.pending_epoch = engine.epoch
            engine.pending = Future()
            engine.pending.set_result(dict(action=[1]*7))
            engine.stop.set()
            with patch.object(engine, 'execute') as execute, patch.object(engine, 'request') as request:
                engine.tick()
                execute.assert_not_called()
                request.assert_not_called()
                self.assertFalse(engine.active)

    def test_instruction_is_preserved_and_checkpoint_must_be_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine('http://127.0.0.1:1', tmp)
            with self.assertRaisesRegex(ValueError, 'finish loading'):
                engine.begin('Put the purple block into the bowl, please.', 1)

    def test_same_checkpoint_keeps_loaded_child_and_loading_refuses_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'staged.json').write_text(json.dumps(dict(training_session='test')))
            models = Models(root)
            models.child = Mock()
            models.child.poll.return_value = None
            models.update(status='ready', model_loaded=True, checkpoint=6000)
            with patch.object(models, 'load') as load:
                models.select(6000)
                load.assert_not_called()
                models.child.terminate.assert_not_called()
            models.lock.acquire()
            try:
                with self.assertRaisesRegex(ValueError, 'in progress'):
                    models.select(8000)
            finally:
                models.lock.release()

    def test_host_rejects_prediction_for_wrong_checkpoint_before_forwarding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'staged.json').write_text(json.dumps(dict(training_session='test')))
            models = Models(root)
            models.update(status='ready', model_loaded=True, checkpoint=6000)
            with patch('cloud.interactive_bowl_host.http.client.HTTPConnection') as connection:
                with self.assertRaisesRegex(ValueError, 'cancelled'):
                    models.predict(dict(checkpoint=1000, instruction='test'))
                connection.assert_not_called()

    def test_bootstrap_does_not_stop_pod_or_merge_on_volume(self):
        text = bootstrap('panda-train-color-bowl-test')
        self.assertIn('nohup', text)
        self.assertIn('--prepare --training-session', text)
        self.assertNotIn('stop_pod', text)
        self.assertNotIn('merge_adapter', text)


class RealSimulatorTests(unittest.TestCase):
    def test_all_counts_render_and_action_uses_typed_instruction(self):
        import numpy as np
        from common.paths import PROJECT_ROOT
        from simulation.color_place_bowl_scene import positions, validate_layout
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'outputs', prefix='interactive_test_') as tmp:
            engine = Engine('http://127.0.0.1:1', tmp)
            engine.health_future = Future()
            try:
                for count in (1, 2, 3, 4):
                    for seed in (1000, 1001, 2001):
                        validate_layout(positions(seed, count), count)
                    colors = ['red', 'purple', 'black', 'orange'][:count]
                    engine.reset(dict(count=count, colors=colors, seed=1001, target=colors[-1]))
                    self.assertTrue(engine.frame.startswith(b'\x89PNG'))
                    self.assertEqual(engine.env.observe().shape, (480, 640, 3))
                engine.view['model'] = dict(model_loaded=True, checkpoint=6000)
                instruction = 'Please put the orange cube into that blue bowl.'
                engine.begin(instruction, 1)
                folder = engine.folder
                before = engine.env.data.qpos.copy()
                result = dict(unnorm_key='panda_grasp_v1', action=[.002, 0, 0, 0, 0, 0, 1], checkpoint=0)
                with patch('cloud.connection_client.request_prediction', return_value=result) as prediction:
                    engine.request()
                    deadline = time.monotonic()+10
                    while engine.active and time.monotonic()<deadline:
                        engine.tick()
                        time.sleep(.01)
                    prediction.assert_called_once()
                    self.assertEqual(prediction.call_args.args[1], instruction)
                    self.assertFalse(engine.active)
                    self.assertEqual(engine.step, 1)
                self.assertFalse(np.array_equal(before, engine.env.data.qpos))
                self.assertEqual(json.loads((folder/'episode.json').read_text())['instruction'], instruction)
                self.assertTrue((folder/'step_001/motion_frames.npz').exists())
                self.assertTrue((folder/'transitions.jsonl').exists())
                self.assertIn('/replay?', (folder/'RUN_REPORT.md').read_text())
                from simulation.action_replay import replay_actions
                replay_actions(folder, no_view=True, output_dir=Path(tmp)/'replayed', max_steps=1)
                held = engine.env.data.qpos.copy()
                engine.stop.set()
                for _ in range(3):
                    engine.tick()
                np.testing.assert_array_equal(held, engine.env.data.qpos)
            finally:
                if engine.env:
                    engine.env.close()


if __name__ == '__main__':
    unittest.main()
