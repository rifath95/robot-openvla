"""Test checkpoint/seed pairing and cleanup without a rented GPU."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from cloud import scripted_touch_session as session


class ScriptedTouchTests(unittest.TestCase):
    def test_stop_targets_environment_pod_after_identity_check(self):
        from cloud import scripted_eval_control as control
        with patch('sys.argv', ['control', '--session', 'scripted_touch_test']), \
                patch.object(control, 'recover_pod_environment', return_value=('inference-pod', 'secret')), \
                patch.object(control, 'api_request', return_value={'id': 'inference-pod'}), \
                patch.object(control.Path, 'mkdir'), \
                patch.object(control, 'stop_pod', return_value=0) as stop:
            self.assertEqual(control.main(), 0)
        self.assertEqual(stop.call_args.args[:2], ('inference-pod', 'secret'))

    def test_wrong_pod_identity_prevents_stop(self):
        from cloud import scripted_eval_control as control
        with patch('sys.argv', ['control', '--session', 'scripted_touch_test']), \
                patch.object(control, 'recover_pod_environment', return_value=('inference-pod', 'secret')), \
                patch.object(control, 'api_request', return_value={'id': 'different-pod'}), \
                patch.object(control, 'stop_pod') as stop:
            self.assertEqual(control.main(), 1)
        stop.assert_not_called()

    def test_identical_unseen_seeds_automatic_greedy_loop_and_failure_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'datasets/panda_touch_cube/manifest.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({'episodes': [{'cube_xy': [0, 0]}]}))
            folder = root / 'outputs/test'
            folder.mkdir(parents=True)
            args = SimpleNamespace(scene_seeds=[1000, 1001], checkpoints=[100, 500],
                steps=50, training_session='panda-train-test', startup_timeout_seconds=60,
                trial_timeout_seconds=10)
            commands = []
            def run(command, **kwargs):
                if 'inference.remote_loop' in command:
                    commands.append(command)
                    path = Path(command[command.index('--output-dir')+1])
                    path.mkdir()
                    seed = int(command[command.index('--scene-seed')+1])
                    (path / 'summary.json').write_text(json.dumps(dict(
                        phase='Completed' if seed == 1000 else 'Failed', completed_steps=18,
                        task_result={'success': True})))
                    return SimpleNamespace(returncode=0 if seed == 1000 else 1)
                return SimpleNamespace(returncode=0)
            tunnel = MagicMock()
            with patch.object(session, 'ROOT', root), patch.object(session.subprocess, 'run', side_effect=run), \
                    patch.object(session.subprocess, 'Popen', return_value=tunnel), \
                    patch.object(session, 'wait_for_server', return_value={'startup_info': {}}):
                session.run_trials(args, folder, ['ssh', 'root@host'])
            self.assertEqual([c[c.index('--scene-seed')+1] for c in commands],
                             ['1000', '1001', '1000', '1001'])
            self.assertTrue(all('--no-view' in c and '--start-immediately' in c for c in commands))
            self.assertTrue(all('--sample-actions' not in c for c in commands))
            rows = json.loads((folder / 'results.json').read_text())['trials']
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[1]['phase'], 'Failed')
            self.assertTrue(rows[1]['contact'])
            tunnel.terminate.assert_called_once()

    def test_stop_attempted_after_trial_error_even_if_volume_copy_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = root / 'key'
            key.touch()
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                if 'python3 -c ' in command[-1]:
                    raise OSError('copy failed')
                return SimpleNamespace(returncode=0)
            argv = ['script', '--ssh', f'ssh root@host -i {key}',
                    '--training-session', 'panda-train-test']
            with patch('sys.argv', argv), patch.object(session, 'ROOT', root), \
                    patch.object(session.socket, 'socket'), \
                    patch.object(session.subprocess, 'run', side_effect=run), \
                    patch.object(session, 'run_trials', side_effect=RuntimeError('trial failed')):
                self.assertEqual(session.main(), 1)
            stops = [c for c in calls if 'cloud.scripted_eval_control' in c[-1]
                     and '--check-only' not in c[-1]]
            self.assertEqual(len(stops), 1)

    def test_collected_position_rejected_before_model_setup(self):
        from data_tools.collect_panda_touch import scenario
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'datasets/panda_touch_cube/manifest.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({'episodes': [{'cube_xy': scenario(1000)['cube_xy']}]}))
            args = SimpleNamespace(scene_seeds=[1000])
            with patch.object(session, 'ROOT', root), patch.object(session.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'collected dataset position'):
                    session.run_trials(args, root, ['ssh', 'root@host'])
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
