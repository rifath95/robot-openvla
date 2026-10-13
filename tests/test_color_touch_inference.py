"""Color selection measurements and scripted pairing without cloud compute."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock,patch
import numpy as np

from inference.color_touch_task import select_instruction,ContactOutcome
from cloud import scripted_touch_session as session


class ColorInferenceTests(unittest.TestCase):
    def test_training_pod_rejection_does_not_arm_stop(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);key=root/'key';key.touch();calls=[]
            def run(command,**kwargs):
                calls.append(command)
                if '--refuse-training' in command[-1]:
                    raise subprocess.CalledProcessError(1,command)
                return SimpleNamespace(returncode=0)
            with patch('sys.argv',['script','--ssh',f'ssh root@host -i {key}',
                '--training-session','panda-train-test']),patch.object(session,'ROOT',root), \
                patch.object(session.socket,'socket'),patch.object(session.subprocess,'run',side_effect=run):
                self.assertEqual(session.main(default_task='color_touch'),1)
            self.assertFalse(any('cloud.scripted_eval_control' in c[-1] and '--check-only' not in c[-1]
                                 for c in calls))

    def test_training_process_guard_uses_exact_arguments(self):
        from cloud.scripted_eval_control import active_training
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'123').mkdir()
            f=root/'123/cmdline';f.write_bytes(b'python\0-m\0training.detached_run\0')
            self.assertTrue(active_training(root))
            f.write_bytes(b'python\0-m\0cloud.connection_server\0')
            self.assertFalse(active_training(root))

    def test_instruction_determines_requested_cube_and_rejects_mismatch(self):
        self.assertEqual(select_instruction('touch the blue cube.'),('touch the blue cube.','blue'))
        self.assertEqual(select_instruction(None,'red'),('touch the red cube.','red'))
        with self.assertRaises(ValueError):select_instruction('touch the red cube.','blue')
        with self.assertRaises(ValueError):select_instruction('move up')

    def test_brief_wrong_contact_is_remembered_even_after_correct_contact(self):
        env=SimpleNamespace(model=None,data=None)
        with patch('inference.color_touch_task.cube_position',return_value=np.array([.5,0,.27])):
            result=ContactOutcome(env,'red')
            with patch('inference.color_touch_task.finger_cube_contacts',side_effect=[[],[{'force':1}]]):
                result.observe(env)
            with patch('inference.color_touch_task.finger_cube_contacts',side_effect=[[{'force':1}],[]]):
                result.observe(env)
            measured=result.result(env)
        self.assertFalse(measured['success'])
        self.assertEqual(measured['outcome'],'both_cubes')
        self.assertEqual(measured['first_contact'],'blue')

    def test_checkpoint_sweep_pairs_both_instructions_in_same_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'datasets/panda_color_touch/manifest.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({'episodes':[{'cube_positions':{'red':[0,0],'blue':[0,0]}}]}))
            folder=root/'outputs/test';folder.mkdir(parents=True)
            args=SimpleNamespace(task='color_touch',scene_seeds=[1000,1001],checkpoints=[50,300],
                steps=50,training_session='panda-train-test',startup_timeout_seconds=60,trial_timeout_seconds=10)
            commands=[]
            def run(command,**kwargs):
                if 'inference.remote_loop' in command:
                    commands.append(command);path=Path(command[command.index('--output-dir')+1]);path.mkdir()
                    color=command[command.index('--target-color')+1]
                    (path/'summary.json').write_text(json.dumps(dict(phase='Completed',completed_steps=18,
                        task_result=dict(correct_cube_contact=color=='red',wrong_cube_contact=color=='blue',
                                         outcome='correct_cube' if color=='red' else 'wrong_cube'))))
                return SimpleNamespace(returncode=0)
            tunnel=MagicMock()
            with patch.object(session,'ROOT',root),patch.object(session.subprocess,'run',side_effect=run), \
                    patch.object(session.subprocess,'Popen',return_value=tunnel), \
                    patch.object(session,'wait_for_server',return_value={'startup_info':{}}):
                session.run_trials(args,folder,['ssh','root@host'])
            self.assertEqual(len(commands),8)
            self.assertEqual([(c[c.index('--scene-seed')+1],c[c.index('--target-color')+1]) for c in commands],
                [('1000','red'),('1000','blue'),('1001','red'),('1001','blue')]*2)
            self.assertTrue(all('--no-view' in c and '--start-immediately' in c for c in commands))
            rows=json.loads((folder/'results.json').read_text())['trials']
            self.assertEqual(sum(r['wrong_contact'] for r in rows),4)
            self.assertIn('Both colors correct in scene',(folder/'REPORT.md').read_text())
            tunnel.terminate.assert_called_once()


if __name__=='__main__':unittest.main()
