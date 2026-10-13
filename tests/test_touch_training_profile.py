import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from common.paths import PROJECT_ROOT
from cloud.download_training_dataset import bundle_source, TOUCH_REVISION
from training.dataset_profile import select_dataset, validate_dataset
from training.checkpoint_schedule import validation_due


class TouchTrainingProfileTests(unittest.TestCase):
    def test_color_profile_has_requested_saves_and_pinned_dataset(self):
        from cloud.download_training_dataset import COLOR_TOUCH_REVISION
        from training.checkpoint_schedule import checkpoint_due
        c=json.loads((PROJECT_ROOT/'training/color_touch_50_500.json').read_text())
        self.assertEqual(select_dataset(c),Path('/workspace/panda-training/training_color_touch_v1'))
        self.assertEqual(c['checkpoint_updates'],[50,100,300,500])
        self.assertEqual(c['validation_updates'],[0,50,100,300,500])
        self.assertEqual(c['expected_split_counts'],dict(
            train=dict(episodes=34,transitions=609),validation=dict(episodes=8,transitions=136),
            test=dict(episodes=8,transitions=142)))
        self.assertEqual((c['batch_size'],c['grad_accumulation_steps'],c['max_steps']),(8,1,500))
        self.assertEqual(c['learning_rate'],.0001)
        self.assertEqual(c['batch_policy'],'config')
        self.assertEqual(bundle_source('training_color_touch_v1'),
                         ('Maxalderone/robot-openvla-panda-color-touch',COLOR_TOUCH_REVISION,''))
        with patch.dict('os.environ',{'PANDA_CHECKPOINT_UPDATES':json.dumps(c['checkpoint_updates'])}):
            self.assertEqual([i for i in range(501) if checkpoint_due(i,100)], [50,100,300,500])

    def config(self):
        return json.loads((PROJECT_ROOT/'training/touch_cube_50_500.json').read_text())

    def test_profile_dataset_batch_and_schedules(self):
        config=self.config()
        self.assertEqual(select_dataset(config),Path('/workspace/panda-training/training_touch_v1'))
        self.assertEqual((config['batch_size'],config['grad_accumulation_steps']),(8,1))
        self.assertEqual(config['batch_policy'],'config')
        self.assertFalse(config['require_benchmark'])
        self.assertEqual(config['max_steps'],500)
        self.assertEqual(config['validation_updates'],config['checkpoint_updates'])
        self.assertEqual(config['expected_split_counts']['train'],dict(episodes=34,transitions=604))
        self.assertEqual(bundle_source('training_touch_v1'),
                         ('Maxalderone/robot-openvla-panda-touch',TOUCH_REVISION,''))

    def test_validation_schedule_and_interval_default(self):
        with patch.dict('os.environ',{'PANDA_VALIDATION_UPDATES':json.dumps(self.config()['validation_updates'])}):
            self.assertTrue(validation_due(10,100))
            self.assertFalse(validation_due(11,100))
            self.assertTrue(validation_due(500,100))
        with patch.dict('os.environ',{'PANDA_VALIDATION_UPDATES':''}):
            self.assertFalse(validation_due(10,100))
            self.assertTrue(validation_due(100,100))

    def test_fixed_batch_reaches_upstream_command_without_auto_selection(self):
        from training.train_panda import main
        config=self.config()
        with tempfile.TemporaryDirectory() as tmp:
            dataset=Path(tmp)/'training_touch_v1';dataset.mkdir()
            counts=config['expected_split_counts']
            from simulation.panda_actions import action_contract
            (dataset/'manifest.json').write_text(json.dumps(dict(split_counts=counts,action_contract=action_contract(camera='touch_camera'))))
            (dataset/'verification.json').write_text(json.dumps(dict(passed=True,split_counts=counts)))
            (dataset/'input_check.json').write_text(json.dumps(dict(passed=True,split_counts=counts,all_source_images_actions_instructions_match=True)))
            output=io.StringIO()
            with patch('sys.argv',['train','--dataset',str(dataset),'--config',
                       str(PROJECT_ROOT/'training/touch_cube_50_500.json'),
                       '--output-root',str(Path(tmp)/'runs')]), \
                 patch('torch.cuda.is_available',return_value=True), \
                 patch('torch.cuda.current_device',return_value=0), \
                 patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(name='NVIDIA A40')), \
                 patch('training.batch_policy.choose_batch',side_effect=AssertionError('Must not choose an automatic batch')), \
                 patch('training.fetch_upstream.fetch'), \
                 patch('training.train_panda.subprocess.run') as run, \
                 patch.object(Path,'write_text'),contextlib.redirect_stdout(output):
                main()
            command=run.call_args.args[0];env=run.call_args.kwargs['env']
            self.assertEqual(command[command.index('--batch_size')+1],'8')
            self.assertEqual(command[command.index('--grad_accumulation_steps')+1],'1')
            self.assertEqual(json.loads(env['PANDA_VALIDATION_UPDATES']),config['checkpoint_updates'])
            self.assertEqual(json.loads(env['PANDA_CHECKPOINT_UPDATES']),config['checkpoint_updates'])
            for forbidden in ('--batch_policy','--validation_updates','--checkpoint_updates'):
                self.assertNotIn(forbidden,command)


if __name__=='__main__':unittest.main()
