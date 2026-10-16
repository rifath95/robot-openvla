"""Standard two-stage launcher dispatch for the matching bowl sweep."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from cloud.bowl_chain_plan import make_uninterrupted_plan

class UninterruptedDeliveryTests(unittest.TestCase):
    def test_dispatch_retains_volume_output_and_checks_source_provenance(self):
        from cloud.bowl_transfer_worker import run
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);folder=p/'sweep';folder.mkdir()
            plan=make_uninterrupted_plan('panda-train-color-bowl-test')
            (folder/'results.json').write_text(json.dumps(dict(plan=plan,trials=[])))
            with patch('shutil.disk_usage',return_value=SimpleNamespace(free=100*1024**3)),patch('subprocess.run') as probe,patch('cloud.bowl_chain_worker.stage_base',return_value={}),patch('cloud.bowl_chain_sweep.run',return_value=0) as sweep:
                self.assertEqual(run(folder,'sim-python',cache_root=p/'cache'),0)
            cfg=sweep.call_args.args[0]
            self.assertEqual(cfg['sweep'],str(folder));self.assertEqual(cfg['session'],str(folder))
            self.assertEqual(cfg['checkpoint_wait_seconds'],43200)
            self.assertIn('source_dataset_sha256',cfg);self.assertTrue(cfg['verify_adapter_copy'])
            self.assertNotIn('hf_token',cfg);probe.assert_called_once()

    def test_same_pod_hf_profile_selects_new_checkpoints_and_uploads_after_trials(self):
        from cloud.color_bowl_sweep import worker
        for fail_upload in (False,True):
            with self.subTest(fail_upload=fail_upload),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);session=root/'job';local=root/'local';source=root/'panda-train-color-bowl-new'
                session.mkdir();local.mkdir();source.mkdir();order=[]
                config=dict(session=str(session),local=str(local),sweep=str(local/'scripted_test'),
                    sweep_profile='inference/sweep_profiles/color_place_bowl_600_uninterrupted.json',
                    global_volume='/workspace-global',hf_token='PRIVATE',sweep_repo='owner/results',
                    selected_training_session=source.name,started_at_utc='now',timeout_budget={},code_commit='code',
                    base_volume='/workspace/openvla-7b',base=str(local/'base'))
                def source_check(profile,**kwargs):
                    self.assertEqual(profile['source_training_config'],'training/color_place_bowl_600_uninterrupted_12000.json')
                    return source
                def command(args,*other,**kw):order.append('trial' if '--config' in args else 'preflight')
                def backup(*args):
                    plan=json.loads((Path(config['sweep'])/'results.json').read_text())['plan']
                    self.assertEqual(plan['checkpoints'],[6000,8000,10000,12000]);order.append('verified-upload')
                    if fail_upload:raise RuntimeError('upload failed')
                from unittest.mock import Mock
                api=Mock();api.repo_info.return_value.private=True
                with patch('huggingface_hub.HfApi',return_value=api),patch('shutil.disk_usage',return_value=SimpleNamespace(free=200*1024**3)),patch('cloud.bowl_chain_worker.global_probe'),patch('cloud.bowl_chain_worker.setup_simulator',return_value='sim'),patch('cloud.bowl_chain_worker.stage_base',return_value={}),patch('cloud.color_bowl_sweep.resolve_training',side_effect=source_check),patch('cloud.bowl_chain_worker.run_command',side_effect=command),patch('cloud.chain_archive.backup_sweep',side_effect=backup),patch('cloud.color_bowl_sweep.publish_reports',side_effect=lambda *a:order.append('verified-reports')):
                    if fail_upload:
                        with self.assertRaisesRegex(RuntimeError,'upload failed'):worker(config)
                        self.assertNotIn('verified-reports',order)
                    else:
                        self.assertEqual(worker(config),0)
                        self.assertEqual(order,['preflight','trial','verified-upload','verified-reports','verified-reports'])
                safe=json.loads((local/'worker_config.json').read_text());self.assertNotIn('hf_token',safe)
                self.assertEqual(json.loads((session/'source_training.json').read_text())['checkpoints'],[6000,8000,10000,12000])

    def test_standard_worker_routes_bowl_plans_before_other_task_setup(self):
        from cloud.sweep_worker import run
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'results.json').write_text(json.dumps(dict(plan=make_uninterrupted_plan('panda-train-color-bowl-test'),trials=[])))
            with patch('cloud.bowl_transfer_worker.run',return_value=0) as dispatch:
                self.assertEqual(run(p,'sim'),0)
            dispatch.assert_called_once_with(p,'sim')

    def test_low_volume_space_rejects_before_probe_or_staging(self):
        from cloud.bowl_transfer_worker import run
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'results.json').write_text(json.dumps(dict(plan=make_uninterrupted_plan('panda-train-color-bowl-test'),trials=[])))
            with patch('shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3)),patch('cloud.bowl_chain_worker.stage_base') as stage:
                with self.assertRaisesRegex(ValueError,'35 GiB'):run(p,'sim',cache_root=p/'cache')
            stage.assert_not_called()

if __name__=='__main__':unittest.main()
