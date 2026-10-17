"""Paired 800-trial sweep, checkpoint discovery/waits, and portable HF delivery."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from common.paths import PROJECT_ROOT
from cloud.bowl_chain_plan import make_colored_plan,validate_saved,trial_specs
from cloud.color_bowl_sweep import PROFILE,resolve_training,worker,publish_reports

class ColoredSweepTests(unittest.TestCase):
    def test_frozen_layouts_exact_inventory_and_historical_baseline(self):
        profile=json.loads(PROFILE.read_text());plan=make_colored_plan('panda-train-color-bowl-test')
        self.assertEqual(plan['groups'],profile['groups']);self.assertEqual(plan['checkpoints'],[1000,2000,4000,6000])
        self.assertFalse(plan['include_base_model']);self.assertEqual(len(validate_saved(dict(plan=plan,trials=[]))),800)
        self.assertEqual(plan['timeout_budget']['inner_seconds'],25*3600);self.assertEqual(plan['timeout_budget']['outer_seconds'],30*3600)
        self.assertEqual(plan['historical_baseline']['trials'],200)
        self.assertIn('300-demo',plan['historical_baseline']['normalization'])
        for damage in ('layout','checkpoint','baseline'):
            bad=copy.deepcopy(plan)
            if damage=='layout':bad['groups'][0]['layouts'][0]['bowl'][0]+=.0001
            elif damage=='checkpoint':bad['checkpoints'][0]=500
            else:bad['historical_baseline']['correct_placements']=200
            with self.assertRaises(ValueError):validate_saved(dict(plan=bad,trials=[]))

    def test_source_discovery_requires_exact_dataset_and_refuses_ambiguous_sessions(self):
        profile=json.loads(PROFILE.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('panda-train-color-bowl-one','panda-train-color-bowl-two'):
                p=root/name;p.mkdir()
                (p/'chain_config.json').write_text(json.dumps(dict(training_only=True,training_config=profile['source_training_config'])))
                (p/'provenance.json').write_text(json.dumps(dict(dataset_repo=profile['source_dataset_repo'],dataset_revision=profile['source_dataset_revision'],checksums_sha256=profile['source_dataset_sha256'],training_episodes=600,training_examples=43344)))
            with self.assertRaisesRegex(ValueError,'one matching'):resolve_training(profile,root)
            chosen=resolve_training(profile,root,'panda-train-color-bowl-one');self.assertEqual(chosen.name,'panda-train-color-bowl-one')
            d=json.loads((chosen/'provenance.json').read_text());d['checksums_sha256']='wrong';(chosen/'provenance.json').write_text(json.dumps(d))
            with self.assertRaises(ValueError):resolve_training(profile,root,chosen.name)

    def test_checkpoint_wait_ends_when_source_has_failed(self):
        from cloud.sweep_worker import wait_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);status=root/'status.json';status.write_text('{"phase":"failed"}')
            with patch('cloud.checkpoint_ready.check_ready',side_effect=ValueError('not ready')),patch('cloud.sweep_worker.time.sleep') as sleep,self.assertRaisesRegex(RuntimeError,'ended without'):
                wait_checkpoint(root/'checkpoint_006000',14400,status)
            sleep.assert_not_called()

    def test_worker_stages_then_sweeps_then_uploads_without_training_or_deletion(self):
        for fail in (False,True):
            with self.subTest(sweep_failed=fail),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);session=root/'reports';local=root/'local';session.mkdir();local.mkdir();source=root/'panda-train-color-bowl-test';source.mkdir();(source/'retained_checkpoint').write_bytes(b'weights')
                config=dict(session=str(session),local=str(local),sweep=str(local/'scripted-evaluations/scripted_test'),hf_token='PRIVATE_TOKEN',pod_key='SECRET_KEY',sweep_repo='owner/results',base_volume='/workspace/openvla-7b',base=str(local/'base'),global_volume=str(root/'global'),started_at_utc='2026-10-15T00:00:00+00:00',timeout_budget={},code_commit='abc')
                order=[];api=Mock();api.repo_info.return_value.private=True
                def command(cmd,*args,**kwargs):
                    self.assertNotIn('HF_TOKEN',kwargs['env']);self.assertNotIn('RUNPOD_API_KEY',kwargs['env'])
                    if '--preflight' in cmd:order.append('preflight')
                    else:
                        order.append('sweep')
                        if fail:raise RuntimeError('sweep failed')
                with patch('huggingface_hub.HfApi',return_value=api),patch('shutil.disk_usage',return_value=SimpleNamespace(free=200*1024**3)),patch('cloud.bowl_chain_worker.global_probe'),patch('cloud.color_bowl_sweep.resolve_training',return_value=source),patch('cloud.bowl_chain_worker.setup_simulator',return_value='/sim/python'),patch('cloud.bowl_chain_worker.stage_base',side_effect=lambda *a:order.append('base') or {}),patch('cloud.bowl_chain_worker.run_command',side_effect=command),patch('cloud.chain_archive.backup_sweep',side_effect=lambda *a:order.append('upload')),patch('cloud.color_bowl_sweep.publish_reports',side_effect=lambda *a:order.append('reports')):
                    if fail:
                        with self.assertRaises(RuntimeError):worker(config)
                        self.assertEqual(order,['base','preflight','sweep'])
                    else:
                        self.assertEqual(worker(config),0);self.assertEqual(order,['base','preflight','sweep','upload','reports','reports'])
                        safe=(local/'worker_config.json').read_text();self.assertNotIn('PRIVATE_TOKEN',safe);self.assertNotIn('SECRET_KEY',safe)
                self.assertEqual((source/'retained_checkpoint').read_bytes(),b'weights')

    def test_runner_waits_copies_and_finishes_at_800_with_controller_failures_retained(self):
        from cloud.bowl_chain_sweep import run
        from cloud.scripted_touch_session import write_reports
        for controller_failure in (False,True):
            with self.subTest(controller_failure=controller_failure),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);sweep=root/'sweep';sweep.mkdir();session=root/'job';session.mkdir();local=root/'local';local.mkdir();source=root/'weights';source.mkdir()
                (source/'training_config.json').write_text(json.dumps(dict(checkpoint_update=6000,dataset_provenance=dict(checksums_sha256='expected',dataset_revision='revision'))))
                (source/'adapter_model.safetensors').write_bytes(b'fixture')
                plan=make_colored_plan('panda-train-color-bowl-test');specs=list(trial_specs(plan));rows=[]
                for spec in specs[:-1]:
                    r={k:v for k,v in spec.items() if k!='layout'};r.update(phase='Completed',outcome='no_lift',placement_success=False);rows.append(r)
                write_reports(sweep,plan,rows)
                config=dict(sweep=str(sweep),session=str(session),local=str(local),base=str(root/'base'),training_run=str(root/'runs/training'),sim_python='/sim/python',source_status=str(root/'status.json'),checkpoint_wait_seconds=14400,source_dataset_sha256='expected',source_dataset_revision='revision',verify_adapter_copy=True)
                server=Mock();server.poll.return_value=None
                def trial(command,**kwargs):
                    self.assertIn('--no-view',command);self.assertEqual(command[command.index('--steps')+1],'120');self.assertNotIn('HF_TOKEN',kwargs['env'])
                    destination=Path(command[command.index('--output-dir')+1]);destination.mkdir()
                    (destination/'summary.json').write_text(json.dumps(dict(phase='Failed' if controller_failure else 'Completed',task_result=dict(outcome='no_lift' if controller_failure else 'correct_placed',success=not controller_failure))))
                    (destination/'status.json').write_text(json.dumps(dict(error='Action executed but tracking failed: 0.006 m' if controller_failure else None)))
                    return SimpleNamespace(returncode=1 if controller_failure else 0)
                with patch('cloud.sweep_worker.wait_checkpoint',return_value=source) as wait,patch('cloud.checkpoint_ready.check_ready'),patch('cloud.bowl_chain_sweep.subprocess.Popen',return_value=server),patch('cloud.bowl_chain_sweep.wait_server',return_value=dict(server_port=12345)),patch('cloud.bowl_chain_sweep.subprocess.run',side_effect=trial):
                    self.assertEqual(run(config),0);self.assertEqual(wait.call_args.args[1],14400)
                saved=json.loads((sweep/'results.json').read_text());self.assertEqual(len(saved['trials']),800);self.assertEqual(saved['trials'][-1]['controller_failed'],controller_failure)
                self.assertFalse(saved['trials'][-1]['infrastructure_failed']);self.assertTrue((local/'adapters/checkpoint_006000/adapter_model.safetensors').is_file())

    def test_archive_restores_800_trials_and_final_metadata_is_verified(self):
        from cloud.archive_sweep import package,digest
        from cloud.scripted_touch_session import write_reports
        from data_tools.restore_sweep_archive import restore
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'outputs') as tmp:
            local=Path(tmp);source=local/'scripted_test';source.mkdir();session=local/'job';session.mkdir();staging=local/'sweep-backup';staging.mkdir()
            plan=make_colored_plan('panda-train-color-bowl-test');rows=[]
            for spec in trial_specs(plan):
                r={k:v for k,v in spec.items() if k!='layout'};r.update(phase='Completed',outcome='no_lift',placement_success=False)
                folder=source/spec['directory'];folder.mkdir(parents=True);(folder/'summary.json').write_text('{}');rows.append(r)
            write_reports(source,plan,rows);package(source,staging,dict(plan=plan,trials=rows))
            before=[r for r in json.loads((staging/'manifest.json').read_text())['files'] if r['path'].startswith('archives/')]
            (session/'TIMINGS.md').write_text('Complete timings');(session/'sweep_hf_receipt.json').write_text('{"verified":true,"revision":"previous"}')
            api=Mock();api.upload_folder.return_value.oid='new-pinned-commit'
            with patch('huggingface_hub.HfApi',return_value=api),patch('cloud.chain_archive.verify_files') as verify:
                publish_reports(source,local,session,'owner/results','TOKEN');self.assertEqual(verify.call_args.args[2],'new-pinned-commit')
            after=json.loads((staging/'manifest.json').read_text());self.assertEqual(before,[r for r in after['files'] if r['path'].startswith('archives/')])
            for entry in after['files']:self.assertEqual(digest(staging/entry['path']),entry['sha256'])
            restored=restore(staging,local/'restored');self.assertEqual(len(json.loads((restored/'results.json').read_text())['trials']),800)
            self.assertEqual((restored/'TIMINGS.md').read_text(),'Complete timings');self.assertIn('Historical pretrained baseline',(restored/'REPORT.md').read_text())

if __name__=='__main__':unittest.main()
