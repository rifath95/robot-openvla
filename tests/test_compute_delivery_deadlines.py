"""Computation stops at 125%; verified delivery can continue to the 150% guard."""
import json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from cloud.bowl_chain_plan import timeout_budget,make_uninterrupted_plan,trial_specs

class ComputeDeliveryDeadlineTests(unittest.TestCase):
    def test_percentages_and_cooperative_clock(self):
        from cloud.compute_deadline import expired,check,ComputeDeadlineReached
        for hours,inner,outer in [(8,10,12),(12,15,18)]:
            budget=timeout_budget(hours*3600)
            self.assertEqual(budget['inner_seconds'],inner*3600);self.assertEqual(budget['outer_seconds'],outer*3600)
        with patch.dict(os.environ,{'PANDA_COMPUTE_DEADLINE_MONOTONIC':'200'}),patch('time.monotonic',return_value=201):
            self.assertTrue(expired())
            with self.assertRaises(ComputeDeadlineReached):check()

    def test_supervisor_waits_for_delivery_beyond_inner_and_marks_partial(self):
        from cloud.bowl_chain import supervise
        for partial in (False,True):
            with self.subTest(partial=partial),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);session=root/'session';local=root/'local';session.mkdir();local.mkdir()
                if partial:(session/'worker_completion.json').write_text(json.dumps(dict(work_complete=False,delivery_verified=True)))
                config=dict(session=str(session),local=str(local),pod_id='p',pod_key='SECRET',hf_token='PRIVATE',code_commit='code',
                    sweep=str(local/'sweep'),global_volume=None,deadline_policy='compute_then_delivery_v1',timeout_budget=dict(inner_seconds=200,outer_seconds=1000))
                guard=Mock(pid=1,stdin=Mock());worker=Mock(pid=2,stdin=Mock());worker.wait.return_value=0
                with patch('cloud.bowl_chain.subprocess.Popen',side_effect=[guard,worker]) as spawn,patch('cloud.bowl_chain.time.monotonic',return_value=100),patch('cloud.bowl_chain.kill_group') as kill,patch('cloud.bowl_chain.attempt_recovery') as recovery,patch('cloud.bowl_chain.stop_request',return_value=True):
                    self.assertEqual(supervise(config),1 if partial else 0)
                worker.wait.assert_called_once_with(timeout=880)
                self.assertEqual(spawn.call_args_list[1].kwargs['env']['PANDA_COMPUTE_DEADLINE_MONOTONIC'],'300')
                kill.assert_not_called();recovery.assert_not_called()
                self.assertEqual(json.loads((session/'status.json').read_text())['phase'],'incomplete' if partial else 'completed')

    def test_stalled_delivery_is_killed_before_outer_stop_reserve(self):
        import subprocess
        from cloud.bowl_chain import supervise
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'session';local=root/'local';session.mkdir();local.mkdir()
            config=dict(session=str(session),local=str(local),pod_id='p',pod_key='SECRET',hf_token='PRIVATE',code_commit='code',
                sweep=str(local/'sweep'),global_volume=None,deadline_policy='compute_then_delivery_v1',timeout_budget=dict(inner_seconds=200,outer_seconds=1000))
            guard=Mock(pid=1,stdin=Mock());worker=Mock(pid=2,stdin=Mock());worker.wait.side_effect=[subprocess.TimeoutExpired('upload',880),-15]
            with patch('cloud.bowl_chain.subprocess.Popen',side_effect=[guard,worker]),patch('cloud.bowl_chain.time.monotonic',return_value=100),patch('cloud.bowl_chain.kill_group') as kill,patch('cloud.bowl_chain.attempt_recovery',return_value='no_time_remaining'),patch('cloud.bowl_chain.stop_request',return_value=True) as stop:
                self.assertEqual(supervise(config),1)
            kill.assert_called_once_with(2);stop.assert_called_once()
            self.assertEqual(json.loads((session/'status.json').read_text())['phase'],'outer_timeout')

    def test_partial_training_verifies_actual_checkpoints_without_claiming_completion(self):
        from cloud.color_bowl_training import verify_computation
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'compute_completion.json').write_text(json.dumps(dict(training_complete=False,last_update=9100,run_update=9100,reason='computation_limit')))
            with patch('cloud.bowl_chain_worker.verify_training',return_value=dict(training_complete=True,last_update=9100)) as verify:
                result,cps=verify_computation(p,dict(max_steps=12000),[6000,8000,10000,12000],True)
            self.assertEqual(cps,[6000,8000,9100]);verify.assert_called_once_with(p,9100,cps)
            self.assertFalse(result['training_complete']);self.assertEqual(result['requested_updates'],12000)

    def test_deadline_stops_sweep_before_model_load_and_retains_valid_partial_plan(self):
        from cloud.bowl_chain_sweep import run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'scripted_test';folder.mkdir();session=root/'session';session.mkdir()
            plan=make_uninterrupted_plan('panda-train-color-bowl-test')
            (folder/'results.json').write_text(json.dumps(dict(plan=plan,trials=[],launcher_error=None)))
            with patch.dict(os.environ,{'PANDA_COMPUTE_DEADLINE_MONOTONIC':'1'}),patch('cloud.bowl_chain_sweep.subprocess.Popen') as model:
                self.assertEqual(run(dict(sweep=str(folder),session=str(session),local=str(root),base='/base',training_run='/training')),0)
            model.assert_not_called();saved=json.loads((folder/'results.json').read_text())
            self.assertEqual(saved['plan']['compute_stop_reason'],'inner_compute_deadline');self.assertIsNone(saved['launcher_error'])
            self.assertIn('INCOMPLETE',(folder/'REPORT.md').read_text())

    def test_partial_sweep_archive_is_verified_and_restorable_but_not_complete(self):
        from cloud.archive_sweep import package
        from cloud.chain_archive import backup_sweep
        from cloud.scripted_touch_session import write_reports
        from data_tools.restore_sweep_archive import restore
        from common.paths import PROJECT_ROOT
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'outputs') as tmp:
            root=Path(tmp);source=root/'scripted_test';source.mkdir();staging=root/'staging';staging.mkdir();job=root/'job';job.mkdir()
            plan=make_uninterrupted_plan('panda-train-color-bowl-test');plan['compute_stop_reason']='inner_compute_deadline'
            row={k:v for k,v in next(trial_specs(plan)).items() if k!='layout'}
            row.update(phase='Completed',outcome='no_lift',placement_success=False)
            trial=source/row['directory'];trial.mkdir();(trial/'summary.json').write_text('{}')
            write_reports(source,plan,[row])
            package(source,staging,json.loads((source/'results.json').read_text()))
            manifest=json.loads((staging/'manifest.json').read_text());self.assertFalse(manifest['complete']);self.assertEqual(manifest['expected_trials'],800)
            restored=restore(staging,root/'restored');self.assertEqual(len(json.loads((restored/'results.json').read_text())['trials']),1)
            api=Mock();api.repo_info.return_value.private=True;api.upload_folder.return_value.oid='commit'
            # Only saved trials are packed; missing trials remain explicitly incomplete.
            with patch('huggingface_hub.HfApi',return_value=api),patch('cloud.chain_archive.verify_files'):
                receipt=backup_sweep(source,'owner/results','SECRET',root,job,allow_partial=True)
            self.assertTrue(receipt['verified']);self.assertFalse(receipt['complete']);self.assertEqual(receipt['trials'],1)
            with self.assertRaises(ValueError):backup_sweep(source,'owner/results','SECRET',root,job)

if __name__=='__main__':unittest.main()
