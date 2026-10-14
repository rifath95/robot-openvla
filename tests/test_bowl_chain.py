"""Chain plan/recovery/timeout tests without models, cloud access or paid compute."""
import copy,json,os,time,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from cloud.bowl_chain_plan import make_plan,trial_specs,validate_saved,timeout_budget
from cloud.chain_timing import Timings
from cloud.chain_archive import recover_to_global
from cloud.bowl_chain import supervise,watchdog,stop_request
from cloud.bowl_chain_worker import verify_training

class ChainTests(unittest.TestCase):
    def test_exact_inventory_layout_matching_and_balanced_target_positions(self):
        plan=make_plan('panda-train-test');specs=list(trial_specs(plan));self.assertEqual(len(specs),1000)
        self.assertEqual(len({s['directory'] for s in specs}),1000)
        validate_saved(dict(plan=plan,trials=[]))
        for cp in plan['checkpoints']:
            own=[s for s in specs if s['checkpoint']==cp];self.assertEqual(len(own),200)
            for name,n in (('single_seen',20),('single_unseen',20),('two_red_present',40),('two_red_absent',40),('four_red_present',40),('four_red_absent',40)):
                self.assertEqual(sum(s['color_set']==name for s in own),n)
        # Each target is requested in each layout, not picked randomly.
        four=[s for s in specs if s['checkpoint']==500 and s['color_set']=='four_red_absent']
        self.assertEqual({s['target_color'] for s in four},{'green','black','purple','orange'})
        for seed in range(1000,1010):self.assertEqual(sum(s['seed']==seed for s in four),4)
        damaged=copy.deepcopy(plan);damaged['groups'][0]['layouts'][0]['bowl'][0]+=.01
        with self.assertRaises(ValueError):validate_saved(dict(plan=damaged,trials=[]))
        damaged=copy.deepcopy(specs[0]);damaged['target_color']='purple'
        with self.assertRaises(ValueError):validate_saved(dict(plan=plan,trials=[damaged]))

    def test_timeout_buffers_and_failed_stage_timings(self):
        budget=timeout_budget();self.assertEqual(budget['inner_seconds'],28*3600);self.assertEqual(budget['outer_seconds'],33*3600)
        with tempfile.TemporaryDirectory() as tmp:
            timer=Timings(tmp)
            with self.assertRaises(RuntimeError):
                with timer.stage('upload'):raise RuntimeError('failure')
            rows=[json.loads(s) for s in (Path(tmp)/'stage_timings.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['status'],'failed');self.assertGreaterEqual(rows[-1]['seconds'],0)

    def test_global_recovery_uses_regular_copies_and_never_deletes_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'recordings';source.mkdir();(source/'state.json').write_text('{"state":1}')
            receipt=recover_to_global(source,root/'global')
            self.assertTrue(receipt['verified']);self.assertTrue((source/'state.json').exists())
            self.assertEqual((root/'global/recordings/state.json').read_bytes(),(source/'state.json').read_bytes())
            self.assertTrue((root/'global/recordings/RECOVERY_COMPLETE.json').exists())
            (source/'bad').symlink_to(source/'state.json')
            with self.assertRaises(ValueError):recover_to_global(source,root/'another')

    def test_inner_timeout_kills_worker_recovers_then_stops(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'session';local=root/'local';session.mkdir();local.mkdir()
            config=dict(session=str(session),local=str(local),pod_id='test',pod_key='SECRET',hf_token='HFSECRET',code_commit='abc',global_volume=str(root/'global'),sweep=str(local/'sweep'),timeout_budget=dict(inner_seconds=.01,outer_seconds=130))
            guard=Mock(pid=100,stdin=Mock());worker=Mock(pid=200,stdin=Mock());recovery=Mock(pid=300,stdin=Mock())
            worker.wait.side_effect=[subprocess.TimeoutExpired('worker',.01),-15];worker.poll.return_value=-15;recovery.wait.return_value=0
            with patch('cloud.bowl_chain.subprocess.Popen',side_effect=[guard,worker,recovery]),patch('cloud.bowl_chain.kill_group') as kill,patch('cloud.bowl_chain.stop_request',return_value=True) as stop:
                self.assertEqual(supervise(config),1)
                kill.assert_called_once_with(200);stop.assert_called_once()
            saved=json.loads((session/'chain_config.json').read_text())
            self.assertNotIn('hf_token',saved);self.assertNotIn('pod_key',saved)
            self.assertEqual(json.loads((session/'status.json').read_text())['phase'],'inner_timeout')

    def test_outer_guard_can_stop_without_worker_or_volume_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=dict(local=tmp,session=str(Path(tmp)/'missing'),outer_deadline_monotonic=time.monotonic()-1,pod_id='test',pod_key='secret')
            with patch('cloud.bowl_chain.api_request',return_value={}) as api:
                self.assertEqual(watchdog(config),0)
                api.assert_called_once_with('test','secret',stop=True)
            self.assertTrue((Path(tmp)/'guard_done').exists())

    def test_controller_failures_are_measured_trials_not_whole_job_failures(self):
        from cloud.bowl_chain_sweep import classify_failure
        summary=dict(phase='Failed',task_result=dict(outcome='no_lift'))
        self.assertEqual(classify_failure(1,summary,dict(error='Action executed but tracking failed: 0.01 m')),(True,False))
        self.assertEqual(classify_failure(1,summary,dict(error='Server returned HTTP 503')),(False,True))
        self.assertEqual(classify_failure(0,{},{}),(False,True))

    def test_worker_orders_stages_and_never_runs_sweep_after_training_failure(self):
        from cloud.bowl_chain_worker import worker
        for fail in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);session=root/'panda-train-test';local=root/'local';session.mkdir();local.mkdir()
                config=dict(session=str(session),local=str(local),sweep=str(local/'scripted-evaluations/scripted_test'),global_volume=str(root/'global'),base_volume='/workspace/openvla-7b',base=str(local/'base'),dataset=str(local/'training_place_bowl_v2_300'),training_run=str(session/'runs/training'),dataset_repo='owner/data',dataset_revision='a'*40,checksums_sha256='b'*64,hf_token='PRIVATE_TOKEN',pod_id='pod',code_commit='abc',checkpoint_repo='owner/weights',sweep_repo='owner/results')
                api=Mock();api.repo_info.return_value.private=True
                calls=[]
                def command(command,log,**kwargs):
                    if 'cloud.bowl_chain_sweep' in command and '--preflight' in command:calls.append('preflight')
                    elif 'training.train_panda' in command:
                        calls.append('training')
                        if fail:raise RuntimeError('training failed')
                    else:calls.append('sweep')
                patches=[patch('huggingface_hub.HfApi',return_value=api),patch('cloud.bowl_chain_worker.shutil.disk_usage',return_value=Mock(free=200*1024**3)),patch('cloud.bowl_chain_worker.global_probe'),patch('cloud.bowl_chain_worker.setup_simulator',return_value='/sim/python'),patch('cloud.bowl_chain_worker.download_dataset'),patch('cloud.bowl_chain_worker.stage_base',return_value={}),patch('cloud.bowl_chain_worker.run_command',side_effect=command),patch('cloud.bowl_chain_worker.verify_training',return_value=dict(training_update_seconds=1,checkpoint_seconds=1)),patch('cloud.chain_archive.verify_files'),patch('cloud.chain_archive.backup_training',side_effect=lambda *a:calls.append('checkpoint_backup')),patch('cloud.chain_archive.backup_sweep',side_effect=lambda *a:calls.append('sweep_backup'))]
                from contextlib import ExitStack
                with ExitStack() as stack:
                    for p in patches:stack.enter_context(p)
                    if fail:
                        with self.assertRaises(RuntimeError):worker(config)
                        self.assertEqual(calls,['preflight','training'])
                    else:
                        self.assertEqual(worker(config),0);self.assertEqual(calls,['preflight','training','sweep','checkpoint_backup','sweep_backup'])
                        self.assertTrue((session/'worker_completion.json').exists())
                self.assertNotIn('PRIVATE_TOKEN',(session/'provenance.json').read_text())

    def test_outer_guard_kills_a_real_owned_process_before_stopping(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as tmp:
            process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
            try:
                Path(tmp,'processes.json').write_text(json.dumps(dict(children=[process.pid])))
                config=dict(local=tmp,session=str(Path(tmp)/'missing'),outer_deadline_monotonic=time.monotonic()-1,pod_id='test',pod_key='secret')
                with patch('cloud.bowl_chain.api_request',return_value={}):self.assertEqual(watchdog(config),0)
                self.assertEqual(process.wait(timeout=5),-9)
            finally:
                if process.poll() is None:process.kill();process.wait()

    def test_incomplete_training_cannot_advance_to_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'metrics.jsonl').write_text('{"update":2999,"train_loss":1}\n')
            with self.assertRaisesRegex(ValueError,'exactly 3000'):verify_training(tmp)

if __name__=='__main__':unittest.main()
