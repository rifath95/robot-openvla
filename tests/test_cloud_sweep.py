"""Detached sweep safety: immutable inputs, timeout budgets, stop paths, archives."""
import io
import json
from pathlib import Path
import tempfile
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,MagicMock
from cloud.sweep_plan import timeout_budget,trial_specs,estimate_remaining,validate_saved
from cloud.sweep_storage import stage_weights
from cloud.cloud_sweep_session import extract_results,setup_script
from cloud import sweep_supervisor as supervisor
from cloud.cloud_session import bootstrap_script
from cloud.sweep_server_setup import setup_script as server_setup


def payload():
    plan=dict(task='color_transfer',checkpoints=[50,100,300,500],scene_seeds=list(range(1000,1010)),
        color_pairs=[['black','purple'],['green','yellow'],['orange','white']],steps=50,sampling=False,
        training_session='panda-train-test',elapsed_seconds=7200)
    specs=list(trial_specs(plan))
    rows=[dict(s,phase='Failed',completed_steps=20,contact=False,outcome='no_contact') for s in specs[:120]]
    return dict(plan=plan,trials=rows)


class CloudSweepTests(unittest.TestCase):
    def test_transfer_stage_acknowledges_without_downloading(self):
        from cloud.cloud_sweep_session import finish_transfer_stage
        with patch('cloud.cloud_sweep_session.subprocess.run') as run,patch('cloud.cloud_sweep_session.fetch') as fetch:
            self.assertEqual(finish_transfer_stage(['ssh'],'/volume/sweep',{'phase':'awaiting_sync','outcome':'completed'}),0)
            self.assertIn('sync_ack',run.call_args.args[0][-1]);fetch.assert_not_called()
            run.reset_mock()
            self.assertEqual(finish_transfer_stage(['ssh'],'/volume/sweep',{'phase':'finished','outcome':'failed'}),1)
            run.assert_not_called();fetch.assert_not_called()

    def test_launcher_default_finishes_without_large_mac_fetch(self):
        from cloud import cloud_sweep_session as session
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'outputs/scripted_test';folder.mkdir(parents=True)
            (folder/'results.json').write_text(json.dumps(payload()))
            def run(command,**kwargs):
                if kwargs.get('capture_output'):
                    return SimpleNamespace(stdout=json.dumps(dict(status=dict(phase='awaiting_sync',outcome='completed'),log='',offset=0)))
                return SimpleNamespace(returncode=0)
            with patch('sys.argv',['script','--ssh','ssh root@host','--resume',str(folder)]), \
                 patch.object(session,'PROJECT_ROOT',root),patch.object(session,'parse_ssh',return_value=('host',22,Path('key'))), \
                 patch.object(session,'ssh_arguments',return_value=['ssh']),patch.object(session.subprocess,'run',side_effect=run), \
                 patch.object(session,'fetch') as fetch:
                self.assertEqual(session.main(),0);fetch.assert_not_called()

    def test_timeout_rule_and_invalid_estimates(self):
        self.assertEqual(timeout_budget(1200),dict(estimated_seconds=1200,inner_seconds=4800,outer_seconds=8400))
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):timeout_budget(value)

    def test_scene_check_accepts_rounding_but_rejects_changed_positions(self):
        from cloud.sweep_plan import validate_scene_positions
        original=[{'red':[.4,.1],'blue':[.6,-.1]}]
        rounded=[{'red':[.4+1e-16,.1],'blue':[.6,-.1]}]
        validate_scene_positions(rounded,original)
        for other in ([{'red':[.400001,.1],'blue':[.6,-.1]}],
                      [{'red':[float('nan'),.1],'blue':[.6,-.1]}]):
            with self.assertRaises(ValueError):validate_scene_positions(other,original)

    def test_resume_retains_failed_outcomes_and_only_remaining_checkpoints(self):
        data=payload();specs=validate_saved(data);done={r['directory'] for r in data['trials']}
        remaining=[s for name,s in specs.items() if name not in done]
        self.assertEqual(len(remaining),120)
        self.assertEqual({s['checkpoint'] for s in remaining},{300,500})
        budget=estimate_remaining(data)
        self.assertEqual(budget['remaining_trials'],120)
        self.assertEqual(budget['inner_seconds']-budget['estimated_seconds'],3600)
        self.assertEqual(budget['outer_seconds']-budget['inner_seconds'],3600)

    def test_reject_altered_saved_colors(self):
        data=payload();data['trials'][0]['cube_colors']=['red','blue']
        with self.assertRaises(ValueError):validate_saved(data)

    def test_stage_once_and_never_change_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);base=p/'base';adapter=p/'adapter';base.mkdir();adapter.mkdir()
            (base/'config.json').write_text('{}');(base/'weights.bin').write_bytes(b'base')
            (adapter/'weights.bin').write_bytes(b'adapter')
            original={f:f.read_bytes() for f in p.rglob('*') if f.is_file()}
            with patch('cloud.sweep_storage.check_ready'),patch('cloud.sweep_storage.shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3)):
                stage_weights(base,{300:adapter},p/'staged')
                open_original=Path.open
                def read_metadata_only(path,*args,**kwargs):
                    if path.name!='staging.json':raise AssertionError('Weights should not be recopied')
                    return open_original(path,*args,**kwargs)
                with patch.object(Path,'open',new=read_metadata_only):
                    stage_weights(base,{300:adapter},p/'staged')
            for f,content in original.items():self.assertEqual(f.read_bytes(),content)
            self.assertEqual((p/'staged/base/weights.bin').read_bytes(),b'base')

    def test_archive_rejects_paths_before_replacing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);destination=p/'out';destination.mkdir();(destination/'results.json').write_text('original')
            archive=p/'bad.tar.gz'
            with tarfile.open(archive,'w:gz') as stream:
                item=tarfile.TarInfo('../escaped');item.size=1;stream.addfile(item,io.BytesIO(b'x'))
            with self.assertRaises(ValueError):extract_results(archive,destination)
            self.assertEqual((destination/'results.json').read_text(),'original')

    def test_fetch_merges_new_results_without_erasing_old_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);destination=p/'out';destination.mkdir();(destination/'old.png').write_bytes(b'old')
            archive=p/'good.tar.gz'
            with tarfile.open(archive,'w:gz') as stream:
                for name,content in [('results.json',json.dumps(payload()).encode()),('new.png',b'new')]:
                    item=tarfile.TarInfo(name);item.size=len(content);stream.addfile(item,io.BytesIO(content))
            extract_results(archive,destination)
            self.assertEqual((destination/'old.png').read_bytes(),b'old')
            self.assertEqual((destination/'new.png').read_bytes(),b'new')

    def test_fixed_checkout_and_in_memory_setup_never_merge_to_disk(self):
        self.assertNotIn('git pull',bootstrap_script(60,update_repo=False))
        script=server_setup('/root/stage/base','/root/stage/checkpoint_000300',60)
        self.assertIn('--adapter',script);self.assertNotIn('training.merge_adapter',script)
        script=setup_script('scripted_test')
        self.assertIn('sparse-checkout set franka_emika_panda',script)
        self.assertIn('fetch --depth 1 origin',script)
        self.assertIn('git clone --shared',script)

    def test_disconnected_client_watchdog_stops_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            with patch.object(supervisor.subprocess,'check_output',return_value='commit'), \
                patch.object(supervisor.time,'monotonic',side_effect=[0,181,181]), \
                patch.object(supervisor,'stop_pod',return_value=0) as stop:
                supervisor.supervise(directory,'pod','key',mode='client_watchdog',estimated_seconds=1200)
            self.assertEqual(stop.call_count,1)
            status=json.loads((directory/'supervisor_status.json').read_text())
            self.assertEqual(status['outcome'],'mac_disconnected');self.assertEqual(status['stop_state'],'accepted')

    def test_cloud_worker_failure_stops_even_without_mac(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);worker=MagicMock();worker.poll.return_value=1;worker.returncode=1
            with patch.object(supervisor.subprocess,'check_output',return_value='commit'), \
                patch.object(supervisor.subprocess,'Popen',return_value=worker), \
                patch.object(supervisor,'stop_pod',return_value=0) as stop:
                supervisor.supervise(directory,'pod','key',mode='cloud',estimated_seconds=1200,sim_python='/sim/python')
            stop.assert_called_once()
            self.assertEqual(json.loads((directory/'supervisor_status.json').read_text())['outcome'],'failed')

    def test_cloud_completion_without_watcher_stops_promptly(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);worker=MagicMock();worker.poll.return_value=0;worker.returncode=0
            with patch.object(supervisor.subprocess,'check_output',return_value='commit'), \
                patch.object(supervisor.subprocess,'Popen',return_value=worker), \
                patch.object(supervisor,'stop_pod',return_value=0) as stop:
                self.assertEqual(supervisor.supervise(directory,'pod','key',mode='cloud',estimated_seconds=1200,sim_python='/sim/python'),0)
            stop.assert_called_once()

    def test_outer_limit_still_requests_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            with patch.object(supervisor.subprocess,'check_output',return_value='commit'), \
                patch.object(supervisor.time,'monotonic',side_effect=[0,8401,8401]), \
                patch.object(supervisor,'stop_pod',return_value=0) as stop:
                supervisor.supervise(directory,'pod','key',mode='client_watchdog',estimated_seconds=1200)
            stop.assert_called_once()
            self.assertEqual(json.loads((directory/'supervisor_status.json').read_text())['outcome'],'outer_limit')

    def test_inner_limit_kills_worker_before_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);worker=MagicMock();worker.poll.return_value=None
            with patch.object(supervisor.subprocess,'check_output',return_value='commit'), \
                patch.object(supervisor.subprocess,'Popen',return_value=worker), \
                patch.object(supervisor.time,'monotonic',side_effect=[0,4801,4801]), \
                patch.object(supervisor,'kill_benchmark_tree') as kill,patch.object(supervisor,'stop_pod',return_value=0) as stop:
                supervisor.supervise(directory,'pod','key',mode='cloud',estimated_seconds=1200,sim_python='/sim/python')
            kill.assert_called_once_with(worker);stop.assert_called_once()
            self.assertEqual(json.loads((directory/'supervisor_status.json').read_text())['outcome'],'inner_limit')

if __name__=='__main__':unittest.main()
