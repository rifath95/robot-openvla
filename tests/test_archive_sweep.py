"""Archive completeness, remote checksum verification and verified/failure stop paths."""
import copy
import json
from pathlib import Path
import tarfile
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch,MagicMock
from cloud import archive_sweep as arch
from cloud.sweep_plan import trial_specs


def fixture(root):
    from simulation.multi_color_scene import positions
    plan=dict(task='multi_color_touch',sampling=False,steps=50,training_session='panda-train-test',
        checkpoints=[50],scene_seeds=[2000],cube_counts=[4],color_sets=[dict(name='seen',colors=['red','blue','green','yellow'])],
        training_colors=['red','blue','green','yellow'],multi_cube_positions={'4':[positions(2000,4)]})
    rows=[dict(s,outcome='correct_cube',phase='Completed') for s in trial_specs(plan)]
    source=root/'source';source.mkdir();payload=dict(plan=plan,trials=rows,launcher_error=None)
    (source/'results.json').write_text(json.dumps(payload));(source/'REPORT.md').write_text('Report')
    (source/'supervisor_status.json').write_text(json.dumps(dict(outcome='completed',exit_code=0)))
    for r in rows:
        trial=source/r['directory'];trial.mkdir();(trial/'summary.json').write_text('{}')
        step=trial/'step_001';step.mkdir();(step/'before.png').write_bytes(b'pixels');(trial/'replay.py').write_text('pass\n')
    staging=root/'package';staging.mkdir()
    return source,staging,payload


class ArchiveTests(unittest.TestCase):
    def test_complete_source_and_reject_partial_or_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,_,p=fixture(Path(tmp));self.assertEqual(len(arch.completed_payload(source)['trials']),4)
            (source/'supervisor_status.json').write_text(json.dumps(dict(outcome='failed',exit_code=1)))
            with self.assertRaises(ValueError):arch.completed_payload(source)
            (source/'supervisor_status.json').write_text(json.dumps(dict(outcome='completed',exit_code=0)))
            p['trials'].pop();(source/'results.json').write_text(json.dumps(p))
            with self.assertRaises(ValueError):arch.completed_payload(source)

    def test_package_preserves_every_trial_and_no_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,staging,p=fixture(Path(tmp));before=(source/'results.json').read_bytes()
            files=arch.package(source,staging,p);parts=list((staging/'archives').iterdir());self.assertEqual(len(parts),1)
            with tarfile.open(parts[0]) as archive:
                for r in p['trials']:self.assertIn(r['directory']+'/step_001/before.png',archive.getnames())
            self.assertEqual((source/'results.json').read_bytes(),before)
            for f in files:self.assertEqual(arch.digest(staging/f['path']),f['sha256'])

    def test_packed_parts_reused_across_container_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,staging,p=fixture(root);cache=root/'volume-cache'
            first=arch.package(source,staging,p,part_cache=cache)
            shutil=__import__('shutil');shutil.rmtree(staging);staging.mkdir()
            with patch.object(arch,'add_checked',side_effect=AssertionError('Should reuse existing archive')):
                second=arch.package(source,staging,p,part_cache=cache)
            self.assertEqual(first,second)
            changed=json.loads((source/'results.json').read_text());changed['plan']['elapsed_seconds']=1
            (source/'results.json').write_text(json.dumps(changed))
            with patch.object(arch,'add_checked',side_effect=ValueError('Repack changed source')):
                with self.assertRaisesRegex(ValueError,'Repack'):arch.package(source,staging,changed,part_cache=cache)

    def test_pipeline_uploads_before_packing_finishes_and_skips_existing_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,staging,p=fixture(root);receipt=root/'receipt';receipt.mkdir()
            api=MagicMock();api.repo_info.return_value=SimpleNamespace(private=True);api.get_paths_info.return_value=[]
            uploaded=threading.Event()
            api.upload_file.side_effect=lambda **kwargs:uploaded.set()
            original=arch.package
            def packing(*args,**kwargs):
                on_part=kwargs['on_part']
                def ready(path,entry):
                    on_part(path,entry)
                    self.assertTrue(uploaded.wait(2),'Upload must start before final reports/manifest')
                    self.assertFalse((staging/'manifest.json').exists())
                kwargs['on_part']=ready
                return original(*args,**kwargs)
            with patch('huggingface_hub.HfApi',return_value=api),patch.object(arch,'package',side_effect=packing), \
                 patch.object(arch,'upload_verify',return_value={'verified':True}) as verify:
                arch.pipeline_archive(source,staging,p,'user/private','secret',receipt)
            api.upload_file.assert_called_once();self.assertTrue(verify.call_args.kwargs['archives_uploaded'])
            manifest=json.loads((staging/'manifest.json').read_text());entry=manifest['files'][0]
            api.get_paths_info.return_value=[SimpleNamespace(size=entry['bytes'],lfs={'sha256':entry['sha256']})]
            api.upload_file.reset_mock()
            with patch('huggingface_hub.HfApi',return_value=api),patch.object(arch,'upload_verify',return_value={'verified':True}):
                arch.pipeline_archive(source,staging,p,'user/private','secret',receipt)
            api.upload_file.assert_not_called()

    def test_failed_background_upload_retains_reusable_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,staging,p=fixture(root);receipt=root/'receipt';receipt.mkdir()
            api=MagicMock();api.repo_info.return_value=SimpleNamespace(private=True);api.get_paths_info.return_value=[]
            api.upload_file.side_effect=RuntimeError('Upload interrupted')
            with patch('huggingface_hub.HfApi',return_value=api),patch.object(arch,'upload_verify') as verify:
                with self.assertRaises(RuntimeError):arch.pipeline_archive(source,staging,p,'user/private','secret',receipt)
            verify.assert_not_called();self.assertEqual(len(list((receipt/'packed-parts').glob('*.tar.gz'))),1)

    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,staging,p=fixture(Path(tmp));(source/p['trials'][0]['directory']/'unsafe').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):arch.package(source,staging,p)

    def test_verify_downloads_exact_commit_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,staging,p=fixture(root);arch.package(source,staging,p)
            api=MagicMock();api.repo_info.return_value=SimpleNamespace(private=True)
            api.upload_folder.return_value=SimpleNamespace(oid='verifiedcommit')
            def download(**args):
                self.assertEqual(args['revision'],'verifiedcommit');self.assertEqual(args['token'],'secret')
                target=root/'cache'/args['filename'];target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((staging/args['filename'].split('/',1)[1]).read_bytes());return str(target)
            with patch('huggingface_hub.HfApi',return_value=api),patch('huggingface_hub.hf_hub_download',side_effect=download):
                result=arch.upload_verify(staging,'user/private','sweep','secret',root/'cache')
                self.assertTrue(result['verified']);self.assertEqual(result['revision'],'verifiedcommit')
            corrupt=root/'bad';corrupt.write_bytes(b'bad')
            with patch('huggingface_hub.HfApi',return_value=api),patch('huggingface_hub.hf_hub_download',return_value=str(corrupt)):
                with self.assertRaises(ValueError):arch.upload_verify(staging,'user/private','sweep','secret',root/'cache')
            api.repo_info.return_value.private=False
            with patch('huggingface_hub.HfApi',return_value=api):
                with self.assertRaises(ValueError):arch.upload_verify(staging,'user/public','sweep','secret',root/'cache')

    def test_supervisor_stops_on_success_or_failure_without_exposing_credentials(self):
        for code,outcome in ((0,'verified'),(1,'failed')):
            with tempfile.TemporaryDirectory() as tmp:
                config=dict(receipt=tmp,estimated_seconds=60,pod_id='pod',pod_key='podsecret',token='hfsecret')
                proc=MagicMock();proc.poll.return_value=code;proc.returncode=code
                with patch.object(arch.subprocess,'Popen',return_value=proc),patch.object(arch,'stop_pod',return_value=0) as stop:
                    self.assertEqual(arch.supervise(config),code)
                stop.assert_called_once();self.assertIn(outcome,stop.call_args.kwargs['reason'])
                status=(Path(tmp)/'supervisor_status.json').read_text()
                self.assertNotIn('secret',status)
                worker=json.loads(proc.stdin.write.call_args.args[0]);self.assertNotIn('pod_key',worker)

    def test_supervisor_timeout_kills_job_then_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=dict(receipt=tmp,estimated_seconds=60,pod_id='pod',pod_key='key')
            proc=MagicMock();proc.poll.return_value=None
            with patch.object(arch.subprocess,'Popen',return_value=proc),patch.object(arch.time,'monotonic',side_effect=[0,3661,3662]), \
                 patch('training.benchmark_supervisor.kill_benchmark_tree') as kill,patch.object(arch,'stop_pod',return_value=0) as stop:
                self.assertEqual(arch.supervise(config),1)
            kill.assert_called_once_with(proc);stop.assert_called_once()
            self.assertEqual(json.loads((Path(tmp)/'supervisor_status.json').read_text())['outcome'],'inner_limit')

if __name__=='__main__':unittest.main()
