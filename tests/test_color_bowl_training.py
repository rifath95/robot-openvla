"""Prepared training pins complete uploads and preserves all artifacts on failures."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from cloud.color_bowl_training import resolve_upload,worker
from common.paths import PROJECT_ROOT


class ColorBowlTrainingTests(unittest.TestCase):
    def test_profile_counts_steps_buffers_and_fixed_batch(self):
        cfg=json.loads((PROJECT_ROOT/'training/color_place_bowl_600_6000.json').read_text())
        self.assertEqual(cfg['expected_split_counts']['train'],dict(episodes=600,transitions=43344))
        self.assertEqual(cfg['checkpoint_updates'],[0,1000,2000,4000,6000]);self.assertEqual(cfg['batch_size'],8)
        self.assertFalse(cfg['validation_enabled']);self.assertEqual(cfg['batch_policy'],'config')
        self.assertEqual(cfg['max_runtime_seconds'],(4+3)*3600);self.assertEqual(cfg['hard_limit_seconds'],(4+3+5)*3600)

    def test_incomplete_upload_rejected_before_reading_checksums(self):
        api=Mock();api.repo_info.return_value=SimpleNamespace(sha='commit',siblings=[])
        profile=dict(dataset_repo='owner/data',file_sizes={'manifest.json':10},checksums_sha256='unused')
        with patch('huggingface_hub.HfApi',return_value=api),patch('huggingface_hub.hf_hub_download') as download,self.assertRaisesRegex(ValueError,'incomplete'):
            resolve_upload(profile,'secret')
        download.assert_not_called()

    def test_checksum_pinned_to_complete_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'SHA256SUMS';f.write_bytes(b'checksums')
            profile=dict(dataset_repo='owner/data',file_sizes={'manifest.json':10},checksums_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            api=Mock();api.repo_info.return_value=SimpleNamespace(sha='commit-123',siblings=[SimpleNamespace(rfilename='manifest.json',size=10)])
            with patch('huggingface_hub.HfApi',return_value=api),patch('huggingface_hub.hf_hub_download',return_value=str(f)) as dl:
                self.assertEqual(resolve_upload(profile,'secret'),'commit-123');self.assertEqual(dl.call_args.kwargs['revision'],'commit-123')
            f.write_bytes(b'corrupt')
            with patch('huggingface_hub.HfApi',return_value=api),patch('huggingface_hub.hf_hub_download',return_value=str(f)),self.assertRaisesRegex(ValueError,'checksums'):
                resolve_upload(profile,'secret')

    def test_worker_orders_download_training_and_verified_backup_and_keeps_secrets_out(self):
        for fail in (False,True):
            with self.subTest(training_failure=fail),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);session=root/'session';local=root/'local';session.mkdir();local.mkdir()
                config=dict(session=str(session),local=str(local),training_config='training/color_place_bowl_600_6000.json',hf_token='never-log-secret',checkpoint_repo='owner/checkpoints',dataset_repo='owner/data',dataset_revision='revision',checksums_sha256='sha',dataset_bundle='training_color_place_bowl_v1_600',dataset=str(local/'dataset'),base=str(local/'base'),base_volume='/workspace/openvla-7b',training_run=str(session/'runs/training'),started_at_utc='2026-10-15T00:00:00+00:00',code_commit='code',timeout_budget={})
                order=[];api=Mock();api.repo_info.return_value.private=True;api.upload_folder.return_value.oid='commit'
                def train(*args):
                    order.append('training');self.assertNotIn('HF_TOKEN',args[2]);self.assertNotIn('RUNPOD_API_KEY',args[2])
                    if fail:raise RuntimeError('training failed')
                def backup(*args):
                    order.append('backup');self.assertEqual(args[4],[0,1000,2000,4000,6000])
                    (session/'checkpoint_hf_receipt.json').write_text('{"verified":true}')
                completion=dict(training_complete=True,last_update=6000,training_update_seconds=12000,checkpoint_seconds=10)
                with patch('huggingface_hub.HfApi',return_value=api),patch('shutil.disk_usage',return_value=SimpleNamespace(free=100*1024**3)),patch('cloud.bowl_chain_worker.download_dataset',side_effect=lambda _:order.append('download')),patch('cloud.bowl_chain_worker.stage_base',side_effect=lambda *a:order.append('base') or {}),patch('cloud.color_bowl_training.train',side_effect=train),patch('cloud.bowl_chain_worker.verify_training',return_value=completion),patch('cloud.chain_archive.backup_training',side_effect=backup),patch('cloud.chain_archive.verify_files'):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'training failed'):worker(config)
                        self.assertNotIn('backup',order)
                    else:
                        self.assertEqual(worker(config),0);self.assertEqual(order,['download','base','training','backup'])
                        self.assertTrue((session/'training_completion.json').exists())
                        self.assertNotIn('never-log-secret',''.join(p.read_text() for p in session.glob('*.json')))

class PreparedDeliveryTests(unittest.TestCase):
    def test_local_launcher_sends_token_only_through_stdin_and_fetches_before_ack(self):
        import subprocess
        from cloud import training_cloud_session as cli
        with tempfile.TemporaryDirectory() as tmp:
            destination=Path(tmp);(destination/'status.json').write_text('{"training_complete":true,"job_complete":true}')
            calls=[]
            def run(command,**kwargs):
                self.assertNotIn('PRIVATE-TOKEN',str(command))
                if 'bash -s' in command:
                    calls.append('setup');self.assertNotIn('PRIVATE-TOKEN',kwargs['input'])
                elif '--token-stdin' in command[-1]:
                    calls.append('launch');self.assertEqual(json.loads(kwargs['input']),{'hf_token':'PRIVATE-TOKEN'})
                    return subprocess.CompletedProcess(command,0,stdout='Cloud session started: panda-train-color-bowl-test\n')
                else:calls.append('ack' if '--ack-download' in command[-1] else 'watch')
                return subprocess.CompletedProcess(command,0)
            def fetch(*args):calls.append('download');return destination
            with patch('sys.argv',['session','--ssh','fake','--prepared-color-bowl']),patch.object(cli,'parse_ssh',return_value=('root@fake',22,Path('/tmp/key'))),patch.object(cli.getpass,'getpass',return_value='PRIVATE-TOKEN'),patch.object(cli.subprocess,'run',side_effect=run),patch.object(cli,'fetch_reports',side_effect=fetch):cli.main()
            self.assertEqual(calls,['setup','launch','watch','download','ack'])

    def test_supervisor_publishes_completion_waits_then_stops_and_uses_training_only_worker(self):
        from cloud.bowl_chain import supervise
        for verified in (True,False):
            with self.subTest(hf_verified=verified),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);session=root/'session';local=root/'local';session.mkdir();local.mkdir()
                (session/'training_completion.json').write_text('{"training_complete":true,"last_update":6000}')
                (session/'checkpoint_hf_receipt.json').write_text(json.dumps(dict(verified=verified)))
                config=dict(session=str(session),local=str(local),sweep=str(local/'no-sweep'),pod_id='pod',pod_key='SECRET',hf_token='HFSECRET',code_commit='code',training_only=True,timeout_budget=dict(inner_seconds=10,outer_seconds=100),download_grace_seconds=120)
                guard=Mock(pid=100,stdin=Mock());child=Mock(pid=200,stdin=Mock());child.wait.return_value=0;child.poll.return_value=0;order=[]
                def wait(directory,status,*args):
                    order.append('wait');self.assertEqual(status['job_complete'],verified);self.assertTrue(status['training_complete'])
                with patch('cloud.bowl_chain.subprocess.Popen',side_effect=[guard,child]) as spawn,patch('training.detached_run.wait_for_download',side_effect=wait),patch('cloud.bowl_chain.stop_request',side_effect=lambda *a:order.append('stop') or True):
                    self.assertEqual(supervise(config),0 if verified else 1)
                    self.assertEqual(spawn.call_args_list[1].args[0][-1],'cloud.color_bowl_training')
                self.assertEqual(order,['wait','stop'])
                status=json.loads((session/'status.json').read_text());self.assertEqual(status['job_complete'],verified)
                self.assertNotIn('HFSECRET',(session/'chain_config.json').read_text())

if __name__=='__main__':unittest.main()
