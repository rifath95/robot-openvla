"""Backups verify pinned files one at a time and keep volume checkpoints on errors."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from cloud.chain_archive import verify_files,backup_training,recover_to_global

class ChainArchiveTests(unittest.TestCase):
    def test_verification_pins_revision_and_releases_each_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files=[dict(path=f'part{i}.bin',bytes=3,sha256=hashlib.sha256(b'abc').hexdigest()) for i in range(2)]
            caches=[]
            def download(**kwargs):
                self.assertEqual(kwargs['revision'],'commit-123');self.assertEqual(kwargs['token'],'secret')
                if caches:self.assertFalse(caches[-1].exists())
                cache=Path(kwargs['cache_dir']);caches.append(cache);p=cache/'file';p.write_bytes(b'abc');return str(p)
            with patch('huggingface_hub.hf_hub_download',side_effect=download):verify_files('owner/repo','dataset','commit-123','sweep',files,'secret',root)
            self.assertTrue(all(not c.exists() for c in caches))
            def corrupt(**kwargs):p=Path(kwargs['cache_dir'])/'file';p.write_bytes(b'bad');return str(p)
            with patch('huggingface_hub.hf_hub_download',side_effect=corrupt),self.assertRaises(ValueError):verify_files('repo','dataset','revision','sweep',files,'secret',root)

    def test_recovery_packs_partial_runs_and_preserves_recordings(self):
        from cloud.chain_archive import prepare_recovery_archive
        import tarfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'scripted_partial';source.mkdir();local=root/'local';local.mkdir()
            (source/'results.json').write_text('{"trials":[]}')
            (source/'state.npz').write_bytes(b'preserved-state')
            with patch('cloud.chain_archive.shutil.disk_usage',return_value=SimpleNamespace(free=3*1024**3)):
                packed=prepare_recovery_archive(source,local)
            with tarfile.open(packed/'recordings.tar.gz') as archive:
                self.assertEqual(archive.extractfile('scripted_partial/state.npz').read(),b'preserved-state')
            self.assertTrue((source/'state.npz').exists())
            recovered=recover_to_global(packed,root/'global')
            self.assertTrue(recovered['verified']);self.assertEqual(len(recovered['files']),2)

    def test_full_1000_trial_archive_restores_with_new_plan_validation(self):
        from common.paths import PROJECT_ROOT
        from cloud.bowl_chain_plan import make_plan,trial_specs
        from cloud.scripted_touch_session import write_reports
        from cloud.archive_sweep import package
        from data_tools.restore_sweep_archive import restore
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'outputs') as tmp:
            root=Path(tmp);source=root/'scripted_chain_test';source.mkdir();staging=root/'archive';staging.mkdir()
            plan=make_plan('panda-train-test');rows=[]
            for spec in trial_specs(plan):
                row={k:v for k,v in spec.items() if k!='layout'};row.update(phase='Completed',outcome='no_lift',placement_success=False)
                trial=source/spec['directory'];trial.mkdir();(trial/'summary.json').write_text('{}');rows.append(row)
            write_reports(source,plan,rows)
            package(source,staging,dict(plan=plan,trials=rows))
            restored=restore(staging,root/'restored')
            self.assertEqual(len(json.loads((restored/'results.json').read_text())['trials']),1000)
            self.assertEqual(len(list(restored.glob('*/summary.json'))),1000)
            self.assertIn('▶ Replay',(restored/'REPORT.md').read_text())

    def test_checkpoints_remain_after_both_verified_and_failed_backups(self):
        for fail in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);session=root/'session';local=root/'local';local.mkdir()
                for cp in (0,500,1000,2000,3000):
                    p=session/f'runs/training/checkpoint_{cp:06d}';p.mkdir(parents=True);(p/'adapter_model.safetensors').write_bytes(b'weights')
                api=Mock();api.upload_folder.return_value=SimpleNamespace(oid='pinned')
                with patch('huggingface_hub.HfApi',return_value=api),patch('cloud.checkpoint_ready.check_ready'),patch('cloud.chain_archive.verify_files',side_effect=ValueError('bad checksum') if fail else None):
                    if fail:
                        with self.assertRaises(ValueError):backup_training(session,'owner/repo','secret',local)
                        self.assertFalse((session/'checkpoint_hf_receipt.json').exists())
                    else:self.assertTrue(backup_training(session,'owner/repo','secret',local)['volume_checkpoints_retained'])
                for cp in (0,500,1000,2000,3000):self.assertEqual((session/f'runs/training/checkpoint_{cp:06d}/adapter_model.safetensors').read_bytes(),b'weights')
                self.assertNotIn('secret',json.dumps([p.name for p in session.rglob('*')]))

if __name__=='__main__':unittest.main()
