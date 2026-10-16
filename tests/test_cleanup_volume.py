"""Fixed cleanup scope, verified archival gate and deletion progress; no SSH."""
import json
import errno
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from cloud import cleanup_volume as c


class CleanupTests(unittest.TestCase):
    def test_nonempty_directory_is_rescanned_before_retry(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(c.time,'sleep'):
            root=Path(tmp);target=root/'cache';target.mkdir();(target/'first').write_bytes(b'one')
            real=Path.rmdir;calls=[]
            def rmdir(path):
                if path==target and not calls:
                    calls.append(path);(target/'late-entry').write_bytes(b'two')
                    raise OSError(errno.ENOTEMPTY,'stale or late entry')
                return real(path)
            counts=dict(files=0,directories=0,bytes=0,last_log=time.monotonic())
            with patch.object(Path,'rmdir',rmdir):c.remove_with_progress(target,counts)
            self.assertFalse(target.exists());self.assertEqual(counts['files'],2)
            self.assertEqual(counts['directories'],1)

    def test_persistent_filesystem_error_leaves_target_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'ROOT',Path(tmp)), \
                patch.object(c,'SELECTED',{}),patch.object(c,'ALREADY_BACKED',()), \
                patch.object(c,'PILOTS',()),patch.object(c,'DATASETS',('busy','good')), \
                patch('huggingface_hub.HfApi') as api,patch.object(c.os,'nice'):
            root=Path(tmp);bad=root/'busy';good=root/'good';bad.mkdir();good.mkdir()
            (bad/'entry').write_text('keep');(good/'entry').write_text('delete')
            reports=root/'reports/cleanup';reports.mkdir(parents=True)
            api.return_value.repo_info.return_value.private=True
            remove=c.remove_with_progress
            def removal(path,counts):
                if path==bad:raise OSError(errno.EBUSY,'busy network cache')
                return remove(path,counts)
            with patch.object(c,'remove_with_progress',side_effect=removal):
                self.assertEqual(c.worker(dict(reports=str(reports),token='test')),0)
            self.assertTrue(bad.exists());self.assertFalse(good.exists())
            result=json.loads((reports/'cleanup_complete.json').read_text())
            self.assertFalse(result['complete']);self.assertEqual(len(result['remaining_targets']),1)
            self.assertNotIn(str(bad),result['removed_targets'])

    def test_targets_exclude_current_training_and_all_other_sweeps(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c,'ROOT',Path(tmp)):
            root=Path(tmp)
            for session in (*c.SELECTED,*c.ALREADY_BACKED,*c.PILOTS,'panda-train-color-bowl-20261016-084536-c17080be'):
                (root/'sessions'/session/'runs/training/checkpoint_001000').mkdir(parents=True)
            items=c.targets()
            self.assertFalse(any('20261016-084536' in str(p) for p in items))
            self.assertEqual([p for p in items if p.parent==root/'scripted-evaluations'],
                             [root/'scripted-evaluations'/c.OLD_SWEEP])
            self.assertTrue(all(p.is_relative_to(root) for p in items))

    def test_archive_failure_prevents_all_deletion(self):
        with patch('huggingface_hub.HfApi') as api, patch.object(c.os,'nice'), \
                patch.object(c,'archive_selected',side_effect=ValueError('verification failed')), \
                patch.object(c,'remove_with_progress') as remove:
            api.return_value.repo_info.return_value.private=True
            with self.assertRaisesRegex(ValueError,'verification failed'):
                c.worker(dict(token='test',inner_deadline=time.monotonic()+100))
            remove.assert_not_called()

    def test_completed_cleanup_preserves_reports_and_current_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c,'ROOT',Path(tmp)), \
                patch('huggingface_hub.HfApi') as api,patch.object(c.os,'nice'), \
                patch.object(c,'archive_selected') as archive:
            root=Path(tmp);api.return_value.repo_info.return_value.private=True
            for session in (*c.SELECTED,*c.ALREADY_BACKED,*c.PILOTS):
                folder=root/'sessions'/session;cp=folder/'runs/training/checkpoint_001000'
                cp.mkdir(parents=True);(cp/'adapter.bin').write_bytes(b'weights')
                (cp/'training_config.json').write_text('{}');(folder/'report.md').write_text('keep')
                if session in c.ALREADY_BACKED:
                    (folder/'checkpoint_hf_receipt.json').write_text(json.dumps(dict(verified=True,repo_id=c.REPO,path_in_repo='sessions/'+session)))
            current=root/'sessions/panda-train-color-bowl-20261016-084536-c17080be/runs/training/checkpoint_012000'
            current.mkdir(parents=True);(current/'state.pt').write_bytes(b'keep')
            for name in c.DATASETS:
                p=root/name;p.mkdir();(p/'record').write_bytes(b'dataset')
                (root/(name+'_hf_source.json')).write_text('{}')
            old=root/'scripted-evaluations'/c.OLD_SWEEP;old.mkdir(parents=True)
            (old/'REPORT.md').write_text('saved report')
            reports=root/'reports/volume-cleanup-20261016';reports.mkdir(parents=True)
            cfg=dict(reports=str(reports),local=str(root/'local'),token='test',inner_deadline=time.monotonic()+100)
            self.assertEqual(c.worker(cfg),0)
            self.assertEqual(archive.call_count,3)
            self.assertTrue((current/'state.pt').exists())
            for session in (*c.SELECTED,*c.ALREADY_BACKED,*c.PILOTS):
                self.assertTrue((root/'sessions'/session/'report.md').exists())
                self.assertTrue((root/'sessions'/session/'checkpoint-metadata/checkpoint_001000.json').exists())
            self.assertTrue((root/'reports/completed-sweeps'/c.OLD_SWEEP/'REPORT.md').exists())
            self.assertTrue(json.loads((reports/'cleanup_complete.json').read_text())['complete'])
            for name in c.DATASETS:
                self.assertFalse((root/name).exists());self.assertTrue((root/(name+'_hf_source.json')).exists())

    def test_existing_remote_backup_is_verified_before_receipt(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c,'ROOT',Path(tmp)), \
                patch('cloud.checkpoint_ready.check_ready',side_effect=Path), \
                patch('cloud.chain_archive.verify_files',side_effect=ValueError('mismatch')):
            root=Path(tmp);session='selected'
            cp=root/'sessions'/session/'runs/training/checkpoint_001000';cp.mkdir(parents=True)
            (cp/'adapter_model.safetensors').write_bytes(b'weight')
            reports=root/'receipts';reports.mkdir();local=root/'local';local.mkdir()
            api=Mock();api.model_info.return_value=SimpleNamespace(sha='pinned')
            api.list_repo_files.return_value=['sessions/selected/runs/training/checkpoint_001000/adapter_model.safetensors']
            with self.assertRaisesRegex(ValueError,'mismatch'):
                c.archive_selected(session,(1000,),dict(reports=str(reports),local=str(local),token='test'),api)
            api.upload_folder.assert_not_called()
            self.assertFalse((reports/'selected.json').exists())
            self.assertTrue(cp.exists())

    def test_missing_checkpoint_protects_entire_session_but_allows_other_targets(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'ROOT',Path(tmp)), \
                patch.object(c,'SELECTED',{'old':(100,)}),patch.object(c,'ALREADY_BACKED',()), \
                patch.object(c,'PILOTS',()),patch.object(c,'DATASETS',('obsolete_dataset',)), \
                patch('huggingface_hub.HfApi') as api,patch.object(c.os,'nice'):
            root=Path(tmp);cp=root/'sessions/old/runs/training/checkpoint_000500'
            cp.mkdir(parents=True);(cp/'adapter_model.safetensors').write_bytes(b'keep remaining weights')
            obsolete=root/'obsolete_dataset';obsolete.mkdir();(obsolete/'record').write_bytes(b'delete')
            reports=root/'reports/cleanup';reports.mkdir(parents=True)
            api.return_value.repo_info.return_value.private=True
            cfg=dict(reports=str(reports),local=str(root/'local'),token='test')
            self.assertEqual(c.worker(cfg),0)
            self.assertTrue((cp/'adapter_model.safetensors').exists())
            self.assertFalse(obsolete.exists())
            result=json.loads((reports/'cleanup_complete.json').read_text())
            self.assertFalse(result['complete']);self.assertEqual(result['protected_sessions'],['old'])
            api.return_value.upload_folder.assert_not_called()

    def test_supervisor_waits_without_cleanup_timeout(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(c.subprocess,'Popen') as launch:
            root=Path(tmp);(root/'cleanup_complete.json').write_text('{"complete": true}')
            launch.return_value.wait.return_value=0
            self.assertEqual(c.supervise(dict(reports=str(root))),0)
            launch.return_value.wait.assert_called_once_with()
            status=json.loads((root/'status.json').read_text())
            self.assertIsNone(status['timeout']);self.assertFalse(status['pod_stop_requested'])

    def test_symlink_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);keep=root/'keep';keep.write_text('safe')
            link=root/'link';link.symlink_to(keep)
            with self.assertRaisesRegex(ValueError,'symlink'):
                c.remove_with_progress(link,{})
            self.assertEqual(keep.read_text(),'safe')


if __name__=='__main__':unittest.main()
