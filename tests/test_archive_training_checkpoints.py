"""Checkpoint deletion must follow exact uploaded-byte verification; shutdown is bounded."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,MagicMock
from types import SimpleNamespace
from cloud import archive_training_checkpoints as arch

class CheckpointArchiveTests(unittest.TestCase):
    def test_verified_copy_deletes_weights_and_keeps_reports(self):
        for corrupt in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/'volume/sessions/panda-train-test';cp=source/'runs/training/checkpoint_000300';cp.mkdir(parents=True)
                (cp/'adapter_model.safetensors').write_bytes(b'weights');(source/'console.log').write_text('training report')
                local=root/'container';local.mkdir();receipts=root/'receipts';receipts.mkdir()
                api=MagicMock();api.upload_folder.return_value=SimpleNamespace(oid='verified-commit')
                def download(repo,path,**kwargs):
                    self.assertEqual(kwargs['revision'],'verified-commit')
                    if corrupt:
                        bad=local/'bad';bad.write_bytes(b'bad');return str(bad)
                    return str(local/'panda-train-test'/path.removeprefix('sessions/panda-train-test/'))
                with patch.object(arch,'ROOT',root/'volume'),patch('cloud.checkpoint_ready.check_ready'),patch('huggingface_hub.hf_hub_download',side_effect=download):
                    if corrupt:
                        with self.assertRaises(ValueError):arch.backup_session('panda-train-test',api,'user/private','secret',receipts,local)
                        self.assertTrue(cp.exists());self.assertFalse((receipts/'panda-train-test/receipt.json').exists())
                    else:
                        arch.backup_session('panda-train-test',api,'user/private','secret',receipts,local)
                        self.assertFalse(cp.exists());self.assertTrue((source/'console.log').exists())
                        self.assertTrue(json.loads((receipts/'panda-train-test/receipt.json').read_text())['verified'])

    def test_supervisor_stops_on_success_and_failure(self):
        for code in (0,1):
            with tempfile.TemporaryDirectory() as tmp:
                proc=MagicMock();proc.wait.return_value=code;proc.poll.return_value=code
                config=dict(receipt=tmp,estimated_seconds=60,pod_id='pod',pod_key='secret',token='hf-secret')
                with patch.object(arch.subprocess,'Popen',return_value=proc),patch.object(arch,'stop_pod') as stop:
                    self.assertEqual(arch.supervise(config),code)
                stop.assert_called_once()
                payload=json.loads(proc.stdin.write.call_args.args[0]);self.assertNotIn('pod_key',payload)

if __name__=='__main__':unittest.main()
