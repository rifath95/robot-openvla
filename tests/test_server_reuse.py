"""Checkpoint identity and live-process checks without cloud access."""
import json
from pathlib import Path
import tempfile
import unittest

from cloud.server_reuse import requested_identity, process_identity, reusable


class ReuseTests(unittest.TestCase):
    def prepare(self,root):
        base=root/'base';base.mkdir()
        (base/'config.json').write_text('{}')
        (base/'export_complete.json').write_text('{"complete":true}')
        adapter=root/'adapter';adapter.mkdir()
        (adapter/'adapter_model.safetensors').write_bytes(b'test weights')
        (adapter/'adapter_config.json').write_text('{}')
        (adapter/'dataset_statistics.json').write_text('{}')
        return base,adapter

    def test_changed_checkpoint_and_base_cannot_reuse_old_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base,adapter=self.prepare(root)
            original=requested_identity(base,adapter)
            self.assertEqual(original,requested_identity(base,adapter))
            self.assertNotEqual(original,requested_identity(base))
            (adapter/'adapter_model.safetensors').write_bytes(b'different weights')
            self.assertNotEqual(original,requested_identity(base,adapter))
            original=requested_identity(base,adapter)
            (adapter/'dataset_statistics.json').write_text('{"changed":true}')
            self.assertNotEqual(original,requested_identity(base,adapter))
            original=requested_identity(base,adapter)
            (base/'export_complete.json').write_text('{"revision":"other"}')
            self.assertNotEqual(original,requested_identity(base,adapter))

    def test_reuse_requires_matching_identity_live_pid_and_ready_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base,adapter=self.prepare(root)
            identity=requested_identity(base,adapter)
            proc=root/'proc/123';proc.mkdir(parents=True)
            (proc/'cmdline').write_bytes(b'python\0-m\0cloud.connection_server\0')
            (proc/'stat').write_text('123 (python) '+' '.join(['S']+['0']*18+['12345']))
            outputs=root/'outputs';outputs.mkdir()
            (outputs/'cloud_server.pid').write_text('123')
            record=dict(identity=identity,process=process_identity(123,root/'proc'))
            marker=outputs/'cloud_server_identity.json';marker.write_text(json.dumps(record))
            ready=lambda:dict(status='ready',mode='openvla',model_loaded=True)
            self.assertTrue(reusable(outputs,identity,proc_root=root/'proc',health_reader=ready))
            self.assertFalse(reusable(outputs,identity,proc_root=root/'proc',health_reader=ready,require_sampling=True))
            sampling_ready=lambda:dict(ready(),action_sampling_available=True)
            self.assertTrue(reusable(outputs,identity,proc_root=root/'proc',health_reader=sampling_ready,require_sampling=True))
            for health in [dict(status='loading',mode='openvla',model_loaded=False),
                           dict(status='ready',mode='connection_test',model_loaded=False)]:
                self.assertFalse(reusable(outputs,identity,proc_root=root/'proc',health_reader=lambda:health))
            self.assertFalse(reusable(outputs,requested_identity(base),proc_root=root/'proc',health_reader=ready))
            (proc/'stat').write_text('123 (python) '+' '.join(['S']+['0']*18+['54321']))
            self.assertFalse(reusable(outputs,identity,proc_root=root/'proc',health_reader=ready))
            (proc/'cmdline').write_bytes(b'python\0other_program\0')
            self.assertFalse(reusable(outputs,identity,proc_root=root/'proc',health_reader=ready))
            marker.unlink()
            self.assertFalse(reusable(outputs,identity,proc_root=root/'proc',health_reader=ready))


if __name__=='__main__':unittest.main()
