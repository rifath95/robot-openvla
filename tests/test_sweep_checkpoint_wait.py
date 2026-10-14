"""Incomplete saves must be awaited; incremental staging must preserve the loaded base copy."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cloud.sweep_worker import wait_checkpoint
from cloud.sweep_storage import stage_weights

class CheckpointWaitTests(unittest.TestCase):
    def test_incomplete_save_then_ready(self):
        with patch('cloud.checkpoint_ready.check_ready',side_effect=[ValueError('metadata partial'),Path('/checkpoint')]) as ready, \
             patch('cloud.sweep_worker.time.sleep'):
            self.assertEqual(wait_checkpoint(Path('/checkpoint'),30),Path('/checkpoint'))
            self.assertEqual(ready.call_count,2)

    def test_timeout_and_ended_training_do_not_load_partial_checkpoint(self):
        with patch('cloud.checkpoint_ready.check_ready',side_effect=ValueError('partial')):
            with self.assertRaisesRegex(RuntimeError,'finish saving'):wait_checkpoint('/checkpoint',0)
            with patch('cloud.sweep_worker.time.monotonic',side_effect=[0,31]):
                with self.assertRaisesRegex(RuntimeError,'finish saving'):wait_checkpoint('/checkpoint',30)
            with tempfile.TemporaryDirectory() as tmp:
                status=Path(tmp)/'status.json';status.write_text('{"phase":"training_failed"}')
                with self.assertRaisesRegex(RuntimeError,'Training ended'):wait_checkpoint('/checkpoint',30,status)

    def test_incremental_staging_copies_new_adapter_without_recopying_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'volume/base';base.mkdir(parents=True);(base/'config.json').write_text('{}')
            (base/'weights.bin').write_bytes(b'base')
            adapters={}
            for cp in (50,100):
                p=root/f'volume/checkpoint_{cp}';p.mkdir();(p/'adapter.bin').write_bytes(str(cp).encode());adapters[cp]=p
            cache=root/'container/cache'
            with patch('cloud.sweep_storage.check_ready'):
                stage_weights(base,{50:adapters[50]},cache)
                original=(cache/'base/weights.bin').stat()
                stage_weights(base,adapters,cache,incremental=True)
                final=(cache/'base/weights.bin').stat()
                self.assertEqual((original.st_ino,original.st_mtime_ns),(final.st_ino,final.st_mtime_ns))
                self.assertEqual((cache/'checkpoint_000100/adapter.bin').read_bytes(),b'100')
                self.assertEqual(len(json.loads((cache/'staging.json').read_text())['sources']),3)
                with self.assertRaisesRegex(ValueError,'identity'):stage_weights(base,{100:adapters[100]},cache,incremental=True)

if __name__=='__main__':unittest.main()
