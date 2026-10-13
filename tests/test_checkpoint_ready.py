import json
from pathlib import Path
import struct
import tempfile
import unittest
from cloud.checkpoint_ready import check_ready
from cloud.cloud_session import bootstrap_script


class CheckpointReadyTests(unittest.TestCase):
    def checkpoint(self,root):
        folder=root/'checkpoint_000010';folder.mkdir()
        for name in ('adapter_config.json','dataset_statistics.json','training_config.json',
                     'processor_config.json','preprocessor_config.json'):
            (folder/name).write_text('{}')
        header=json.dumps({'weight':dict(dtype='F32',shape=[2],data_offsets=[0,8])}).encode()
        (folder/'adapter_model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+bytes(8))
        return folder

    def test_complete_checkpoint_and_truncated_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=self.checkpoint(Path(tmp));self.assertEqual(check_ready(folder),folder)
            path=folder/'adapter_model.safetensors';path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError,'incomplete'):check_ready(folder)

    def test_final_metadata_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=self.checkpoint(Path(tmp));(folder/'training_config.json').unlink()
            with self.assertRaisesRegex(ValueError,'not fully saved'):check_ready(folder)

    def test_readiness_checked_before_old_server_is_stopped(self):
        script=bootstrap_script(1800,'/workspace/checkpoint_000010')
        self.assertLess(script.index('cloud.checkpoint_ready'),script.index('cloud.server_reuse'))
        self.assertLess(script.index('cloud.checkpoint_ready'),script.index('bash scripts/cloud_stop.sh'))


if __name__=='__main__':unittest.main()
