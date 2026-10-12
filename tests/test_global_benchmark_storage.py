"""Global storage uses copies; local benchmark work retains watchdog behavior."""
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from cloud.prepare_cloud_model import MODEL_ID,REVISION
from training.global_benchmark_storage import stage_inputs,copy_reports,global_preflight
from training.benchmark_supervisor import supervise
from cloud.benchmark_cloud_session import remote_benchmark_script


class GlobalBenchmarkTests(unittest.TestCase):
    def test_staged_inputs_match_and_corrupt_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);volume=root/'volume';model=volume/'openvla-7b';model.mkdir(parents=True)
            (model/'config.json').write_text('{}')
            (model/'model.safetensors').write_bytes(b'base model')
            record=dict(model=MODEL_ID,revision=REVISION,files={'config.json':2,'model.safetensors':10})
            (model/'export_complete.json').write_text(json.dumps(record))
            dataset=volume/'panda-training/training_v1';dataset.mkdir(parents=True)
            (dataset/'verification.json').write_text('{"passed":true}')
            (dataset/'data.bin').write_bytes(b'image+action')
            checks=''.join(f'{hashlib.sha256((dataset/n).read_bytes()).hexdigest()}  {n}\n' for n in ('data.bin','verification.json'))
            (dataset/'SHA256SUMS').write_text(checks)
            staged,local_dataset,metadata=stage_inputs(volume,dataset,root/'local')
            self.assertEqual((staged/'model.safetensors').read_bytes(),b'base model')
            self.assertEqual((local_dataset/'data.bin').read_bytes(),b'image+action')
            self.assertEqual(metadata['training_reads'],'container_disk')
            self.assertEqual(metadata['source_storage'],'global_volume')
            bowl=volume/'panda-training/training_bowl_v1'
            shutil.copytree(dataset,bowl)
            _,local_bowl,_=stage_inputs(volume,bowl,root/'local-bowl')
            self.assertEqual((local_bowl/'data.bin').read_bytes(),b'image+action')
            (dataset/'data.bin').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):stage_inputs(volume,dataset,root/'other')

    def test_report_publication_uses_copies_and_excludes_checkpoint_weights_and_locks(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'local';destination=root/'global';source.mkdir()
            run=source/'run';run.mkdir();(run/'benchmark.json').write_text('[]')
            (run/'console.log').write_text('complete')
            (source/'supervisor.lock').write_text('lock')
            checkpoint=run/'checkpoint_000025';checkpoint.mkdir()
            (checkpoint/'adapter_model.safetensors').write_bytes(b'weights')
            (checkpoint/'adapter_config.json').write_text('{}')
            with patch.object(Path,'rename',side_effect=AssertionError('No global rename')):
                self.assertEqual(copy_reports(source,destination),2)
            self.assertTrue((destination/'run/benchmark.json').is_file())
            self.assertFalse((destination/'supervisor.lock').exists())
            self.assertFalse((destination/'run/checkpoint_000025').exists())

    def test_global_preflight_requires_real_mount_and_matching_pod_identity(self):
        with patch('training.global_benchmark_storage.subprocess.run',return_value=SimpleNamespace(stdout='/workspace fuse.geesefs')), \
             patch('training.global_benchmark_storage.recover_pod_environment',return_value=('pod123','secret')), \
             patch('training.global_benchmark_storage.api_request',return_value={'id':'pod123','volumeMountPath':'/workspace'}):
            self.assertEqual(global_preflight('/workspace'),('pod123','secret'))
        with patch('training.global_benchmark_storage.subprocess.run',return_value=SimpleNamespace(stdout='/ overlay')):
            with self.assertRaises(RuntimeError):global_preflight('/workspace')

    def test_global_report_copy_failure_still_requests_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)/'local/sessions/test';directory.mkdir(parents=True)
            with patch('training.benchmark_supervisor.subprocess.run',side_effect=subprocess.TimeoutExpired('copy',1)), \
                 patch('training.benchmark_supervisor.stop_pod',return_value=0) as stop:
                result=supervise(['unused'],directory,'pod123','secret',sync_grace=0,persistent_root=Path(folder)/'global')
            self.assertEqual(result,1);stop.assert_called_once()
            self.assertIn('report_publication_error',json.loads((directory/'status.json').read_text()))

    def test_completed_and_failed_benchmarks_publish_before_stopping(self):
        for source,code in [('print("done")',0),('raise SystemExit(2)',1)]:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);directory=root/'local/sessions/test';directory.mkdir(parents=True)
                destination=root/'global'
                def stopped(*args,**kwargs):
                    self.assertTrue((destination/'sessions/test/console.log').is_file())
                    self.assertEqual(json.loads((destination/'sessions/test/status.json').read_text())['phase'],'stopping')
                    return 0
                with patch('training.benchmark_supervisor.stop_pod',side_effect=stopped):
                    self.assertEqual(supervise([sys.executable,'-c',source],directory,'pod123','secret',sync_grace=0,persistent_root=destination),code)

    def test_global_launcher_carries_mount_to_preflight_and_watchdog(self):
        source=remote_benchmark_script('/workspace/panda-training/training_v1',64,auto_stop=True,global_volume='/workspace')
        self.assertIn('training.benchmark_stop --check-only --global-volume /workspace',source)
        self.assertIn('training.benchmark_supervisor --launch',source)
        self.assertIn('--global-volume /workspace',source)

    def test_global_preparation_downloads_bowl_bundle_by_default(self):
        from cloud.prepare_global_benchmark import main
        with patch('sys.argv',['prepare','--volume-root','/workspace-global']), \
             patch('cloud.prepare_global_benchmark.subprocess.run') as run:
            main()
        command=run.call_args_list[-1].args[0]
        self.assertEqual(command[command.index('--bundle')+1],'training_bowl_v1')
        self.assertEqual(command[command.index('--volume-root')+1],'/workspace-global')


if __name__=='__main__':unittest.main()
