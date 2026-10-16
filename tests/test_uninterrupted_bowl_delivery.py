"""Standard two-stage launcher dispatch for the matching bowl sweep."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from cloud.bowl_chain_plan import make_uninterrupted_plan

class UninterruptedDeliveryTests(unittest.TestCase):
    def test_dispatch_retains_volume_output_and_checks_source_provenance(self):
        from cloud.bowl_transfer_worker import run
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);folder=p/'sweep';folder.mkdir()
            plan=make_uninterrupted_plan('panda-train-color-bowl-test')
            (folder/'results.json').write_text(json.dumps(dict(plan=plan,trials=[])))
            with patch('shutil.disk_usage',return_value=SimpleNamespace(free=100*1024**3)),patch('subprocess.run') as probe,patch('cloud.bowl_chain_worker.stage_base',return_value={}),patch('cloud.bowl_chain_sweep.run',return_value=0) as sweep:
                self.assertEqual(run(folder,'sim-python',cache_root=p/'cache'),0)
            cfg=sweep.call_args.args[0]
            self.assertEqual(cfg['sweep'],str(folder));self.assertEqual(cfg['session'],str(folder))
            self.assertEqual(cfg['checkpoint_wait_seconds'],43200)
            self.assertIn('source_dataset_sha256',cfg);self.assertTrue(cfg['verify_adapter_copy'])
            self.assertNotIn('hf_token',cfg);probe.assert_called_once()

    def test_standard_worker_routes_bowl_plans_before_other_task_setup(self):
        from cloud.sweep_worker import run
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'results.json').write_text(json.dumps(dict(plan=make_uninterrupted_plan('panda-train-color-bowl-test'),trials=[])))
            with patch('cloud.bowl_transfer_worker.run',return_value=0) as dispatch:
                self.assertEqual(run(p,'sim'),0)
            dispatch.assert_called_once_with(p,'sim')

    def test_low_volume_space_rejects_before_probe_or_staging(self):
        from cloud.bowl_transfer_worker import run
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'results.json').write_text(json.dumps(dict(plan=make_uninterrupted_plan('panda-train-color-bowl-test'),trials=[])))
            with patch('shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3)),patch('cloud.bowl_chain_worker.stage_base') as stage:
                with self.assertRaisesRegex(ValueError,'35 GiB'):run(p,'sim',cache_root=p/'cache')
            stage.assert_not_called()

if __name__=='__main__':unittest.main()
