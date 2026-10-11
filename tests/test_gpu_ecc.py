"""ECC provenance must never silently label unknown hardware as disabled."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from training.gpu_ecc import collect_ecc,ecc_label
from training.benchmark_training_batches import write_report
from analysis.summarize_gpu_benchmarks import comparison


class EccTests(unittest.TestCase):
    def test_current_and_pending_are_distinct_and_query_is_read_only(self):
        response=subprocess.CompletedProcess([],0,'0, GPU-abc, NVIDIA A40, Enabled, Disabled\n','')
        with patch('training.gpu_ecc.subprocess.run',return_value=response) as run:
            record=collect_ecc()
        self.assertEqual(record['current'],'Enabled')
        self.assertEqual(record['pending'],'Disabled')
        command=run.call_args.args[0]
        self.assertIn('--query-gpu=index,uuid,name,ecc.mode.current,ecc.mode.pending',command)
        self.assertFalse(any(arg.startswith('--ecc-config') or arg=='-e' for arg in command))

    def test_selects_cuda_uuid_and_does_not_guess_for_multiple_gpus(self):
        response=subprocess.CompletedProcess([],0,
            '0, GPU-a, NVIDIA A40, Enabled, Enabled\n1, GPU-b, NVIDIA A40, Disabled, Disabled\n','')
        with patch('training.gpu_ecc.subprocess.run',return_value=response),patch.dict('os.environ',{},clear=True):
            self.assertEqual(collect_ecc('GPU-b')['current'],'Disabled')
            self.assertEqual(collect_ecc()['current'],'Unknown')

    def test_missing_unsupported_and_failed_queries_are_not_disabled(self):
        self.assertEqual(ecc_label(None),'Unknown')
        self.assertEqual(ecc_label({'current':'N/A'}),'Unavailable (N/A)')
        self.assertEqual(ecc_label({'current':'[Not Supported]'}),'Not supported')
        for error in (FileNotFoundError(),subprocess.TimeoutExpired('nvidia-smi',10)):
            with patch('training.gpu_ecc.subprocess.run',side_effect=error):
                self.assertEqual(collect_ecc()['current'],'Unknown')

    def test_ecc_reaches_run_and_combined_reports_and_legacy_stays_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,record in [('new',dict(current='Enabled',pending='Disabled')),('legacy',None)]:
                run=root/name;run.mkdir()
                metadata={'context':{'gpu':'NVIDIA A40'}}
                if record:metadata['gpu_ecc']=record
                (run/'benchmark_metadata.json').write_text(json.dumps(metadata))
                write_report(run,[])
                report=(run/'BATCH_BENCHMARK.md').read_text()
                self.assertIn('GPU: **NVIDIA A40**',report)
                self.assertIn('GPU ECC: **Enabled**' if record else 'GPU ECC: **Unknown**',report)
                candidate=run/'batch_1_accum_1';candidate.mkdir()
                (candidate/'model_info.json').write_text(json.dumps({'gpu':'NVIDIA A40','total_vram_gib':47.4}))
            output=root/'comparison.md'
            comparison(root,output)
            report=output.read_text()
            self.assertIn('| GPU | ECC current | VRAM GiB |',report)
            self.assertIn('| NVIDIA A40 | Enabled | 47.4 |',report)
            self.assertIn('| NVIDIA A40 | Unknown | 47.4 |',report)
            self.assertIn('pending mode: **Disabled**',report)


if __name__=='__main__':unittest.main()
