"""Verify runtime provenance and truthful historical report fallbacks."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from training.benchmark_software import collect_software,software_lines
from training.benchmark_training_batches import write_report
from analysis.summarize_gpu_benchmarks import comparison


class SoftwareTests(unittest.TestCase):
    def test_collects_installed_versions_driver_and_code_revision(self):
        with patch('training.benchmark_software.packages.version',side_effect=lambda name:'actual-'+name),\
             patch('training.benchmark_software.subprocess.run',side_effect=[
                 subprocess.CompletedProcess([],0,'580.1\n580.1\n'),
                 subprocess.CompletedProcess([],0,'abc123\n')]):
            stack=collect_software({'cuda_runtime':'13.0','precision':'bfloat16','attention':'eager'})
        self.assertEqual(stack['packages']['torch'],'actual-torch')
        self.assertEqual(stack['driver'],'580.1')
        self.assertEqual(stack['benchmark_code_revision'],'abc123')
        self.assertEqual(stack['cuda_runtime'],'13.0')

    def test_missing_commands_and_legacy_versions_remain_explicit(self):
        with patch('training.benchmark_software.subprocess.run',side_effect=FileNotFoundError()):
            self.assertEqual(collect_software(None)['driver'],'Unknown')
        lines='\n'.join(software_lines({'context':{'cuda_runtime':'12.1','transformers_version':'4.40.1'}},
            {'torch_version':'2.2.2+cu121'},
            {'gpu_snapshot':'name, driver_version\nNVIDIA A40, 580.1\n'}))
        self.assertIn('| torch | 2.2.2+cu121 |',lines)
        self.assertIn('| transformers | 4.40.1 |',lines)
        self.assertIn('| NVIDIA driver | 580.1 |',lines)
        self.assertIn('| peft | Unknown |',lines)

    def test_software_table_is_in_both_report_details_before_candidate_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir()
            (run/'benchmark_metadata.json').write_text(json.dumps({
                'software_stack':{'packages':{'torch':'2.14.1+cu130','peft':'0.11.1'},
                                  'driver':'580.1','cuda_runtime':'13.0'}}))
            write_report(run,[])
            output=root/'comparison.md';comparison(root,output)
            for report in ((run/'BATCH_BENCHMARK.md').read_text(),output.read_text()):
                self.assertIn('| torch | 2.14.1+cu130 |',report)
                self.assertIn('| NVIDIA driver | 580.1 |',report)
                self.assertLess(report.index('Software stack ('),report.index('| Microbatch |'))


if __name__=='__main__':unittest.main()
