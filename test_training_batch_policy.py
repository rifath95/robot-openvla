"""Automatic microbatches require matching provenance and memory headroom."""
import json
from pathlib import Path
import tempfile
import unittest

from training.batch_policy import choose_batch


class BatchPolicyTests(unittest.TestCase):
    config=dict(batch_size=1,grad_accumulation_steps=4)
    context=dict(gpu='NVIDIA A40',total_vram_bytes=48*2**30,dataset_checksums_sha256='v2',
                 model_identity_sha256='model',settings=dict(lora_rank=32),torch_version='2.2.2')

    def benchmark(self,root,context=None,mode='capacity'):
        root.mkdir(parents=True)
        (root/'benchmark_metadata.json').write_text(json.dumps(dict(context=context or self.context,mode=mode)))
        rows=[dict(microbatch=b,accumulation=1,status='completed',
                   headroom_screen_passed=b<9,examples_per_second=s)
              for b,s in ((4,3.69),(8,3.74),(9,3.94))]
        (root/'benchmark.json').write_text(json.dumps(rows))

    def test_fastest_safe_compatible_candidate_sets_accumulation_one(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.benchmark(root/'run')
            config,record=choose_batch(self.config,self.context,root)
            self.assertEqual((config['batch_size'],config['grad_accumulation_steps']),(8,1))
            self.assertEqual(record['source'],'compatible_benchmark')
            self.assertEqual(self.config['batch_size'],1)

    def test_other_gpu_dataset_model_rank_and_environment_are_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.benchmark(root/'run')
            for key,value in [('gpu','A100'),('total_vram_bytes',80*2**30),
                              ('dataset_checksums_sha256','v1'),('model_identity_sha256','other-model'),
                              ('settings',dict(lora_rank=16)),('torch_version','other-version')]:
                config,record=choose_batch(self.config,dict(self.context,**{key:value}),root)
                self.assertEqual(config,self.config)
                self.assertEqual(record['source'],'configured_fallback')

    def test_legacy_missing_provenance_and_malformed_files_fall_back(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.benchmark(root/'legacy')
            (root/'legacy/benchmark_metadata.json').write_text('{"mode":"capacity"}')
            broken=root/'broken';broken.mkdir();(broken/'benchmark_metadata.json').write_text('partial')
            config,record=choose_batch(self.config,self.context,root)
            self.assertEqual(config,self.config)
            self.assertEqual(record['source'],'configured_fallback')

    def test_fixed_effective_batch_benchmarks_are_not_used_as_capacity_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.benchmark(root/'run',mode='controlled')
            self.assertEqual(choose_batch(self.config,self.context,root)[0],self.config)


if __name__=='__main__':unittest.main()
