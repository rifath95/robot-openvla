"""Automatic microbatches require matching provenance and memory headroom."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from training.batch_policy import choose_batch,benchmark_context,context_differences


class BatchPolicyTests(unittest.TestCase):
    def test_minor_vram_drift_is_tolerated_with_adjusted_headroom_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.benchmark(root/'run')
            current=dict(self.context,total_vram_bytes=self.context['total_vram_bytes']-1024**2)
            config,record=choose_batch(self.config,current,root)
            self.assertEqual(record['source'],'compatible_benchmark')
            self.assertEqual(record['benchmarks_considered'][0]['differences'],
                             {'total_vram_bytes':dict(saved=self.context['total_vram_bytes'],
                                                     current=current['total_vram_bytes'])})
            self.assertEqual(config['batch_size'],8)
            self.assertEqual(record['adjusted_minimum_logged_free_gib'],5-1/1024)
            self.assertEqual(context_differences(self.context,dict(self.context,model_path='different',dataset_checksums_sha256='new')), {})

    def test_capacity_tolerance_cannot_remove_required_headroom(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.benchmark(root/'run')
            path=root/'run/benchmark.json';rows=json.loads(path.read_text())
            rows[1]['minimum_logged_free_gib']=4.005
            path.write_text(json.dumps(rows))
            current=dict(self.context,total_vram_bytes=self.context['total_vram_bytes']-10*2**20)
            self.assertEqual(choose_batch(self.config,current,root)[0]['batch_size'],4)
            rows[0].pop('reserved_headroom_gib')
            path.write_text(json.dumps(rows))
            self.assertEqual(choose_batch(self.config,current,root)[1]['source'],'configured_fallback')

    def test_capacity_changes_beyond_tolerance_remain_incompatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.benchmark(root/'run')
            for reduction in (17*2**20,3*2**30):
                current=dict(self.context,total_vram_bytes=self.context['total_vram_bytes']-reduction)
                self.assertEqual(choose_batch(self.config,current,root)[1]['source'],'configured_fallback')

    config=dict(batch_size=1,grad_accumulation_steps=4)
    context=dict(gpu='NVIDIA A40',total_vram_bytes=48*2**30,dataset_checksums_sha256='v2',
                 model_identity_sha256='model',settings=dict(lora_rank=32),torch_version='2.2.2')

    def benchmark(self,root,context=None,mode='capacity'):
        root.mkdir(parents=True)
        (root/'benchmark_metadata.json').write_text(json.dumps(dict(context=context or self.context,mode=mode)))
        rows=[dict(microbatch=b,accumulation=1,status='completed',
                   headroom_screen_passed=b<9,examples_per_second=s,
                   reserved_headroom_gib=6,minimum_logged_free_gib=5)
              for b,s in ((4,3.69),(8,3.74),(9,3.94))]
        (root/'benchmark.json').write_text(json.dumps(rows))

    def test_context_uses_input_format_without_requiring_dataset_checksums(self):
        gpu=SimpleNamespace(name='NVIDIA A40',total_memory=48*2**30)
        torch=SimpleNamespace(__version__='2.2.2+cu121',version=SimpleNamespace(cuda='12.1'),
            cuda=SimpleNamespace(is_available=lambda:True,current_device=lambda:0,
                get_device_properties=lambda _:gpu,get_device_capability=lambda:(8,6)))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset=root/'dataset';model=root/'model';dataset.mkdir();model.mkdir()
            contract=dict(pixel_shape=[6,224,224],supervised_tokens=8,input_tokens=48)
            (dataset/'input_check.json').write_text(json.dumps(dict(passed=True,
                official_processor_batches=dict(train=contract,validation=contract))))
            (model/'config.json').write_text('{}');(model/'export_complete.json').write_text('{}')
            settings=dict(lora_rank=32,lora_dropout=0.,use_lora=True,use_quantization=False,image_aug=False)
            with patch.dict('sys.modules',{'torch':torch,'transformers':SimpleNamespace(__version__='4.40.1')}):
                context=benchmark_context(settings,dataset,model)
                self.assertEqual(context['input_contract'],dict(pixel_shape=[6,224,224],supervised_tokens=8))
                self.assertNotIn('dataset_checksums_sha256',context)
                contract['pixel_shape']=[6,448,448]
                (dataset/'input_check.json').write_text(json.dumps(dict(passed=True,
                    official_processor_batches=dict(train=contract,validation=dict(pixel_shape=[6,224,224],supervised_tokens=8)))))
                self.assertIsNone(benchmark_context(settings,dataset,model))

    def test_fastest_safe_compatible_candidate_sets_accumulation_one(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.benchmark(root/'run')
            config,record=choose_batch(self.config,self.context,root)
            self.assertEqual((config['batch_size'],config['grad_accumulation_steps']),(8,1))
            self.assertEqual(record['source'],'compatible_benchmark')
            self.assertEqual(self.config['batch_size'],1)

    def test_other_gpu_model_rank_environment_and_input_shape_are_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.benchmark(root/'run')
            for key,value in [('gpu','A100'),('total_vram_bytes',80*2**30),
                              ('input_contract',dict(pixel_shape=[6,448,448],supervised_tokens=8)),
                              ('model_identity_sha256','other-model'),
                              ('settings',dict(lora_rank=16)),('torch_version','other-version')]:
                config,record=choose_batch(self.config,dict(self.context,**{key:value}),root)
                self.assertEqual(config,self.config)
                self.assertEqual(record['source'],'configured_fallback')

    def test_dataset_changes_and_missing_dataset_identity_reuse_benchmark(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.benchmark(root/'run')
            for context in [dict(self.context,dataset_checksums_sha256='v1'),
                            {k:v for k,v in self.context.items() if k!='dataset_checksums_sha256'}]:
                config,record=choose_batch(self.config,context,root)
                self.assertEqual(config['batch_size'],8)
                self.assertEqual(record['source'],'compatible_benchmark')

    def test_reviewed_legacy_a40_run_is_reused_with_original_candidate_artifacts(self):
        context=dict(self.context,model_path='/workspace/openvla-7b',transformers_version='4.40.1',
                     cuda_runtime='12.1',precision='bfloat16',attention='eager',
                     input_contract=dict(pixel_shape=[6,224,224],supervised_tokens=8))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'20261011_054257_777818';self.benchmark(run)
            (run/'benchmark_metadata.json').write_text(json.dumps(dict(mode='capacity',model=context['model_path'])))
            candidate=run/'batch_8_accum_1';candidate.mkdir()
            info=dict(gpu=context['gpu'],total_vram_gib=48,torch_version=context['torch_version'],
                      config=dict(context['settings'],vla_path=context['model_path'],batch_size=8,
                                  grad_accumulation_steps=1,dataset_name='panda_grasp_v1'))
            (candidate/'model_info.json').write_text(json.dumps(info))
            config,record=choose_batch(self.config,context,root)
            self.assertEqual(config['batch_size'],8)
            self.assertEqual(record['source'],'compatible_benchmark')
            for key,value in [('gpu','A100'),('settings',dict(lora_rank=16)),
                              ('total_vram_bytes',40*2**30),('transformers_version','other')]:
                self.assertEqual(choose_batch(self.config,dict(context,**{key:value}),root)[0],self.config)
            (candidate/'model_info.json').unlink()
            self.assertEqual(choose_batch(self.config,context,root)[0],self.config)

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
