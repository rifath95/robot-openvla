"""Real CPU adapter reload plus training-launch configuration checks."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig
from transformers import PreTrainedModel,PretrainedConfig

from training.continuation import attach_lora,validate_adapter
from training.integrate_upstream import integrated_source
import train_panda
import training.detached_run as detached


class TinyModel(PreTrainedModel):
    config_class=PretrainedConfig
    def __init__(self):
        super().__init__(PretrainedConfig());self.layer=torch.nn.Linear(4,3)
    def forward(self,x):return self.layer(x)


class ContinuationTests(unittest.TestCase):
    def test_reloaded_adapter_preserves_learned_output_and_remains_trainable(self):
        torch.manual_seed(123);base=TinyModel();base_weights={k:v.clone() for k,v in base.state_dict().items()}
        cfg=LoraConfig(r=2,target_modules='all-linear')
        with patch.dict(os.environ,{'PANDA_INITIAL_ADAPTER':''}):model=attach_lora(base,cfg)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        x=torch.ones(2,4);model(x).square().sum().backward();optimizer.step();model.eval()
        expected=model(x).detach()
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            fresh=TinyModel();fresh.load_state_dict(base_weights)
            with patch.dict(os.environ,{'PANDA_INITIAL_ADAPTER':folder}):continued=attach_lora(fresh,cfg)
            continued.eval();torch.testing.assert_close(continued(x),expected)
            trainable=[p for p in continued.parameters() if p.requires_grad]
            self.assertTrue(trainable)
            before=[p.detach().clone() for p in trainable]
            new_optimizer=torch.optim.AdamW(trainable,lr=.01)
            continued(x).square().sum().backward();new_optimizer.step()
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,trainable)))
            self.assertTrue(all(not p.requires_grad for n,p in continued.named_parameters() if 'lora_' not in n))

    def test_invalid_adapter_rank_and_base_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=dict(lora_rank=32,lora_dropout=0.)
            (root/'adapter_model.safetensors').write_bytes(b'fixture')
            (root/'adapter_config.json').write_text(json.dumps(dict(peft_type='LORA',r=32,lora_dropout=0.)))
            (root/'training_config.json').write_text(json.dumps(dict(vla_path='/workspace/openvla-7b')))
            self.assertEqual(validate_adapter(root,config,Path('/workspace/openvla-7b')),root.resolve())
            for changed,model in [(dict(config,lora_rank=16),'/workspace/openvla-7b'),(config,'/different-base')]:
                with self.assertRaises(ValueError):validate_adapter(root,changed,Path(model))

    def test_new_profile_and_supervisor_flags_do_not_leak_into_upstream_command(self):
        profile=json.loads(Path('training/pick_place_v2_5000.json').read_text())
        self.assertEqual(profile['max_steps']*profile['batch_size']*profile['grad_accumulation_steps'],40000)
        self.assertEqual(profile['max_runtime_seconds'],4*3600)
        self.assertEqual(profile['hard_limit_seconds'],5*3600)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset=root/'dataset';dataset.mkdir()
            (dataset/'verification.json').write_text('{"passed":true}')
            argv=['train','--dataset',str(dataset),'--config','training/pick_place_v2_5000.json',
                  '--output-root',str(root/'runs'),'--prepare-only']
            with patch('sys.argv',argv), \
                 patch('training.batch_policy.benchmark_context',return_value={'gpu':'NVIDIA A40'}), \
                 patch('training.batch_policy.choose_batch',return_value=(profile,dict(source='compatible_benchmark',microbatch=8,accumulation=1))), \
                 patch('train_panda.subprocess.run') as run, \
                 patch('sys.stdout',new_callable=io.StringIO) as output:
                # chooser copies the real configuration; return profile stripped of wrapper keys.
                selected={k:v for k,v in profile.items() if k not in ('hard_limit_seconds','minimum_free_disk_gib','require_benchmark','expected_gpu')}
                with patch('training.batch_policy.choose_batch',return_value=(selected,dict(source='compatible_benchmark',microbatch=8,accumulation=1))):
                    train_panda.main()
                command=output.getvalue().split('Command:',1)[1]
                for option in ('--hard_limit_seconds','--minimum_free_disk_gib','--require_benchmark','--expected_gpu','--target_effective_batch'):
                    self.assertNotIn(option,command)
                self.assertIn('--max_steps 5000',command)
                run.assert_not_called()
        self.assertIn('attach_lora(vla, lora_config)',integrated_source())

    def test_supervisor_uses_profile_outer_limit_and_forwards_continuation(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)/'new'
            config=json.loads(Path('training/pick_place_v2_5000.json').read_text())
            checkpoint=Path('/workspace/panda-training/runs/previous/checkpoint_005000')
            argv=['supervisor','--session-dir',str(directory),'--config','training/pick_place_v2_5000.json',
                  '--dataset','/workspace/panda-training/training_v2','--adapter',str(checkpoint)]
            read=Path.read_text
            def content(path,*args,**kwargs):
                if path.name=='verification.json':return '{"passed":true}'
                return read(path,*args,**kwargs)
            with patch('sys.argv',argv),patch.object(Path,'read_text',content), \
                 patch.object(Path,'is_relative_to',return_value=True), \
                 patch.object(detached,'preflight',return_value=('pod123','secret')), \
                 patch.object(detached.shutil,'disk_usage',return_value=type('Usage',(),{'free':10*2**30})()), \
                 patch('training.continuation.validate_adapter',return_value=checkpoint), \
                 patch.object(detached,'supervise',return_value=0) as supervise:
                self.assertEqual(detached.main(),0)
            self.assertEqual(supervise.call_args.kwargs['hard_limit'],18000)
            self.assertEqual(supervise.call_args.kwargs['expected_updates'],5000)
            self.assertIn('--adapter',supervise.call_args.args[0])


if __name__=='__main__':unittest.main()
