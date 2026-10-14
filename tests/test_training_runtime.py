"""CPU checks for adapter freezing and persistent plot/event logging."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
from types import SimpleNamespace

import torch
from peft import LoraConfig,get_peft_model
from transformers import PreTrainedModel,PretrainedConfig

from training.runtime import Metrics,save_adapter


class TrainingRuntimeTests(unittest.TestCase):
    def test_benchmark_skips_saves_and_normal_training_keeps_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=Mock();processor=Mock()
            cfg=SimpleNamespace(lora_rank=32)
            dataset=SimpleNamespace(dataset_statistics={'train':{}})
            model.module.save_pretrained.side_effect=lambda folder:(folder/'adapter_model.safetensors').write_bytes(b'adapter')
            with patch('training.runtime.metrics',SimpleNamespace(directory=root,model_details={'gpu':'test'},plot=Mock())):
                with patch.dict(os.environ,{'PANDA_SAVE_CHECKPOINTS':'0'}):
                    save_adapter(model,processor,cfg,dataset,25)
                    self.assertFalse((root/'checkpoint_000025').exists())
                    model.module.save_pretrained.assert_not_called()
                    processor.save_pretrained.assert_not_called()
                with patch.dict(os.environ,{'PANDA_SAVE_CHECKPOINTS':'1','PANDA_INPUT_STAGING':'{\"sources\":{}}',
                    'PANDA_SOURCE_MODEL':'/workspace/openvla-7b','PANDA_SOURCE_DATASET':'/workspace/panda-training/bundle'}):
                    save_adapter(model,processor,cfg,dataset,1000)
                    folder=root/'checkpoint_001000'
                    self.assertTrue((folder/'adapter_model.safetensors').is_file())
                    self.assertTrue((folder/'dataset_statistics.json').is_file())
                    self.assertTrue((folder/'training_config.json').is_file())
                    saved=json.loads((folder/'training_config.json').read_text())
                    self.assertEqual(saved['source_model_path'],'/workspace/openvla-7b')
                    self.assertEqual(saved['source_dataset_path'],'/workspace/panda-training/bundle')
                    self.assertTrue((root/'model_info.json').is_file())

    def test_training_only_never_opens_a_validation_dataset(self):
        from training.runtime import validate
        from training.checkpoint_schedule import validation_due
        with patch.dict(os.environ,{'PANDA_VALIDATION_ENABLED':'0'}),patch('training.runtime.PandaRLDSDataset',side_effect=AssertionError('No validation split')):
            self.assertIsNone(validate(None,None,None,None,None,0))
            self.assertFalse(validation_due(1000,100))

    def test_lora_update_keeps_base_weights_fixed(self):
        class TinyModel(PreTrainedModel):
            config_class=PretrainedConfig
            def __init__(self):
                super().__init__(PretrainedConfig())
                self.layer=torch.nn.Linear(4,3)
            def forward(self,x):return self.layer(x)
        model=get_peft_model(TinyModel(),
                            LoraConfig(r=2,target_modules='all-linear'))
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        trainable=[p for p in model.parameters() if p.requires_grad]
        before=[p.detach().clone() for p in trainable]
        optimizer=torch.optim.AdamW(trainable,lr=.01)
        loss=model(torch.ones(2,4)).square().sum()
        loss.backward();optimizer.step()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,trainable)))
        for n,p in model.named_parameters():
            if n in frozen:self.assertTrue(torch.equal(frozen[n],p))

    def test_plots_events_and_existing_run_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            env=dict(PANDA_RUN_DIR=directory,PANDA_UPSTREAM_REVISION='test',
                     PANDA_VALIDATE_EVERY='10',PANDA_VALIDATION_MAX_BATCHES='64',PANDA_MAX_RUNTIME='30',
                     PANDA_BATCH_SELECTION=json.dumps(dict(policy='auto',microbatch=8,accumulation=1)))
            with patch.dict(os.environ,env):
                metrics=Metrics();metrics.init(name='test')
                selection=json.loads((Path(directory)/'batch_selection.json').read_text())
                self.assertEqual(selection['microbatch'],8)
                metrics.log({'train_loss':2.0},step=1)
                metrics.log({'validation_loss':2.5,'validation_examples':64,'validation_complete':False},step=10)
                self.assertTrue((Path(directory)/'loss.png').is_file())
                self.assertTrue((Path(directory)/'loss_smoothed.png').is_file())
                self.assertTrue(any((Path(directory)/'tensorboard').glob('events*')))
                rows=[json.loads(line) for line in (Path(directory)/'metrics.jsonl').read_text().splitlines()]
                self.assertFalse(rows[-1]['validation_complete'])
                with self.assertRaises(ValueError):Metrics().init(name='second')
                metrics.writer.close()


if __name__=='__main__':unittest.main()
