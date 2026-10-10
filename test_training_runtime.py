"""CPU checks for adapter freezing and persistent plot/event logging."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig,get_peft_model
from transformers import PreTrainedModel,PretrainedConfig

from training.runtime import Metrics


class TrainingRuntimeTests(unittest.TestCase):
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
                     PANDA_VALIDATE_EVERY='10',PANDA_VALIDATION_MAX_BATCHES='64',PANDA_MAX_RUNTIME='30')
            with patch.dict(os.environ,env):
                metrics=Metrics();metrics.init(name='test')
                metrics.log({'train_loss':2.0},step=1)
                metrics.log({'validation_loss':2.5,'validation_examples':64,'validation_complete':False},step=10)
                self.assertTrue((Path(directory)/'loss.png').is_file())
                self.assertTrue(any((Path(directory)/'tensorboard').glob('events*')))
                rows=[json.loads(line) for line in (Path(directory)/'metrics.jsonl').read_text().splitlines()]
                self.assertFalse(rows[-1]['validation_complete'])
                with self.assertRaises(ValueError):Metrics().init(name='second')
                metrics.writer.close()


if __name__=='__main__':unittest.main()
