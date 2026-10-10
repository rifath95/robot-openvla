"""Validation and local metrics for the pinned official training loop."""
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.panda_rlds import PandaRLDSDataset


class Metrics:
    def init(self, *, name, **kwargs):
        self.started=time.monotonic()
        self.directory=Path(os.environ['PANDA_RUN_DIR'])
        self.directory.mkdir(parents=True,exist_ok=True)
        if (self.directory/'experiment.json').exists():
            raise ValueError('Run already exists; refusing to mix experiment logs')
        self.rows=[]
        from torch.utils.tensorboard import SummaryWriter
        self.writer=SummaryWriter(str(self.directory/'tensorboard'))
        (self.directory/'experiment.json').write_text(json.dumps(dict(name=name,
            upstream_revision=os.environ['PANDA_UPSTREAM_REVISION'],
            validation_every=int(os.environ['PANDA_VALIDATE_EVERY']),
            validation_max_batches=int(os.environ['PANDA_VALIDATION_MAX_BATCHES']),
            max_runtime_seconds=int(os.environ['PANDA_MAX_RUNTIME']),
            note='Runtime guard is checked between updates; it does not stop pod billing.'),indent=2)+'\n')
        if hasattr(self,'model_details'):
            (self.directory/'model_info.json').write_text(json.dumps(self.model_details,default=str,indent=2)+'\n')

    def log(self, values, step):
        row=dict(update=step,elapsed_seconds=time.monotonic()-self.started,**values)
        if torch.cuda.is_available():
            row['gpu_allocated_gib']=torch.cuda.memory_allocated()/2**30
            row['gpu_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        self.rows.append(row)
        for key,value in values.items():
            if isinstance(value,(int,float)):
                self.writer.add_scalar(key,value,step)
        self.writer.flush()
        with (self.directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if 'validation_loss' in values:self.plot()

    def plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        for field,label in [('train_loss','Training'),('validation_loss','Validation')]:
            rows=[r for r in self.rows if field in r]
            if rows:ax.plot([r['update'] for r in rows],[r[field] for r in rows],label=label)
        ax.set(xlabel='Parameter updates',ylabel='Action-token cross entropy',title='OpenVLA Panda LoRA')
        ax.legend();fig.tight_layout();fig.savefig(self.directory/'loss.png');plt.close(fig)


metrics=Metrics()


def record_model(model, cfg):
    properties=torch.cuda.get_device_properties(torch.cuda.current_device())
    details=dict(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),
        gpu=properties.name,total_vram_gib=properties.total_memory/2**30,
        torch_version=torch.__version__,config=vars(cfg))
    # Logging is initialized later in the official loop, so carry this in memory.
    metrics.model_details=details


def validate(model, cfg, batch_transform, collator, device_id, update):
    dataset=PandaRLDSDataset(cfg.data_root_dir,cfg.dataset_name,batch_transform,
                            train=False,split='validation')
    loader=DataLoader(dataset,batch_size=cfg.batch_size,collate_fn=collator,num_workers=0)
    max_batches=int(os.environ['PANDA_VALIDATION_MAX_BATCHES'])
    losses=[];weights=[];examples=0
    was_training=model.training
    model.eval()
    try:
        with torch.inference_mode():
            for i,batch in enumerate(loader):
                if max_batches and i>=max_batches:break
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    result=model(input_ids=batch['input_ids'].to(device_id),
                        attention_mask=batch['attention_mask'].to(device_id),
                        pixel_values=batch['pixel_values'].to(device_id,dtype=torch.bfloat16),
                        labels=batch['labels'].to(device_id))
                # Weight CE by supervised tokens, including the stop token.
                weight=int((batch['labels'][:,1:]!=-100).sum())
                losses.append(float(result.loss)*weight);weights.append(weight)
                examples+=len(batch['input_ids'])
    finally:
        model.train(was_training)
    metrics.log(dict(validation_loss=sum(losses)/sum(weights),validation_examples=examples,
                     validation_complete=examples==len(dataset)),step=update)


def save_adapter(model, processor, cfg, dataset, update):
    folder=metrics.directory/f'checkpoint_{update:06d}'
    folder.mkdir(parents=True,exist_ok=False)
    model.module.save_pretrained(folder)
    processor.save_pretrained(folder)
    (folder/'dataset_statistics.json').write_text(json.dumps(dataset.dataset_statistics,indent=2)+'\n')
    (folder/'training_config.json').write_text(json.dumps(vars(cfg),default=str,indent=2)+'\n')
    (metrics.directory/'model_info.json').write_text(json.dumps(metrics.model_details,default=str,indent=2)+'\n')
    print(f'Saved adapter checkpoint: {folder}',flush=True)


def budget_expired():
    return time.monotonic()-metrics.started>=int(os.environ['PANDA_MAX_RUNTIME'])
