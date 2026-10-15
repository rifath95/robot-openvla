"""Validation and local metrics for the pinned official training loop."""
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.panda_rlds import PandaRLDSDataset
from training.update_numbering import initial_update, cumulative_update


class Metrics:
    def init(self, *, name, **kwargs):
        self.started=time.monotonic()
        self.update_started=self.started
        self.directory=Path(os.environ['PANDA_RUN_DIR'])
        self.directory.mkdir(parents=True,exist_ok=True)
        if (self.directory/'experiment.json').exists():
            raise ValueError('Run already exists; refusing to mix experiment logs')
        self.rows=[]
        if os.environ.get('PANDA_BATCH_SELECTION'):
            (self.directory/'batch_selection.json').write_text(json.dumps(
                json.loads(os.environ['PANDA_BATCH_SELECTION']),indent=2)+'\n')
        from torch.utils.tensorboard import SummaryWriter
        self.writer=SummaryWriter(str(self.directory/'tensorboard'))
        (self.directory/'experiment.json').write_text(json.dumps(dict(name=name,
            upstream_revision=os.environ['PANDA_UPSTREAM_REVISION'],
            validation_enabled=os.environ.get('PANDA_VALIDATION_ENABLED','1')!='0',
            dataset_provenance=json.loads(os.environ['PANDA_PROVENANCE']) if os.environ.get('PANDA_PROVENANCE') else None,
            validation_every=int(os.environ['PANDA_VALIDATE_EVERY']),
            validation_updates=json.loads(os.environ['PANDA_VALIDATION_UPDATES']) if os.environ.get('PANDA_VALIDATION_UPDATES') else None,
            validation_max_batches=int(os.environ['PANDA_VALIDATION_MAX_BATCHES']),
            max_runtime_seconds=int(os.environ['PANDA_MAX_RUNTIME']),
            initial_adapter=os.environ.get('PANDA_INITIAL_ADAPTER') or None,
            initial_update=initial_update(),
            continuation_mode='adapter_weights_fresh_optimizer' if os.environ.get('PANDA_INITIAL_ADAPTER') else 'fresh_base',
            checkpoint_updates=[cumulative_update(u) for u in json.loads(os.environ['PANDA_CHECKPOINT_UPDATES'])] if os.environ.get('PANDA_CHECKPOINT_UPDATES') else None,
            checkpoint_saving=os.environ.get('PANDA_SAVE_CHECKPOINTS','1')!='0',
            note='Runtime guard is checked between updates; it does not stop pod billing.'),indent=2)+'\n')
        if hasattr(self,'model_details'):
            (self.directory/'model_info.json').write_text(json.dumps(self.model_details,default=str,indent=2)+'\n')

    def log(self, values, step):
        local_step=step
        step=cumulative_update(step)
        row=dict(update=step,elapsed_seconds=time.monotonic()-self.started,**values)
        if initial_update():row.update(run_update=local_step,initial_update=initial_update())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            row['gpu_allocated_gib']=torch.cuda.memory_allocated()/2**30
            row['gpu_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
            row['gpu_reserved_gib']=torch.cuda.memory_reserved()/2**30
            row['gpu_peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
            free,total=torch.cuda.mem_get_info()
            row['gpu_free_gib']=free/2**30
            row['gpu_total_gib']=total/2**30
            row['elapsed_seconds']=time.monotonic()-self.started
        if 'train_loss' in values:
            row['training_update_wall_seconds']=time.monotonic()-self.update_started
            self.update_started=time.monotonic()
        self.rows.append(row)
        for key,value in values.items():
            if isinstance(value,(int,float)):
                self.writer.add_scalar(key,value,step)
        self.writer.flush()
        with (self.directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if 'validation_loss' in values:self.plot()

    def plot(self):
        from training.loss_plots import write_loss_plots
        write_loss_plots(self.rows,self.directory)


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
    if os.environ.get('PANDA_VALIDATION_ENABLED','1')=='0':return
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
    metrics.update_started=time.monotonic()


def save_adapter(model, processor, cfg, dataset, update):
    if os.environ.get('PANDA_SAVE_CHECKPOINTS','1')=='0':
        print(f'Benchmark: adapter checkpoint saving disabled at update {update}',flush=True)
        return
    save_started=time.monotonic()
    local_update=update
    update=cumulative_update(update)
    folder=metrics.directory/f'checkpoint_{update:06d}'
    folder.mkdir(parents=True,exist_ok=False)
    model.module.save_pretrained(folder)
    processor.save_pretrained(folder)
    (folder/'dataset_statistics.json').write_text(json.dumps(dataset.dataset_statistics,indent=2)+'\n')
    saved_config=dict(vars(cfg))
    if os.environ.get('PANDA_INPUT_STAGING'):
        saved_config.update(source_model_path=os.environ['PANDA_SOURCE_MODEL'],
            source_dataset_path=os.environ['PANDA_SOURCE_DATASET'],
            input_staging=json.loads(os.environ['PANDA_INPUT_STAGING']))
    saved_config.update(checkpoint_update=update,training_elapsed_seconds_at_save=time.monotonic()-metrics.started if hasattr(metrics,'started') else None)
    if initial_update():
        saved_config.update(initial_update=initial_update(),run_update=local_update,
            initial_adapter=os.environ.get('PANDA_INITIAL_ADAPTER'),continuation_mode='adapter_weights_fresh_optimizer')
    saved_config['validation_enabled']=os.environ.get('PANDA_VALIDATION_ENABLED','1')!='0'
    if os.environ.get('PANDA_PROVENANCE'):
        provenance=json.loads(os.environ['PANDA_PROVENANCE'])
        saved_config.update(dataset_provenance=provenance,source_model_path=provenance['base_model_volume_path'])
        (folder/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (folder/'training_config.json').write_text(json.dumps(saved_config,default=str,indent=2)+'\n')
    (metrics.directory/'model_info.json').write_text(json.dumps(metrics.model_details,default=str,indent=2)+'\n')
    with (metrics.directory/'checkpoint_timings.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(update=update,seconds=time.monotonic()-save_started))+'\n')
    metrics.plot()
    metrics.update_started=time.monotonic()
    print(f'Saved adapter checkpoint: {folder}',flush=True)


def budget_expired():
    return time.monotonic()-metrics.started>=int(os.environ['PANDA_MAX_RUNTIME'])
