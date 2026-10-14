"""Continue learning from adapter weights; optimizer and data stream start fresh."""
import json
import os
from pathlib import Path


def validate_adapter(path,config,model):
    path=Path(path).resolve()
    for name in ('adapter_config.json','adapter_model.safetensors','training_config.json'):
        if not (path/name).is_file() or not (path/name).stat().st_size:
            raise ValueError(f'Adapter checkpoint is missing {name}')
    adapter=json.loads((path/'adapter_config.json').read_text())
    saved=json.loads((path/'training_config.json').read_text())
    if adapter.get('peft_type')!='LORA' or adapter.get('r')!=config['lora_rank']:
        raise ValueError('Checkpoint LoRA type/rank does not match the training configuration')
    if adapter.get('lora_dropout')!=config['lora_dropout']:
        raise ValueError('Checkpoint LoRA dropout does not match the training configuration')
    if saved.get('source_model_path',saved.get('vla_path'))!=str(Path(model).resolve()):
        raise ValueError('Checkpoint must use the same original unmerged base-model path')
    return path


def attach_lora(model,lora_config):
    from peft import PeftModel,get_peft_model
    checkpoint=os.environ.get('PANDA_INITIAL_ADAPTER','')
    if checkpoint:
        print('Continuing from adapter weights with a fresh optimizer:',checkpoint,flush=True)
        return PeftModel.from_pretrained(model,checkpoint,is_trainable=True)
    return get_peft_model(model,lora_config)
