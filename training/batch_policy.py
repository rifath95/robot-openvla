"""Select a measured microbatch only for a matching GPU/data/model setup."""
import hashlib
import json
from pathlib import Path


def benchmark_context(config,dataset,model):
    import torch
    import transformers
    dataset=Path(dataset);model=Path(model)
    if not torch.cuda.is_available():
        return None
    required=[dataset/'SHA256SUMS',model/'config.json',model/'export_complete.json']
    if not all(p.is_file() for p in required):
        return None
    gpu=torch.cuda.get_device_properties(torch.cuda.current_device())
    return dict(gpu=gpu.name,total_vram_bytes=gpu.total_memory,
        compute_capability=list(torch.cuda.get_device_capability()),
        torch_version=torch.__version__,transformers_version=transformers.__version__,
        cuda_runtime=torch.version.cuda,dataset_checksums_sha256=hashlib.sha256(required[0].read_bytes()).hexdigest(),
        model_identity_sha256=hashlib.sha256(required[1].read_bytes()+b'\n'+required[2].read_bytes()).hexdigest(),
        precision='bfloat16',attention='eager',
        settings={k:config[k] for k in ('lora_rank','lora_dropout','use_lora','use_quantization','image_aug')})


def choose_batch(config,context,benchmark_root):
    selected=dict(config)
    record=dict(policy='auto',source='configured_fallback',
        microbatch=config['batch_size'],accumulation=config['grad_accumulation_steps'],
        effective_batch=config['batch_size']*config['grad_accumulation_steps'],
        reason='No compatible GPU/dataset/model benchmark; using the explicit configuration.')
    if context is None:
        record['reason']='GPU or dataset/model identity unavailable; using the explicit configuration.'
        return selected,record
    candidates=[]
    for path in sorted(Path(benchmark_root).glob('*/benchmark_metadata.json')):
        try:
            metadata=json.loads(path.read_text())
            if metadata.get('context')!=context or metadata.get('mode')!='capacity':continue
            for row in json.loads((path.parent/'benchmark.json').read_text()):
                if (row.get('status')=='completed' and row.get('headroom_screen_passed') is True
                    and row.get('accumulation')==1 and row.get('microbatch',0)>=1
                    and row.get('examples_per_second',0)>0):
                    candidates.append((row['examples_per_second'],path,row))
        except (OSError,ValueError,KeyError,TypeError):
            continue  # An incomplete/malformed benchmark cannot configure training.
    if candidates:
        speed,path,row=max(candidates,key=lambda item:(item[0],str(item[1])))
        selected.update(batch_size=row['microbatch'],grad_accumulation_steps=1)
        record.update(source='compatible_benchmark',microbatch=row['microbatch'],accumulation=1,
            effective_batch=row['microbatch'],examples_per_second=speed,
            benchmark=str(path.parent),context=context,
            reason='Fastest measured candidate passing the memory-headroom screen; accumulation one baseline.')
    return selected,record
