"""Select a measured microbatch only for a compatible GPU/input/model setup."""
import hashlib
import json
from pathlib import Path


def benchmark_context(config,dataset,model):
    import torch
    import transformers
    dataset=Path(dataset);model=Path(model)
    if not torch.cuda.is_available():
        return None
    required=[dataset/'input_check.json',model/'config.json',model/'export_complete.json']
    if not all(p.is_file() for p in required):
        return None
    checks=json.loads(required[0].read_text())
    batches=checks.get('official_processor_batches',{}).values()
    contracts=[dict(pixel_shape=b['pixel_shape'],supervised_tokens=b['supervised_tokens']) for b in batches]
    if not checks.get('passed') or not contracts or any(c!=contracts[0] for c in contracts):
        return None
    gpu=torch.cuda.get_device_properties(torch.cuda.current_device())
    return dict(gpu=gpu.name,total_vram_bytes=gpu.total_memory,
        compute_capability=list(torch.cuda.get_device_capability()),
        torch_version=torch.__version__,transformers_version=transformers.__version__,
        cuda_runtime=torch.version.cuda,input_contract=contracts[0],model_path=str(model.resolve()),
        model_identity_sha256=hashlib.sha256(required[1].read_bytes()+b'\n'+required[2].read_bytes()).hexdigest(),
        precision='bfloat16',attention='eager',
        settings={k:config[k] for k in ('lora_rank','lora_dropout','use_lora','use_quantization','image_aug')})


DEFAULT_INPUT_CONTRACT=dict(pixel_shape=[6,224,224],supervised_tokens=8)


def compatible_context(saved,current,allow_reduced_memory=False):
    """Dataset identity is provenance, not a batch-capacity requirement."""
    if not isinstance(saved,dict):return False
    def normalized(value):
        result={k:v for k,v in value.items()
                if k not in ('dataset_checksums_sha256','model_path')}
        # Earlier context-bearing benchmarks used this fixed Panda input format.
        result.setdefault('input_contract',DEFAULT_INPUT_CONTRACT)
        return result
    a,b=normalized(saved),normalized(current)
    if allow_reduced_memory:
        old,new=a.pop('total_vram_bytes',None),b.pop('total_vram_bytes',None)
        if not old or not new or new>old:return False
    return a==b


def memory_screen(row,saved_gib,current_gib):
    """Conservatively project measured headroom onto a smaller-memory same GPU."""
    if abs(saved_gib-current_gib)<0.01:return row.get('headroom_screen_passed') is True
    if current_gib>saved_gib:return False
    try:
        headroom=min(current_gib-row['gpu_peak_reserved_gib'],
                     row['minimum_logged_free_gib']-(saved_gib-current_gib))
        return headroom>=4
    except (KeyError,TypeError):return False


def compatible_reviewed_legacy(root,metadata,row,current):
    """Reuse the user's reviewed A40 run, which predates context metadata.

    Check its original candidate artifacts; arbitrary legacy runs remain excluded.
    Transformer/CUDA/attention settings were established by its pinned training
    image, not recorded independently in these older artifacts.
    """
    if current.get('input_contract')!=DEFAULT_INPUT_CONTRACT:return False
    if current.get('transformers_version')!='4.40.1' or current.get('cuda_runtime')!='12.1':return False
    if current.get('precision')!='bfloat16' or current.get('attention')!='eager':return False
    if metadata.get('model')!=current.get('model_path'):return False
    name=f"batch_{row['microbatch']}_accum_{row['accumulation']}"
    try:
        info=json.loads((root/name/'model_info.json').read_text())
        cfg=info['config']
        return (info['gpu']==current['gpu'] and info['torch_version']==current['torch_version']
            and memory_screen(row,info['total_vram_gib'],current['total_vram_bytes']/2**30)
            and cfg['vla_path']==current['model_path']
            and all(cfg[k]==v for k,v in current['settings'].items())
            and cfg['batch_size']==row['microbatch'] and cfg['grad_accumulation_steps']==1
            and cfg['dataset_name']=='panda_grasp_v1')
    except (OSError,ValueError,KeyError,TypeError):return False


def choose_batch(config,context,benchmark_root):
    selected=dict(config)
    record=dict(policy='auto',source='configured_fallback',
        microbatch=config['batch_size'],accumulation=config['grad_accumulation_steps'],
        effective_batch=config['batch_size']*config['grad_accumulation_steps'],
        reason='No compatible GPU/input/model benchmark; using the explicit configuration.')
    if context is None:
        record['reason']='GPU or input/model identity unavailable; using the explicit configuration.'
        return selected,record
    candidates=[]
    for path in sorted(Path(benchmark_root).glob('*/benchmark_metadata.json')):
        try:
            metadata=json.loads(path.read_text())
            if metadata.get('mode')!='capacity':continue
            saved_context=metadata.get('context')
            compatible=compatible_context(saved_context,context) if saved_context else False
            reduced=(compatible_context(saved_context,context,allow_reduced_memory=True)
                     if saved_context and not compatible else False)
            reviewed_legacy=(not saved_context and path.parent.name=='20261011_054257_777818')
            if not compatible and not reduced and not reviewed_legacy:continue
            for row in json.loads((path.parent/'benchmark.json').read_text()):
                if (row.get('status')=='completed' and row.get('headroom_screen_passed') is True
                    and row.get('accumulation')==1 and row.get('microbatch',0)>=1
                    and row.get('examples_per_second',0)>0):
                    target=config.get('target_effective_batch')
                    if target and (row['microbatch']>target or target%row['microbatch']):continue
                    projected=(reduced and memory_screen(row,saved_context['total_vram_bytes']/2**30,context['total_vram_bytes']/2**30))
                    if compatible or projected or (reviewed_legacy and compatible_reviewed_legacy(path.parent,metadata,row,context)):
                        candidates.append((row['examples_per_second'],path,row))
        except (OSError,ValueError,KeyError,TypeError):
            continue  # An incomplete/malformed benchmark cannot configure training.
    if candidates:
        speed,path,row=max(candidates,key=lambda item:(item[0],str(item[1])))
        selected.update(batch_size=row['microbatch'],grad_accumulation_steps=1)
        if config.get('target_effective_batch'):
            selected['grad_accumulation_steps']=config['target_effective_batch']//row['microbatch']
        record.update(source='compatible_benchmark',microbatch=row['microbatch'],accumulation=selected['grad_accumulation_steps'],
            effective_batch=row['microbatch']*selected['grad_accumulation_steps'],examples_per_second=speed,
            benchmark=str(path.parent),context=context,
            compatibility_basis='gpu_model_training_and_input_format',
            reason='Fastest measured candidate passing the current-VRAM headroom screen; accumulation preserves the configured effective batch when requested.',
            throughput_note='Measured at accumulation one on the benchmark pod; not a new measurement on this pod.')
    return selected,record
