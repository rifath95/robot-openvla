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
VRAM_TOLERANCE_BYTES=16*2**20


def normalized_context(value):
    result={k:v for k,v in value.items() if k not in ('dataset_checksums_sha256','model_path')}
    result.setdefault('input_contract',DEFAULT_INPUT_CONTRACT)
    return result


def context_differences(saved,current):
    if not isinstance(saved,dict):return {'context':dict(saved='Not recorded',current=current)}
    saved,current=normalized_context(saved),normalized_context(current)
    return {key:dict(saved=saved.get(key),current=current.get(key))
            for key in sorted(saved.keys()|current.keys()) if saved.get(key)!=current.get(key)
            or (key in saved)!=(key in current)}


def compatible_context(saved,current):
    """Dataset identity is provenance, not a batch-capacity requirement."""
    if not isinstance(saved,dict):return False
    differences=context_differences(saved,current)
    if not differences:return True
    if set(differences)!={'total_vram_bytes'}:return False
    a,b=saved.get('total_vram_bytes'),current.get('total_vram_bytes')
    return (type(a) is int and type(b) is int and a>0 and b>0
            and abs(a-b)<=VRAM_TOLERANCE_BYTES)


def adjusted_headroom(row,saved,current):
    """Recheck both memory margins when tolerating capacity accounting drift."""
    delta=(current['total_vram_bytes']-saved['total_vram_bytes'])/2**30
    if delta==0:return row
    adjusted=dict(row,vram_adjustment_gib=delta)
    for key in ('reserved_headroom_gib','minimum_logged_free_gib'):
        value=row.get(key)
        if not isinstance(value,(int,float)):return None
        adjusted[key]=value+delta
        if not adjusted[key]>=4:return None
    return adjusted


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
            and abs(info['total_vram_gib']-current['total_vram_bytes']/2**30)<0.01
            and cfg['vla_path']==current['model_path']
            and all(cfg[k]==v for k,v in current['settings'].items())
            and cfg['batch_size']==row['microbatch'] and cfg['grad_accumulation_steps']==1
            and cfg['dataset_name']=='panda_grasp_v1')
    except (OSError,ValueError,KeyError,TypeError):return False


def choose_batch(config,context,benchmark_root):
    selected=dict(config)
    record=dict(policy='auto',source='configured_fallback',
        context=context,benchmarks_considered=[],
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
            reviewed_legacy=(not saved_context and path.parent.name=='20261011_054257_777818')
            record['benchmarks_considered'].append(dict(benchmark=str(path.parent),
                compatible_context=compatible,reviewed_legacy=reviewed_legacy,
                differences=context_differences(saved_context,context)))
            if not compatible and not reviewed_legacy:continue
            for row in json.loads((path.parent/'benchmark.json').read_text()):
                if (row.get('status')=='completed' and row.get('headroom_screen_passed') is True
                    and row.get('accumulation')==1 and row.get('microbatch',0)>=1
                    and row.get('examples_per_second',0)>0):
                    if compatible:
                        row=adjusted_headroom(row,saved_context,context)
                        if row is None:continue
                        candidates.append((row['examples_per_second'],path,row))
                    elif compatible_reviewed_legacy(path.parent,metadata,row,context):
                        candidates.append((row['examples_per_second'],path,row))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            record['benchmarks_considered'].append(dict(benchmark=str(path.parent),error=type(exc).__name__))
            continue  # An incomplete/malformed benchmark cannot configure training.
    if candidates:
        speed,path,row=max(candidates,key=lambda item:(item[0],str(item[1])))
        selected.update(batch_size=row['microbatch'],grad_accumulation_steps=1)
        record.update(source='compatible_benchmark',microbatch=row['microbatch'],accumulation=1,
            effective_batch=row['microbatch'],examples_per_second=speed,
            benchmark=str(path.parent),context=context,
            compatibility_basis='gpu_model_training_and_input_format',
            reason='Fastest measured candidate passing the memory-headroom screen; accumulation one baseline.')
        if 'vram_adjustment_gib' in row:
            record.update(vram_tolerance_bytes=VRAM_TOLERANCE_BYTES,
                vram_adjustment_gib=row['vram_adjustment_gib'],
                adjusted_reserved_headroom_gib=row['reserved_headroom_gib'],
                adjusted_minimum_logged_free_gib=row['minimum_logged_free_gib'])
    return selected,record
