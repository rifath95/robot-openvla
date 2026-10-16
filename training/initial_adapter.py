"""Restore the original untrained LoRA matrices from a pinned verified HF backup."""
import hashlib
import json
from pathlib import Path


def stage_initial_adapter(spec,local,profile,*,token=None):
    from huggingface_hub import hf_hub_download
    from cloud.checkpoint_ready import check_ready
    import torch
    from safetensors import safe_open
    if spec.get('checkpoint_update')!=0:raise ValueError('Initialization source must be checkpoint zero')
    prefix=spec['path_in_repo'];root=Path(local)/'initial-adapter-hf';folder=root/prefix
    if Path(prefix).is_absolute() or '..' in Path(prefix).parts:raise ValueError('Unsafe initialization prefix')
    names=set()
    for entry in spec['files']:
        relative=Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe initialization file')
        downloaded=Path(hf_hub_download(spec['repo_id'],prefix+'/'+relative.as_posix(),repo_type='model',
            revision=spec['revision'],local_dir=root,token=token))
        if downloaded.resolve()!=(folder/relative).resolve() or downloaded.is_symlink():raise ValueError('Unexpected initialization download location')
        with downloaded.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if downloaded.stat().st_size!=entry['bytes'] or sha!=entry['sha256']:raise ValueError('Initial adapter differs from original verified backup')
        names.add(relative.as_posix())
    if not {'adapter_model.safetensors','adapter_config.json','training_config.json'}<=names:raise ValueError('Incomplete initialization inventory')
    check_ready(folder)
    saved=json.loads((folder/'training_config.json').read_text());source=saved.get('dataset_provenance',{})
    if saved.get('checkpoint_update')!=0 or source.get('checksums_sha256')!=profile['checksums_sha256'] or source.get('dataset_revision')!=profile['dataset_revision']:
        raise ValueError('Checkpoint zero dataset/update provenance differs')
    cfg=json.loads((folder/'adapter_config.json').read_text())
    training=json.loads(Path(profile['training_config']).read_text())
    if cfg.get('peft_type')!='LORA' or cfg.get('r')!=training['lora_rank'] or cfg.get('lora_dropout')!=training['lora_dropout']:
        raise ValueError('Checkpoint zero LoRA settings differ')
    a_count=b_count=0
    with safe_open(folder/'adapter_model.safetensors',framework='pt',device='cpu') as tensors:
        for name in tensors.keys():
            if 'lora_B' in name:
                b_count+=1
                if torch.count_nonzero(tensors.get_tensor(name)).item():raise ValueError('Checkpoint zero has a nonzero LoRA B matrix')
            if 'lora_A' in name:
                a_count+=1
    if not a_count or a_count!=b_count:raise ValueError('Unexpected initial LoRA tensor inventory')
    identity=dict(repo_id=spec['repo_id'],revision=spec['revision'],path_in_repo=prefix,
        source_training_session=spec['source_training_session'],checkpoint_update=0,
        adapter_sha256=next(e['sha256'] for e in spec['files'] if e['path']=='adapter_model.safetensors'),
        exact_initial_adapter=True,zero_b_matrices_verified=b_count)
    return folder,identity
