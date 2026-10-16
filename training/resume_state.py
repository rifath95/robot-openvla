"""Portable final-checkpoint optimizer, RNG and input-stream state (single rank)."""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def _cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu(v) for v in value)
    return value


def save_state(folder,model,optimizer,dataset,update,cfg):
    """Called at an optimizer boundary, before checkpoint-ready metadata exists."""
    root=Path(folder)/'training_state';root.mkdir()
    numpy_state=np.random.get_state()
    state=dict(format_version=1,update=update,optimizer= _cpu(optimizer.state_dict()),
        optimizer_class=type(optimizer).__name__,input_root=str(dataset.root.resolve()) if hasattr(dataset,'root') else None,
        trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
        batch_size=cfg.batch_size,grad_accumulation_steps=cfg.grad_accumulation_steps,
        scheduler=None,scheduler_note='Constant learning rate; no scheduler in this trainer',
        python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        numpy_rng=(numpy_state[0],torch.tensor(numpy_state[1].astype('int64')),int(numpy_state[2]),int(numpy_state[3]),float(numpy_state[4])))
    dataset.save_input_state(root/'input')
    torch.save(state,root/'state.pt')
    records=[]
    for p in sorted(root.rglob('*')):
        if p.is_file():
            with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
            records.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha))
    (root/'manifest.json').write_text(json.dumps(dict(format_version=1,update=update,files=records),indent=2)+'\n')


def verify_state(folder,expected_update=None):
    root=Path(folder)/'training_state';manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('format_version')!=1 or (expected_update is not None and manifest['update']!=expected_update):raise ValueError('Resume update/format mismatch')
    names=set()
    for entry in manifest['files']:
        rel=Path(entry['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe resume-state path')
        p=root/rel
        if p.is_symlink() or p.stat().st_size!=entry['bytes']:raise ValueError('Incomplete resume state')
        with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        if sha!=entry['sha256']:raise ValueError('Resume-state checksum mismatch')
        names.add(rel.as_posix())
    if not {'state.pt','input.index'}<=names or not any(n.startswith('input.data-') for n in names):raise ValueError('Missing optimizer or input-stream state')
    return manifest


INPUT_CACHE_ROOTS=(Path('/root/panda-training-work'),Path('/root/panda-chain-work'),Path('/root/panda-training-cache'))


def ensure_input_alias(saved_root,current_root):
    """TF iterator snapshots retain absolute TFRecord filenames from staging."""
    if not saved_root or current_root is None:return
    saved=Path(saved_root);current=Path(current_root).resolve()
    if saved.resolve()==current:return
    if (not saved.is_absolute() or '..' in saved.parts or
        not any(saved.is_relative_to(parent) and saved!=parent for parent in INPUT_CACHE_ROOTS)):
        raise ValueError('Resume input iterator requires dataset at its original path: '+str(saved))
    if saved.exists() or saved.is_symlink():
        raise ValueError('Original input path already exists; use it as --dataset or relocate it explicitly: '+str(saved))
    saved.parent.mkdir(parents=True,exist_ok=True)
    saved.symlink_to(current,target_is_directory=True)


def restore_state(folder,model,optimizer,dataset,cfg):
    from training.update_numbering import initial_update
    verify_state(folder,initial_update())
    root=Path(folder)/'training_state'
    state=torch.load(root/'state.pt',map_location='cpu',weights_only=True)
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    if (state['update']!=initial_update() or state['trainable_names']!=names or
        state['optimizer_class']!=type(optimizer).__name__ or state['batch_size']!=cfg.batch_size or
        state['grad_accumulation_steps']!=cfg.grad_accumulation_steps):raise ValueError('Resume model, optimizer or batch configuration differs')
    optimizer.load_state_dict(state['optimizer'])
    ensure_input_alias(state.get('input_root'),getattr(dataset,'root',None))
    dataset.restore_input_state(root/'input')
    random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
    nr=state['numpy_rng'];np.random.set_state((nr[0],nr[1].numpy().astype('uint32'),nr[2],nr[3],nr[4]))
    if state['cuda_rng']:
        if not torch.cuda.is_available() or len(state['cuda_rng'])!=torch.cuda.device_count():raise ValueError('Resume CUDA device count differs')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    print('Restored optimizer, RNG and input stream at update',state['update'],flush=True)


def restore_if_requested(model,optimizer,dataset,cfg):
    path=os.environ.get('PANDA_RESUME_CHECKPOINT')
    if path:restore_state(path,model,optimizer,dataset,cfg)
