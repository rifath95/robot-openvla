"""Audit exported observations/labels and test official OpenVLA batch preparation.

Loads processor/tokenizer only, never the 7B model. The official batch transform
is isolated from its package's OXE imports so this check runs on a CPU Mac.
"""
import argparse
import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, Type

import numpy as np
from PIL import Image

from prepare_training_dataset import load_episodes, training_statistics, normalize, verify


def official_batch_transform(processor):
    import torch
    from transformers import PreTrainedTokenizerBase
    from training.upstream.prompting import PurePromptBuilder
    root=Path('training/upstream')
    tokenizer_scope={'np':np,'List':list,'Union':__import__('typing').Union,
                     'PreTrainedTokenizerBase':PreTrainedTokenizerBase}
    exec(compile((root/'action_tokenizer.py').read_text(),str(root/'action_tokenizer.py'),'exec'),tokenizer_scope)
    tree=ast.parse((root/'datasets.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RLDSBatchTransform')
    scope=dict(dataclass=dataclass,Type=Type,Any=Any,Dict=Dict,np=np,torch=torch,Image=Image,
               ActionTokenizer=tokenizer_scope['ActionTokenizer'],PreTrainedTokenizerBase=PreTrainedTokenizerBase,
               ImageTransform=Any,PromptBuilder=Any,IGNORE_INDEX=-100)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(root/'datasets.py'),'exec'),scope)
    return scope['RLDSBatchTransform'](tokenizer_scope['ActionTokenizer'](processor.tokenizer),
        processor.tokenizer,processor.image_processor.apply_transform,PurePromptBuilder)


def check(dataset, source, processor_path=None):
    import tensorflow_datasets as tfds
    result=verify(dataset)
    episodes=load_episodes(source)
    expected={e['id']:e for e in episodes}
    manifest=json.loads((dataset/'manifest.json').read_text())
    stats=json.loads((dataset/'dataset_statistics.json').read_text())
    if stats!=training_statistics(episodes):raise ValueError('Statistics do not match training-only sources')
    builder=tfds.builder_from_directory(str(dataset/manifest['builder_directory']))
    samples={}
    for split in ('train','validation','test'):
        for ep in tfds.as_numpy(builder.as_dataset(split=split,shuffle_files=False)):
            source_episode=expected[ep['episode_id'].decode()]
            if source_episode['split']!=split:raise ValueError('Changed split')
            rows=[row for row in ep['steps'] if row['is_action_valid']]
            if len(rows)!=len(source_episode['rows']):raise ValueError('Dropped or duplicated steps')
            for row,original in zip(rows,source_episode['rows']):
                with Image.open(source_episode['path']/original['observation']) as im:
                    if not np.array_equal(row['observation']['image'],np.asarray(im)):
                        raise ValueError('Image/action alignment mismatch')
                if (not np.array_equal(row['action'],np.asarray(original['action'],np.float32))
                        or row['language_instruction'].decode()!=original['instruction']):
                    raise ValueError('Label/instruction mismatch')
            samples[split]=rows[0]
    result['all_source_images_actions_instructions_match']=True
    if processor_path is not None:
        os.environ.setdefault('HF_MODULES_CACHE',str(Path('.cache-openvla/modules').resolve()))
        from transformers import AutoProcessor
        from training.panda_rlds import PandaRLDSDataset
        processor=AutoProcessor.from_pretrained(str(processor_path),trust_remote_code=True,local_files_only=True)
        transform=official_batch_transform(processor)
        checked={}
        for split in samples:
            ds=PandaRLDSDataset(dataset,'panda_grasp_v1',transform,train=False,split=split)
            batch=next(iter(ds))
            if batch['pixel_values'].shape!=(6,224,224):raise ValueError('Unexpected fused vision input shape')
            if (batch['labels']!=-100).sum()!=8:raise ValueError('Expected seven action labels plus stop token')
            if not __import__('torch').isfinite(batch['pixel_values']).all():raise ValueError('Nonfinite pixels')
            checked[split]=dict(pixel_shape=list(batch['pixel_values'].shape),
                supervised_tokens=int((batch['labels']!=-100).sum()),input_tokens=len(batch['input_ids']))
        result['official_processor_batches']=checked
    (dataset/'input_check.json').write_text(json.dumps(result,indent=2)+'\n')
    # Include the check report in the upload integrity manifest.
    verify(dataset)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('datasets/panda_pick_place/training_v1'))
    p.add_argument('--source-manifest',type=Path,default=Path('datasets/panda_pick_place/panda_expansion_20261010_104720/combined_manifest.json'))
    p.add_argument('--processor',type=Path,help='Existing exported model directory or local HF snapshot (weights are not loaded)')
    args=p.parse_args(); print(json.dumps(check(args.dataset,args.source_manifest,args.processor),indent=2))
