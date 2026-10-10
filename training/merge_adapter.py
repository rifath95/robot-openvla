"""Merge an adapter once after training and attach Panda action statistics."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Output must be a new directory')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq,AutoProcessor
    model=AutoModelForVision2Seq.from_pretrained(str(args.base_model),trust_remote_code=True,
        local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='eager')
    merged=PeftModel.from_pretrained(model,str(args.adapter)).merge_and_unload()
    stats=json.loads((args.adapter/'dataset_statistics.json').read_text())
    merged.norm_stats.update(stats);merged.config.norm_stats=merged.norm_stats
    merged.save_pretrained(args.output)
    AutoProcessor.from_pretrained(str(args.base_model),trust_remote_code=True,local_files_only=True).save_pretrained(args.output)
    for file in args.base_model.glob('*.py'):shutil.copy2(file,args.output/file.name)
    shutil.copy2(args.adapter/'dataset_statistics.json',args.output/'dataset_statistics.json')
    print('Merged model:',args.output,'; inference unnorm_key: panda_grasp_v1')


if __name__=='__main__':main()
