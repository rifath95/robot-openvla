"""Launch the project integration of OpenVLA's pinned official LoRA script."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from training.integrate_upstream import integrated_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('/workspace/openvla-7b'))
    p.add_argument('--output-root',type=Path,default=Path('/workspace/panda-training/runs'))
    p.add_argument('--config',type=Path,default=Path('training/smoke.json'))
    p.add_argument('--upstream',type=Path,default=Path('/opt/openvla-source'))
    p.add_argument('--prepare-only',action='store_true',help='Print launch command without running training')
    p.add_argument('--run-name',help='Optional unique run name inside output-root (used by detached supervisor)')
    args=p.parse_args()
    if not (args.dataset/'verification.json').is_file() or not json.loads((args.dataset/'verification.json').read_text())['passed']:
        p.error('Dataset must have a passing export verification')
    config=json.loads(args.config.read_text())
    for key in ['batch_size','grad_accumulation_steps','max_steps','save_steps','validation_every','max_runtime_seconds']:
        if config[key]<1:p.error(f'{key} must be positive')
    if config['validation_max_batches']<0:p.error('validation_max_batches must be zero (full split) or positive')
    if not config['use_lora'] or config['use_quantization'] or config['image_aug']:
        p.error('This initial integration requires LoRA, unquantized weights, and disabled augmentation')
    pin=json.loads(Path('training/upstream/PIN.json').read_text())
    generated=Path('training/generated/panda_finetune.py').resolve()
    generated.parent.mkdir(parents=True,exist_ok=True);generated.write_text(integrated_source())
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in ('.','..')):
        p.error('Run name must be a single directory name')
    run=args.output_root.resolve()/(args.run_name or datetime.now(timezone.utc).strftime('lora_%Y%m%d_%H%M%S_%f'))
    if run.exists():p.error('Run already exists; choose a new run name')
    env=os.environ.copy()
    env.update(PANDA_RUN_DIR=str(run),PANDA_UPSTREAM_REVISION=pin['revision'],
        PANDA_VALIDATE_EVERY=str(config.pop('validation_every')),
        PANDA_VALIDATION_MAX_BATCHES=str(config.pop('validation_max_batches')),
        PANDA_MAX_RUNTIME=str(config.pop('max_runtime_seconds')),
        PYTHONPATH=os.pathsep.join([str(Path.cwd()),str(args.upstream.resolve()),env.get('PYTHONPATH','')]),
        TF_CPP_MIN_LOG_LEVEL='2',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes','1','--nproc-per-node','1',str(generated),
        '--vla_path',str(args.model.resolve()),'--data_root_dir',str(args.dataset.resolve()),
        '--dataset_name','panda_grasp_v1','--run_root_dir',str(run/'upstream'),
        '--adapter_tmp_dir',str(run/'adapter-tmp')]
    for key,value in config.items():command.extend(['--'+key,str(value).lower() if isinstance(value,bool) else str(value)])
    print('Run directory:',run)
    print('Command:',__import__('shlex').join(command))
    print('Limits end this process; they do not terminate a RunPod pod or stop billing.')
    if not args.prepare_only:
        from training.fetch_upstream import fetch
        fetch(args.upstream)
        subprocess.run(command,env=env,check=True)


if __name__=='__main__':main()
