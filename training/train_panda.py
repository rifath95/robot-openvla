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
    p.add_argument('--provenance',type=Path,help='Chain dataset/model provenance copied into checkpoints')
    p.add_argument('--adapter',type=Path,help='Continue from LoRA weights in a new run with a fresh optimizer')
    p.add_argument('--initial-update',type=int,default=0,help='Cumulative starting update; requires matching adapter metadata')
    p.add_argument('--base-staging-receipt',type=Path,help='Verified base copy receipt when continuing from a staged base')
    p.add_argument('--no-checkpoints',action='store_true',help='Benchmark only: skip adapter saves while retaining metrics and reports')
    p.add_argument('--prepare-only',action='store_true',help='Print launch command without running training')
    p.add_argument('--run-name',help='Optional unique run name inside output-root (used by detached supervisor)')
    p.add_argument('--batch-policy',choices=('auto','config'),default=None,help='Reuse matching benchmark recommendation, or keep configured batch settings')
    p.add_argument('--benchmark-root',type=Path,default=Path('/workspace/panda-training/batch-benchmarks'))
    p.add_argument('--require-benchmark',action='store_true',help='Refuse training if no compatible automatic recommendation exists')
    args=p.parse_args()
    if not (args.dataset/'verification.json').is_file() or not json.loads((args.dataset/'verification.json').read_text())['passed']:
        p.error('Dataset must have a passing export verification')
    config=json.loads(args.config.read_text())
    from training.dataset_profile import validate_dataset, PROFILE_KEYS
    try:validate_dataset(config,args.dataset)
    except (ValueError,OSError) as exc:p.error(str(exc))
    for key in PROFILE_KEYS:config.pop(key,None)
    validation_enabled=config.pop('validation_enabled',True)
    if type(validation_enabled) is not bool:p.error('validation_enabled must be boolean')
    config.pop('minimum_free_disk_gib',None)
    config.pop('hard_limit_seconds',None)  # Supervisor option, not an upstream trainer flag.
    stage_local=config.pop('stage_inputs_on_container',False)
    if type(stage_local) is not bool:p.error('stage_inputs_on_container must be boolean')
    profile_requires_benchmark=config.pop('require_benchmark',False)
    require_benchmark=args.require_benchmark or profile_requires_benchmark
    expected_gpu=config.pop('expected_gpu',None)
    args.batch_policy=args.batch_policy or config.pop('batch_policy','auto')
    config.pop('batch_policy',None)
    if args.batch_policy not in ('auto','config'):p.error('batch_policy must be auto or config')
    adapter=None
    if args.adapter:
        from training.continuation import validate_adapter
        try:adapter=validate_adapter(args.adapter,config,args.model,staging_receipt=args.base_staging_receipt)
        except (ValueError,OSError) as exc:p.error(str(exc))
    from training.update_numbering import validate_initial_update
    try:validate_initial_update(adapter,args.initial_update)
    except (ValueError,OSError) as exc:p.error(str(exc))
    if args.base_staging_receipt and not adapter:p.error('--base-staging-receipt requires --adapter')
    for key in ['batch_size','grad_accumulation_steps','max_steps','save_steps','validation_every','max_runtime_seconds']:
        if config[key]<1:p.error(f'{key} must be positive')
    from training.checkpoint_schedule import validate_schedule
    try:
        checkpoint_updates=validate_schedule(config.pop('checkpoint_updates',None),config['max_steps'])
        validation_updates=validate_schedule(config.pop('validation_updates',None),config['max_steps'],name='validation_updates')
    except ValueError as exc:p.error(str(exc))
    if config['validation_max_batches']<0:p.error('validation_max_batches must be zero (full split) or positive')
    if not config['use_lora'] or config['use_quantization'] or config['image_aug']:
        p.error('This initial integration requires LoRA, unquantized weights, and disabled augmentation')
    if args.batch_policy=='auto':
        from training.batch_policy import benchmark_context,choose_batch
        context=benchmark_context(config,args.dataset,args.model)
        config,batch_selection=choose_batch(config,context,args.benchmark_root)
        if (require_benchmark and batch_selection['source']!='compatible_benchmark') or (
                expected_gpu and (context is None or context.get('gpu')!=expected_gpu)):
            args.output_root.mkdir(parents=True,exist_ok=True)
            diagnostic=args.output_root/'batch_compatibility.json'
            diagnostic.write_text(json.dumps(dict(batch_selection,expected_gpu=expected_gpu),indent=2)+'\n')
            print(f'Benchmark compatibility diagnostics: {diagnostic}',flush=True)
        if expected_gpu and (context is None or context.get('gpu')!=expected_gpu):
            p.error(f'This training profile requires {expected_gpu} with verified model/input metadata')
        if require_benchmark and batch_selection['source']!='compatible_benchmark':
            p.error('No compatible safe benchmark recommendation. Training was not started; see batch_compatibility.json for exact differences.')
    else:
        if require_benchmark:p.error('This profile requires automatic benchmark selection')
        if expected_gpu:
            import torch
            if not torch.cuda.is_available() or torch.cuda.get_device_properties(torch.cuda.current_device()).name!=expected_gpu:
                p.error(f'This training profile requires {expected_gpu}')
        batch_selection=dict(policy='config',source='configuration',microbatch=config['batch_size'],
            accumulation=config['grad_accumulation_steps'],
            effective_batch=config['batch_size']*config['grad_accumulation_steps'])
    print('Batch selection:',json.dumps(batch_selection),flush=True)
    pin=json.loads(Path('training/upstream/PIN.json').read_text())
    generated=Path('training/generated/panda_finetune.py').resolve()
    generated.parent.mkdir(parents=True,exist_ok=True);generated.write_text(integrated_source())
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in ('.','..')):
        p.error('Run name must be a single directory name')
    run=args.output_root.resolve()/(args.run_name or datetime.now(timezone.utc).strftime('lora_%Y%m%d_%H%M%S_%f'))
    if run.exists():p.error('Run already exists; choose a new run name')
    source_model,source_dataset=args.model.resolve(),args.dataset.resolve()
    staging_receipt=None
    if stage_local and not args.prepare_only:
        from training.input_staging import stage_inputs
        cache=Path('/root/panda-training-cache')/args.output_root.resolve().parent.name/run.name
        args.model,args.dataset,staging_receipt=stage_inputs(source_model,source_dataset,cache)
        run.mkdir(parents=True,exist_ok=False)
        (run/'input_staging.json').write_text(json.dumps(staging_receipt,indent=2)+'\n')
    elif stage_local:
        print('Container staging will copy model and dataset before training; outputs remain:',run,flush=True)
    env=os.environ.copy()
    env.update(PANDA_VALIDATION_ENABLED='1' if validation_enabled else '0',
        PANDA_PROVENANCE=args.provenance.read_text() if args.provenance else '',PANDA_SOURCE_MODEL=str(source_model),PANDA_SOURCE_DATASET=str(source_dataset),
        PANDA_INPUT_STAGING=json.dumps(staging_receipt) if staging_receipt else '',
        PANDA_RUN_DIR=str(run),PANDA_UPSTREAM_REVISION=pin['revision'],
        PANDA_CHECKPOINT_UPDATES=json.dumps(checkpoint_updates) if checkpoint_updates is not None else '',
        PANDA_VALIDATION_UPDATES=json.dumps(validation_updates) if validation_updates is not None else '',
        PANDA_VALIDATE_EVERY=str(config.pop('validation_every')),
        PANDA_VALIDATION_MAX_BATCHES=str(config.pop('validation_max_batches')),
        PANDA_MAX_RUNTIME=str(config.pop('max_runtime_seconds')),
        PANDA_BATCH_SELECTION=json.dumps(batch_selection),
        PANDA_INITIAL_ADAPTER=str(adapter) if adapter else '',
        PANDA_INITIAL_UPDATE=str(args.initial_update),
        PANDA_SAVE_CHECKPOINTS='0' if args.no_checkpoints else '1',
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
