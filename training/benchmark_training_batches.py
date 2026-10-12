"""Compare microbatches on the real GPU using fresh short LoRA runs.

Run on the pod, after the dataset download. This does not stop pod billing.
"""

from common.paths import PROJECT_ROOT
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def summarize(rows, microbatch, accumulation, warmup=5):
    train=[r for r in rows if 'train_loss' in r]
    measured=[r for r in train if r['update'] >= warmup]
    if len(measured)<2:
        raise ValueError('Not enough completed updates after warmup')
    updates=measured[-1]['update']-measured[0]['update']
    seconds=measured[-1]['elapsed_seconds']-measured[0]['elapsed_seconds']
    if seconds<=0:
        raise ValueError('Update timing must be positive')
    result=dict(microbatch=microbatch,accumulation=accumulation,
        effective_batch=microbatch*accumulation,measured_updates=updates,
        seconds_per_update=seconds/updates,
        examples_per_second=updates*microbatch*accumulation/seconds)
    for key in ('gpu_peak_allocated_gib','gpu_peak_reserved_gib'):
        result[key]=max(r[key] for r in train)
    result['minimum_logged_free_gib']=min(r['gpu_free_gib'] for r in train)
    # Reserved peak accounts for the allocator cache; other processes and
    # untracked allocations mean this is a screening rule, not a guarantee.
    result['reserved_headroom_gib']=min(r['gpu_total_gib'] for r in train)-result['gpu_peak_reserved_gib']
    result['headroom_screen_passed']=result['reserved_headroom_gib']>=4 and result['minimum_logged_free_gib']>=4
    return result


def next_capacity_batch(results, maximum=64):
    """Double until OOM, then binary-search the fitting/failing interval."""
    if maximum<1:
        raise ValueError('Maximum batch must be positive')
    if any(r['status'] not in ('completed','out_of_memory') for r in results):
        raise ValueError('Capacity search cannot continue after a non-OOM failure')
    fitting=[r['microbatch'] for r in results if r['status']=='completed']
    failing=[r['microbatch'] for r in results if r['status']=='out_of_memory']
    low=max(fitting,default=0)
    if failing:
        high=min(failing)
        if high<=low:
            raise ValueError('Inconsistent fit results; inspect the GPU before retrying')
        return None if high-low<=1 else (low+high)//2
    if low>=maximum:
        return None
    return min(maximum,low*2) if low else 1


def missed_speed_candidates(results):
    """Refine above the largest power of two passing the headroom screen.

    Skipping smaller sizes is a search-cost heuristic, not a throughput guarantee.
    Without a passing anchor, conservatively start at one.
    """
    fitting=[r['microbatch'] for r in results if r['status']=='completed']
    if not fitting:
        return []
    largest=max(fitting)
    anchors=[r['microbatch'] for r in results
             if r['status']=='completed' and r.get('headroom_screen_passed') is True
             and r['microbatch']>0 and r['microbatch'] & (r['microbatch']-1)==0]
    lower=max(anchors,default=1)
    tested={r['microbatch'] for r in results}
    return [batch for batch in range(lower,largest+1) if batch not in tested]


def write_report(root, results, mode='controlled', maximum=64):
    (root/'benchmark.json').write_text(json.dumps(results,indent=2)+'\n')
    description=('Fresh LoRA runs from the same base model; effective batch stays at four examples.'
                 if mode=='controlled' else
                 'Adaptive fit test: accumulation one; effective batch changes with microbatch. This compares hardware throughput, not training convergence.')
    lines=['# Training batch benchmark','',description,
        'First five updates excluded from throughput. Validation excluded. Adapter checkpoint saving is disabled for benchmark candidates.',
        'Timings include data loading, forward/backward, optimizer and normal training logging.',
        'These short runs measure throughput, not model quality. No benchmark adapter is promoted.', '',
        '| Microbatch | Accumulation | Effective batch | Status | Seconds/update | Examples/s | Peak allocated GiB | Peak reserved GiB | Headroom screen |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    metadata_path=root/'benchmark_metadata.json'
    metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    from training.gpu_ecc import ecc_details
    info_path=next(iter(sorted(root.glob('batch_*/model_info.json'))),None)
    info=json.loads(info_path.read_text()) if info_path else {}
    gpu=(metadata.get('context') or {}).get('gpu')
    if not gpu:
        gpu=info.get('gpu')
    lines[2:2]=[f"GPU: **{gpu or 'Unknown'}**.",
                ecc_details(metadata.get('gpu_ecc')), '']
    from training.benchmark_software import software_lines
    host_path=root/'batch_host_context.json'
    host=json.loads(host_path.read_text()) if host_path.exists() else {}
    lines[-3:-3]=software_lines(metadata,info,host)
    for row in results:
        if row['status']=='completed':
            lines.append(f"| {row['microbatch']} | {row['accumulation']} | {row['effective_batch']} | completed | {row['seconds_per_update']:.3f} | {row['examples_per_second']:.2f} | {row['gpu_peak_allocated_gib']:.2f} | {row['gpu_peak_reserved_gib']:.2f} | {row['headroom_screen_passed']} |")
        else:
            lines.append(f"| {row['microbatch']} | {row['accumulation']} | {row['microbatch']*row['accumulation']} | {row['status']} | — | — | — | — | — |")
    if mode=='capacity':
        fitting=[r['microbatch'] for r in results if r['status']=='completed']
        failing=[r['microbatch'] for r in results if r['status']=='out_of_memory']
        lines+=['',f'Largest tested fitting microbatch: {max(fitting,default=0)}.',
                f'Smallest tested OOM microbatch: {min(failing) if failing else "not observed"}.',
                f'Search bound: {maximum}. If the bound fits, capacity is at least that size; the maximum was not established.',
                'Fit results assume approximately monotonic memory demand. They apply to the tested data/settings and short runs, not every possible instruction length.']
    eligible=[r for r in results if r['status']=='completed' and r['headroom_screen_passed']]
    if eligible:
        best=max(eligible,key=lambda r:r['examples_per_second'])
        lines+=['',f"Fastest candidate passing the 4 GiB headroom screen: microbatch {best['microbatch']}, accumulation {best['accumulation']}.",
                'Confirm stability in a longer trial before choosing it for the full training run.']
    else:
        lines+=['','No candidate passed the memory headroom screen; inspect logs before training.']
    (root/'BATCH_BENCHMARK.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('/workspace/panda-training/training_v2'))
    p.add_argument('--model',type=Path,default=Path('/workspace/openvla-7b'))
    p.add_argument('--output-root',type=Path,default=Path('/workspace/panda-training/batch-benchmarks'))
    p.add_argument('--find-capacity',action='store_true',help='Double until OOM, then refine the boundary; accumulation one')
    p.add_argument('--max-batch',type=int,default=64,help='Bound the adaptive search (default: 64)')
    args=p.parse_args()
    if args.max_batch<1:
        p.error('--max-batch must be positive')
    benchmark_started=time.monotonic()
    verification=args.dataset/'verification.json'
    if not verification.is_file() or not json.loads(verification.read_text()).get('passed'):
        p.error(f'Download and verify the selected dataset first: {args.dataset}')
    root=args.output_root/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path('training/pick_place_1000.json').read_text())
    results=[]
    mode='capacity' if args.find_capacity else 'controlled'
    from training.batch_policy import benchmark_context
    context=benchmark_context(config,args.dataset,args.model)
    from training.benchmark_software import collect_software
    from training.gpu_ecc import collect_ecc
    import torch
    gpu_uuid=getattr(torch.cuda.get_device_properties(0),'uuid',None)
    metadata=dict(mode=mode,dataset=str(args.dataset.resolve()),model=str(args.model.resolve()),
        max_batch=args.max_batch,updates_per_candidate=25,warmup_updates=5,context=context,
        checkpoint_saving=False,gpu_ecc=collect_ecc(str(gpu_uuid) if gpu_uuid else None),
        software_stack=collect_software(context),
        speed_refinement_policy='largest_headroom_power_of_two_to_capacity' if args.find_capacity else None)
    if os.environ.get('PANDA_BENCHMARK_STAGING_METADATA'):
        metadata['storage']=json.loads(Path(os.environ['PANDA_BENCHMARK_STAGING_METADATA']).read_text())
    def report(outcome='running'):
        metadata.update(benchmark_elapsed_seconds=time.monotonic()-benchmark_started,benchmark_outcome=outcome)
        (root/'benchmark_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        write_report(root,results,mode,args.max_batch)
        from analysis.summarize_gpu_benchmarks import comparison
        comparison(args.output_root,args.output_root/'GPU_TRAINING_COMPARISON.md')
    print(f'Benchmark outputs: {root}',flush=True)
    print('This benchmark alone does not stop the pod; the Mac launcher handles download and automatic stop when enabled.',flush=True)
    if (os.environ.get('OPENVLA_TRAINING_STACK')=='blackwell'
            or Path('/opt/training/build_stack_check.json').is_file()):
        print('Checking Blackwell training dependencies and a tiny CUDA LoRA update...',flush=True)
        try:
            subprocess.run([sys.executable,str(PROJECT_ROOT/'training/check_training_stack.py'),
                            '--device','cuda','--expect-blackwell-stack',
                            '--output',str(root/'batch_stack_check.json')],check=True)
        except subprocess.CalledProcessError:
            report('stack_check_failed')
            raise
    controlled=iter(((1,4),(2,2),(4,1)))
    speed_candidates=None
    while True:
        if args.find_capacity:
            if speed_candidates is None:
                microbatch=next_capacity_batch(results,args.max_batch)
                if microbatch is None:
                    speed_candidates=iter(missed_speed_candidates(results))
                    print('Capacity search finished; measuring missing sizes from the largest power-of-two batch with headroom up to capacity...',flush=True)
            if speed_candidates is not None:
                microbatch=next(speed_candidates,None)
            if microbatch is None:break
            accumulation=1
        else:
            pair=next(controlled,None)
            if pair is None:
                break
            microbatch,accumulation=pair
        name=f'batch_{microbatch}_accum_{accumulation}'
        candidate=dict(config,batch_size=microbatch,grad_accumulation_steps=accumulation,
            max_steps=25,save_steps=25,validation_every=25,validation_max_batches=1,
            max_runtime_seconds=600)
        config_path=root/f'{name}.json'
        config_path.write_text(json.dumps(candidate,indent=2)+'\n')
        command=[sys.executable,'-u','train_panda.py','--dataset',str(args.dataset),
            '--model',str(args.model),'--config',str(config_path),
            '--output-root',str(root),'--run-name',name,'--batch-policy','config','--no-checkpoints']
        print(f'Testing microbatch {microbatch}, accumulation {accumulation} (25 updates)...',flush=True)
        started=time.monotonic()
        with (root/f'{name}.log').open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                process.wait(timeout=900)
                status='completed' if process.returncode==0 else 'failed'
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
                results.append(dict(microbatch=microbatch,accumulation=accumulation,status='timeout',wall_seconds=time.monotonic()-started))
                report('timeout')
                raise SystemExit('Benchmark timed out. Stop the pod; do not start another GPU job.')
            except KeyboardInterrupt:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise
        row=dict(microbatch=microbatch,accumulation=accumulation,status=status,
                 phase='speed_refinement' if speed_candidates is not None else 'capacity' if args.find_capacity else 'controlled',
                 wall_seconds=time.monotonic()-started)
        if status=='completed':
            rows=[json.loads(line) for line in (root/name/'metrics.jsonl').read_text().splitlines()]
            if max((r['update'] for r in rows if 'train_loss' in r),default=0)!=25:
                results.append(dict(row,status='incomplete'));report('incomplete')
                raise SystemExit('Candidate ended before 25 updates; capacity was not established.')
            row.update(summarize(rows,microbatch,accumulation))
            info=json.loads((root/name/'model_info.json').read_text())
            row['gpu']=info['gpu'];row['torch_version']=info['torch_version']
        else:
            log_text=(root/f'{name}.log').read_text()
            if 'CUDA out of memory' in log_text or 'torch.cuda.OutOfMemoryError' in log_text:
                row['status']='out_of_memory'
            else:
                results.append(row);report('failed')
                raise SystemExit(f'Benchmark failed; inspect {root / (name + ".log")}')
        results.append(row);report()
        print(json.dumps(row),flush=True)
    report('completed')
    print(f'Report: {root / "BATCH_BENCHMARK.md"}',flush=True)


if __name__=='__main__':
    main()
