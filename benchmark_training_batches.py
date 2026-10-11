"""Compare microbatches on the real GPU using fresh short LoRA runs.

Run on the pod, after the dataset download. This does not stop pod billing.
"""
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


def write_report(root, results):
    (root/'benchmark.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=['# A40 training batch benchmark','',
        'Fresh LoRA runs from the same base model; effective batch stays at four examples.',
        'First five updates excluded from throughput. Final validation and checkpoint writes excluded.',
        'Timings include data loading, forward/backward, optimizer and normal training logging.',
        'These short runs measure throughput, not model quality. No benchmark adapter is promoted.', '',
        '| Microbatch | Accumulation | Status | Seconds/update | Examples/s | Peak allocated GiB | Peak reserved GiB | Headroom screen |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in results:
        if row['status']=='completed':
            lines.append(f"| {row['microbatch']} | {row['accumulation']} | completed | {row['seconds_per_update']:.3f} | {row['examples_per_second']:.2f} | {row['gpu_peak_allocated_gib']:.2f} | {row['gpu_peak_reserved_gib']:.2f} | {row['headroom_screen_passed']} |")
        else:
            lines.append(f"| {row['microbatch']} | {row['accumulation']} | {row['status']} | — | — | — | — | — |")
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
    args=p.parse_args()
    verification=args.dataset/'verification.json'
    if not verification.is_file() or not json.loads(verification.read_text()).get('passed'):
        p.error('Download and verify training_v2 first')
    root=args.output_root/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path('training/pick_place_1000.json').read_text())
    results=[]
    print(f'Benchmark outputs: {root}',flush=True)
    print('This command does not stop the pod. Stop it in RunPod afterward.',flush=True)
    for microbatch,accumulation in ((1,4),(2,2),(4,1)):
        name=f'batch_{microbatch}_accum_{accumulation}'
        candidate=dict(config,batch_size=microbatch,grad_accumulation_steps=accumulation,
            max_steps=25,save_steps=25,validation_every=25,validation_max_batches=1,
            max_runtime_seconds=600)
        config_path=root/f'{name}.json'
        config_path.write_text(json.dumps(candidate,indent=2)+'\n')
        command=[sys.executable,'-u','train_panda.py','--dataset',str(args.dataset),
            '--model',str(args.model),'--config',str(config_path),
            '--output-root',str(root),'--run-name',name]
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
                results.append(dict(microbatch=microbatch,accumulation=accumulation,status='timeout'))
                write_report(root,results)
                raise SystemExit('Benchmark timed out. Stop the pod; do not start another GPU job.')
            except KeyboardInterrupt:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise
        row=dict(microbatch=microbatch,accumulation=accumulation,status=status,
                 wall_seconds=time.monotonic()-started)
        if status=='completed':
            rows=[json.loads(line) for line in (root/name/'metrics.jsonl').read_text().splitlines()]
            row.update(summarize(rows,microbatch,accumulation))
            info=json.loads((root/name/'model_info.json').read_text())
            row['gpu']=info['gpu'];row['torch_version']=info['torch_version']
        else:
            log_text=(root/f'{name}.log').read_text()
            if 'CUDA out of memory' in log_text or 'torch.cuda.OutOfMemoryError' in log_text:
                row['status']='out_of_memory'
            else:
                results.append(row);write_report(root,results)
                raise SystemExit(f'Benchmark failed; inspect {root / (name + ".log")}')
        results.append(row);write_report(root,results)
        print(json.dumps(row),flush=True)
    print(f'Report: {root / "BATCH_BENCHMARK.md"}',flush=True)


if __name__=='__main__':
    main()
