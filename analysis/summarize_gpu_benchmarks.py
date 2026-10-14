"""Build a readable GPU comparison from saved batch benchmark directories."""

from common.output_layout import output_path,resolve_output_path

from common.paths import PROJECT_ROOT
import argparse
import json
import math
import os
from pathlib import Path
from training.gpu_ecc import ecc_label,ecc_details
from training.benchmark_software import software_lines


def cell(value):
    return str(value).replace('|','/').replace('\n',' ')



def duration_line(rows,metadata):
    seconds=metadata.get('benchmark_elapsed_seconds')
    recorded=metadata.get('benchmark_outcome')
    if seconds is None:
        values=[row.get('wall_seconds') for row in rows]
        if not values or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values):
            return 'Total benchmark time: **not recorded**.'
        seconds=sum(values)
        note=' (sum of batch-test wall times).'
    else:
        if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0:
            return 'Total benchmark time: **not recorded**.'
        note='.'
    minutes,remainder=divmod(round(seconds),60)
    label='Benchmark time recorded so far' if recorded=='running' else 'Total benchmark time'
    return f'{label}: **{minutes} min {remainder:02d} sec**'+note



def cost_estimate(examples_per_second,hourly_rate,examples=10000):
    if examples_per_second<=0 or hourly_rate<=0:raise ValueError('Throughput and hourly rate must be positive')
    seconds=examples/examples_per_second
    return seconds,seconds/3600*hourly_rate


def cost_section(runs):
    rate_file=PROJECT_ROOT/'training/gpu_benchmark_rates.json'
    rates=json.loads(rate_file.read_text())['runs'] if rate_file.exists() else {}
    lines=['','## Cost for the same amount of training','',
        'Estimates below process **10,000 training examples**, using the recommended microbatch and measured throughput for each run.',
        'Time = examples ÷ examples/second. Cost = time in hours × hourly GPU rate.',
        'Rates are the user-confirmed prices for these particular pod runs, saved in `training/gpu_benchmark_rates.json`; they are not live quotes and are not assumed for later runs.', '',
        '| GPU | ECC | Run | GPU rate USD/hour | Examples/s | Time for 10,000 examples | Training cost USD |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    priced=[]
    exceptions=[]
    for gpu,path,best in runs:
        metadata_path=path.parent/'benchmark_metadata.json'
        metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        ecc=ecc_label(metadata.get('gpu_ecc'))
        rate=rates.get(path.parent.name,{})
        hourly=rate.get('hourly_rate_usd') if rate.get('gpu')==gpu else None
        ignoring_headroom = False
        if best is None:
            completed=[r for r in json.loads(path.read_text()) if r['status']=='completed']
            if completed:
                best=max(completed,key=lambda r:r['examples_per_second'])
                ignoring_headroom = True
        if best and isinstance(hourly,(int,float)) and math.isfinite(hourly) and hourly>0:
            seconds,cost=cost_estimate(best['examples_per_second'],hourly)
            minutes,remainder=divmod(round(seconds),60)
            values=[gpu,ecc,path.parent.name,f'${hourly:.2f}',f"{best['examples_per_second']:.2f}",
                    f'{minutes} min {remainder:02d} sec',f'${cost:.2f}']
            if ignoring_headroom:
                values[4:]=[value+'*' for value in values[4:]]
                exceptions.append(f'{gpu} (run {path.parent.name}): microbatch {best["microbatch"]}, accumulation {best["accumulation"]}')
            else:
                priced.append((cost,gpu,best['examples_per_second']))
        else:
            values=[gpu,ecc,path.parent.name,'Not recorded' if hourly is None else f'${hourly:.2f}',
                    f"{best['examples_per_second']:.2f}" if best else '—','—','—']
        lines.append('| '+' | '.join(cell(v) for v in values)+' |')
    if exceptions:
        lines+=['', '* Asterisked estimates ignore the memory-headroom requirement because no tested microbatch passed it, including microbatch 1. They use the fastest fitting candidate for comparison only, are not training recommendations, and are excluded from the cheapest-GPU ranking.',
                'Asterisked candidates: '+ '; '.join(exceptions)+'.']
    if priced:
        cheapest=min(priced)
        lines+=['',f'Lowest estimated training cost among these priced runs: **{cell(cheapest[1])}**, **${cheapest[0]:.2f} per 10,000 examples**.']
        fastest=max(priced,key=lambda item:item[2])
        if fastest!=cheapest:
            lines+=[f'**{cell(fastest[1])}** processes examples **{fastest[2]/cheapest[2]:.2f}× faster** than {cell(cheapest[1])}, but costs **{fastest[0]/cheapest[0]:.2f}× as much per example**.']
    lines+=['',
        'These are training-only estimates, excluding deployment, model loading, validation, checkpoint saving, idle time, and storage charges. Total billed cost includes those additional items.',
        'Different microbatches/effective batches and datasets were benchmarked. Equal examples processed does not guarantee equal model quality; a convergence comparison should match effective batch and training settings.',
        'Future report refreshes retain the confirmed rates. Record each new run’s actual hourly rate before estimating its cost.']
    return lines


def comparison(root, output, gpu_filter=None):
    root=resolve_output_path(root)
    lines=['# GPU batch capacity and training throughput','',
        'This table collects measured OpenVLA LoRA batch benchmarks. Each row is one run; different datasets or settings remain separate.', '',
        '**Throughput** means training examples processed per second after warmup. One example is one camera image, its task instruction, and seven target action values—not one trajectory.',
        '**Examples/second = effective batch ÷ seconds per completed parameter update.** Effective batch = microbatch × gradient accumulation (single GPU).',
        'Measured update time includes data loading, image/text processing, forward/backward, optimizer, and ordinary training logging. Model loading, final validation, and checkpoint saving are excluded.', '',
        'The speed columns describe the fastest tested candidate passing the 4 GiB memory-headroom screen. That candidate may differ from the largest fitting batch.',
        'A fitting batch completed 25 real updates. The short-run capacity applies to tested inputs/settings; confirm it on the final dataset and a longer trial.', '',
        '| GPU | ECC current | VRAM GiB | Dataset | Mode | Fitting capacity | First OOM | Suggested microbatch / accumulation | Seconds/update | Examples/s | Peak allocated / reserved GiB | Run / details |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    paths=sorted(root.rglob('benchmark.json')) if root.exists() else []
    groups={}
    selected=0
    cost_runs=[]
    for path in paths:
        rows=json.loads(path.read_text())
        metadata_path=path.parent/'benchmark_metadata.json'
        metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        complete=[r for r in rows if r['status']=='completed']
        failures=[r['microbatch'] for r in rows if r['status']=='out_of_memory']
        largest=max((r['microbatch'] for r in complete),default=0)
        mode=metadata.get('mode','unknown')
        if not largest:
            capacity='None fitted' if failures else 'Pending'
        elif failures and min(failures)<=largest:
            capacity='Inconsistent fits; recheck'
        elif mode=='capacity' and failures and min(failures)==largest+1:
            capacity=f'{largest} (boundary found)'
        elif mode=='capacity':
            capacity=f'≥{largest} (boundary not established)'
        else:
            capacity=f'≥{largest} (fixed candidates only)'
        eligible=[r for r in complete if r['headroom_screen_passed']]
        best=max(eligible,key=lambda r:r['examples_per_second']) if eligible else None
        info={}
        for info_path in sorted(path.parent.glob('batch_*/model_info.json')):
            info=json.loads(info_path.read_text());break
        gpu=info.get('gpu',complete[0].get('gpu','Unknown') if complete else 'Unknown')
        if gpu_filter is not None and gpu!=gpu_filter:
            continue
        selected+=1
        vram=info.get('total_vram_gib')
        reference=os.path.relpath(path.parent/'BATCH_BENCHMARK.md',output.parent).replace(os.sep,'/')
        values=[gpu,ecc_label(metadata.get('gpu_ecc')),f'{vram:.1f}' if vram is not None else '—',
            Path(metadata.get('dataset','unknown')).name,mode,capacity,min(failures) if failures else 'Not observed',
            f"{best['microbatch']} / {best['accumulation']}" if best else 'None with headroom',
            f"{best['seconds_per_update']:.3f}" if best else '—',
            f"{best['examples_per_second']:.2f}" if best else '—',
            f"{best['gpu_peak_allocated_gib']:.2f} / {best['gpu_peak_reserved_gib']:.2f}" if best else '—',
            f'[{path.parent.name}]({reference})']
        lines.append('| '+' | '.join(cell(v) for v in values)+' |')
        cost_runs.append((gpu,path,best))
        groups.setdefault((gpu,vram),[]).append((path,rows,metadata,info,capacity,best))
    if not selected:
        lines+=['| NVIDIA A40 | Unknown | — | — | Planned | Not measured | — | — | — | — | — | Pending |',
                '| NVIDIA A100 (variant to be recorded) | Unknown | — | — | Planned | Not measured | — | — | — | — | — | Pending |']
    if cost_runs:lines+=cost_section(cost_runs)
    for (gpu,vram),runs in groups.items():
        lines+=['',f'## {cell(gpu)}'+(f' — {vram:.1f} GiB' if vram is not None else ''),'']
        for path,rows,metadata,info,capacity,best in runs:
            config=info.get('config',{})
            lines += [f'### Run {path.parent.name}','',
                f"Dataset: `{metadata.get('dataset','unknown')}`. Model: `{metadata.get('model','unknown')}`.",
                f"Mode: {metadata.get('mode','unknown')}; PyTorch: {info.get('torch_version','unknown')}; LoRA rank: {config.get('lora_rank','unknown')}; quantization: {config.get('use_quantization','unknown')}; image augmentation: {config.get('image_aug','unknown')}.",
                duration_line(rows,metadata),
                ecc_details(metadata.get('gpu_ecc')),
                'Training uses BF16 autocast and eager attention. See the saved candidate configs for all settings.',
                f'Capacity finding: **{capacity}**.',
                (f"Fastest candidate with headroom: **microbatch {best['microbatch']}, accumulation {best['accumulation']}**, {best['examples_per_second']:.2f} examples/s." if best else 'No candidate passed the headroom screen.'),'',
                '| Microbatch | Accumulation | Phase | Status | Seconds/update | Examples/s | Peak allocated / reserved GiB | Headroom screen |',
                '| --- | --- | --- | --- | --- | --- | --- | --- |']
            storage=metadata.get('storage',{})
            context_path=path.parent/'batch_host_context.json'
            context=json.loads(context_path.read_text()) if context_path.exists() else {}
            notes=software_lines(metadata,info,context)
            if storage:
                notes.append(f"Input storage: **{cell(storage.get('source_storage','unknown'))}** at `{cell(storage.get('volume_mount','unknown'))}`; training reads staged inputs from **{cell(storage.get('training_reads','unknown'))}**.")
                staging=storage.get('input_staging_seconds')
                if isinstance(staging,(int,float)):
                    notes.append(f"Input staging: **{staging:.2f} seconds**, excluded from total benchmark time above. Model staging read source: `{cell(storage.get('model_staging_reads','unknown'))}`.")
            if context:
                notes.append(f"Pod CPU: {cell(context.get('cpu_model','unknown'))}; CPU quota (`cpu.max`): `{cell(context.get('/sys/fs/cgroup/cpu.max','unknown'))}`.")
                notes.append('CPU and GPU snapshot diagnostics: `batch_host_context.json` in this run’s output folder. Throughput describes this deployment and can depend on host resources and software as well as the GPU.')
            if notes:
                # Keep provenance before the candidate table, outside its Markdown rows.
                lines[-3:-3]=notes+['']
            for row in sorted(rows,key=lambda r:(r['microbatch'],r['accumulation'])):
                values=[row['microbatch'],row['accumulation'],row.get('phase','not recorded'),row['status']]
                if row['status']=='completed':
                    values += [f"{row['seconds_per_update']:.3f}",f"{row['examples_per_second']:.2f}",
                        f"{row['gpu_peak_allocated_gib']:.2f} / {row['gpu_peak_reserved_gib']:.2f}",row['headroom_screen_passed']]
                else:
                    values += ['—','—','—','—']
                lines.append('| '+' | '.join(cell(v) for v in values)+' |')
    lines+=['','## Reading the results','',
        '- “Boundary found” means adjacent sizes were tested: N fitted and N+1 ran out of memory. Intermediate untested sizes are inferred using approximately monotonic memory demand.',
        '- “Boundary not established” means the search is incomplete or its ceiling fitted. It is a lower bound, not a measured maximum.',
        '- Suggested batches leave at least 4 GiB by peak reserved and sampled free VRAM. This is a screening rule, not a stability guarantee.',
        '- ECC reports the current NVIDIA error-correction mode; pending mode is shown in each run section. Unknown means it was not recorded or could not be queried. Unavailable (N/A) is not treated as disabled. The benchmark does not change ECC settings.',
        '- Capacity mode uses accumulation one; effective batch changes. Controlled mode keeps effective batch four. These speed tests do not compare learning quality.',
        '- After capacity search, speed refinement tests missing integer sizes from the largest tested power-of-two batch passing the headroom screen up to fitting capacity. Earlier runs refined only the final capacity interval and may have missed sizes below a fitting batch without headroom. Skipping smaller sizes is a search-cost heuristic, not a guarantee that powers of two are faster. Recommendations describe the fastest tested candidate with headroom.',
        '- Compare GPUs with the same dataset, model, LoRA rank, precision, attention and software stack. Changing PyTorch/CUDA or other libraries can affect throughput; comparisons across stacks do not isolate GPU hardware. A100 40 GB and 80 GB variants should be recorded as separate runs.',
        '- Total benchmark time in each run section covers setup/loading and all batch tests, excluding pod deployment, report downloading and the supervisor shutdown window. Older runs use summed batch-test wall times, which omit between-test overhead.',
        '- Each linked report has per-batch timings and memory. Its sibling benchmark_metadata.json, benchmark.json, and model_info.json inside candidate folders preserve the exact settings and GPU details.', '',
        '## Updating this file','',
        'Use the local launcher for automatic cloud execution, report download and workspace refresh:', '',
        '```bash','.venv/bin/python benchmark_cloud_session.py \\',
        "  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519'",'```','',
        'Get POD_IP and POD_PORT from RunPod → Pod → Connect → SSH over exposed TCP. The launcher selects this Mac\'s existing private key.',
        'Keep the Mac awake and connected. By default a detached cloud supervisor requests a pod stop after completion, failure, or its 60-minute outer limit; the Mac downloads reports and also requests a stop. It uses the RUNPOD_API_KEY secret already configured on the pod. Confirm stopped status in RunPod.',
        'Use `--keep-pod-running` to disable that stop. For an existing run or interrupted download, use `--fetch-only`; add `--stop-after-fetch` to stop after retrieval. No benchmark is run in fetch-only mode.',
        'If the Mac disconnects, the cloud supervisor continues and saved reports can be fetched later. If stop requests fail, stop the pod manually in RunPod.',
        'The download includes all saved benchmark reports, JSON measurements, configs and logs on the attached volume; adapter weights are excluded.',
        'Timestamped report folders and a combined report are saved in outputs/gpu-benchmarks. This workspace file is rebuilt automatically from the downloaded JSON.',
        'For report folders already downloaded, the standalone refresh command remains available:', '',
        '```bash','.venv-training/bin/python -m analysis.summarize_gpu_benchmarks \\',
        '  --root outputs/benchmarks/training/gpu-benchmarks \\',
        '  --output docs/benchmarks/GPU_TRAINING_COMPARISON.md','```','',
        'Historical training memory usage alone does not establish maximum batch capacity.']
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines)+'\n')
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=output_path('gpu-benchmarks'))
    p.add_argument('--output',type=Path,default=PROJECT_ROOT / 'docs/benchmarks/GPU_TRAINING_COMPARISON.md')
    args=p.parse_args()
    count=comparison(args.root,args.output)
    print(f'Wrote {args.output}: {count} measured benchmark runs')


if __name__=='__main__':main()
