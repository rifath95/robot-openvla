"""Build a readable GPU comparison from saved batch benchmark directories."""
import argparse
import json
import os
from pathlib import Path


def cell(value):
    return str(value).replace('|','/').replace('\n',' ')


def comparison(root, output):
    lines=['# GPU batch capacity and training throughput','',
        'This table collects measured OpenVLA LoRA batch benchmarks. Each row is one run; different datasets or settings remain separate.', '',
        '**Throughput** means training examples processed per second after warmup. One example is one camera image, its task instruction, and seven target action values—not one trajectory.',
        '**Examples/second = effective batch ÷ seconds per completed parameter update.** Effective batch = microbatch × gradient accumulation (single GPU).',
        'Measured update time includes data loading, image/text processing, forward/backward, optimizer, and ordinary training logging. Model loading, final validation, and checkpoint saving are excluded.', '',
        'The speed columns describe the fastest tested candidate passing the 4 GiB memory-headroom screen. That candidate may differ from the largest fitting batch.',
        'A fitting batch completed 25 real updates. The short-run capacity applies to tested inputs/settings; confirm it on the final dataset and a longer trial.', '',
        '| GPU | VRAM GiB | Dataset | Mode | Fitting capacity | First OOM | Suggested microbatch / accumulation | Seconds/update | Examples/s | Peak allocated / reserved GiB | Run / details |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    paths=sorted(root.rglob('benchmark.json')) if root.exists() else []
    groups={}
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
        vram=info.get('total_vram_gib')
        reference=os.path.relpath(path.parent/'BATCH_BENCHMARK.md',output.parent).replace(os.sep,'/')
        values=[gpu,f'{vram:.1f}' if vram is not None else '—',
            Path(metadata.get('dataset','unknown')).name,mode,capacity,min(failures) if failures else 'Not observed',
            f"{best['microbatch']} / {best['accumulation']}" if best else 'None with headroom',
            f"{best['seconds_per_update']:.3f}" if best else '—',
            f"{best['examples_per_second']:.2f}" if best else '—',
            f"{best['gpu_peak_allocated_gib']:.2f} / {best['gpu_peak_reserved_gib']:.2f}" if best else '—',
            f'[{path.parent.name}]({reference})']
        lines.append('| '+' | '.join(cell(v) for v in values)+' |')
        groups.setdefault((gpu,vram),[]).append((path,rows,metadata,info,capacity,best))
    if not paths:
        lines+=['| NVIDIA A40 | — | — | Planned | Not measured | — | — | — | — | — | Pending |',
                '| NVIDIA A100 (variant to be recorded) | — | — | Planned | Not measured | — | — | — | — | — | Pending |']
    for (gpu,vram),runs in groups.items():
        lines+=['',f'## {cell(gpu)}'+(f' — {vram:.1f} GiB' if vram is not None else ''),'']
        for path,rows,metadata,info,capacity,best in runs:
            config=info.get('config',{})
            lines += [f'### Run {path.parent.name}','',
                f"Dataset: `{metadata.get('dataset','unknown')}`. Model: `{metadata.get('model','unknown')}`.",
                f"Mode: {metadata.get('mode','unknown')}; PyTorch: {info.get('torch_version','unknown')}; LoRA rank: {config.get('lora_rank','unknown')}; quantization: {config.get('use_quantization','unknown')}; image augmentation: {config.get('image_aug','unknown')}.",
                'Training uses BF16 autocast and eager attention. See the saved candidate configs for all settings.',
                f'Capacity finding: **{capacity}**.',
                (f"Fastest candidate with headroom: **microbatch {best['microbatch']}, accumulation {best['accumulation']}**, {best['examples_per_second']:.2f} examples/s." if best else 'No candidate passed the headroom screen.'),'',
                '| Microbatch | Accumulation | Phase | Status | Seconds/update | Examples/s | Peak allocated / reserved GiB | Headroom screen |',
                '| --- | --- | --- | --- | --- | --- | --- | --- |']
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
        '- Capacity mode uses accumulation one; effective batch changes. Controlled mode keeps effective batch four. These speed tests do not compare learning quality.',
        '- After capacity search, missed fitting sizes in the final doubling interval are timed too. The recommendation is the fastest tested candidate with headroom; smaller sizes outside that interval are not exhaustively scanned.',
        '- Compare GPUs with the same dataset, model, LoRA rank, precision and attention settings. A100 40 GB and 80 GB variants should be recorded as separate runs.',
        '- Each linked report has per-batch timings and memory. Its sibling benchmark_metadata.json, benchmark.json, and model_info.json inside candidate folders preserve the exact settings and GPU details.', '',
        '## Updating this file','',
        'Every cloud benchmark refreshes a comparison file in the network-volume benchmark directory. To refresh this workspace copy after downloading benchmark folders into outputs/gpu-benchmarks:', '',
        '```bash','.venv-training/bin/python summarize_gpu_benchmarks.py \\',
        '  --root outputs/gpu-benchmarks \\',
        '  --output GPU_BATCH_COMPARISON.md','```','',
        'Historical training memory usage alone does not establish maximum batch capacity.']
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines)+'\n')
    return len(paths)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/gpu-benchmarks'))
    p.add_argument('--output',type=Path,default=Path('GPU_BATCH_COMPARISON.md'))
    args=p.parse_args()
    count=comparison(args.root,args.output)
    print(f'Wrote {args.output}: {count} measured benchmark runs')


if __name__=='__main__':main()
