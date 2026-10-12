"""Compare recorded single-image inference and simulator cycle timings offline."""
import argparse
import json
from pathlib import Path
from statistics import mean

from common.paths import PROJECT_ROOT
from analysis.run_report import number, read_json

REMOTE = [('receive_body', 'Receive body'), ('preprocess', 'Decode image'),
          ('processor_and_device_transfer', 'Processor + GPU transfer'),
          ('model_inference', 'Model prediction'), ('server_processing', 'Server total')]
LOCAL = [('capture_and_state_save', 'Input capture/save'), ('ik', 'IK'),
         ('robot_execution_wall', 'Movement'),
         ('next_capture_and_state_save', 'Result capture/save')]


def average(rows, group, key):
    values = [(r.get(group) or {}).get(key) for r in rows]
    values = [v for v in values if number(v)]
    return mean(values) if values else None


def difference(rows, parts):
    values = []
    for row in rows:
        fields = [(row.get(group) or {}).get(key) for group, key, sign in parts]
        if all(number(v) for v in fields):
            values.append(sum(v * part[2] for v, part in zip(fields, parts)))
    return mean(values) if values else None


def percentile(values, fraction):
    values = sorted(v for v in values if number(v))
    if not values:
        return None
    index = (len(values)-1)*fraction
    lo = int(index); hi = min(lo+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(index-lo)


def measurement(folder, warmup=5, benchmark=None):
    startup = benchmark['startup_info'] if benchmark else read_json(folder/'startup_info.json')
    cloud_only=bool(benchmark and not benchmark.get('mac_requests'))
    rows = ((benchmark.get('mac_requests') or benchmark['cloud_requests']) if benchmark else
            [read_json(p) for p in sorted(folder.glob('step_*/execution.json'))])
    if not rows or not startup.get('gpu', {}).get('name'):
        return None
    # Only complete prediction requests qualify; rejected IK actions still qualify.
    rows = [r for r in rows if number((r.get('request_timings_seconds') or {}).get('request_round_trip'))]
    if not rows:
        return None
    measured = [r for r in rows if not r['warmup']] if benchmark else rows[warmup:]
    if cloud_only:
        # Cloud-loopback RTT is not Mac-to-cloud latency. Preserve server stages only.
        measured=[dict(r,request_timings_seconds={}) for r in measured]
    local = [('timings_seconds', key, 1) for key, _ in LOCAL]
    request = average(measured, 'request_timings_seconds', 'request_round_trip')
    cycle = average(measured, 'timings_seconds', 'cycle')
    return dict(folder=folder, startup=startup, count=len(rows), measured=len(measured),
                samples=measured,
                kind='Benchmark (cloud only)' if cloud_only else 'Benchmark (Mac)' if benchmark else 'Simulator',
                model_identity=benchmark['model_identity'] if benchmark else
                    startup.get('adapter_checkpoint') or 'base OpenVLA',
                fixture_identity=benchmark.get('image_sha256') if benchmark else 'live simulator',
                request=request, cycle=cycle,
                p95=percentile([(r.get('request_timings_seconds') or {}).get('request_round_trip')
                                for r in measured], .95),
                remote={key:average(measured, 'server_timings_seconds', key) for key, _ in REMOTE},
                server_overhead=difference(measured, [('server_timings_seconds', 'server_processing', 1)] +
                                           [('server_timings_seconds', key, -1) for key, _ in REMOTE[:-1]]),
                local={key:average(measured, 'timings_seconds', key) for key, _ in LOCAL},
                local_total=difference(measured, local),
                network=difference(measured, [('request_timings_seconds', 'request_round_trip', 1),
                                              ('server_timings_seconds', 'server_processing', -1)]),
                overhead=difference(measured, [('timings_seconds', 'cycle', 1),
                                               ('request_timings_seconds', 'request_round_trip', -1)] +
                                              [(g, k, -s) for g, k, s in local]),
                request_stages={k:average(measured, 'request_timings_seconds', k) for k in
                                ('encode_request','connect','upload_send','wait_for_response_headers','response_body_read')})


def fmt(value, digits=3):
    return f'{value:.{digits}f}' if number(value) else '—'


def table(headers, rows):
    def escape(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] + [
        '| '+' | '.join(escape(v) for v in row)+' |' for row in rows]


def comparison(root, output, warmup=5):
    if warmup < 0:
        raise ValueError('Warmup count must be nonnegative')
    root, output = Path(root), Path(output)
    rates = read_json(PROJECT_ROOT/'inference/gpu_inference_rates.json').get('gpus', {})
    runs = [m for path in sorted(root.rglob('startup_info.json'))
            if (m := measurement(path.parent, warmup)) is not None]
    runs += [m for path in sorted(root.rglob('sessions/*/inference.json'))
             if (m := measurement(path.parent,warmup,read_json(path))) is not None]
    lines = ['# GPU inference comparison', '',
             'One request = one RGB image + instruction → one seven-value action. Inference batch size is 1.', '',
             f'Timings are seconds. The first {warmup} recorded requests per trial are excluded as warmup. '
             'Short trials without remaining samples show missing timings. These are observed simulator runs, '
             'not identical-input hardware benchmarks; model/checkpoint, instruction, network, Mac, storage and '
             'software differences can affect results. Rejected/held actions remain prediction requests. '
             'Combined benchmarks start with 5 warmup requests and 20 measured inference requests, both on the cloud and '
             'from the Mac when connected. Cloud-loopback RTT is never reported as Mac latency. '
             'When Mac measurements exist, the server breakdown uses those same responses to keep '
             'RTT and server timing paired; otherwise it uses the cloud samples. Both raw sets are retained.', '']
    groups={}
    for m in runs:
        groups.setdefault(m['startup']['gpu']['name'],[]).append(m)
    def pooled(group, category, key):
        values=[(row.get(category) or {}).get(key) for m in group for row in m['samples']]
        values=[value for value in values if number(value)]
        return mean(values) if values else None
    summary=[]
    for gpu,group in groups.items():
        cycle=pooled(group,'timings_seconds','cycle')
        request=pooled(group,'request_timings_seconds','request_round_trip')
        prediction=pooled(group,'server_timings_seconds','model_inference')
        rate=rates.get(gpu,{}).get('hourly_rate_usd')
        def cost(duration):
            return 100*duration/3600*rate if number(duration) and number(rate) else None
        summary.append([gpu,len(group),sum(m['measured'] for m in group),
                        fmt(cycle),fmt(request),fmt(prediction),fmt(rate,2),fmt(cost(request),4),fmt(cost(cycle),4)])
    lines += ['', '## Average by GPU type', '',
              'Means are weighted by measured request counts, not an average of rounded table values. '
              'There is one row per reported GPU type: all A40 runs are combined, for example. '
              'These overall averages mix VRAM/ECC configurations, model checkpoints, software versions, '
              'inputs, and simulator/benchmark runs; use the detailed tables below to inspect those differences. '
              'They describe our observed runs rather than an isolated hardware ranking. Local timing is Mac-dependent. '
              'Each metric includes only runs that recorded it; cloud-only runs contribute server timing '
              'but not Mac RTT or simulator time. The per-run table below preserves each observation.', '']
    lines += table(['GPU','Runs','Measured requests','Mean cycle s','Mean request RTT s','Mean prediction s',
                    'USD/hour','100 requests USD','100 simulator cycles USD'],summary)
    lines += ['', '## Full cycle and estimated cost for 100 steps', '']
    overview = []
    for m in runs:
        s=m['startup']; gpu=s['gpu']['name']; rate=rates.get(gpu, {}).get('hourly_rate_usd')
        rate=rate if number(rate) and rate>0 else None
        duration=m['cycle'] if number(m['cycle']) else m['request']
        active=100*duration if number(duration) else None
        setup=s.get('model_preparation_seconds'); load=s.get('model_load_seconds')
        total=active+setup+load if all(number(v) for v in (active,setup,load)) else None
        overview.append([gpu,m['folder'].name,f'{m["measured"]}/{m["count"]}',fmt(m['cycle']),
                         fmt(m['request']),fmt(m['p95']),fmt(m['remote']['model_inference']),
                         fmt(rate,2),fmt(active,1),fmt(active/3600*rate,4) if active is not None and rate else '—',
                         fmt(total/3600*rate,4) if total is not None and rate else '—',m['kind']])
    observed={m['startup']['gpu']['name'] for m in runs}
    for gpu, rate in rates.items():
        if gpu not in observed:
            overview.append([gpu,'Not measured','0','—','—','—','—',fmt(rate['hourly_rate_usd'],2),'—','—','—','—'])
    lines += table(['GPU','Run','Measured/all requests','Full cycle s','Request RTT s','Request p95 s',
                    'Model prediction s','USD/hour','100 cycles s','100 active cycles USD',
                    '100 cycles + preparation/loading USD','Type'], overview)
    lines += ['', 'Cost = estimated seconds ÷ 3600 × user-confirmed hourly rate. '
              '100-cycle estimates extrapolate the measured mean, including partial trials; they do not assert task success. '
              'Setup-inclusive cost adds recorded model preparation and loading once. It excludes image deployment, '
              'adapter merge/export unless covered by recorded preparation, other setup, manual pauses, idle time, '
              'shutdown delays and storage charges. Rates are comparison inputs, not historical billing records. '
              'Multiple trials can reuse one model loading. For benchmark rows without simulator movements, '
              '100-cycle columns mean 100 requests only: local IK/movement/capture time is unmeasured, not zero. '
              'Cloud-only rows cannot estimate Mac RTT costs. Missing measurements stay missing.', '',
              '## Remote request breakdown — averages in seconds', '']
    lines += table(['Run','Request RTT']+[label for _,label in REMOTE]+['Remaining server overhead','RTT minus server total'],
                   [[m['folder'].name,fmt(m['request'])]+[fmt(m['remote'][key]) for key,_ in REMOTE]+
                    [fmt(m['server_overhead']),fmt(m['network'])] for m in runs])
    lines += ['', 'Server total is inside request RTT, not added to it. RTT minus server total is residual '
              'network/tunnel/client overhead, not pure network latency. Receive-body time can itself include '
              'waiting for uploaded bytes. Processor and GPU transfer are measured together. Model prediction '
              'includes vision encoding, autoregressive action-token generation and decoding to seven values.', '',
              '## Client request stages — averages in seconds', '']
    lines += table(['Run','Encode (outside RTT)','Connect (outside RTT)','Send','Wait for headers','Read response'],
                   [[m['folder'].name]+[fmt(v) for v in m['request_stages'].values()] for m in runs])
    lines += ['', 'Send, wait for response headers and read response are included in request RTT. '
              'Sending may buffer bytes locally; it is not a separately measured network upload duration. '
              'Waiting includes server processing and network delays.', '',
              '## Local simulator breakdown — averages in seconds', '']
    lines += table(['Run']+[label for _,label in LOCAL]+['Local subtotal','Remaining cycle overhead'],
                   [[m['folder'].name]+[fmt(m['local'][key]) for key,_ in LOCAL]+
                    [fmt(m['local_total']),fmt(m['overhead'])] for m in runs])
    lines += ['', 'Full cycle = request RTT + local subtotal + remaining cycle overhead, using records '
              'with all required fields. Remaining overhead includes client encoding/connection work and '
              'recording outside named stages. Movement may be deliberately paced by the live viewer. '
              'Pauses or interruptions within a recorded cycle can affect timing. Each column averages '
              'available measurements only; different coverage can prevent exact subtotal agreement.', '',
              '## Setup and comparison context', '']
    lines += table(['Run','VRAM GiB','ECC','Model / checkpoint','Load s','Preparation s','Storage',
                    'PyTorch / Transformers','Precision / attention'],
                   [[m['folder'].name,fmt(m['startup']['gpu'].get('total_vram_bytes',0)/2**30)
                     if number(m['startup']['gpu'].get('total_vram_bytes')) else '—',
                     m['startup'].get('gpu_ecc',{}).get('current','Not recorded'),
                     m['startup'].get('adapter_checkpoint') or m['startup'].get('model_dir','Not recorded'),
                     fmt(m['startup'].get('model_load_seconds')),fmt(m['startup'].get('model_preparation_seconds')),
                     ', '.join(fs.get('fstype','unknown') for fs in m['startup'].get('volume_mount',{}).get('filesystems',[])) or 'Not recorded',
                     ' / '.join(m['startup'].get(k,'Not recorded') for k in ('pytorch_version','transformers_version')),
                     ' / '.join(m['startup'].get(k,'Not recorded') for k in ('dtype','attention'))] for m in runs])
    lines += ['', 'Model loading includes processor setup, checkpoint reading, construction and GPU transfer; '
              'it does not isolate storage-to-VRAM time. Historical ECC is not inferred from unrelated training pods.', '',
              '## Refreshing this report', '',
              'New local simulator trials refresh this report automatically after saving their run report. '
              'No cloud connections or predictions are made by report generation.', '',
              '```bash', '.venv/bin/python -m analysis.summarize_gpu_inference', '```', '',
              'Hourly rates are stored in `inference/gpu_inference_rates.json`. Raw measurements remain in '
              '`outputs/<run>/startup_info.json` and `step_*/execution.json`. Each run also keeps `RUN_REPORT.md`.', '']
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines))
    return len(runs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=PROJECT_ROOT/'outputs')
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'docs/benchmarks/GPU_INFERENCE_COMPARISON.md')
    parser.add_argument('--warmup',type=int,default=5)
    args=parser.parse_args()
    print(f'Saved {comparison(args.root,args.output,args.warmup)} inference trials: {args.output}')


if __name__=='__main__':
    main()
