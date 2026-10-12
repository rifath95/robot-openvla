"""Run a cloud GPU benchmark and automatically sync its reports to this Mac.

Requires an already-running training-image pod with its network volume attached.
A detached cloud supervisor stops the pod on completion, failure, or timeout.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import tempfile
import base64
import socket
import time

from cloud.cloud_session import ROOT, REPOSITORY, parse_ssh, ssh_arguments
from analysis.summarize_gpu_benchmarks import comparison


# Send report files only, not the large diagnostic adapter checkpoints. Use a
# remote Python tar stream so download does not require container disk space.
ARCHIVE_SCRIPT=r'''
from pathlib import Path
import sys, tarfile
root=Path('/workspace/panda-training/batch-benchmarks')
if not root.is_dir():
    raise SystemExit('No saved benchmark directory on this volume')
files=[]
for run in sorted(root.iterdir()):
    if not run.is_dir() or not (run/'benchmark.json').is_file():
        continue
    for name in ('benchmark.json','benchmark_metadata.json','BATCH_BENCHMARK.md'):
        if (run/name).is_file(): files.append(run/name)
    files.extend(run.glob('batch_*.json'))
    files.extend(run.glob('batch_*.log'))
    for candidate in run.glob('batch_*_accum_*'):
        if not candidate.is_dir(): continue
        for name in ('metrics.jsonl','model_info.json','experiment.json'):
            if (candidate/name).is_file(): files.append(candidate/name)
for session in (root/'sessions').glob('*'):
    if not session.is_dir():continue
    for name in ('status.json','console.log','supervisor.log','pod_stop_status.json','inference.json'):
        if (session/name).is_file():files.append(session/name)
if not files:
    raise SystemExit('No saved benchmark results to download')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for path in files:
        if path.is_symlink(): raise SystemExit('Refusing a symlink in benchmark reports')
        archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
'''


def extract_reports(archive_path, destination):
    """Reject links/traversal before writing any report into the local folder."""
    with tarfile.open(archive_path,'r:gz') as archive:
        members=archive.getmembers()
        if len(members)>10000 or sum(m.size for m in members)>256*1024**2:
            raise ValueError('Benchmark report archive exceeds expected size')
        for member in members:
            path=PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                or len(path.parts) not in (2,3)
                or not (re.fullmatch(r'\d{8}_\d{6}_\d{6}',path.parts[0])
                        or (len(path.parts)==3 and path.parts[0]=='sessions'
                            and re.fullmatch(r'\d{8}_\d{6}_\d{6}',path.parts[1])))
                or path.suffix not in ('.json','.jsonl','.md','.log')):
                raise ValueError(f'Unexpected report archive entry: {member.name}')
            target=destination.joinpath(*path.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Report target escapes the local output directory')
        for member in members:
            target=destination.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as output:
                while chunk:=source.read(1024**2): output.write(chunk)


def fetch_reports(ssh, local_root, workspace_report,volume_root='/workspace'):
    local_root.mkdir(parents=True,exist_ok=True)
    print('Downloading benchmark reports, JSON measurements and logs (excluding adapter weights)...',flush=True)
    with tempfile.TemporaryDirectory(prefix='benchmark-download-',dir=local_root) as temp:
        archive_path=Path(temp)/'reports.tar.gz'
        archive_script=ARCHIVE_SCRIPT.replace("Path('/workspace/panda-training/batch-benchmarks')",repr_path(volume_root))
        remote='/opt/openvla/bin/python -c '+shlex.quote(archive_script)
        with archive_path.open('wb') as output:
            subprocess.run(ssh+[remote],stdout=output,check=True)
        extract_reports(archive_path,local_root)
    # Check all imported JSON before replacing the tracked comparison document.
    for path in local_root.rglob('benchmark.json'):
        json.loads(path.read_text())
    comparison(local_root,local_root/'GPU_TRAINING_COMPARISON.md')
    gpus=set()
    for path in local_root.glob('*/batch_*/model_info.json'):
        gpu=json.loads(path.read_text()).get('gpu')
        if gpu:gpus.add(gpu)
    for gpu in sorted(gpus):
        label=re.sub(r'[^A-Za-z0-9]+','_',gpu.removeprefix('NVIDIA ')).strip('_')
        comparison(local_root,local_root/f'{label}_COMPARISON.md',gpu_filter=gpu)
    count=comparison(local_root,workspace_report)
    print(f'Downloaded reports: {local_root}',flush=True)
    print(f'Updated {workspace_report} with {count} saved benchmark runs.',flush=True)
    from analysis.summarize_gpu_inference import comparison as inference_comparison
    inference_comparison(ROOT/'outputs',ROOT/'docs/benchmarks/GPU_INFERENCE_COMPARISON.md')


def measure_mac_inference(ssh,local_root,global_volume=None):
    """Measure real Mac/tunnel latency while the detached cloud server is waiting."""
    from cloud.connection_client import request_prediction
    reports=sorted(local_root.glob('sessions/*/inference.json'))
    if not reports:return
    path=reports[-1];report=json.loads(path.read_text())
    status=json.loads((path.parent/'status.json').read_text())
    if status.get('phase')!='inference_ready':return
    remote=report['live_session_dir']
    if not re.fullmatch(r'/((workspace|workspace-global)/panda-training|root/robot-openvla/\.benchmark-local)/batch-benchmarks/sessions/\d{8}_\d{6}_\d{6}',remote):
        raise ValueError('Unexpected inference session directory')
    port=report['server_port']
    if type(port)!=int or not 1<=port<=65535:raise ValueError('Invalid inference port')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));local_port=probe.getsockname()[1]
    tunnel=subprocess.Popen(ssh[:-1]+['-o','ExitOnForwardFailure=yes','-N','-L',
                            f'127.0.0.1:{local_port}:127.0.0.1:{port}',ssh[-1]])
    rows=[]
    try:
        deadline=time.monotonic()+15
        while True:
            if tunnel.poll() is not None:raise RuntimeError('Inference SSH tunnel exited')
            try:
                with socket.create_connection(('127.0.0.1',local_port),timeout=.5):break
            except OSError:
                if time.monotonic()>deadline:raise TimeoutError('Inference tunnel did not start')
                time.sleep(.1)
        with tempfile.TemporaryDirectory() as tmp:
            image=Path(tmp)/'input.png';image.write_bytes(base64.b64decode(report['image_png_base64'],validate=True))
            deadline=time.monotonic()+100
            for i in range(report['warmup_requests']+report['measured_requests']):
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Mac inference measurement window expired')
                response=request_prediction(image,report['instruction'],server_url=f'http://127.0.0.1:{local_port}',
                                            timeout_seconds=min(30,remaining))
                rows.append(dict(warmup=i<report['warmup_requests'],
                                 request_timings_seconds=response['client_timings_seconds'],
                                 server_timings_seconds=response['timings_seconds']))
                print(f'Mac inference request {i+1}/25',flush=True)
        report['mac_requests']=rows
        path.write_text(json.dumps(report,indent=2)+'\n')
        # Preserve the Mac measurements on the volume before acknowledging completion.
        script=f'from pathlib import Path; import sys,json; from training.detached_run import write_status; p=Path({remote!r}); write_status(p/"inference.json",json.load(sys.stdin))'
        if global_volume:
            script+=f'; from training.global_benchmark_storage import copy_reports; copy_reports(p.parent.parent,Path({global_volume!r})/"panda-training/batch-benchmarks")'
        script+='; (p/"inference_client_done").touch()'
        subprocess.run(ssh+['cd /root/robot-openvla && /opt/openvla/bin/python -c '+shlex.quote(script)],
                       input=json.dumps(report),text=True,check=True,timeout=30)
    finally:
        tunnel.terminate()
        try:tunnel.wait(timeout=5)
        except subprocess.TimeoutExpired:tunnel.kill();tunnel.wait()
        from analysis.summarize_gpu_inference import comparison as inference_comparison
        inference_comparison(ROOT/'outputs',ROOT/'docs/benchmarks/GPU_INFERENCE_COMPARISON.md')


def repr_path(volume_root):
    if volume_root not in ('/workspace','/workspace-global'):raise ValueError('Invalid global-volume mount')
    return 'Path('+repr(volume_root+'/panda-training/batch-benchmarks')+')'


def remote_benchmark_script(dataset, maximum, controlled=False, auto_stop=False, prepare_only=False, hard_limit=3600,global_volume=None):
    volume_root=global_volume or '/workspace'
    repr_path(volume_root)
    if not re.fullmatch(re.escape(volume_root)+r'/panda-training/[A-Za-z0-9][A-Za-z0-9_-]*',dataset):
        raise ValueError('Dataset must be a named bundle directly inside /workspace/panda-training')
    if hard_limit<1:raise ValueError('Outer timeout must be positive')
    if maximum<1:
        raise ValueError('Maximum batch must be positive')
    mode='' if controlled else '--find-capacity'
    check='/opt/openvla/bin/python -m training.benchmark_stop --check-only' if auto_stop else ''
    global_option=' --global-volume '+shlex.quote(global_volume) if global_volume else ''
    check+=global_option if check else ''
    benchmark='' if prepare_only else f'/opt/openvla/bin/python -u -m training.benchmark_training_batches {mode} --max-batch {maximum} --dataset {shlex.quote(dataset)}'
    if auto_stop and not prepare_only:
        benchmark=f'/opt/openvla/bin/python -u -m training.benchmark_supervisor --launch --max-batch {maximum} --dataset {shlex.quote(dataset)} --hard-limit-seconds {hard_limit}'
        benchmark+=global_option
        if controlled:benchmark+=' --controlled'
    if global_volume and not auto_stop and not prepare_only:
        raise ValueError('Global benchmarks require the watchdog; omit --keep-pod-running')
    return f'''set -euo pipefail
mountpoint -q {shlex.quote(volume_root)}
test -x /opt/openvla/bin/python
if [[ ! -d /root/robot-openvla ]]; then
  git clone {shlex.quote(REPOSITORY)} /root/robot-openvla
else
  git -C /root/robot-openvla pull --ff-only
fi
cd /root/robot-openvla
{check}
{benchmark}
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ssh',required=True,help='SSH command copied from RunPod Connect')
    p.add_argument('--key',type=Path,help='Optional override of the Mac private-key path')
    p.add_argument('--dataset',default='/workspace/panda-training/training_v1')
    p.add_argument('--max-batch',type=int,default=64)
    p.add_argument('--hard-limit-seconds',type=int,default=3600,help='Cloud supervisor outer limit (default: 60 minutes)')
    p.add_argument('--controlled',action='store_true',help='Use fixed effective-batch-four comparison instead')
    p.add_argument('--fetch-only',action='store_true',help='Download saved results without running any GPU training')
    p.add_argument('--keep-pod-running',action='store_true',help='Disable automatic stop after a new benchmark')
    p.add_argument('--stop-after-fetch',action='store_true',help='Also request a pod stop after --fetch-only, including download errors')
    p.add_argument('--global-volume',choices=('/workspace','/workspace-global'),help='Global mount: stage inputs locally and publish reports before stop')
    args=p.parse_args()
    try:
        host,port,key=parse_ssh(args.ssh,args.key)
        auto_stop=(not args.fetch_only and not args.keep_pod_running) or args.stop_after_fetch
        if args.keep_pod_running and args.stop_after_fetch:
            p.error('--keep-pod-running conflicts with --stop-after-fetch')
        if args.global_volume=='/workspace-global' and args.dataset=='/workspace/panda-training/training_v1':
            args.dataset='/workspace-global/panda-training/training_v1'
        script=remote_benchmark_script(args.dataset,args.max_batch,args.controlled,auto_stop,prepare_only=args.fetch_only,hard_limit=args.hard_limit_seconds,global_volume=args.global_volume)
    except ValueError as exc:
        p.error(str(exc))
    ssh=ssh_arguments(host,port,key)
    print(f'Connecting to {host}:{port}. The pod must already be running.',flush=True)
    print('Automatic stop after completion, failure, or cloud outer timeout.' if auto_stop else
          'Automatic stop disabled. Stop the pod manually in RunPod afterward.',flush=True)
    code=0
    error=None
    try:
        if args.fetch_only and auto_stop:
            setup=remote_benchmark_script(args.dataset,args.max_batch,args.controlled,True,prepare_only=True,global_volume=args.global_volume)
            subprocess.run(ssh+['bash -s'],input=setup,text=True,check=True)
        if not args.fetch_only:
            result=subprocess.run(ssh+['bash -s'],input=script,text=True)
            code=result.returncode
            if code==75:
                raise SystemExit('Another benchmark supervisor is running. It was left running; this launch did not stop it.')
            if code:
                print('Cloud command ended with an error; attempting to collect saved diagnostics.',flush=True)
        if args.global_volume:
            fetch_reports(ssh,ROOT/'outputs/gpu-benchmarks',ROOT/'docs/benchmarks/GPU_TRAINING_COMPARISON.md',args.global_volume)
        else:fetch_reports(ssh,ROOT/'outputs/gpu-benchmarks',ROOT/'docs/benchmarks/GPU_TRAINING_COMPARISON.md')
        if not args.fetch_only and auto_stop:
            measure_mac_inference(ssh,ROOT/'outputs/gpu-benchmarks',args.global_volume)
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError,tarfile.TarError) as exc:
        error=f'Results were not fully synchronized: {exc}. Saved results remain on the volume.'
    except KeyboardInterrupt:
        # Leave the detached supervisor responsible for killing workers and stopping.
        if auto_stop and not args.fetch_only:
            raise SystemExit('Local watcher interrupted. If the cloud supervisor started, it continues and will request a pod stop after completion or its outer limit. Confirm stopped status in RunPod.')
        raise SystemExit('Local launcher interrupted. Check and stop the pod manually in RunPod.')
    if error:print(error,flush=True)
    if auto_stop:
        print('Requesting pod stop (including when the benchmark or download failed)...',flush=True)
        stop_command='cd /root/robot-openvla && /opt/openvla/bin/python -m training.benchmark_stop'
        if args.global_volume:stop_command+=' --global-volume '+shlex.quote(args.global_volume)
        try:
            stopped=subprocess.run(ssh+[stop_command],timeout=100)
            if stopped.returncode:
                raise RuntimeError('Stop command did not confirm success')
        except (OSError,RuntimeError,subprocess.TimeoutExpired):
            raise SystemExit('Stop could not be confirmed. The cloud supervisor also attempts cleanup if it started. Check RunPod and stop manually if still running.')
    if error:raise SystemExit(error)
    if code:raise SystemExit(code)



if __name__=='__main__':main()
