"""Archive a completed multi-cube sweep to HF; detached, verified, bounded stop."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import getpass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
from common.paths import PROJECT_ROOT
from cloud.sweep_plan import validate_saved,timeout_budget
from training.detached_run import preflight,write_status
from training.benchmark_stop import stop_pod

SWEEPS=Path('/workspace/panda-training/scripted-evaluations')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(4*1024**2):h.update(block)
    return h.hexdigest()


def completed_payload(source):
    p=json.loads((source/'results.json').read_text());specs=validate_saved(p)
    if p['plan']['task']!='multi_color_touch' or p.get('launcher_error') or len(p['trials'])!=len(specs):
        raise ValueError('Archive requires a completed multi-cube sweep')
    if any(r.get('outcome')=='unmeasured' or r.get('phase')=='Missing summary' for r in p['trials']):
        raise ValueError('Sweep contains unmeasured trials')
    status=json.loads((source/'supervisor_status.json').read_text())
    if status.get('outcome')!='completed' or status.get('exit_code')!=0:raise ValueError('Cloud sweep did not complete successfully')
    for r in p['trials']:
        if not (source/r['directory']/'summary.json').is_file():raise ValueError('Missing trial artifacts')
    return p


def add_checked(archive,path,relative):
    if path.is_symlink():raise ValueError('Refusing symlink in sweep results')
    if path.is_dir():
        archive.add(path,arcname=str(relative),recursive=False)
        for child in sorted(path.iterdir()):add_checked(archive,child,relative/child.name)
    elif path.is_file():archive.add(path,arcname=str(relative),recursive=False)
    else:raise ValueError('Unsupported archive entry')


def package(source,staging,payload):
    groups=defaultdict(list)
    for r in payload['trials']:
        model='base' if r['checkpoint']==0 else f"checkpoint_{r['checkpoint']:06d}"
        groups[f"{model}_{r['cube_count']}c_{r['color_set']}"].append(r['directory'])
    manifest=[]
    for label,folders in sorted(groups.items()):
        target=staging/'archives'/f'{label}.tar.gz';target.parent.mkdir(parents=True,exist_ok=True)
        print(f'Packing {label}: {len(folders)} trials',flush=True)
        temporary=target.with_suffix('.partial')
        with tarfile.open(temporary,'w:gz',compresslevel=1) as archive:
            for folder in folders:add_checked(archive,source/folder,Path(folder))
        temporary.replace(target)
        manifest.append(dict(path=str(target.relative_to(staging)),bytes=target.stat().st_size,sha256=digest(target),trials=len(folders)))
    reports=staging/'reports';reports.mkdir(exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and not path.is_symlink() and path.suffix in ('.json','.jsonl','.md','.log','.txt'):
            shutil.copyfile(path,reports/path.name)
    for path in sorted(reports.iterdir()):manifest.append(dict(path=str(path.relative_to(staging)),bytes=path.stat().st_size,sha256=digest(path)))
    (staging/'manifest.json').write_text(json.dumps(dict(sweep=source.name,trials=len(payload['trials']),files=manifest),indent=2)+'\n')
    return manifest


def upload_verify(staging,repo,prefix,token,cache):
    from huggingface_hub import HfApi,hf_hub_download
    api=HfApi(token=token)
    info=api.repo_info(repo,repo_type='dataset')
    if not info.private:raise ValueError('Archive repository must be private')
    print('Uploading archive parts and reports to Hugging Face...',flush=True)
    commit=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=str(staging),path_in_repo=prefix,
        commit_message=f'Archive completed simulator sweep {prefix}')
    revision=commit.oid
    manifest=json.loads((staging/'manifest.json').read_text())
    files=manifest['files']+[dict(path='manifest.json',bytes=(staging/'manifest.json').stat().st_size,sha256=digest(staging/'manifest.json'))]
    for f in files:
        print('Verifying from HF:',f['path'],flush=True)
        downloaded=Path(hf_hub_download(repo_id=repo,repo_type='dataset',revision=revision,
            filename=prefix+'/'+f['path'],token=token,cache_dir=str(cache)))
        if downloaded.stat().st_size!=f['bytes'] or digest(downloaded)!=f['sha256']:
            raise ValueError('Uploaded archive checksum mismatch: '+f['path'])
    return dict(repo_id=repo,repo_type='dataset',revision=revision,path_in_repo=prefix,
        verified_files=len(files),verified_bytes=sum(f['bytes'] for f in files),trials=manifest['trials'],verified=True)


def worker(config):
    source=Path(config['source']);receipt=Path(config['receipt']);status=receipt/'status.json'
    try:
        payload=completed_payload(source)
        staging=Path(config['staging']);staging.mkdir(parents=True,exist_ok=True)
        # Avoid an extra full NFS metadata traversal before packing.
        minimum_gib=max(16,16*len(payload['trials'])/1080)
        if shutil.disk_usage(staging).free<minimum_gib*1024**3:raise ValueError('Insufficient container-disk space for packing and verification')
        write_status(status,dict(phase='packing',source=str(source),minimum_container_free_gib=minimum_gib))
        package(source,staging,payload)
        write_status(status,dict(phase='uploading_and_verifying',source=str(source)))
        verified=upload_verify(staging,config['repo'],source.name,config['token'],staging.parent/(source.name+'-verification'))
        verified['verified_at_utc']=datetime.now(timezone.utc).isoformat()
        write_status(receipt/'receipt.json',verified)
        write_status(status,dict(phase='verified',**verified))
        print('Upload verified. Original sweep remains on the network volume.',flush=True)
        return 0
    except Exception as exc:
        # Log type only: third-party HTTP exceptions can contain credential-bearing requests.
        write_status(status,dict(phase='failed',error_type=type(exc).__name__))
        print(f'Archive failed ({type(exc).__name__}); original results are retained.',flush=True)
        return 1


def supervise(config):
    receipt=Path(config['receipt']);budget=timeout_budget(config['estimated_seconds']);process=None
    started=time.monotonic();outcome='supervisor_failed'
    try:
        worker_config={k:v for k,v in config.items() if k not in ('pod_id','pod_key','lock_fd')}
        process=subprocess.Popen([sys.executable,'-u','-m','cloud.archive_sweep','--worker'],
            cwd=PROJECT_ROOT,stdin=subprocess.PIPE,start_new_session=True)
        process.stdin.write(json.dumps(worker_config).encode());process.stdin.close()
        while process.poll() is None:
            elapsed=time.monotonic()-started
            if elapsed>=budget['outer_seconds']:
                outcome='outer_limit';break
            if elapsed>=budget['inner_seconds']:
                outcome='inner_limit';break
            time.sleep(1)
        else:outcome='verified' if process.returncode==0 else 'failed'
    finally:
        if process is not None and process.poll() is None:
            from training.benchmark_supervisor import kill_benchmark_tree
            kill_benchmark_tree(process)
        try:write_status(receipt/'supervisor_status.json',dict(outcome=outcome,elapsed_seconds=time.monotonic()-started,timeout_budget=budget))
        except OSError:print('Could not save supervisor status; still requesting stop.',flush=True)
        # Failed archives remain on the volume for another attempt; do not leave the GPU billing indefinitely.
        stop_pod(config['pod_id'],config['pod_key'],receipt/'pod_stop_status.json',reason=f'HF sweep archive: {outcome}')
    return 0 if outcome=='verified' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sweep');p.add_argument('--repo',default='Maxalderone/robot-openvla-simulator-results')
    p.add_argument('--estimated-seconds',type=float,default=3600)
    internal=p.add_mutually_exclusive_group();internal.add_argument('--supervise',action='store_true');internal.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.supervise or a.worker:
        config=json.load(sys.stdin)
        return supervise(config) if a.supervise else worker(config)
    if not a.sweep or not re.fullmatch('scripted_multi_color_models_[A-Za-z0-9_]+',a.sweep):p.error('Choose a saved multi-cube sweep name')
    if not re.fullmatch('[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+',a.repo):p.error('Invalid HF repository name')
    budget=timeout_budget(a.estimated_seconds)
    source=SWEEPS/a.sweep;completed_payload(source)
    pod_id,key=preflight()
    from cloud.scripted_eval_control import active_training
    if active_training():p.error('Refusing to stop a pod with active training')
    token=getpass.getpass('Hugging Face WRITE token (hidden): ').strip()
    if not token or '{{' in token:raise ValueError('Provide an HF write token')
    from huggingface_hub import HfApi
    api=HfApi(token=token);api.whoami()
    api.create_repo(a.repo,repo_type='dataset',private=True,exist_ok=True)
    if not api.repo_info(a.repo,repo_type='dataset').private:raise ValueError('Choose a private repository')
    receipt=Path('/workspace/panda-training/reports/hf-archives')/a.sweep;receipt.mkdir(parents=True,exist_ok=True)
    lock=(receipt/'upload.lock').open('a')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('An archive upload is already running')
    config=dict(lock_fd=lock.fileno(),source=str(source),receipt=str(receipt),staging=str(Path('/root/panda-hf-archives')/a.sweep),
        repo=a.repo,token=token,pod_id=pod_id,pod_key=key,estimated_seconds=a.estimated_seconds)
    env=os.environ.copy()
    for name in ('HF_TOKEN','RUNPOD_API_KEY'):env.pop(name,None)
    with (receipt/'upload.log').open('a',buffering=1) as log:
        process=subprocess.Popen([sys.executable,'-u','-m','cloud.archive_sweep','--supervise'],
            cwd=PROJECT_ROOT,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        process.stdin.write(json.dumps(config).encode());process.stdin.close()
    print('Detached HF archive upload started. Mac may disconnect.',flush=True)
    print('Log:',receipt/'upload.log',flush=True)
    print(f"Inner limit: {budget['inner_seconds']/3600:g} h; outer budget: {budget['outer_seconds']/3600:g} h.",flush=True)
    print('Pod stop requested after verified upload, failure or timeout; original results remain on the volume.',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
