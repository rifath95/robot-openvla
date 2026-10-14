"""Archive a completed multi-cube sweep to HF; detached, verified, bounded stop."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
    if p.get('launcher_error') or len(p['trials'])!=len(specs):
        raise ValueError('Archive requires a completed simulator sweep')
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


def package(source,staging,payload,*,part_cache=None,on_part=None):
    groups=defaultdict(list)
    for r in payload['trials']:
        model='base' if r['checkpoint']==0 and payload['plan'].get('include_base_model') else f"checkpoint_{r['checkpoint']:06d}"
        count=r.get('cube_count') or len(r.get('cube_colors') or (['red'] if payload['plan']['task'] in ('touch','lift','place_bowl') else ['red','blue']))
        palette=r.get('color_set') or (r.get('color_pair') or payload['plan']['task']).replace(':','-')
        groups[f'{model}_{count}c_{palette}'].append(r['directory'])
    manifest=[]
    source_revision=digest(source/'results.json')
    if part_cache is not None:part_cache.mkdir(parents=True,exist_ok=True)
    for label,folders in sorted(groups.items()):
        target=staging/'archives'/f'{label}.tar.gz';target.parent.mkdir(parents=True,exist_ok=True)
        cached=part_cache/(label+'.tar.gz') if part_cache is not None else None
        marker=part_cache/(label+'.json') if part_cache is not None else None
        saved=json.loads(marker.read_text()) if marker is not None and marker.exists() else {}
        if (cached is not None and cached.is_file() and saved.get('source_revision')==source_revision
            and saved.get('trial_directories')==folders and saved.get('bytes')==cached.stat().st_size):
            print(f'Reusing packed part {label}',flush=True)
            shutil.copyfile(cached,target)
            entry={k:saved[k] for k in ('path','bytes','sha256','trials')}
        else:
            print(f'Packing {label}: {len(folders)} trials',flush=True)
            temporary=target.with_suffix('.partial')
            with tarfile.open(temporary,'w:gz',compresslevel=1) as archive:
                for folder in folders:add_checked(archive,source/folder,Path(folder))
            temporary.replace(target)
            entry=dict(path=str(target.relative_to(staging)),bytes=target.stat().st_size,sha256=digest(target),trials=len(folders))
            if cached is not None:
                temporary_cache=cached.with_suffix('.partial')
                shutil.copyfile(target,temporary_cache)
                temporary_cache.replace(cached)
                write_status(marker,dict(entry,source_revision=source_revision,trial_directories=folders))
        manifest.append(entry)
        if on_part is not None:on_part(target,entry)
    reports=staging/'reports';reports.mkdir(exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and not path.is_symlink() and path.suffix in ('.json','.jsonl','.md','.log','.txt'):
            shutil.copyfile(path,reports/path.name)
    for path in sorted(reports.iterdir()):manifest.append(dict(path=str(path.relative_to(staging)),bytes=path.stat().st_size,sha256=digest(path)))
    (staging/'manifest.json').write_text(json.dumps(dict(sweep=source.name,source_revision=source_revision,trials=len(payload['trials']),files=manifest),indent=2)+'\n')
    return manifest


def upload_verify(staging,repo,prefix,token,cache,*,archives_uploaded=False):
    from huggingface_hub import HfApi,hf_hub_download
    api=HfApi(token=token)
    info=api.repo_info(repo,repo_type='dataset')
    if not info.private:raise ValueError('Archive repository must be private')
    print('Uploading final manifest and reports...' if archives_uploaded else 'Uploading archive parts and reports to Hugging Face...',flush=True)
    selection=dict(allow_patterns=['reports/*','manifest.json']) if archives_uploaded else {}
    commit=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=str(staging),path_in_repo=prefix,
        commit_message=f'Archive completed simulator sweep {prefix}',**selection)
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
        verified_files=len(files),verified_bytes=sum(f['bytes'] for f in files),trials=manifest['trials'],
        source_revision=manifest.get('source_revision'),verified=True)



def remote_part_matches(api,repo,remote,entry):
    """One remote metadata lookup, rather than retransferring a committed part."""
    records=api.get_paths_info(repo_id=repo,repo_type='dataset',paths=[remote])
    if not records:return False
    lfs=getattr(records[0],'lfs',None)
    sha=lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
    return getattr(records[0],'size',None)==entry['bytes'] and sha==entry['sha256']


def pipeline_archive(source,staging,payload,repo,token,receipt):
    from huggingface_hub import HfApi
    api=HfApi(token=token)
    if not api.repo_info(repo,repo_type='dataset').private:raise ValueError('Archive repository must be private')
    source_revision=digest(source/'results.json')
    part_cache=receipt/'packed-parts'
    uploaded=receipt/'uploaded-parts';uploaded.mkdir(parents=True,exist_ok=True)
    futures=[]
    def upload_part(path,entry):
        remote=source.name+'/'+entry['path']
        if remote_part_matches(api,repo,remote,entry):
            print('Already uploaded:',entry['path'],flush=True)
        else:
            print('Uploading ready part:',entry['path'],flush=True)
            api.upload_file(repo_id=repo,repo_type='dataset',path_or_fileobj=str(path),path_in_repo=remote,
                commit_message=f'Archive part {path.name}')
        write_status(uploaded/(path.stem+'.json'),dict(entry,source_revision=source_revision,repo_id=repo,path_in_repo=remote))
    with ThreadPoolExecutor(max_workers=1) as uploader:
        def ready(path,entry):
            for f in futures:
                if f.done():f.result()  # Stop preparing new parts after an upload failure.
            futures.append(uploader.submit(upload_part,path,entry))
        package(source,staging,payload,part_cache=part_cache,on_part=ready)
        for f in futures:f.result()
    return upload_verify(staging,repo,source.name,token,staging.parent/(source.name+'-verification'),archives_uploaded=True)

def cleanup_verified(config):
    """Delete only a verified sweep and its packed cache, retaining provenance/reports."""
    source=Path(config['source']);receipt=Path(config['receipt'])
    verified=json.loads((receipt/'receipt.json').read_text())
    if (verified.get('verified') is not True or verified.get('repo_id')!=config['repo']
        or verified.get('path_in_repo')!=source.name or not verified.get('revision')
        or source.is_symlink() or source.resolve().parent!=SWEEPS.resolve()
        or not re.fullmatch('scripted_[A-Za-z0-9_]+',source.name)):
        raise ValueError('Refusing cleanup without the matching verified archive and sweep path')
    reports=Path(config['staging'])/'reports'
    if not reports.is_dir():raise ValueError('Retained compact reports are missing; no cleanup performed')
    shutil.copytree(reports,receipt/'reports',dirs_exist_ok=True)
    removed=[]
    folders=(source,receipt/'packed-parts')
    if any(folder.is_symlink() for folder in folders):raise ValueError('Refusing cleanup through a symlink')
    for folder in folders:
        if folder.exists():
            print(f'Deleting verified volume copy: {folder}',flush=True)
            shutil.rmtree(folder);removed.append(str(folder))
    write_status(receipt/'cleanup.json',dict(verified_revision=verified['revision'],removed=removed,
        reports_retained=str(receipt/'reports'),completed_at_utc=datetime.now(timezone.utc).isoformat()))
    print('Verified source and packing cache cleaned; reports, receipt and logs retained.',flush=True)


def worker(config):
    source=Path(config['source']);receipt=Path(config['receipt']);status=receipt/'status.json'
    try:
        payload=completed_payload(source)
        staging=Path(config['staging']);staging.mkdir(parents=True,exist_ok=True)
        # Avoid an extra full NFS metadata traversal before packing.
        minimum_gib=max(16,16*len(payload['trials'])/1080)
        if shutil.disk_usage(staging).free<minimum_gib*1024**3:raise ValueError('Insufficient container-disk space for packing and verification')
        cached_bytes=sum(p.stat().st_size for p in (receipt/'packed-parts').glob('*.tar.gz'))
        minimum_volume_gib=max(.5,6*len(payload['trials'])/1080-cached_bytes/1024**3)
        if shutil.disk_usage(receipt).free<minimum_volume_gib*1024**3:
            raise ValueError('Insufficient volume space for reusable archive parts')
        write_status(status,dict(phase='packing_and_uploading',source=str(source),minimum_container_free_gib=minimum_gib,
            minimum_volume_free_gib=minimum_volume_gib))
        verified=pipeline_archive(source,staging,payload,config['repo'],config['token'],receipt)
        verified['verified_at_utc']=datetime.now(timezone.utc).isoformat()
        write_status(receipt/'receipt.json',verified)
        write_status(status,dict(phase='verified',**verified))
        if config.get('delete_after_verification'):
            cleanup_verified(config)
        else:
            print('Upload verified. Original sweep remains on the network volume.',flush=True)
        return 0
    except Exception as exc:
        # Log type only: third-party HTTP exceptions can contain credential-bearing requests.
        write_status(status,dict(phase='failed',error_type=type(exc).__name__))
        print(f'Archive failed ({type(exc).__name__}); original results are retained.',flush=True)
        return 1


def worker_batch(config):
    """Sequential archives under one shutdown supervisor; retain per-sweep receipts."""
    receipt=Path(config['receipt']);verified=[]
    for job in config['jobs']:
        print(f"Archiving sweep: {Path(job['source']).name}",flush=True)
        selected=dict(job,repo=config['repo'],token=config['token'],delete_after_verification=False)
        if worker(selected)!=0:
            write_status(receipt/'status.json',dict(phase='failed',completed_sweeps=[v['sweep'] for v in verified],failed_sweep=Path(job['source']).name))
            return 1
        result=json.loads((Path(job['receipt'])/'receipt.json').read_text())
        if result.get('verified') is not True:raise ValueError('Sweep verification receipt missing')
        verified.append(dict(sweep=Path(job['source']).name,receipt=result))
    write_status(receipt/'receipt.json',dict(verified=True,sweeps=verified))
    if config.get('delete_after_verification'):
        for job in config['jobs']:
            cleanup_verified(dict(job,repo=config['repo']))
    write_status(receipt/'status.json',dict(phase='verified',completed_sweeps=[v['sweep'] for v in verified]))
    print('All requested sweep uploads verified.',flush=True)
    return 0


def supervise(config):
    receipt=Path(config['receipt']);budget=timeout_budget(config['estimated_seconds']);process=None
    started=time.monotonic();outcome='supervisor_failed'
    try:
        worker_config={k:v for k,v in config.items() if k not in ('pod_id','pod_key','lock_fd','lock_fds')}
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
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--sweep');selection.add_argument('--sweeps',nargs='+',help='Upload multiple sweeps sequentially and stop once afterward')
    p.add_argument('--repo',default='Maxalderone/robot-openvla-simulator-results')
    p.add_argument('--delete-after-verification',action='store_true',help='Delete verified sweep source and packing cache; retain receipts, logs and compact reports')
    p.add_argument('--estimated-seconds',type=float,default=3600)
    internal=p.add_mutually_exclusive_group();internal.add_argument('--supervise',action='store_true');internal.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.supervise or a.worker:
        config=json.load(sys.stdin)
        return supervise(config) if a.supervise else worker_batch(config) if 'jobs' in config else worker(config)
    names=a.sweeps or ([a.sweep] if a.sweep else [])
    if not names or len(set(names))!=len(names) or any(not re.fullmatch('scripted_[A-Za-z0-9_]+',name) for name in names):
        p.error('Choose unique saved simulator sweep names')
    if not re.fullmatch('[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+',a.repo):p.error('Invalid HF repository name')
    budget=timeout_budget(a.estimated_seconds)
    for name in names:completed_payload(SWEEPS/name)
    pod_id,key=preflight()
    from cloud.scripted_eval_control import active_training
    if active_training():p.error('Refusing to stop a pod with active training')
    token=getpass.getpass('Hugging Face WRITE token (hidden): ').strip()
    if not token or '{{' in token:raise ValueError('Provide an HF write token')
    from huggingface_hub import HfApi
    api=HfApi(token=token);api.whoami()
    api.create_repo(a.repo,repo_type='dataset',private=True,exist_ok=True)
    if not api.repo_info(a.repo,repo_type='dataset').private:raise ValueError('Choose a private repository')
    receipts=Path('/workspace/panda-training/reports/hf-archives')
    jobs=[];locks=[]
    for name in names:
        saved=receipts/name;saved.mkdir(parents=True,exist_ok=True)
        lock=(saved/'upload.lock').open('a')
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'An archive upload is already running for {name}')
        locks.append(lock)
        jobs.append(dict(source=str(SWEEPS/name),receipt=str(saved),staging=str(Path('/root/panda-hf-archives')/name)))
    if len(jobs)==1:
        receipt=Path(jobs[0]['receipt'])
        config=dict(jobs[0],lock_fd=locks[0].fileno())
    else:
        batch='batch_'+hashlib.sha256(json.dumps(names).encode()).hexdigest()[:12]
        receipt=receipts/batch;receipt.mkdir(parents=True,exist_ok=True)
        config=dict(jobs=jobs,receipt=str(receipt),lock_fds=[lock.fileno() for lock in locks])
    config.update(repo=a.repo,token=token,pod_id=pod_id,pod_key=key,estimated_seconds=a.estimated_seconds,delete_after_verification=a.delete_after_verification)
    env=os.environ.copy()
    for name in ('HF_TOKEN','RUNPOD_API_KEY'):env.pop(name,None)
    with (receipt/'upload.log').open('a',buffering=1) as log:
        process=subprocess.Popen([sys.executable,'-u','-m','cloud.archive_sweep','--supervise'],
            cwd=PROJECT_ROOT,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(lock.fileno() for lock in locks))
        process.stdin.write(json.dumps(config).encode());process.stdin.close()
    print(f'Detached HF archive upload started for {len(names)} sweep(s). Mac may disconnect.',flush=True)
    print('Log:',receipt/'upload.log',flush=True)
    print(f"Inner limit: {budget['inner_seconds']/3600:g} h; outer budget: {budget['outer_seconds']/3600:g} h.",flush=True)
    print('Verified sources and packing caches will be deleted; receipts, logs and compact reports retained.' if a.delete_after_verification else 'Original results remain on the volume.',flush=True)
    print('Pod stop requested after all selected uploads verify, or on failure/timeout.',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
