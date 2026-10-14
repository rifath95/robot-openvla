"""Back up approved training checkpoints to private HF, verify, clean volume and stop."""
import argparse
import getpass
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common.paths import PROJECT_ROOT
from cloud.archive_sweep import digest
from training.detached_run import preflight,write_status
from training.benchmark_stop import stop_pod
from cloud.sweep_plan import timeout_budget

ROOT=Path('/workspace/panda-training')
SESSIONS=('panda-train-20261013-074359-896','panda-train-20261013-110217-1084',
          'panda-train-20261014-064135-2604','panda-train-20261014-095425-1136',
          'panda-train-20261014-104416-1195')
DATASETS=('training_four_color_touch_v1','training_lift_v1','training_lift_v2_300','training_place_bowl_v1')


def backup_session(session,api,repo,token,receipt_root,local):
    source=ROOT/'sessions'/session
    checkpoints=sorted((source/'runs/training').glob('checkpoint_*'))
    if not checkpoints:
        previous=receipt_root/session/'receipt.json'
        if previous.is_file() and json.loads(previous.read_text()).get('verified') is True:return
        raise ValueError(f'No checkpoints or verified backup receipt for {session}')
    from cloud.checkpoint_ready import check_ready
    for checkpoint in checkpoints:check_ready(checkpoint)
    staging=local/session;staging.mkdir(parents=True,exist_ok=True)
    records=[]
    for parent,dirs,names in os.walk(source):
        dirs[:]=[name for name in dirs if name not in ('upstream','adapter-tmp','.cache')]
        for name in names:
            original=Path(parent)/name;relative=original.relative_to(source)
            in_checkpoint=any(p.startswith('checkpoint_') for p in relative.parts)
            if not in_checkpoint and original.suffix not in ('.json','.jsonl','.md','.png','.log','.txt') and not name.startswith('events.out.tfevents.'):continue
            if original.is_symlink():raise ValueError('Refusing checkpoint/report symlink')
            target=staging/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original,target)
            records.append(dict(path=str(relative),bytes=target.stat().st_size,sha256=digest(target)))
    required_bytes=sum(r['bytes'] for r in records)
    if shutil.disk_usage(local).free<required_bytes+2*1024**3:raise ValueError('Need local space for verified HF download plus 2 GiB')
    prefix='sessions/'+session
    print(f'Uploading {session}: {len(checkpoints)} checkpoints, {required_bytes/1024**3:.2f} GiB',flush=True)
    commit=api.upload_folder(repo_id=repo,repo_type='model',folder_path=str(staging),path_in_repo=prefix,
        commit_message=f'Back up {session} checkpoints and training reports')
    from huggingface_hub import hf_hub_download
    for record in records:
        downloaded=Path(hf_hub_download(repo, prefix+'/'+record['path'],repo_type='model',revision=commit.oid,
            token=token,cache_dir=str(local/'verification')))
        if downloaded.stat().st_size!=record['bytes'] or digest(downloaded)!=record['sha256']:
            raise ValueError('HF checkpoint verification failed')
    saved=receipt_root/session;saved.mkdir(parents=True,exist_ok=True)
    write_status(saved/'receipt.json',dict(verified=True,repo_id=repo,repo_type='model',revision=commit.oid,
        path_in_repo=prefix,checkpoints=[p.name for p in checkpoints],files=records))
    print(f'Verified {session}; deleting volume checkpoints, retaining training reports.',flush=True)
    for checkpoint in checkpoints:shutil.rmtree(checkpoint)
    write_status(saved/'cleanup.json',dict(removed_checkpoints=[str(p) for p in checkpoints]))
    shutil.rmtree(staging)
    # Verification cache is container-only and can be released between sessions.
    shutil.rmtree(local/'verification',ignore_errors=True)


def worker(config):
    from huggingface_hub import HfApi
    api=HfApi(token=config['token']);repo=config['repo'];receipt=Path(config['receipt'])
    local=Path('/root/panda-checkpoint-backup');local.mkdir(parents=True,exist_ok=True)
    for session in SESSIONS:backup_session(session,api,repo,config['token'],receipt,local)
    # Datasets are already committed to their separate HF repositories; keep provenance records.
    for name in DATASETS:
        path=ROOT/name
        if path.exists():shutil.rmtree(path)
    for name in ('diagnostics','runs'):
        path=ROOT/name
        if path.exists():shutil.rmtree(path)
    write_status(receipt/'cleanup_complete.json',dict(complete=True,removed_datasets=list(DATASETS)))
    print('Checkpoint backups verified and approved volume cleanup complete.',flush=True)
    return 0


def supervise(config):
    receipt=Path(config['receipt']);budget=timeout_budget(config['estimated_seconds']);process=None
    outcome='failed'
    try:
        payload={k:v for k,v in config.items() if k not in ('pod_id','pod_key')}
        process=subprocess.Popen([sys.executable,'-u','-m','cloud.archive_training_checkpoints','--worker'],
            cwd=PROJECT_ROOT,stdin=subprocess.PIPE,start_new_session=True)
        process.stdin.write(json.dumps(payload).encode());process.stdin.close()
        try:outcome='completed' if process.wait(timeout=budget['inner_seconds'])==0 else 'failed'
        except subprocess.TimeoutExpired:outcome='timeout'
    finally:
        if process is not None and process.poll() is None:
            from training.benchmark_supervisor import kill_benchmark_tree
            kill_benchmark_tree(process)
        try:write_status(receipt/'status.json',dict(outcome=outcome,timeout_budget=budget))
        except OSError:print('Could not save status; still requesting stop.',flush=True)
        stop_pod(config['pod_id'],config['pod_key'],receipt/'pod_stop_status.json',reason='Checkpoint backup and cleanup: '+outcome)
    return 0 if outcome=='completed' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='Maxalderone/robot-openvla-checkpoints')
    p.add_argument('--estimated-seconds',type=float,default=3600)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--worker',action='store_true');mode.add_argument('--supervise',action='store_true')
    a=p.parse_args()
    if a.worker or a.supervise:
        config=json.load(sys.stdin)
        try:return worker(config) if a.worker else supervise(config)
        except Exception as exc:
            print(f'Checkpoint archive failed: {type(exc).__name__}',flush=True)
            return 1
    timeout_budget(a.estimated_seconds)
    pod,key=preflight()
    from cloud.scripted_eval_control import active_training
    if active_training():p.error('Use a separate transfer pod without active training')
    token=getpass.getpass('Hugging Face WRITE token (hidden): ').strip()
    if not token:raise ValueError('Write token required')
    from huggingface_hub import HfApi
    api=HfApi(token=token);api.whoami();api.create_repo(a.repo,repo_type='model',private=True,exist_ok=True)
    if not api.repo_info(a.repo,repo_type='model').private:raise ValueError('Use a private model repository')
    receipt=ROOT/'reports/hf-checkpoints';receipt.mkdir(parents=True,exist_ok=True)
    lock=(receipt/'backup.lock').open('a')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A checkpoint backup is already running')
    config=dict(repo=a.repo,token=token,pod_id=pod,pod_key=key,receipt=str(receipt),estimated_seconds=a.estimated_seconds)
    env=os.environ.copy()
    for name in ('HF_TOKEN','RUNPOD_API_KEY'):env.pop(name,None)
    with (receipt/'backup.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u','-m','cloud.archive_training_checkpoints','--supervise'],
            cwd=PROJECT_ROOT,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        child.stdin.write(json.dumps(config).encode());child.stdin.close()
    print('Detached checkpoint backup and cleanup started. Mac may disconnect.',flush=True)
    print('Log:',receipt/'backup.log',flush=True)
    print('Pod stops after backup/cleanup, failure or timeout. Base model and training reports retained.',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
