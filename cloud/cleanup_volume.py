"""Execute the agreed October 16 cleanup; never stop a pod or a sweep."""
import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from common.paths import PROJECT_ROOT
from cloud.bowl_chain_plan import timeout_budget
from training.detached_run import write_status

ROOT = Path('/workspace/panda-training')
REPO = 'Maxalderone/robot-openvla-checkpoints'
SELECTED = {
    'panda-train-20261013-074359-896': (100,),
    'panda-train-20261013-110217-1084': (300, 1000),
    'panda-train-20261014-095425-1136': (1000,),
}
ALREADY_BACKED = (
    'panda-train-chain-20261014-174530-4acde087',
    'panda-train-color-bowl-20261015-094416-4344d31f',
    'panda-train-color-bowl-cont-20261015-151409-140b3657',
)
PILOTS = ('panda-train-20261014-064135-2604',
          'panda-train-20261014-104416-1195',
          'panda-train-20261013-034928-1786')
DATASETS = ('training_four_color_touch_v1', 'training_lift_v1',
            'training_lift_v2_300', 'training_place_bowl_v1')
OLD_SWEEP = 'scripted_color_lift_models_20261014_144240_51b35ee5'


def check_deadline(config):
    if time.monotonic() >= config['inner_deadline']:
        raise TimeoutError('Inner deadline reached; no new cleanup work started. Rerun to continue.')


def checkpoint_folders(session):
    return sorted(p for p in (ROOT/'sessions'/session/'runs/training').glob('checkpoint_*')
                  if re.fullmatch(r'checkpoint_[0-9]{6}', p.name))


def archive_selected(session, updates, config, api):
    """Archive selected weights plus reports, without staging other checkpoints."""
    from cloud.archive_sweep import digest
    from cloud.chain_archive import verify_files
    from cloud.checkpoint_ready import check_ready
    receipt = Path(config['reports'])/(session+'.json')
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if (saved.get('verified') is True and saved.get('repo_id') == REPO and
                saved.get('path_in_repo') == 'sessions/'+session and
                saved.get('checkpoints') == list(updates)):
            print('Reusing verified cleanup receipt:', session, flush=True)
            return
    source = ROOT/'sessions'/session
    for cp in updates:check_ready(source/f'runs/training/checkpoint_{cp:06d}')
    stage = Path(config['local'])/session
    stage.mkdir(parents=True, exist_ok=True)
    records = []
    for parent, dirs, names in os.walk(source):
        dirs[:] = [d for d in dirs if d not in ('upstream', 'adapter-tmp', '.cache') and
                   (not d.startswith('checkpoint_') or d in {f'checkpoint_{cp:06d}' for cp in updates})]
        for name in names:
            original = Path(parent)/name
            relative = original.relative_to(source)
            in_checkpoint = any(p.startswith('checkpoint_') for p in relative.parts)
            if not in_checkpoint and original.suffix not in ('.json', '.jsonl', '.md', '.png', '.log', '.txt') and not name.startswith('events.out.tfevents.'):
                continue
            if original.is_symlink():raise ValueError('Refusing symlink in backup')
            target = stage/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            records.append(dict(path=relative.as_posix(), bytes=target.stat().st_size, sha256=digest(target)))
    for cp in updates:check_ready(stage/f'runs/training/checkpoint_{cp:06d}')
    prefix = 'sessions/'+session
    # If the same files are already on HF, verify them instead of uploading again.
    info = api.model_info(REPO)
    remote = set(api.list_repo_files(REPO, repo_type='model', revision=info.sha))
    if all(prefix+'/'+r['path'] in remote for r in records):
        revision = info.sha
        print('Selected backup already exists; verifying:', session, flush=True)
    else:
        print('Uploading selected checkpoints:', session, list(updates), flush=True)
        revision = api.upload_folder(repo_id=REPO, repo_type='model', folder_path=stage,
            path_in_repo=prefix, commit_message='Archive selected best checkpoints before volume cleanup').oid
    # A presence check alone never permits source deletion.
    verify_files(REPO, 'model', revision, prefix, records, config['token'], config['local'])
    write_status(receipt, dict(verified=True, repo_id=REPO, path_in_repo=prefix,
                              revision=revision, checkpoints=list(updates), files=records))
    print('Verified selected backup:', session, flush=True)
    shutil.rmtree(stage)


def targets():
    result = []
    for session in (*SELECTED, *ALREADY_BACKED, *PILOTS):
        result.extend(checkpoint_folders(session))
    result.extend(ROOT/name for name in DATASETS)
    result.extend((ROOT/'scripted-evaluations'/OLD_SWEEP,
                   ROOT/'reports/hf-archives'/OLD_SWEEP/'packed-parts'))
    return result


def remove_with_progress(path, counters):
    """One traversal, with progress instead of an expensive preliminary inventory."""
    if path.is_symlink():raise ValueError('Refusing deletion through symlink: '+str(path))
    if path.is_dir():
        with os.scandir(path) as entries:
            for entry in entries:remove_with_progress(Path(entry.path), counters)
        path.rmdir()
        counters['directories'] += 1
    elif path.exists():
        size = path.stat().st_size
        path.unlink()
        counters['files'] += 1
        counters['bytes'] += size
    if time.monotonic()-counters['last_log'] >= 3:
        print(f"Removed {counters['files']:,} files / {counters['directories']:,} directories; "
              f"{counters['bytes']/1024**3:.2f} GiB of file contents. Current: {path}", flush=True)
        counters['last_log'] = time.monotonic()


def worker(config):
    from huggingface_hub import HfApi
    os.nice(19)
    api = HfApi(token=config['token'])
    if not api.repo_info(REPO, repo_type='model').private:raise ValueError('Expected private checkpoint repository')
    # All archival requirements must pass before any volume deletion.
    for session, updates in SELECTED.items():
        check_deadline(config)
        archive_selected(session, updates, config, api)
    for session in ALREADY_BACKED:
        receipt = json.loads((ROOT/'sessions'/session/'checkpoint_hf_receipt.json').read_text())
        if (receipt.get('verified') is not True or receipt.get('repo_id') != REPO or
                receipt.get('path_in_repo') != 'sessions/'+session):
            raise ValueError('Existing checkpoint backup not confirmed: '+session)
    # Retain the completed old sweep's aggregate reports outside its recordings.
    old = ROOT/'scripted-evaluations'/OLD_SWEEP
    reports = ROOT/'reports/completed-sweeps'/OLD_SWEEP
    if old.exists():
        reports.mkdir(parents=True, exist_ok=True)
        for p in old.iterdir():
            if p.is_file() and p.suffix in ('.json', '.jsonl', '.md', '.log', '.txt'):
                shutil.copyfile(p, reports/p.name)
    items = targets()
    counters = dict(files=0, directories=0, bytes=0, last_log=time.monotonic())
    for index, path in enumerate(items, 1):
        check_deadline(config)
        # This allowlist excludes the base, current training, active sweep and global volume.
        if not path.is_relative_to(ROOT):raise ValueError('Invalid cleanup path')
        if path.name.startswith('checkpoint_') and path.exists():
            metadata = path/'training_config.json'
            if metadata.is_file():
                saved = path.parent.parent.parent/'checkpoint-metadata'
                saved.mkdir(exist_ok=True)
                shutil.copyfile(metadata, saved/(path.name+'.json'))
        print(f'Removing target {index}/{len(items)}: {path}', flush=True)
        remove_with_progress(path, counters)
        print(f'Finished target {index}/{len(items)}', flush=True)
    write_status(Path(config['reports'])/'cleanup_complete.json',
                 dict(complete=True, removed_targets=[str(p) for p in items],
                      files_removed=counters['files'], logical_bytes_removed=counters['bytes']))
    print('CLEANUP COMPLETE. Reports and current checkpoints retained. Pod and sweep left running.', flush=True)
    print('Network-volume free space: %.2f GiB' % (shutil.disk_usage(ROOT).free/1024**3), flush=True)
    return 0


def supervise(config):
    start = time.monotonic()
    config['inner_deadline'] = start+config['budget']['inner_seconds']
    child = subprocess.Popen([sys.executable, '-u', '-m', 'cloud.cleanup_volume', '--worker'],
        cwd=PROJECT_ROOT, stdin=subprocess.PIPE, start_new_session=True,
        pass_fds=(config['lock_fd'],) if 'lock_fd' in config else ())
    child.stdin.write(json.dumps(config).encode());child.stdin.close()
    try:
        code = child.wait(timeout=config['budget']['outer_seconds'])
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(child.pid, signal.SIGKILL);child.wait()
        print('Outer cleanup deadline reached; cleanup terminated. Pod/sweep untouched.', flush=True)
        code = 124
    write_status(Path(config['reports'])/'status.json', dict(exit_code=code, complete=code==0,
                 elapsed_seconds=time.monotonic()-start, budget=config['budget'], pod_stop_requested=False))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--launch', action='store_true')
    group.add_argument('--supervise', action='store_true')
    group.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:return worker(json.load(sys.stdin))
    if args.supervise:return supervise(json.load(sys.stdin))
    if not ROOT.is_dir():raise SystemExit('Run this command inside the pod, not on the Mac.')
    reports = ROOT/'reports/volume-cleanup-20261016'
    reports.mkdir(parents=True, exist_ok=True)
    lock = (reports/'cleanup.lock').open('a')
    try:fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Cleanup is already running; watch its cleanup.log.')
    from huggingface_hub import HfApi
    token = getpass.getpass('Hugging Face WRITE token (hidden): ').strip()
    if not token:raise SystemExit('Write token required; no cleanup started.')
    HfApi(token=token).whoami()
    local = Path('/root/panda-volume-cleanup-20261016')
    local.mkdir(exist_ok=True)
    if shutil.disk_usage(local).free < 3*1024**3:
        raise SystemExit('Need 3 GiB free on container disk for selective backup and verification.')
    budget = timeout_budget(3600)
    config = dict(reports=str(reports), local=str(local), token=token, budget=budget,lock_fd=lock.fileno())
    env = os.environ.copy()
    for key in ('HF_TOKEN', 'RUNPOD_API_KEY'):env.pop(key, None)
    with (reports/'cleanup.log').open('a', buffering=1) as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'cloud.cleanup_volume', '--supervise'],
            cwd=PROJECT_ROOT, env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, pass_fds=(lock.fileno(),))
        child.stdin.write(json.dumps(config).encode());child.stdin.close()
    print('Detached selective archive + cleanup started. Mac may disconnect.')
    print('Watch:', reports/'cleanup.log')
    print('Budget: 60 min; inner: 75 min; outer: 90 min. This command never stops the pod.')
    return 0


if __name__ == '__main__':raise SystemExit(main())
