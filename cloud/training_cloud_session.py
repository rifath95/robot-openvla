"""Launch detached cloud training, download reports, and acknowledge automatic stop.

The pod must already be running with the training image, dataset, base weights,
network volume, and RunPod stop secret. Sleeping the Mac leaves training running.
"""

from common.output_layout import output_path
import argparse
import getpass
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import tempfile

from cloud.cloud_session import ROOT, REPOSITORY, parse_ssh, ssh_arguments


SESSION_PATTERN = r'panda-train-[A-Za-z0-9_-]+'
SESSION_ROOT = '/workspace/panda-training/sessions'
LOCAL_ROOT = output_path('phase3_training')

ARCHIVE_SCRIPT = r'''
from pathlib import Path
import sys,tarfile
root=Path(sys.argv[1])
if not (root/'status.json').is_file():raise SystemExit('Training session not found')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for path in sorted(root.rglob('*')):
        relative=path.relative_to(root)
        if any(part in ('upstream','adapter-tmp') for part in relative.parts):continue
        if path.is_symlink():raise SystemExit('Refusing a symlink in training reports')
        if not path.is_file():continue
        if path.suffix not in ('.json','.jsonl','.md','.log','.png') and not path.name.startswith('events.out.tfevents.'):
            continue
        archive.add(path,arcname=str(relative),recursive=False)
'''


def session_path(name):
    if not re.fullmatch(SESSION_PATTERN, name):
        raise ValueError('Use the panda-train-... session name printed by the launcher')
    return f'{SESSION_ROOT}/{name}'


def setup_script():
    return f'''set -euo pipefail
mountpoint -q /workspace
test -x /opt/openvla/bin/python
if [[ ! -d /root/robot-openvla ]]; then
  git clone {shlex.quote(REPOSITORY)} /root/robot-openvla
else
  git -C /root/robot-openvla pull --ff-only
fi
cd /root/robot-openvla
'''


def launch_script(config, dataset, adapter=None, download_window=120):
    if not re.fullmatch(r'training/[A-Za-z0-9_-]+\.json', config):
        raise ValueError('Select a JSON configuration directly inside training/')
    if not re.fullmatch(r'/workspace/panda-training/[A-Za-z0-9_-]+', dataset):
        raise ValueError('Select a dataset bundle directly inside /workspace/panda-training')
    if not 1 <= download_window <= 3600:
        raise ValueError('Download window must be between 1 and 3600 seconds')
    if adapter and (not re.fullmatch(r'/workspace/panda-training/[A-Za-z0-9_./-]+', adapter)
                    or '..' in PurePosixPath(adapter).parts):
        raise ValueError('Select an adapter checkpoint inside /workspace/panda-training')
    args = ' '.join(shlex.quote(value) for value in (config, dataset))
    if adapter:
        args += ' ' + shlex.quote(adapter)
    return setup_script() + f'''command -v tmux >/dev/null || (apt-get update && apt-get install -y tmux)
PANDA_TRAIN_DOWNLOAD_GRACE_SECONDS={download_window} bash scripts/train_detached.sh {args}
'''


def extract_reports(archive_path, destination):
    """Validate every member before replacing any local report; never import weights."""
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) > 10000 or sum(member.size for member in members) > 256 * 1024**2:
            raise ValueError('Training report archive exceeds expected size')
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                or any(part in ('upstream', 'adapter-tmp') for part in path.parts)
                or (path.suffix not in ('.json', '.jsonl', '.md', '.log', '.png')
                    and not (path.parent == PurePosixPath('runs/training/tensorboard')
                             and path.name.startswith('events.out.tfevents.')))
                or not destination.joinpath(*path.parts).resolve().is_relative_to(destination.resolve())):
                raise ValueError(f'Unexpected training archive entry: {member.name}')
        if not any(member.name == 'status.json' for member in members):
            raise ValueError('Training report archive is missing session status')
        # Validate JSON content before replacing a previous successful download.
        for member in members:
            if PurePosixPath(member.name).suffix == '.json':
                json.loads(archive.extractfile(member).read())
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as output:
                while chunk := source.read(1024**2):
                    output.write(chunk)


def fetch_reports(ssh, name, local_root=LOCAL_ROOT):
    remote_root = session_path(name)
    local_root.mkdir(parents=True, exist_ok=True)
    destination = local_root / name
    print('Downloading training plots, metrics, logs, TensorBoard events and checkpoint metadata; adapter weights stay on the volume.', flush=True)
    command = '/opt/openvla/bin/python -c ' + shlex.quote(ARCHIVE_SCRIPT) + ' ' + shlex.quote(remote_root)
    with tempfile.TemporaryDirectory(prefix='training-download-', dir=local_root) as temp:
        archive_path = Path(temp) / 'reports.tar.gz'
        with archive_path.open('wb') as output:
            subprocess.run(ssh + [command], stdout=output, check=True, timeout=90)
        extract_reports(archive_path, destination)
    print(f'Local training reports: {destination}', flush=True)
    return destination


def supervisor_command(option, name):
    return ('cd /root/robot-openvla && /opt/openvla/bin/python -u -m training.detached_run '
            + option + ' ' + shlex.quote(session_path(name)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh', required=True, help='Current RunPod direct TCP SSH command')
    parser.add_argument('--key', type=Path)
    parser.add_argument('--config', default='training/smoke.json')
    parser.add_argument('--dataset', help='Override the dataset selected by the training profile')
    parser.add_argument('--adapter', help='Continue adapter weights with a fresh optimizer')
    parser.add_argument('--download-window-seconds', type=int, default=120)
    parser.add_argument('--fetch-only', action='store_true')
    parser.add_argument('--prepared-color-bowl',action='store_true',help='Launch the prepared 600-demo training/download/backup workflow on container disk')
    parser.add_argument('--session', help='Saved session name; required for --fetch-only')
    parser.add_argument('--stop-after-fetch', action='store_true', help='Stop the resumed pod after a successful fetch-only download')
    args = parser.parse_args()
    if args.prepared_color_bowl:
        if args.fetch_only or args.adapter or args.dataset or args.config!='training/smoke.json':
            parser.error('--prepared-color-bowl selects its own profile and inputs; use ordinary --fetch-only for later downloads')
        args.config='training/color_place_bowl_600_6000.json'
    try:
        from training.dataset_profile import select_dataset
        args.dataset=str(select_dataset(json.loads((ROOT/args.config).read_text()),args.dataset))
        host, port, key = parse_ssh(args.ssh, args.key)
        script = launch_script(args.config, args.dataset, args.adapter, args.download_window_seconds)
        if args.fetch_only:
            if not args.session:
                parser.error('--fetch-only requires --session')
            session_path(args.session)
        elif args.session or args.stop_after_fetch:
            parser.error('--session and --stop-after-fetch require --fetch-only')
    except ValueError as exc:
        parser.error(str(exc))
    ssh = ssh_arguments(host, port, key)
    print(f'Connecting to {host}:{port}. The pod must already be running.', flush=True)
    name = args.session
    code = 0
    try:
        if not args.fetch_only:
            if args.prepared_color_bowl:
                subprocess.run(ssh+['bash -s'],input=setup_script(),text=True,check=True)
                token=getpass.getpass('Hugging Face WRITE token (hidden): ').strip()
                if not token:raise ValueError('A Hugging Face write token is required')
                remote='cd /root/robot-openvla && /opt/openvla/bin/python -u -m cloud.color_bowl_training --launch --token-stdin --download-window-seconds '+str(args.download_window_seconds)
                result=subprocess.run(ssh+[remote],input=json.dumps(dict(hf_token=token)),text=True,stdout=subprocess.PIPE)
            else:
                result = subprocess.run(ssh + ['bash -s'], input=script, text=True,
                                        stdout=subprocess.PIPE)
            print(result.stdout, end='', flush=True)
            match = re.search(r'^Cloud session started: (' + SESSION_PATTERN + r')$', result.stdout, re.MULTILINE)
            if result.returncode or not match:
                raise RuntimeError('Training was not confirmed started. Check the pod; this launcher did not stop another running job.')
            name = match.group(1)
            print('Detached cloud training confirmed. You may disconnect; cloud saving and cleanup continue.', flush=True)
            print(f'For later download: --fetch-only --session {name}', flush=True)
            code = subprocess.run(ssh + [supervisor_command('--watch-session', name)]).returncode
            if code:
                print('Training watcher ended with an error; attempting to download saved diagnostics.', flush=True)
        destination = fetch_reports(ssh, name)
        status = json.loads((destination / 'status.json').read_text())
        if not args.fetch_only:
            try:
                acknowledged = subprocess.run(ssh + [supervisor_command('--ack-download', name)], timeout=30)
                acknowledged_ok = acknowledged.returncode == 0
            except (OSError, subprocess.SubprocessError):
                acknowledged_ok = False
            if not acknowledged_ok:
                print('Acknowledgment unavailable; cloud supervisor will stop after its bounded download window.', flush=True)
        elif args.stop_after_fetch:
            subprocess.run(ssh + ['bash -s'], input=setup_script(), text=True, check=True)
            subprocess.run(ssh + ['cd /root/robot-openvla && /opt/openvla/bin/python -m training.detached_run --stop-only'], check=True, timeout=90)
        if not args.fetch_only:
            print('Reports saved locally. Cloud supervisor handles pod stopping; confirm Stopped in RunPod.', flush=True)
        if status.get('job_complete') is False:
            raise SystemExit('Reports downloaded. The prepared training/backup job did not fully complete; inspect status.json and console.log. Saved checkpoints remain on the volume.')
        if not status.get('training_complete') or code:
            raise SystemExit('Reports downloaded, but successful training completion was not confirmed. Inspect status.json and console.log.')
    except KeyboardInterrupt:
        raise SystemExit('Local watcher interrupted. If detached training started, it continues and will save results and request pod stop. Fetch reports later using the printed session name.')
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as exc:
        raise SystemExit(f'Training reports were not fully synchronized: {exc}. Saved results remain on the volume. If training started, cloud cleanup still runs; confirm pod status in RunPod.')


if __name__ == '__main__':
    main()
