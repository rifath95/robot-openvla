"""Check or stop the pod used by an explicitly requested scripted evaluation."""
import argparse
from pathlib import Path

from training.benchmark_stop import stop_pod
from training.detached_run import api_request, preflight
from training.pod_environment import recover_pod_environment


def active_training(proc_root=Path('/proc')):
    targets={b'training.detached_run',b'training.train_panda',
             b'/root/robot-openvla/training/detached_run.py',
             b'/root/robot-openvla/training/train_panda.py',
             b'/root/robot-openvla/train_panda.py'}
    for process in proc_root.iterdir():
        if not process.name.isdigit():continue
        try:arguments=set((process/'cmdline').read_bytes().split(b'\0'))
        except OSError:continue
        if arguments.intersection(targets):return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--session', required=True)
    parser.add_argument('--refuse-training',action='store_true')
    args = parser.parse_args()
    if not args.session.replace('_', '').isalnum():
        parser.error('Invalid session name')
    try:
        if args.check_only:
            if args.refuse_training and active_training():
                raise RuntimeError('This pod is running training. Use a separate inference pod; training was left running')
            preflight()
            print('Scripted evaluation API and volume checks passed. Pod remains running.')
            return 0
        # A report-copy or volume error must not prevent requesting a stop.
        pod_id, key = recover_pod_environment()
        if not pod_id or not key or api_request(pod_id, key).get('id') != pod_id:
            raise RuntimeError('Cannot identify this pod with its API secret')
        folder = Path('/workspace/panda-training/scripted-evaluations') / args.session
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return stop_pod(pod_id, key, folder / 'pod_stop_status.json',
                        reason='Scripted simulator evaluation ended; local reports saved')
    except (RuntimeError, OSError) as exc:
        print(f'Pod stop/check failed: {exc}. Check RunPod and stop manually if needed.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
