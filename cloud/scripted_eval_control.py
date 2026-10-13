"""Check or stop the pod used by an explicitly requested scripted evaluation."""
import argparse
from pathlib import Path

from training.benchmark_stop import stop_pod
from training.detached_run import api_request, preflight
from training.pod_environment import recover_pod_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--session', required=True)
    args = parser.parse_args()
    if not args.session.replace('_', '').isalnum():
        parser.error('Invalid session name')
    try:
        if args.check_only:
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
