"""Cloud-only training supervisor: persistent logs, wall limit, and RunPod stop."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def api_request(pod_id, key, *, stop=False):
    """Never put the credential in a URL, command line, or logged exception."""
    request = Request(
        f'https://rest.runpod.io/v1/pods/{pod_id}' + ('/stop' if stop else ''),
        # The tested RunPod/Cloudflare route rejects urllib's default agent.
        headers={'Authorization': f'Bearer {key}', 'User-Agent': 'Mozilla/5.0'},
        method='POST' if stop else 'GET',
    )
    try:
        with urlopen(request, timeout=20) as response:
            data = response.read()
        return json.loads(data) if data else {}
    except HTTPError as exc:
        raise RuntimeError(f'RunPod API returned HTTP {exc.code}') from None
    except (URLError, TimeoutError, OSError, ValueError):
        raise RuntimeError('RunPod API connection or response failed') from None


def preflight():
    pod_id = os.environ.get('RUNPOD_POD_ID', '')
    key = os.environ.get('RUNPOD_API_KEY', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', pod_id) or not key:
        raise RuntimeError('RUNPOD_POD_ID and RUNPOD_API_KEY must be set on the pod')
    mount = subprocess.run(['findmnt', '-n', '-o', 'TARGET', '-T', '/workspace'],
                           capture_output=True, text=True, check=True)
    if mount.stdout.strip() != '/workspace':
        raise RuntimeError('Attach the network volume at /workspace first')
    pod = api_request(pod_id, key)
    if pod.get('id') != pod_id or not pod.get('networkVolumeId'):
        raise RuntimeError('API must identify this pod and its attached network volume')
    # A read check cannot prove stop permission; failures are recorded and retried.
    return pod_id, key


def write_status(path, values):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(values, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def verify_results(run_dir, expected_updates, *, require_full_validation=True):
    """Report complete versus partial runs; never treat a zero exit as 1,000 updates."""
    rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text().splitlines()]
    updates = [row['update'] for row in rows if 'train_loss' in row]
    last = max(updates, default=0)
    checkpoint = run_dir / f'checkpoint_{last:06d}'
    required = ['adapter_config.json', 'adapter_model.safetensors',
                'dataset_statistics.json', 'training_config.json']
    required_run = ['loss.png', 'experiment.json', 'model_info.json']
    missing = [str(checkpoint / name) for name in required
               if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size]
    missing += [str(run_dir / name) for name in required_run
                if not (run_dir / name).is_file() or not (run_dir / name).stat().st_size]
    full_validation = any(row.get('update') == last and row.get('validation_complete') is True
                          for row in rows)
    validation_present = any(row.get('update') == last and 'validation_complete' in row for row in rows)
    return dict(last_update=last, checkpoint=str(checkpoint), missing_artifacts=missing,
                final_full_validation=full_validation,
                validation_requirement='full_split' if require_full_validation else 'configured_sample',
                training_complete=last >= expected_updates and not missing and validation_present
                and (full_validation or not require_full_validation))


def supervise(command, directory, pod_id, key, *, hard_limit, expected_updates, require_full_validation=True):
    """Always attempt to stop this pod after the child exits, fails, or times out."""
    status = dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
                  pod_id=pod_id, command=command, hard_limit_seconds=hard_limit,
                  phase='training', stop_state='not_requested')
    status_path = directory / 'status.json'
    write_status(status_path, status)
    process = None
    try:
        # Remove the control credential from the training subprocess environment.
        env = os.environ.copy()
        env.pop('RUNPOD_API_KEY', None)
        with (directory / 'console.log').open('w', buffering=1) as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                result = process.wait(timeout=hard_limit)
            except subprocess.TimeoutExpired:
                status['phase'] = 'wall_limit_reached'
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                result = process.returncode
            status['exit_code'] = result
            if status['phase'] == 'training':
                status['phase'] = 'training_exited' if result == 0 else 'training_failed'
        if result == 0:
            status.update(verify_results(directory / 'runs' / 'training', expected_updates,
                                         require_full_validation=require_full_validation))
            status['phase'] = 'completed' if status['training_complete'] else 'incomplete'
    except Exception as exc:
        # Internal errors cannot contain the API credential (which stays here).
        status.update(phase='wrapper_failed', error=str(exc))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        status['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        status['stop_state'] = 'requested'
        write_status(status_path, status)
        os.sync()  # Flush persistent output before asking RunPod to stop.
        for attempt in range(1, 4):
            status['stop_attempt'] = attempt
            try:
                api_request(pod_id, key, stop=True)
                status['stop_state'] = 'accepted'
                write_status(status_path, status)
                print('RunPod accepted the stop request. Confirm stopped status in the portal.', flush=True)
                break
            except RuntimeError as exc:
                status.update(stop_state='failed', stop_error=str(exc))
                write_status(status_path, status)
                if attempt < 3:
                    time.sleep(5)
        else:
            print('Automatic stop failed; stop the pod manually in RunPod.', flush=True)
    return 0 if status.get('training_complete') and status['stop_state'] == 'accepted' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--session-dir', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'training/pick_place_1000.json')
    parser.add_argument('--hard-limit-seconds', type=int, default=5400)
    args = parser.parse_args()
    if args.hard_limit_seconds <= 0:
        parser.error('Hard wall limit must be positive')
    try:
        config = json.loads(args.config.read_text())
        if config['max_steps'] < 1 or config['validation_max_batches'] < 0:
            raise ValueError('Invalid update count or validation limit')
        pod_id, key = preflight()
        if args.check_only:
            print('API and network-volume checks passed. This did not stop the pod.')
            return 0
        if args.session_dir is None:
            parser.error('--session-dir is required')
        directory = args.session_dir.resolve()
        if not directory.is_relative_to(Path('/workspace/panda-training/sessions')):
            parser.error('Session directory must be inside /workspace/panda-training/sessions')
        directory.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, '-u', str(ROOT / 'train_panda.py'),
                   '--dataset', '/workspace/panda-training/training_v1',
                   '--model', '/workspace/openvla-7b', '--config', str(args.config.resolve()),
                   '--output-root', str(directory / 'runs'),
                   '--run-name', 'training']
        return supervise(command, directory, pod_id, key,
                         hard_limit=args.hard_limit_seconds, expected_updates=config['max_steps'],
                         require_full_validation=config['validation_max_batches'] == 0)
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'Not started: {exc}. The pod still bills; stop it manually if needed.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
