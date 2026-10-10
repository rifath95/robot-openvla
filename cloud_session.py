"""One Mac command: set up a GPU host, tunnel, then run the paused local loop."""

import argparse
from datetime import datetime, timezone
import uuid
import json
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
REPOSITORY = 'https://github.com/rifath95/robot-openvla.git'


def parse_ssh(command, key_override=None):
    """Parse RunPod's direct TCP command without executing pasted shell text."""
    tokens = shlex.split(command)
    if not tokens or tokens.pop(0) != 'ssh':
        raise ValueError('Paste the complete SSH over exposed TCP command, starting with ssh')
    host = None
    port = 22
    key = Path.home() / '.ssh/id_ed25519'
    while tokens:
        token = tokens.pop(0)
        if token in ('-p', '-i'):
            if not tokens:
                raise ValueError(f'Missing value for {token}')
            value = tokens.pop(0)
            if token == '-p':
                port = int(value)
            else:
                key = Path(value).expanduser()
        elif host is None and re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_-]*@[a-zA-Z0-9.:-]+', token):
            host = token
        else:
            raise ValueError('Use the direct TCP command with only user@host, -p, and -i')
    if not host or not 1 <= port <= 65535:
        raise ValueError('Invalid SSH host or port')
    personal_key = Path.home() / '.ssh/id_ed25519_maxalderone'
    if key_override:
        key = Path(key_override).expanduser()
    elif key == Path.home() / '.ssh/id_ed25519' and personal_key.is_file():
        key = personal_key  # RunPod displays a generic key path, not this Mac's key.
    if not key.is_file() or key.suffix == '.pub':
        raise ValueError(f'Private key not found: {key}. Supply --key with its actual path.')
    return host, port, key


def ssh_arguments(host, port, key):
    return ['ssh', '-F', '/dev/null', '-o', 'IdentitiesOnly=yes',
            '-o', 'PasswordAuthentication=no', '-o', 'KbdInteractiveAuthentication=no',
            '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
            '-o', 'ServerAliveCountMax=3', '-i', str(key), '-p', str(port), host]


def bootstrap_script(runtime, adapter=None):
    # All variable arguments are shell-quoted; pasted SSH text is never a shell command.
    model_setup = ''
    model_args = ''
    if adapter:
        # Merge on container disk, leaving the persistent base and adapter intact.
        model_setup = f'''test -x /opt/openvla/bin/python || {{ echo 'Use the training Docker template for adapter evaluation.'; exit 1; }}
test -f {shlex.quote(str(adapter) + '/adapter_config.json')} || {{ echo 'Adapter checkpoint not found on the attached volume.'; exit 1; }}
bash scripts/cloud_stop.sh
merged_dir="$(mktemp -d /root/panda-evaluation-XXXXXXXX)"
/opt/openvla/bin/python -m training.merge_adapter --base-model /workspace/openvla-7b --adapter {shlex.quote(str(adapter))} --output "$merged_dir/model"
'''
        model_args = ' "$merged_dir/model" panda_grasp_v1'
    return f'''set -euo pipefail
if [[ ! -d /root/robot-openvla ]]; then
  git clone {shlex.quote(REPOSITORY)} /root/robot-openvla
else
  git -C /root/robot-openvla pull --ff-only
fi
cd /root/robot-openvla
{model_setup}bash scripts/cloud_start.sh {int(runtime)}{model_args}
'''


def wait_for_server(tunnel, seconds, check_liveness=None):
    deadline = time.monotonic() + seconds
    next_update = 0
    while time.monotonic() < deadline:
        if tunnel.poll() is not None:
            raise RuntimeError('SSH tunnel exited. Check the connection and whether local port 8000 is free.')
        try:
            with urlopen('http://127.0.0.1:8000/health', timeout=2) as response:
                health = json.load(response)
            if health.get('status') == 'ready' and health.get('mode') == 'openvla' and health.get('model_loaded') is True:
                return health
        except (URLError, OSError, ValueError):
            pass
        if time.monotonic() >= next_update:
            if check_liveness is not None:
                check_liveness()
            print('Waiting for OpenVLA to load into GPU memory...', flush=True)
            next_update = time.monotonic() + 20
        time.sleep(1)
    raise TimeoutError('Model did not become ready before the startup timeout. Inspect the cloud server log.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh', help='Complete SSH over exposed TCP command copied from RunPod')
    parser.add_argument('--key', type=Path, help='Override the local private-key path')
    parser.add_argument('--adapter', help='Cloud checkpoint directory to merge and evaluate; requires the training template')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--max-consecutive-rejections', type=int, default=3)
    parser.add_argument('--instruction', default='pick up the red cube')
    parser.add_argument('--max-runtime-seconds', type=int, default=900, help='Local active limit, excluding pauses')
    parser.add_argument('--server-runtime-seconds', type=int, default=1800, help='Server wall limit AFTER readiness; does not stop pod billing')
    parser.add_argument('--startup-timeout-seconds', type=int, default=600, help='Wait for loading after dependency setup')
    parser.add_argument('--setup-only', action='store_true', help='Keep server/tunnel ready without opening the simulator')
    args = parser.parse_args()
    if any(value <= 0 for value in (args.steps, args.max_consecutive_rejections, args.max_runtime_seconds, args.server_runtime_seconds, args.startup_timeout_seconds)):
        parser.error('Step counts and time limits must be positive')
    try:
        host, port, key = parse_ssh(args.ssh or input('Paste SSH over exposed TCP command: '), args.key)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.setup_only and not (ROOT / '.venv/bin/python').is_file():
        parser.error('The local simulator .venv is missing; prepare the Mac simulation environment first')
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', 8000))
        except OSError:
            parser.error('Local port 8000 is occupied. Stop the old tunnel/server first.')
    ssh = ssh_arguments(host, port, key)
    tunnel = None
    remote_started = False
    print(f'Connecting to {host}:{port} using {key}', flush=True)
    print('This command does not launch or terminate a rented pod. Stop it in RunPod after testing.', flush=True)
    try:
        # Host-key confirmation and key passphrase, if any, use the terminal.
        subprocess.run(ssh + ['true'], check=True)
        print('Updating cloud code and preparing the server...', flush=True)
        remote_started = True  # Also clean up if interrupted just after background launch.
        subprocess.run(ssh + ['bash -s'], input=bootstrap_script(args.server_runtime_seconds, args.adapter).encode(), check=True)
        tunnel = subprocess.Popen(ssh[:-1] + ['-o', 'ExitOnForwardFailure=yes', '-N',
                                  '-L', '127.0.0.1:8000:127.0.0.1:8000', host])
        def check_liveness():
            command = ('pid=$(cat /root/robot-openvla/outputs/cloud_server.pid); '
                       'case "$pid" in ""|*[!0-9]*) exit 1;; esac; '
                       'kill -0 "$pid" && test -r /proc/"$pid"/cmdline && '
                       'tr "\\0" " " </proc/"$pid"/cmdline | grep -q connection_server.py')
            check = subprocess.run(ssh + [command], timeout=20, check=False)
            if check.returncode:
                raise RuntimeError('The cloud server exited during loading; see its log below.')
        health = wait_for_server(tunnel, args.startup_timeout_seconds, check_liveness)
        print(f"Ready. Model loading took {health.get('model_load_seconds')} seconds.", flush=True)
        startup_info = dict(health.get('startup_info', {}))
        startup_info.setdefault('model_load_seconds', health.get('model_load_seconds'))
        if args.adapter:
            startup_info['adapter_checkpoint'] = args.adapter
        startup_info['ready_at_utc'] = datetime.now(timezone.utc).isoformat()
        startup_info['note'] = 'Session startup measurements reused for subsequent trials; the model is not reloaded per trial.'
        session_dir = ROOT / 'outputs' / f"cloud_session_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        session_dir.mkdir(parents=True, exist_ok=False)
        startup_path = session_dir / 'startup_info.json'
        startup_path.write_text(json.dumps(startup_info, indent=2, allow_nan=False) + '\n')
        print(f'Startup measurements saved: {startup_path}', flush=True)
        if args.setup_only:
            print('Server and tunnel ready. Run remote_loop.py in another Mac terminal; Ctrl+C disconnects.', flush=True)
            while tunnel.poll() is None:
                time.sleep(1)
            raise RuntimeError('SSH tunnel closed')
        while True:
            print('Opening paused simulator: Space starts/pauses/resumes; Esc stops the local trial.', flush=True)
            result = subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'remote_loop.py'),
                                     '--steps', str(args.steps), '--instruction', args.instruction,
                                     '--startup-info', str(startup_path),
                                     '--max-consecutive-rejections', str(args.max_consecutive_rejections),
                                     '--max-runtime-seconds', str(args.max_runtime_seconds)], cwd=ROOT)
            if result.returncode:
                print('Trial failed; inspect its status.json and the terminal output.', flush=True)
            input('Trial finished. Server stays loaded. Enter runs another trial; Ctrl+C ends the session: ')
            wait_for_server(tunnel, 5)
    except KeyboardInterrupt:
        print('\nEnding session...', flush=True)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError, TimeoutError, EOFError) as exc:
        print(f'Session ended: {exc}', file=sys.stderr, flush=True)
        if remote_started:
            try:
                subprocess.run(ssh + ['tail -n 30 /root/robot-openvla/outputs/cloud_server.log'], timeout=20, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        return 1
    finally:
        if remote_started:
            try:
                subprocess.run(ssh + ['if [ -f /root/robot-openvla/scripts/cloud_stop.sh ]; then bash /root/robot-openvla/scripts/cloud_stop.sh; fi'], timeout=20, check=False)
            except (OSError, subprocess.TimeoutExpired):
                print('Could not confirm server shutdown. Check the pod portal.', flush=True)
        if tunnel is not None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait()
        print('Stop or terminate the pod in RunPod to stop GPU billing.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
