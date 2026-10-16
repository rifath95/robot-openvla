"""Prepare a separate cloud GPU and open local interactive bowl controls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import socket
import subprocess
import time
from urllib.request import urlopen

from common.paths import PROJECT_ROOT
from cloud.cloud_session import REPOSITORY, parse_ssh, ssh_arguments
from inference.playground_config import STAGING_DIRECTORY,validate_host


def bootstrap(selected=None):
    option = ' --training-session '+shlex.quote(selected) if selected else ''
    check = shlex.quote('import json; from urllib.request import urlopen; '
        'from inference.playground_config import validate_host; '
        f'validate_host(json.load(urlopen("http://127.0.0.1:8010/health", timeout=5)),{selected!r})')
    return f'''set -euo pipefail
if [ -d /root/robot-openvla/.git ]; then
  git -C /root/robot-openvla pull --ff-only
else
  git clone {shlex.quote(REPOSITORY)} /root/robot-openvla
fi
cd /root/robot-openvla
if curl -fsS http://127.0.0.1:8010/health >/dev/null; then
  /opt/openvla/bin/python -c {check}
  echo 'Existing interactive host retained.'
else
  /opt/openvla/bin/python -u -m cloud.interactive_bowl_host --prepare{option}
  nohup /opt/openvla/bin/python -u -m cloud.interactive_bowl_host > {STAGING_DIRECTORY}/host.log 2>&1 < /dev/null &
fi
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh', required=True)
    parser.add_argument('--key', type=Path)
    parser.add_argument('--training-session')
    parser.add_argument('--reports-only', action='store_true', help='Fetch training reports and leave pod running')
    args = parser.parse_args()
    host, port, key = parse_ssh(args.ssh, args.key)
    ssh = ssh_arguments(host, port, key)
    if args.reports_only:
        # Resolve the source without staging/loading 17 GiB of weights.
        command = '''set -euo pipefail
if [ -d /root/robot-openvla/.git ]; then git -C /root/robot-openvla pull --ff-only; else git clone https://github.com/rifath95/robot-openvla.git /root/robot-openvla; fi
cd /root/robot-openvla
/opt/openvla/bin/python -c '''+shlex.quote(
            'import json; from cloud.interactive_bowl_host import PROFILE; from cloud.color_bowl_sweep import resolve_training; '
            f'print("SOURCE_SESSION="+resolve_training(json.loads(PROFILE.read_text()),selected={args.training_session!r}).name)')
        result = subprocess.run(ssh+['bash -s'], input=command, text=True, capture_output=True, check=True)
        names = [line.split('=', 1)[1] for line in result.stdout.splitlines() if line.startswith('SOURCE_SESSION=')]
        if len(names) != 1:
            raise RuntimeError('Could not identify the 600-demo training session')
        from cloud.training_cloud_session import fetch_reports
        fetch_reports(ssh, names[0])
        print('Reports downloaded. Pod remains running; no model loaded yet.', flush=True)
        return 0
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', 8010))
        except OSError:
            parser.error('Local port 8010 occupied; close the former interactive command first')
    print('Preparing container-disk weights. Pod remains running after testing.', flush=True)
    subprocess.run(ssh+['bash -s'], input=bootstrap(args.training_session), text=True, check=True)
    tunnel = subprocess.Popen(ssh[:-1]+['-o', 'ExitOnForwardFailure=yes', '-N',
        '-L', '127.0.0.1:8010:127.0.0.1:8010', host])
    try:
        deadline = time.monotonic()+60
        while True:
            if tunnel.poll() is not None:
                raise RuntimeError('SSH tunnel exited')
            try:
                with urlopen('http://127.0.0.1:8010/health', timeout=2) as response:
                    health = json.load(response)
                validate_host(health,args.training_session)
                print('Training session:', health['training_session'], flush=True)
                break  # Controls can open while the first checkpoint loads.
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError(f'Host did not start; inspect {STAGING_DIRECTORY}/host.log')
                time.sleep(1)
        destination = PROJECT_ROOT/'outputs/simulator_runs/interactive'/datetime.now(timezone.utc).strftime('bowl_%Y%m%d_%H%M%S')
        from inference.interactive_bowl import serve
        serve('http://127.0.0.1:8010', destination)
    finally:
        tunnel.terminate()
        try:
            tunnel.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tunnel.kill()
            tunnel.wait()
        print('Local session ended. Model remains loaded on cloud pod. Stop the pod in RunPod when finished.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
