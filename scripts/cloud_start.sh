#!/usr/bin/env bash
# Run on a Linux GPU host. Dependencies/caches stay on its container disk.
set -euo pipefail
cd "$(dirname "$0")/.."
runtime="${1:-1800}"
model_dir="${2:-/workspace/openvla-7b}"
unnorm_key="${3:-bridge_orig}"
[[ "$runtime" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid server runtime'; exit 1; }
mkdir -p outputs
exec 9>outputs/cloud_setup.lock
flock -n 9 || { echo 'Another setup is running'; exit 1; }
[[ "$(findmnt -n -o TARGET -T /workspace)" == /workspace ]] || {
  echo 'Attach the existing persistent model volume at /workspace first.'; exit 1;
}
nvidia-smi
if [[ -s outputs/cloud_server.pid ]]; then
  pid="$(cat outputs/cloud_server.pid)"
  if [[ "$pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'connection_server.py'; then
    echo 'An existing model server is running; reusing it.'
    exit 0
  fi
fi
desired="$(sha256sum requirements-openvla.txt | cut -d ' ' -f 1)"
if [[ -x /opt/openvla/bin/python ]] && [[ -f /opt/openvla/requirements.sha256 ]] && [[ "$(cat /opt/openvla/requirements.sha256)" == "$desired" ]]; then
  python=/opt/openvla/bin/python
  echo 'Using dependencies preinstalled in the Docker image.'
else
  python=.venv-openvla/bin/python
  if [[ ! -x "$python" ]]; then
    python3.11 -m venv .venv-openvla
  fi
  if [[ ! -f .venv-openvla/requirements.sha256 ]] || [[ "$(cat .venv-openvla/requirements.sha256)" != "$desired" ]]; then
    "$python" -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
    "$python" -m pip install -r requirements-openvla.txt
    "$python" -m pip check
    printf '%s\n' "$desired" > .venv-openvla/requirements.sha256
  fi
fi
"$python" -m pip check
"$python" openvla_mac.py --preflight --device cuda
"$python" cloud_startup_metrics.py
# Fail before expensive loading if port 8000 belongs to another process.
"$python" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",8000)); s.close()'
nohup "$python" -u connection_server.py --mode openvla --model-dir "$model_dir" --unnorm-key "$unnorm_key" --device cuda --max-runtime-seconds "$runtime" >outputs/cloud_server.log 2>&1 < /dev/null 9>&- &
printf '%s\n' "$!" >outputs/cloud_server.pid
echo 'Server loading; log: /root/robot-openvla/outputs/cloud_server.log'
echo 'The server timer does not stop the pod or billing.'
