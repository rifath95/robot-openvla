#!/usr/bin/env bash
# Stop only this project's recorded server, never the rented pod.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -s outputs/cloud_server.pid ]]; then
  pid="$(cat outputs/cloud_server.pid)"
  if [[ "$pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'connection_server.py'; then
    kill "$pid"
    echo 'Model server stopped. The pod still bills until stopped in the portal.'
  fi
fi
