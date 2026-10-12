#!/usr/bin/env bash
# Stop only this project's recorded server, never the rented pod.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -s outputs/cloud_server.pid ]]; then
  pid="$(cat outputs/cloud_server.pid)"
  if [[ "$pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Eq 'connection_server.py|cloud.connection_server'; then
    kill "$pid"
    # Wait for GPU resources to be released before a replacement model loads.
    for attempt in {1..100}; do
      if [[ ! -r "/proc/$pid/cmdline" ]] || ! tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Eq 'connection_server.py|cloud.connection_server'; then
        break
      fi
      sleep 0.1
    done
    if [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Eq 'connection_server.py|cloud.connection_server'; then
      echo 'Server has not exited yet; aborting rather than loading another model alongside it.'
      exit 1
    fi
    echo 'Model server stopped. The pod still bills until stopped in the portal.'
  fi
fi
