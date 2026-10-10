#!/usr/bin/env bash
# Run inside a training pod; tmux owns the supervisor after SSH disconnects.
set -euo pipefail
cd "$(dirname "$0")/.."
python=/opt/openvla/bin/python
config="${1:-training/pick_place_1000.json}"
[[ "$#" -le 1 && -f "$config" ]] || { echo 'Usage: bash scripts/train_detached.sh [CONFIG_JSON]'; exit 1; }
[[ -x "$python" ]] || { echo 'Use the training Docker template.'; exit 1; }
command -v tmux >/dev/null || { echo 'Install tmux first: apt-get update && apt-get install -y tmux'; exit 1; }
# Each job uses its own tmux server so it inherits the current secret environment.
if pgrep -f '[t]raining.detached_run --session-dir' >/dev/null; then
  echo 'A detached training supervisor is already running; refusing a second job.'; exit 1
fi
"$python" -m training.detached_run --check-only --config "$config"
session="panda-train-$(date -u +%Y%m%d-%H%M%S)-$$"
session_dir="/workspace/panda-training/sessions/$session"
printf -v worker_command '%q ' "$python" -u -m training.detached_run \
  --session-dir "$session_dir" --config "$config"
tmux -L "$session" new-session -d -s train "$worker_command"
echo "Training configuration: $config"
echo "Cloud session started: $session"
echo "Attach: tmux -L $session attach -t train"
echo 'Detach without stopping: Ctrl+B, then D'
echo "Watch training: tail -F $session_dir/console.log"
echo "Persistent status: $session_dir/status.json"
echo 'Training has a 90-minute hard wall limit, including setup; then the wrapper requests a pod stop.'
echo 'Confirm the pod is stopped in RunPod afterward; an API/network failure can prevent automatic stopping.'
