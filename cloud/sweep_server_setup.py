"""In-memory checkpoint switching for Mac-driven scripted sweeps only."""
import shlex


def setup_script(base,adapter,runtime):
    return f'''set -euo pipefail
cd /root/robot-openvla
/opt/openvla/bin/python -m cloud.checkpoint_ready {shlex.quote(str(adapter))}
bash scripts/cloud_stop.sh
/opt/openvla/bin/python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",8000)); s.close()'
nohup /opt/openvla/bin/python -u -m cloud.connection_server --mode openvla --device cuda --unnorm-key panda_grasp_v1 --model-dir {shlex.quote(str(base))} --adapter {shlex.quote(str(adapter))} --max-runtime-seconds {int(runtime)} > outputs/cloud_server.log 2>&1 < /dev/null &
printf '%s\\n' "$!" > outputs/cloud_server.pid
'''
