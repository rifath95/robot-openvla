"""Prepare the persistent checkpoint and record cloud startup measurements."""
import json
import os
import socket
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def prepare():
    model_dir = Path('/workspace/openvla-7b')
    existed = (model_dir / 'export_complete.json').is_file()
    started = time.perf_counter()
    subprocess.run([sys.executable, str(ROOT / 'prepare_cloud_model.py'),
                    '--export-dir', str(model_dir)], check=True)
    preparation_seconds = time.perf_counter() - started
    mount = subprocess.run(['findmnt', '-J', '-T', '/workspace', '-o', 'TARGET,SOURCE,FSTYPE'],
                           capture_output=True, text=True, check=True)
    info = {
        'model_dir': str(model_dir),
        'hostname': socket.gethostname(),
        'visible_cpu_count': os.cpu_count(),
        'visible_host_ram_bytes': os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'),
        'host_note': 'Visible host resources may exceed the pod allocation; not provider billing specifications.',
        'completed_export_present_before_setup': existed,
        'model_preparation_seconds': preparation_seconds,
        'preparation_note': 'Includes export validation; when missing, download and export. Separate from model loading.',
        'volume_mount': json.loads(mount.stdout),
    }
    (ROOT / 'outputs/cloud_startup.json').write_text(json.dumps(info, indent=2) + '\n')


if __name__ == '__main__':
    prepare()
