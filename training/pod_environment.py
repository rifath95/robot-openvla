"""Recover only the pod-control variables omitted by some SSH environments."""
import os
from pathlib import Path


VARIABLES=('RUNPOD_POD_ID','RUNPOD_API_KEY')


def recover_pod_environment(environ=None,container_environment=Path('/proc/1/environ')):
    environ=os.environ if environ is None else environ
    missing=[name for name in VARIABLES if not environ.get(name)]
    if missing:
        try:
            entries=container_environment.read_bytes().split(b'\0')
            for name in missing:
                prefix=(name+'=').encode()
                value=next((entry[len(prefix):].decode() for entry in entries if entry.startswith(prefix)), '')
                if value:environ[name]=value
        except (OSError,UnicodeError):
            pass
    for name in VARIABLES:
        value=environ.get(name,'').strip()
        if not value:
            raise RuntimeError(f'{name} is missing; configure it on the pod before starting')
        if '{{' in value or 'RUNPOD_SECRET' in value:
            raise RuntimeError(f'{name} contains an unresolved secret placeholder')
        environ[name]=value
    return environ['RUNPOD_POD_ID'],environ['RUNPOD_API_KEY']
