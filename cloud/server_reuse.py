"""Identify a live server before reusing its already loaded checkpoint."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.request import urlopen


def requested_identity(base,adapter=None):
    base=Path(base).resolve()
    identity={'base':str(base),'adapter':None,'unnorm_key':'bridge_orig'}
    identity['base_identity']=hashlib.sha256(
        (base/'config.json').read_bytes()+(base/'export_complete.json').read_bytes()).hexdigest()
    if adapter:
        adapter=Path(adapter).resolve()
        weights=adapter/'adapter_model.safetensors'
        stat=weights.stat()
        identity.update(adapter=str(adapter),unnorm_key='panda_grasp_v1',
            adapter_config_sha256=hashlib.sha256((adapter/'adapter_config.json').read_bytes()).hexdigest(),
            adapter_weights_size=stat.st_size,adapter_weights_mtime_ns=stat.st_mtime_ns,
            adapter_weights_inode=stat.st_ino)
        identity['statistics_sha256']=hashlib.sha256((adapter/'dataset_statistics.json').read_bytes()).hexdigest()
    return identity


def process_identity(pid,proc_root=Path('/proc')):
    root=Path(proc_root)/str(pid)
    cmdline=(root/'cmdline').read_bytes().split(b'\0')
    if not any(value in (b'cloud.connection_server',b'connection_server.py') for value in cmdline):
        raise ValueError('Recorded PID is not the model server')
    # stat's command name may contain spaces; starttime is field 22.
    fields=(root/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=fields[19],cmdline=[x.decode() for x in cmdline if x])


def read_health():
    with urlopen('http://127.0.0.1:8000/health',timeout=2) as response:
        return json.load(response)


def reusable(outputs,identity,*,proc_root=Path('/proc'),health_reader=read_health,require_sampling=False):
    try:
        outputs=Path(outputs)
        pid=int((outputs/'cloud_server.pid').read_text().strip())
        record=json.loads((outputs/'cloud_server_identity.json').read_text())
        if record['identity']!=identity or record['process']!=process_identity(pid,proc_root):
            return False
        health=health_reader()
        return (health.get('status')=='ready' and health.get('mode')=='openvla'
                and health.get('model_loaded') is True
                and (not require_sampling or health.get('action_sampling_available') is True))
    except (OSError,ValueError,KeyError,IndexError):
        return False


def record_identity(outputs,identity):
    outputs=Path(outputs)
    pid=int((outputs/'cloud_server.pid').read_text().strip())
    deadline=time.monotonic()+5
    while True:
        try:
            process=process_identity(pid)
            break
        except (OSError,ValueError,IndexError):
            if time.monotonic()>=deadline:raise
            time.sleep(.1)
    record=dict(identity=identity,process=process)
    temporary=outputs/'cloud_server_identity.json.tmp'
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    temporary.replace(outputs/'cloud_server_identity.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='/workspace/openvla-7b')
    parser.add_argument('--adapter')
    parser.add_argument('--outputs',type=Path,default=Path('outputs'))
    parser.add_argument('--record',action='store_true')
    parser.add_argument('--require-sampling',action='store_true')
    args=parser.parse_args()
    try:
        identity=requested_identity(args.base,args.adapter)
    except (OSError,ValueError):
        if args.record:raise
        return 1
    if args.record:
        record_identity(args.outputs,identity)
        return 0
    return 0 if reusable(args.outputs,identity,require_sampling=args.require_sampling) else 1


if __name__=='__main__':raise SystemExit(main())
