"""Copy a verified training bundle to the pod's attached persistent volume."""
import argparse
import json
from pathlib import Path
import subprocess
import re
import shlex

from cloud_session import parse_ssh, ssh_arguments


def remote_destination(dataset, override=None):
    destination = override or f'/workspace/panda-training/{Path(dataset).name}'
    if not re.fullmatch(r'/workspace/panda-training/[A-Za-z0-9][A-Za-z0-9_-]*', destination):
        raise ValueError('Remote dataset must be a named directory directly inside /workspace/panda-training')
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ssh',required=True,help='SSH over exposed TCP command from RunPod Connect')
    p.add_argument('--dataset',type=Path,default=Path('datasets/panda_pick_place/training_v1'))
    p.add_argument('--remote-dir',help='Defaults to /workspace/panda-training/<local dataset folder name>')
    args=p.parse_args()
    if not json.loads((args.dataset/'verification.json').read_text())['passed']:
        p.error('Dataset has not passed verification')
    host,port,key=parse_ssh(args.ssh)
    try:
        destination=remote_destination(args.dataset,args.remote_dir)
    except ValueError as exc:
        p.error(str(exc))
    ssh=ssh_arguments(host,port,key)
    # Refuse container-only /workspace or an existing destination; no overwrites.
    remote=f"mountpoint -q /workspace && test ! -e {shlex.quote(destination)} && mkdir -p /workspace/panda-training"
    subprocess.run(ssh+[remote],check=True)
    subprocess.run(['scp','-F','/dev/null','-o','IdentitiesOnly=yes','-o','PasswordAuthentication=no',
                    '-i',str(key),'-P',str(port),'-r',str(args.dataset.resolve()),host+':'+destination],check=True)
    subprocess.run(ssh+[f'cd {shlex.quote(destination)} && sha256sum -c SHA256SUMS'],check=True)
    print('Verified uploaded bundle:',destination)


if __name__=='__main__':main()
