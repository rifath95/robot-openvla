"""Fetch the pinned upstream package; do not install its conflicting dependency pins."""
import argparse
import json
from pathlib import Path
import subprocess


def fetch(destination):
    pin=json.loads((Path(__file__).parent/'upstream/PIN.json').read_text())
    path=Path(destination).resolve()
    if not path.exists():
        subprocess.run(['git','clone','--no-checkout',pin['repository'],str(path)],check=True)
        subprocess.run(['git','-C',str(path),'checkout','--detach',pin['revision']],check=True)
    actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if actual!=pin['revision']:raise ValueError('Upstream checkout is not the pinned revision')
    if (path/'vla-scripts/finetune.py').read_bytes()!=(Path(__file__).parent/'upstream/finetune.py').read_bytes():
        raise ValueError('Upstream script differs from pinned source')
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,default=Path('training/vendor/openvla'))
    print(fetch(p.parse_args().destination))
